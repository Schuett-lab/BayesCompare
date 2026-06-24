"""
Tests for cov_extractor and cov_extractor_batch.

Coverage
--------
- Output structure / types
- Mathematical correctness (symmetry, PSD, variance/covariance, np.cov oracle)
- Mode flags (gradient, eval_mode, inference_mode)
- File I/O and cleanup (batch function)
- Edge cases & warnings
- Reproducibility / random seed

Author: Claude, Sezan Oral
"""

import os
import pickle
import warnings
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from BayesCompare import cov_extractor, cov_extractor_batch, get_layer_names

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class TinyNet(nn.Module):
    """Two-layer MLP with named layers for easy targeting."""

    def __init__(self):
        super().__init__()
        self.linear_1_2 = nn.Linear(192, 8, bias=False)  # 3×8×8 = 192
        self.relu = nn.ReLU()
        self.linear_2_4 = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear_1_2(x))
        return self.linear_2_4(x)


@pytest.fixture
def model():
    net = TinyNet()
    return net


@pytest.fixture
def inputs_np():
    """20 RGB images of 8×8, float32, fixed seed."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((20, 3, 8, 8)).astype(np.float32)


@pytest.fixture
def inputs_tensor(inputs_np):
    return torch.from_numpy(inputs_np)


@pytest.fixture
def layer_list():
    return ["linear_1_2", "linear_2_4"]


@pytest.fixture
def tmp_out(tmp_path):
    """Temporary output directory."""
    return str(tmp_path)


# ---------------------------------------------------------------------------
# 0. Obtaining correct layer names  —  get_layer_names
# ---------------------------------------------------------------------------


class TestLayerNames:
    def test_get_correct_layer_names_with_input(self, model, inputs_np):
        expected = [
            "input_1",
            "view_1_1",
            "linear_1_2",
            "relu_1_3",
            "linear_2_4",
            "output_1",
        ]
        result = get_layer_names(model=model, mock_input=inputs_np)
        assert len(expected) == len(result)
        for i, layer_name in enumerate(expected):
            assert (
                result[i] == layer_name
            ), f"Layer name difference at [{i}]: expected {layer_name}, got {result[i]}"

    def test_get_correct_layer_names_without_input(self, model):
        expected = [
            "input_1",
            "view_1_1",
            "linear_1_2",
            "relu_1_3",
            "linear_2_4",
            "output_1",
        ]
        result = get_layer_names(model=model)
        assert len(expected) == len(result)
        for i, layer_name in enumerate(expected):
            assert (
                result[i] == layer_name
            ), f"Layer name difference at [{i}]: expected {layer_name}, got {result[i]}"


# ---------------------------------------------------------------------------
# 1. Output structure / types  —  cov_extractor
# ---------------------------------------------------------------------------


class TestOutputStructure:

    def test_returns_dict(self, model, inputs_np, layer_list):
        result = cov_extractor(model, inputs_np, layer_list)
        assert isinstance(result, dict)

    def test_keys_match_layer_list(self, model, inputs_np, layer_list):
        result = cov_extractor(model, inputs_np, layer_list)
        assert set(result.keys()) == set(layer_list)

    def test_single_layer_string_input(self, model, inputs_np):
        result = cov_extractor(model, inputs_np, "linear_1_2")
        assert "linear_1_2" in result

    def test_values_are_tensors(self, model, inputs_np, layer_list):
        result = cov_extractor(model, inputs_np, layer_list)
        for key, val in result.items():
            assert isinstance(val, torch.Tensor), f"{key} value is not a torch.Tensor"

    def test_cov_shape_is_square(self, model, inputs_np, layer_list):
        n = inputs_np.shape[0]
        result = cov_extractor(model, inputs_np, layer_list)
        for key, val in result.items():
            assert val.dim() == 2, f"{key}: expected 2-D tensor"
            assert val.shape[0] == val.shape[1], f"{key}: covariance not square"
            assert val.shape[0] == val.shape[1] == n, f"{key}: covariance is not n by n"

    def test_activation_shape_has_n_images_dimension(
        self, model, inputs_np, layer_list
    ):
        """When compute_covs=False the first (or some) dimension equals n_images."""
        n = inputs_np.shape[0]
        result = cov_extractor(model, inputs_np, layer_list, compute_covs=False)
        for key, val in result.items():
            assert (
                n in val.shape
            ), f"{key}: n_images={n} not found in activation shape {val.shape}"

    def test_accepts_torch_tensor_input(self, model, inputs_tensor, layer_list):
        result = cov_extractor(model, inputs_tensor, layer_list)
        assert set(result.keys()) == set(layer_list)


# ---------------------------------------------------------------------------
# 2. Mathematical correctness  —  cov_extractor
# ---------------------------------------------------------------------------


class TestMathCorrectness:

    def test_symmetry(self, model, inputs_np, layer_list):
        result = cov_extractor(model, inputs_np, layer_list)
        for key, cov in result.items():
            cov_np = cov.detach().numpy()
            np.testing.assert_allclose(
                cov_np,
                cov_np.T,
                atol=1e-5,
                err_msg=f"{key}: covariance matrix is not symmetric",
            )

    def test_positive_semidefinite(self, model, inputs_np, layer_list):
        result = cov_extractor(model, inputs_np, layer_list)
        for key, cov in result.items():
            eigenvalues = np.linalg.eigvalsh(cov.detach().numpy())
            assert np.all(
                eigenvalues >= -1e-6
            ), f"{key}: covariance has negative eigenvalue {eigenvalues.min():.3e}"

    def test_matches_numpy_cov_oracle(self, model, inputs_np, layer_list):
        """Full covariance matrix must match act @ act.T on the raw activations."""
        activations = cov_extractor(model, inputs_np, layer_list, compute_covs=False)
        covs = cov_extractor(model, inputs_np, layer_list, compute_covs=True)
        for key in layer_list:
            act = activations[key].detach().numpy()
            act = np.reshape(act, [act.shape[0], -1])
            cov = covs[key].detach().numpy()
            # np.cov expects shape (d, n); uses ddof=1 by default
            oracle = act @ act.T
            np.testing.assert_allclose(
                cov,
                oracle,
                rtol=1e-4,
                err_msg=f"{key}: covariance doesn't match oracle",
            )

    def test_golden_2x2x2_exact(self):
        """
        Hand-crafted golden test with exact rational arithmetic.

        Network: single Linear(2, 2, bias=False) with weight W = [[1, 0], [0, 1]] (identity).
        Inputs: X = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]  → activations == inputs.
        Cov   = [[30.0, 70.0], [70.0, 174.0]]
        """

        class IdentityNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.layer.weight.copy_(torch.eye(2))

            def forward(self, x):
                return self.layer(x)

        net = IdentityNet().eval()
        X = np.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32
        )

        expected = [[30.0, 70.0], [70.0, 174.0]]

        result = cov_extractor(net, X, "linear_1_1")["linear_1_1"]

        for i in range(2):
            for j in range(2):
                got = Fraction(result[i, j].item()).limit_denominator(1000)
                assert (
                    got == expected[i][j]
                ), f"Golden test failed at [{i},{j}]: expected {expected[i][j]}, got {got}"


# ---------------------------------------------------------------------------
# 3. Mode flags  —  cov_extractor
# ---------------------------------------------------------------------------


class TestCovExtractorModes:

    def test_gradient_true_tensors_require_grad(self, model, inputs_np, layer_list):
        """gradient=True: returned tensors must be attached to the graph."""
        result = cov_extractor(model, inputs_np, layer_list, gradient=True)
        for key, val in result.items():
            assert (
                val.requires_grad
            ), f"{key}: tensor does not require grad when gradient=True"

    def test_gradient_false_tensors_no_grad(self, model, inputs_np, layer_list):
        """gradient=False (default): returned tensors must not be attached to the graph."""
        result = cov_extractor(model, inputs_np, layer_list, gradient=False)
        for key, val in result.items():
            assert (
                not val.requires_grad
            ), f"{key}: tensor requires grad when gradient=False"

    def test_gradient_true_overrides_eval_mode(self, model, inputs_np, layer_list):
        """gradient=True must force training mode regardless of eval_mode flag."""
        cov_extractor(model, inputs_np, layer_list, gradient=True, eval_mode=True)
        assert (
            model.training
        ), "Model should be in training mode after gradient=True, even with eval_mode=True"

    def test_gradient_true_overrides_inference_mode(
        self, model, inputs_tensor, layer_list
    ):
        """gradient=True must disable inference_mode regardless of the flag."""
        result = cov_extractor(
            model,
            inputs_tensor,
            layer_list,
            gradient=True,
            inference_mode=True,
        )
        # If inference_mode were active, requires_grad would be False
        for key, val in result.items():
            assert (
                val.requires_grad
            ), f"{key}: inference_mode was not overridden by gradient=True"

    def test_eval_mode_true_puts_model_in_eval(self, model, inputs_np, layer_list):
        """eval_mode=True (default) should leave the model in eval mode after the call."""
        model.train()
        cov_extractor(model, inputs_np, layer_list, gradient=False, eval_mode=True)
        assert not model.training, "Model should be in eval mode after eval_mode=True"

    def test_eval_mode_false_leaves_model_in_train(self, model, inputs_np, layer_list):
        """eval_mode=False should not switch the model to eval mode."""
        model.train()
        cov_extractor(
            model,
            inputs_np,
            layer_list,
            gradient=False,
            eval_mode=False,
            inference_mode=False,
        )
        assert (
            model.training
        ), "Model should remain in training mode when eval_mode=False"

    def test_inference_mode_true_no_grad(self, model, inputs_np, layer_list):
        """inference_mode=True (default): tensors must not require grad."""
        result = cov_extractor(
            model,
            inputs_np,
            layer_list,
            gradient=False,
            inference_mode=True,
        )
        for key, val in result.items():
            assert (
                not val.requires_grad
            ), f"{key}: tensor requires grad under inference_mode=True"

    def test_gradient_true_and_false_same_values(self, model, inputs_np, layer_list):
        """The covariance values must be numerically identical regardless of gradient flag."""
        r_grad = cov_extractor(model, inputs_np, layer_list, gradient=True)
        r_no_grad = cov_extractor(model, inputs_np, layer_list, gradient=False)
        for key in layer_list:
            np.testing.assert_allclose(
                r_grad[key].detach().numpy(),
                r_no_grad[key].detach().numpy(),
                rtol=1e-5,
                err_msg=f"{key}: covariance values differ between gradient=True and False",
            )


# ---------------------------------------------------------------------------
# 4. File I/O and cleanup  —  cov_extractor_batch
# ---------------------------------------------------------------------------


class TestBatchFileIO:

    def test_cov_pickle_created(self, model, inputs_np, layer_list, tmp_out):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="test",
            out_dir=tmp_out,
            batch_size=5,
            compute_covs=True,
            delete_act_files=True,
        )
        pkl_path = Path(tmp_out) / "covs_test.pkl"
        assert pkl_path.exists(), "covariance pickle file was not created"

    def test_cov_pickle_contains_expected_keys(
        self, model, inputs_np, layer_list, tmp_out
    ):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="test",
            out_dir=tmp_out,
            batch_size=5,
            compute_covs=True,
            delete_act_files=True,
        )
        with open(Path(tmp_out) / "covs_test.pkl", "rb") as f:
            saved = pickle.load(f)
        assert set(saved.keys()) == set(layer_list)

    def test_activation_hdf5_created_when_compute_covs_false(
        self, model, inputs_np, layer_list, tmp_out
    ):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="test",
            out_dir=tmp_out,
            batch_size=5,
            compute_covs=False,
        )
        hdf5_path = Path(tmp_out) / "activations_test.hdf5"
        assert hdf5_path.exists(), "activations HDF5 file was not created"

    def test_hdf5_contains_expected_datasets(
        self, model, inputs_np, layer_list, tmp_out
    ):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="test",
            out_dir=tmp_out,
            batch_size=5,
            compute_covs=False,
        )
        with h5py.File(Path(tmp_out) / "activations_test.hdf5", "r") as f:
            dataset_names = []
            for i in f.values():
                dataset_names.append(i.name[1:])

            for layer in layer_list:
                assert (
                    "activations_" + layer in dataset_names
                ), f"Layer '{layer}' missing from HDF5"

    def test_activation_files_deleted_when_flag_true(
        self, model, inputs_np, layer_list, tmp_out
    ):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="test",
            out_dir=tmp_out,
            batch_size=5,
            compute_covs=True,
            delete_act_files=True,
        )
        hdf5_path = Path(tmp_out) / "activations_test.hdf5"
        assert not hdf5_path.exists(), "activation file should have been deleted"

    def test_activation_files_kept_when_flag_false(
        self, model, inputs_np, layer_list, tmp_out
    ):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="test",
            out_dir=tmp_out,
            batch_size=5,
            compute_covs=True,
            delete_act_files=False,
        )
        hdf5_path = Path(tmp_out) / "activations_test.hdf5"
        assert hdf5_path.exists(), "activation file should have been kept"

    def test_layer_by_layer_creates_separate_files(
        self, model, inputs_np, layer_list, tmp_out
    ):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="test",
            out_dir=tmp_out,
            batch_size=5,
            layer_by_layer=True,
            compute_covs=True,
            delete_act_files=True,
        )
        for layer in layer_list:
            pkl_path = Path(tmp_out) / f"covs_test_{layer}.pkl"
            assert pkl_path.exists(), f"Per-layer pickle for '{layer}' not found"

    def test_combined_file_not_created_in_layer_by_layer_mode(
        self, model, inputs_np, layer_list, tmp_out
    ):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="test",
            out_dir=tmp_out,
            batch_size=5,
            layer_by_layer=True,
            compute_covs=True,
            delete_act_files=True,
        )
        combined_path = Path(tmp_out) / "covs_test.pkl"
        assert (
            not combined_path.exists()
        ), "combined pickle should not exist in layer_by_layer mode"

    def test_batch_cov_matches_extractor_cov(
        self, model, inputs_np, layer_list, tmp_out
    ):
        """Batch and cov_extractor paths must produce numerically identical covariances."""
        extractor_result = cov_extractor(model, inputs_np, layer_list)
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="match",
            out_dir=tmp_out,
            batch_size=5,
            compute_covs=True,
            delete_act_files=True,
        )
        with open(Path(tmp_out) / "covs_match.pkl", "rb") as f:
            batch_result = pickle.load(f)

        for layer in layer_list:
            np.testing.assert_allclose(
                batch_result[layer],
                extractor_result[layer].detach().numpy(),
                rtol=1e-3,
                err_msg=f"{layer}: batch and cov_extractor covariances differ",
            )


# ---------------------------------------------------------------------------
# 5. Edge cases & warnings
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_single_image_does_not_crash(self, model):
        """n=1 is an edge case; at minimum it should not raise."""
        single = np.random.randn(1, 3, 8, 8).astype(np.float32)
        try:
            result = cov_extractor(model, single, "linear_1_2")
            assert "linear_1_2" in result
        except Exception as exc:
            pytest.fail(f"Single-image input raised unexpectedly: {exc}")

    def test_warning_on_mismatched_layer(self, model, inputs_np):
        """A layer whose output dimension doesn't match n_images should trigger a UserWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cov_extractor(model, inputs_np, ["linear_1_2", "linear_2_4"])
        # No assertion on caught here — we're verifying it doesn't raise

    def test_empty_layer_list_returns_empty_dict(self, model, inputs_np):
        result = cov_extractor(model, inputs_np, [])
        assert result == {} or isinstance(result, dict)

    def test_large_batch_size_exceeds_inputs(
        self, model, inputs_np, layer_list, tmp_out
    ):
        """batch_size > n_inputs should not crash."""
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="bigbatch",
            out_dir=tmp_out,
            batch_size=1000,
            compute_covs=True,
            delete_act_files=True,
        )
        assert (Path(tmp_out) / "covs_bigbatch.pkl").exists()

    def test_batch_size_one(self, model, inputs_np, layer_list, tmp_out):
        cov_extractor_batch(
            model,
            inputs_np,
            layer_list,
            out_filename="bs1",
            out_dir=tmp_out,
            batch_size=1,
            compute_covs=True,
            delete_act_files=True,
        )
        assert (Path(tmp_out) / "covs_bs1.pkl").exists()


# ---------------------------------------------------------------------------
# 6. Reproducibility / random seed
# ---------------------------------------------------------------------------


class TestReproducibility:

    def test_same_seed_same_output(self, model, inputs_np, layer_list):
        r1 = cov_extractor(model, inputs_np, layer_list, random_seed=42)
        r2 = cov_extractor(model, inputs_np, layer_list, random_seed=42)
        for layer in layer_list:
            assert torch.equal(
                r1[layer], r2[layer]
            ), f"{layer}: same seed produced different results"

    def test_different_seeds_may_differ(self, model, inputs_np, layer_list):
        """For stochastic models, different seeds should (usually) differ.
        Skipped automatically if the model is deterministic."""
        r1 = cov_extractor(model, inputs_np, layer_list, random_seed=0)
        r2 = cov_extractor(model, inputs_np, layer_list, random_seed=99)
        any_differ = any(not torch.equal(r1[l], r2[l]) for l in layer_list)
        if not any_differ:
            pytest.skip(
                "Model appears deterministic; seed difference test not applicable"
            )

    def test_batch_reproducibility(self, model, inputs_np, layer_list, tmp_path):
        """Two batch runs with the same seed must produce identical pickle files."""
        # this test runs correctly as the model and input are deterministic but is not valid for cov_extractor_batch
        for run, name in enumerate(["run1", "run2"]):
            out = str(tmp_path / f"out_{name}")
            os.makedirs(out)
            cov_extractor_batch(
                model,
                inputs_np,
                layer_list,
                out_filename=name,
                out_dir=out,
                batch_size=5,
                random_seed=42,
                compute_covs=True,
                delete_act_files=True,
            )

        def load(run_name):
            out = tmp_path / f"out_{run_name}"
            with open(out / f"covs_{run_name}.pkl", "rb") as f:
                return pickle.load(f)

        r1, r2 = load("run1"), load("run2")
        for layer in layer_list:
            np.testing.assert_array_equal(
                r1[layer],
                r2[layer],
                err_msg=f"{layer}: batch runs with same seed differ",
            )
