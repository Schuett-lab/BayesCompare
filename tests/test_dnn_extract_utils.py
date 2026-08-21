import numpy as np
import torch
import os
import pytest
import h5py
from functools import partial
from BayesCompare.dnn_extract_utils import (
    check_act_dims,
    get_cov,
    create_covs_dict,
    get_model_device_dtype,
    make_mock_input,
    check_hdf_exists_save_acts,
)
import torchlens as tl
import torch.nn as nn
from numpy.testing import assert_allclose, assert_array_equal
from torch.testing import assert_close


class TestCheckActDims:
    input_dimensions = [2, 3, 4, 5, 6]
    input_types = ["slim", "fat", "square"]
    mtx_dtypes = ["numpy", "torch"]

    valid_cases = []

    for ndim in input_dimensions:
        for shape_type in input_types:
            for mtx_dtype in mtx_dtypes:
                for N_axis in range(ndim):
                    N = 7

                    if shape_type == "slim":
                        other_dims = list(range(2, ndim + 1))
                    elif shape_type == "fat":
                        other_dims = list(range(10, 10 + ndim - 1))
                    else:
                        other_dims = [4] * (ndim - 1)

                    shape = other_dims.copy()
                    shape.insert(N_axis, N)

                    valid_cases.append(
                        pytest.param(
                            tuple(shape),
                            N,
                            N_axis,
                            mtx_dtype,
                            id=f"{mtx_dtype}-{ndim}D-{shape_type}-N_axis={N_axis}",
                        )
                    )

    @staticmethod
    def make_act(shape, mtx_dtype):
        if mtx_dtype == "numpy":
            return np.random.randn(*shape).astype(np.float32)
        return torch.randn(*shape, dtype=torch.float32)

    @pytest.mark.parametrize(
        "shape, N, N_axis, mtx_dtype",
        valid_cases,
    )
    def test_check_act_dims_valid(self, shape, N, N_axis, mtx_dtype):
        act = self.make_act(shape, mtx_dtype)
        result = check_act_dims(
            act=act,
            N=N,
            module_name=mtx_dtype,
        )
        # result exists
        assert result is not None
        # N has been moved to the first dimension
        assert result.shape[0] == N
        # rank is unchanged
        assert result.ndim == act.ndim
        # number of elements is unchanged
        assert (
            result.size == act.size
            if mtx_dtype == "numpy"
            else result.numel() == act.numel()
        )

    @pytest.mark.parametrize(
        "shape, N, N_axis, mtx_dtype",
        valid_cases,
    )
    def test_check_act_dims_valid(self, shape, N, N_axis, mtx_dtype):
        act = self.make_act(shape, mtx_dtype)

        result = check_act_dims(
            act=act,
            N=N,
            module_name=mtx_dtype,
        )
        assert result is not None

        if N_axis == 0:
            expected = act
        else:
            axes = list(range(act.ndim))
            axes.insert(0, axes.pop(N_axis))

            if mtx_dtype == "numpy":
                expected = np.transpose(act, axes)
            else:
                expected = act.permute(*axes)

        assert result.shape == expected.shape

        if mtx_dtype == "numpy":
            assert_array_equal(result, expected)
        else:
            assert_close(result, expected)

    @pytest.mark.parametrize("mtx_dtypes", mtx_dtypes)
    def test_check_act_dims_N_already_first(self, mtx_dtypes):
        N = 7
        act = self.make_act((N, 3, 4, 5), mtx_dtypes)

        result = check_act_dims(
            act,
            N=N,
            module_name=mtx_dtypes,
        )
        assert result is act

    @pytest.mark.parametrize("mtx_dtypes", mtx_dtypes)
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6),
            (2, 3, 4, 5, 6, 8),
        ],
    )
    def test_check_act_dims_N_missing(self, mtx_dtypes, shape):
        act = self.make_act(shape, mtx_dtypes)
        N = 7

        with pytest.warns(
            UserWarning, match="This layer does not have a number of images dimension."
        ):
            result = check_act_dims(
                act,
                N=N,
                module_name=mtx_dtypes,
            )
        assert result is None

    @pytest.mark.parametrize("mtx_dtypes", mtx_dtypes)
    def test_check_act_dims_1d_raises(self, mtx_dtypes):
        N = 7
        act = self.make_act((N,), mtx_dtypes)

        with pytest.raises(
            ValueError,
            match="Activations cannot be one dimensional arrays! Please check the input activations.",
        ):
            check_act_dims(
                act,
                N=N,
                module_name=mtx_dtypes,
            )


class TestGetCov:
    activations = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    expected = [[5.0, 10.0, 15.0], [10.0, 20.0, 30.0], [15.0, 30.0, 45.0]]

    def test_get_cov_numpy(self):
        assert_allclose(get_cov(np.array(self.activations)), np.array(self.expected))

    def test_get_cov_tensor(self):
        assert_close(
            get_cov(torch.tensor(self.activations, dtype=torch.float64)),
            torch.tensor(self.expected, dtype=torch.float64),
        )

    def test_get_cov_N_matches_first_dimension(self):
        assert_allclose(
            get_cov(np.array(self.activations), N=3), np.array(self.expected)
        )

    def test_get_cov_reorders_dimension_with_N(self):
        assert_allclose(
            get_cov(np.array(self.activations).T, N=3), np.array(self.expected).T
        )

    def test_get_cov_tensor_reorders_dimension_with_N(self):
        assert_close(
            get_cov(torch.tensor(self.activations, dtype=torch.float64), N=3),
            torch.tensor(self.expected, dtype=torch.float64),
        )

    def test_get_cov_tensor_not_detached_by_default(self):
        result = get_cov(
            torch.tensor(self.activations, dtype=torch.float64, requires_grad=True)
        )
        assert result.requires_grad

    def test_get_cov_tensor_detached(self):
        result = get_cov(
            torch.tensor(self.activations, dtype=torch.float64, requires_grad=True),
            detach=True,
        )
        assert not result.requires_grad
        assert result.grad_fn is None

    def test_get_cov_tensor_allows_gradient(self):
        acts = torch.tensor(self.activations, dtype=torch.float64, requires_grad=True)
        result = get_cov(acts)
        result.sum().backward()
        assert acts.grad is not None

    @pytest.mark.parametrize(
        "acts",
        [
            [[1.0, 2.0], [2.0, 3.0]],
            "invalid",
            42,
        ],
    )
    def test_get_cov_invalid_type(self, acts):
        with pytest.raises(NotImplementedError):
            get_cov(acts)

    def test_get_cov_numpy_returns_numpy(self):
        assert isinstance(get_cov(np.array(self.activations)), np.ndarray)

    def test_get_cov_tensor_returns_tensor(self):
        assert isinstance(
            get_cov(torch.tensor(self.activations, dtype=torch.float64)), torch.Tensor
        )


class TestCreateCovsDict:
    input_dim = 25
    output_dim = 10
    N = 5
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    width = 32
    linear_model = nn.Linear(input_dim, output_dim)
    linear_inputs = torch.rand(N, input_dim)
    cnn = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        ),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 30 * 30, output_dim),
    )
    conv_inputs = torch.randn(N, in_channels, width, width)

    @pytest.mark.parametrize(
        "compute_covs, flatten_acts",
        [
            (compute_covs, flatten_acts)
            for compute_covs in [True, False]
            for flatten_acts in [True, False]
        ],
    )
    def test_create_covs_dict_fcn(self, compute_covs, flatten_acts):
        if compute_covs:
            out_transform = partial(get_cov, N=self.N)
        else:
            out_transform = None

        out = tl.trace(
            self.linear_model,
            self.linear_inputs,
            layers_to_save=["linear_1"],
            out_transform=out_transform,
            random_seed=20,
        )
        covs_dict = create_covs_dict(
            out,
            ["linear_1"],
            compute_covs=compute_covs,
            flatten_acts=flatten_acts,
        )
        assert isinstance(covs_dict["linear_1"], torch.Tensor)
        if compute_covs == True:
            assert covs_dict["linear_1"].shape == (self.N, self.N)
        if compute_covs == False:
            assert covs_dict["linear_1"].shape == (self.N, self.output_dim)

    @pytest.mark.parametrize(
        "compute_covs, flatten_acts, save_network_output",
        [
            (compute_covs, flatten_acts, save_network_output)
            for compute_covs in [True, False]
            for flatten_acts in [True, False]
            for save_network_output in [True, False]
        ],
    )
    def test_create_covs_dict_cnn(
        self, compute_covs, flatten_acts, save_network_output
    ):
        if compute_covs:
            out_transform = partial(get_cov, N=self.N)
        else:
            out_transform = None

        out = tl.trace(
            self.cnn,
            self.conv_inputs,
            layers_to_save=["conv2d_1"],
            out_transform=out_transform,
            random_seed=20,
        )
        covs_dict = create_covs_dict(
            out,
            ["conv2d_1"],
            compute_covs=compute_covs,
            flatten_acts=flatten_acts,
            save_network_output=save_network_output,
        )
        assert "conv2d_1" in covs_dict
        assert isinstance(covs_dict["conv2d_1"], torch.Tensor)

        if compute_covs:
            assert covs_dict["conv2d_1"].shape == (self.N, self.N)
        if not (compute_covs) and flatten_acts:
            assert covs_dict["conv2d_1"].shape == (
                self.N,
                self.out_channels
                * (self.width - self.kernel_size + 1)
                * (self.width - self.kernel_size + 1),
            )
        if not (compute_covs) and not (flatten_acts):
            assert covs_dict["conv2d_1"].shape == (
                self.N,
                self.out_channels,
                self.width - self.kernel_size + 1,
                self.width - self.kernel_size + 1,
            )

        if save_network_output:
            assert "output_1" in covs_dict
            assert covs_dict["output_1"].shape == (self.N, self.output_dim)
            assert isinstance(covs_dict["output_1"], torch.Tensor)
        else:
            assert "output_1" not in covs_dict


class TestGetModelDeviceDtype:
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ],
    )
    def test_cpu_models(self, dtype):
        model = nn.Linear(5, 10, dtype=dtype)
        res_device, res_dtype = get_model_device_dtype(model)
        assert res_device == torch.device("cpu")
        assert res_dtype is dtype

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ],
    )
    def test_gpu_models(self, dtype):
        model = nn.Linear(5, 10, device=torch.device("cuda"), dtype=dtype)
        res_device, res_dtype = get_model_device_dtype(model)
        assert res_device == torch.device("cuda", index=0)
        assert res_dtype is dtype

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ],
    )
    def test_no_param_models(self, dtype):
        model = nn.Identity()
        res_device, res_dtype = get_model_device_dtype(model)
        assert res_device == torch.device("cpu")
        assert res_dtype is torch.float32


class TestMakeMockInput:
    input_dim = 25
    output_dim = 10
    N = 5
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    width = 32
    linear = nn.Sequential(nn.Linear(input_dim, output_dim))
    cnn_1 = nn.Sequential(
        nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        ),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 30 * 30, output_dim),
    )
    cnn_2 = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        ),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 30 * 30, output_dim),
    )
    cnn_3 = nn.Sequential(
        nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        ),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 30 * 30, output_dim),
    )
    undefined = nn.Embedding(
        num_embeddings=10,
        embedding_dim=16,
    )

    @pytest.mark.parametrize(
        "model, model_name, input_shape, batch_size",
        [
            (
                model,
                model_name,
                input_shape,
                batch_size,
            )
            for model, model_name in [
                (linear, "linear"),
                (cnn_1, "cnn_1"),
                (cnn_2, "cnn_2"),
                (cnn_3, "cnn_3"),
                (undefined, "unidentified"),
            ]
            for input_shape in [None, (5, 65, 42)]
            for batch_size in [None, 4]
        ],
    )
    def test_different_architectures(
        self,
        model,
        model_name,
        input_shape,
        batch_size,
    ):
        if not (input_shape) and isinstance(model, nn.Embedding):
            with pytest.raises(
                NotImplementedError,
                match="Could not infer a valid input shape. Please provide input_shape explicitly.",
            ):
                output = make_mock_input(model, input_shape, batch_size)

        elif not (input_shape) and not (batch_size) and not (model_name == "undefined"):
            output = make_mock_input(model, input_shape, batch_size)
            if model_name == "linear":
                assert output.shape == (
                    1,
                    self.input_dim,
                )
            elif model_name == "ccn_1":
                assert output.shape == (
                    1,
                    self.in_channels,
                    224,
                )
            elif model_name == "ccn_2":
                assert output.shape == (
                    1,
                    self.in_channels,
                    224,
                    224,
                )
            elif model_name == "ccn_3":
                assert output.shape == (
                    1,
                    self.in_channels,
                    16,
                    224,
                    224,
                )

        elif not (input_shape) and model_name == "undefined":
            with pytest.raises(
                NotImplementedError,
                match="Could not infer a valid input shape. Please provide input_shape explicitly.",
            ):
                output = make_mock_input(model, input_shape, batch_size)

        elif not (input_shape) and batch_size and not (model_name == "undefined"):
            output = make_mock_input(model, input_shape, batch_size)
            if model_name == "linear":
                assert output.shape == (
                    batch_size,
                    self.input_dim,
                )
            elif model_name == "ccn_1":
                assert output.shape == (
                    batch_size,
                    self.in_channels,
                    224,
                )
            elif model_name == "ccn_2":
                assert output.shape == (
                    batch_size,
                    self.in_channels,
                    224,
                    224,
                )
            elif model_name == "ccn_3":
                assert output.shape == (
                    batch_size,
                    self.in_channels,
                    16,
                    224,
                    224,
                )

        elif input_shape:
            output = make_mock_input(model, input_shape, batch_size)
            assert output.shape == input_shape


class TestCheckHdfExistsSaveActs:
    hdf_dir = os.path.join(
        os.getcwd(),
        "tests/sample_data/test_input_activation_hdf.hdf5",
    )
    mock_act = np.zeros((5, 3))
    dset_name = "tests_set"

    def test_non_existing_hdf(self):
        if os.path.exists(self.hdf_dir):
            os.remove(self.hdf_dir)
        check_hdf_exists_save_acts(
            self.hdf_dir, self.mock_act, 10, self.dset_name, True
        )
        with h5py.File(self.hdf_dir, "r") as f:
            dset = f[self.dset_name]
            assert dset.shape == (5, 3)
        with pytest.raises(FileExistsError):
            check_hdf_exists_save_acts(
                self.hdf_dir, self.mock_act, 5, self.dset_name, True
            )

    def test_existing_hdf(self):
        check_hdf_exists_save_acts(
            self.hdf_dir, self.mock_act, 10, self.dset_name, False
        )
        assert os.path.exists(self.hdf_dir)
        with h5py.File(self.hdf_dir, "r") as f:
            dset = f[self.dset_name]
            assert dset.shape == (10, 3)
