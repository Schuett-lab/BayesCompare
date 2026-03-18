import torch
import numpy as np
from BayesCompare.distances import measure_dist
from numpy.testing import assert_allclose
from torch.testing import assert_close
import pytest

ALL_MEASURES = [
    "cka",
    "rsa_arccos",
    "rsa_cos",
    "rsa_corr",
    "rsa_rank",
    "gulp",
    "dist_corr",
    "jaccard",
    "procrustes",
]

lmbd = 0.001
k = 3


def generate_covs(
    N: int,
    dim: int,
    eig_min: float = 0.5,
    eig_max: float = 2.0,
    cov_dtype: np.dtype = np.float64,
    specific_eigs: int | None = None,
    scale: int = 1,
):
    rng = np.random.default_rng(0)
    covs = np.empty((N, dim, dim), dtype=cov_dtype)

    for i in range(N):
        A = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(A)

        eigs = rng.uniform(eig_min, eig_max, size=dim)

        if specific_eigs:
            eigs[0] = specific_eigs[i]

        covs[i] = scale * (Q @ np.diag(eigs) @ Q.T)

    return covs


@pytest.fixture()
def inputs():
    dim = 25

    # Large input test
    test_large_input_data = generate_covs(N=10, dim=dim, scale=1e7)

    # Small input test
    test_small_input_data = generate_covs(N=10, dim=dim, scale=1e-4)

    # One small one large input test
    test_large_and_small_input = [
        (A, B) for A, B in zip(test_large_input_data, test_small_input_data)
    ]

    return {
        "test_large_input_data": test_large_input_data,
        "test_small_input_data": test_small_input_data,
        "test_large_and_small_input": test_large_and_small_input,
    }


# Large scale input
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_large_scale_inputs(inputs, meas_name):
    output_np = []
    output_th = []

    if meas_name == "gulp" or meas_name == "jaccard" or meas_name == "procrustes":
        output_np.append(
            measure_dist(
                inputs["test_large_input_data"],
                meas_name=meas_name,
                show_progress=False,
                normalize=False,
                lmbd=lmbd,
                k=k,
            )
        )
        output_th.append(
            measure_dist(
                [torch.from_numpy(A) for A in inputs["test_large_input_data"]],
                meas_name=meas_name,
                show_progress=False,
                normalize=False,
                lmbd=lmbd,
                k=k,
            )
        )
    else:
        output_np.append(
            measure_dist(
                inputs["test_large_input_data"],
                meas_name=meas_name,
                show_progress=False,
                b=1 / 100,
            )
        )
        output_th.append(
            measure_dist(
                [torch.from_numpy(A) for A in inputs["test_large_input_data"]],
                meas_name=meas_name,
                show_progress=False,
                b=1 / 100,
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, output_th.numpy(), rtol=4e-2, atol=4e-2)
    if meas_name == "rsa_corr" or meas_name == "rsa_rank":
        assert np.all(
            abs(output_np) <= 1
        ), "RSA result NPArray cannot contain values greater than 1."
        assert torch.all(
            abs(output_th) <= 1
        ), "RSA result Torch Tensor cannot contain values greater than 1."
    elif meas_name == "rsa_cos":
        assert np.all(output_np <= 1) and np.all(
            output_np >= 0
        ), "RSA result NPArray cannot contain values greater than 1 or smaller than 0."
        assert torch.all(output_th <= 1) and torch.all(
            output_th >= 0
        ), "RSA result Torch Tensor cannot contain values greater than 1 or smaller than 0."
    elif meas_name == "rsa_arccos":
        assert np.all(output_np <= np.pi / 2) and np.all(
            output_np >= 0
        ), "RSA result NPArray cannot contain values greater than pi/2 or smaller than 0."
        assert torch.all(output_th <= torch.pi / 2) and torch.all(
            output_th >= 0
        ), "RSA result Torch Tensor cannot contain values greater than pi/2 or smaller than 0."
    else:
        assert np.all(output_np >= 0), "NPArray contains negative values"
        assert torch.all(output_th >= 0), "Torch Tensor contains negative values"


# # Small scale input
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_small_scale_inputs(inputs, meas_name):
    output_np = []
    output_th = []

    if meas_name == "gulp" or meas_name == "jaccard" or meas_name == "procrustes":
        output_np.append(
            measure_dist(
                inputs["test_small_input_data"],
                meas_name=meas_name,
                show_progress=False,
                normalize=False,
                lmbd=lmbd,
                k=k,
            )
        )
        output_th.append(
            measure_dist(
                [torch.from_numpy(A) for A in inputs["test_small_input_data"]],
                meas_name=meas_name,
                show_progress=False,
                normalize=False,
                lmbd=lmbd,
                k=k,
            )
        )
    else:
        output_np.append(
            measure_dist(
                inputs["test_small_input_data"],
                meas_name=meas_name,
                show_progress=False,
                b=1 / 100,
            )
        )
        output_th.append(
            measure_dist(
                [torch.from_numpy(A) for A in inputs["test_small_input_data"]],
                meas_name=meas_name,
                show_progress=False,
                b=1 / 100,
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, output_th.numpy(), rtol=4e-2, atol=4e-2)
    if meas_name == "rsa_corr" or meas_name == "rsa_rank":
        assert np.all(
            abs(output_np) <= 1
        ), "RSA result NPArray cannot contain values greater than 1."
        assert torch.all(
            abs(output_th) <= 1
        ), "RSA result Torch Tensor cannot contain values greater than 1."
    elif meas_name == "rsa_cos":
        assert np.all(output_np <= 1) and np.all(
            output_np >= 0
        ), "RSA result NPArray cannot contain values greater than 1 or smaller than 0."
        assert torch.all(output_th <= 1) and torch.all(
            output_th >= 0
        ), "RSA result Torch Tensor cannot contain values greater than 1 or smaller than 0."
    elif meas_name == "rsa_arccos":
        assert np.all(output_np <= np.pi / 2) and np.all(
            output_np >= 0
        ), "RSA result NPArray cannot contain values greater than pi/2 or smaller than 0."
        assert torch.all(output_th <= torch.pi / 2) and torch.all(
            output_th >= 0
        ), "RSA result Torch Tensor cannot contain values greater than pi/2 or smaller than 0."
    else:
        assert np.all(output_np >= 0), "NPArray contains negative values"
        assert torch.all(output_th >= 0), "Torch Tensor contains negative values"


# One large and one small scale input
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_large_and_small_scale_inputs(inputs, meas_name):
    output_np = []
    output_th = []

    for pair in inputs["test_large_and_small_input"]:
        if meas_name == "gulp" or meas_name == "jaccard" or meas_name == "procrustes":
            output_np.append(
                measure_dist(
                    pair,
                    meas_name=meas_name,
                    show_progress=False,
                    normalize=False,
                    lmbd=lmbd,
                    k=k,
                )
            )
            output_th.append(
                measure_dist(
                    [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
                    meas_name=meas_name,
                    show_progress=False,
                    normalize=False,
                    lmbd=lmbd,
                    k=k,
                )
            )
        else:
            output_np.append(
                measure_dist(
                    pair,
                    meas_name=meas_name,
                    show_progress=False,
                    b=1 / 100,
                )
            )
            output_th.append(
                measure_dist(
                    [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
                    meas_name=meas_name,
                    show_progress=False,
                    b=1 / 100,
                )
            )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, output_th.numpy(), rtol=4e-2, atol=4e-2)
    if meas_name == "rsa_corr" or meas_name == "rsa_rank":
        assert np.all(
            abs(output_np) <= 1
        ), "RSA result NPArray cannot contain values greater than 1."
        assert torch.all(
            abs(output_th) <= 1
        ), "RSA result Torch Tensor cannot contain values greater than 1."
    elif meas_name == "rsa_cos":
        assert np.all(output_np <= 1) and np.all(
            output_np >= 0
        ), "RSA result NPArray cannot contain values greater than 1 or smaller than 0."
        assert torch.all(output_th <= 1) and torch.all(
            output_th >= 0
        ), "RSA result Torch Tensor cannot contain values greater than 1 or smaller than 0."
    elif meas_name == "rsa_arccos":
        assert np.all(output_np <= np.pi / 2) and np.all(
            output_np >= 0
        ), "RSA result NPArray cannot contain values greater than pi/2 or smaller than 0."
        assert torch.all(output_th <= torch.pi / 2) and torch.all(
            output_th >= 0
        ), "RSA result Torch Tensor cannot contain values greater than pi/2 or smaller than 0."
    else:
        assert np.all(output_np >= 0), "NPArray contains negative values"
        assert torch.all(output_th >= 0), "Torch Tensor contains negative values"
