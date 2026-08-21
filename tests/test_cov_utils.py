import numpy as np
import torch
import pathlib
import os
import glob
import re
import pytest
from BayesCompare.cov_utils import (
    trace_norm,
    cov_sigma,
    cov_sigma_N,
    trace_norm_N,
    cov_trace_norm_sigma_N,
    check_cov_normalized,
    check_cov_symmetry,
    check_input_format,
    check_and_change_input_format,
)
from numpy.testing import assert_allclose
from torch.testing import assert_close


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
    sample_path = os.path.join(os.getcwd(), "tests/sample_data")
    input_args = {}
    for input_file in ["utils_covs_np", "utils_covs_th"]:
        filename = glob.glob(os.path.join(sample_path, f"test_{input_file}*"))[0]
        file_ext = pathlib.Path(filename).suffix

        if file_ext == ".np" or file_ext == ".npy" or file_ext == ".npz":
            sample_file = np.load(filename)
        elif file_ext == ".pt":
            sample_file = torch.load(filename)
        else:
            return NotImplementedError("Input file type not implemented for loading")

        input_args[input_file] = sample_file

    return input_args


def test_trace_norm():
    dim = 6
    input_np = generate_covs(N=1, dim=dim, scale=5)
    input_th = torch.from_numpy(input_np)
    assert_close(
        np.trace(trace_norm(input_np[0, :, :])), float(dim), rtol=1e-6, atol=1e-6
    )
    assert_close(
        torch.trace(trace_norm(input_th[0, :, :])),
        torch.tensor(dim, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_cov_sigma(inputs, weighted):
    norm_cov_np = trace_norm(inputs["utils_covs_np"][0])
    norm_cov_th = trace_norm(torch.Tensor(inputs["utils_covs_np"][0]))
    noise_var = 0.5
    if weighted:
        weights = np.random.randint(1, 4, size=norm_cov_np.shape[0])
        noise_weights = np.diag(1 / weights)
    else:
        weights = None
        noise_weights = np.eye(norm_cov_np.shape[0])

    expected_np = (1 - noise_var) * norm_cov_np + noise_var * noise_weights
    result_np = cov_sigma(norm_cov_np, noise_var=noise_var, img_weights=weights)
    expected_th = (1 - noise_var) * norm_cov_th + noise_var * torch.Tensor(
        noise_weights
    )
    result_th = cov_sigma(norm_cov_th, noise_var=noise_var, img_weights=weights)

    assert_allclose(result_np, expected_np, rtol=1e-4, atol=1e-4)
    assert_allclose(result_th, expected_th, rtol=1e-4, atol=1e-4)
    assert_allclose(torch.Tensor(result_np), result_th, rtol=1e-4, atol=1e-4)
    assert_allclose(torch.Tensor(expected_np), expected_th, rtol=1e-4, atol=1e-4)


def test_trace_norm_N(inputs):
    single_output_np = []
    single_output_th = []
    single_output_th_np = []

    for cov in inputs["utils_covs_np"]:
        single_output_np.append(trace_norm(cov))
    for cov in inputs["utils_covs_th"]:
        single_output_th.append(trace_norm(cov))
    for cov in inputs["utils_covs_np"]:
        single_output_th_np.append(trace_norm(torch.Tensor(cov)))

    single_output_np = np.array(single_output_np)
    single_output_th = torch.stack(single_output_th, dim=0)
    single_output_th_np = torch.stack(single_output_th_np, dim=0)

    batch_output_np = trace_norm_N(inputs["utils_covs_np"])
    batch_output_th = trace_norm_N(inputs["utils_covs_th"])
    batch_output_th_np = trace_norm_N(torch.Tensor(inputs["utils_covs_np"]))

    idx = np.random.randint(len(batch_output_np))

    assert check_cov_normalized(single_output_np[idx, :, :])
    assert check_cov_normalized(single_output_th[idx, :, :])
    assert check_cov_normalized(batch_output_np[idx, :, :])
    assert check_cov_normalized(batch_output_th[idx, :, :])
    assert check_cov_normalized(single_output_th_np[idx, :, :])
    assert check_cov_normalized(batch_output_th_np[idx, :, :])

    assert_allclose(single_output_np, batch_output_np, rtol=1e-4, atol=1e-4)
    assert_close(single_output_th, batch_output_th, rtol=1e-4, atol=1e-4)
    assert_allclose(
        torch.Tensor(single_output_np), single_output_th_np, rtol=1e-4, atol=1e-4
    )
    assert_allclose(
        torch.Tensor(batch_output_np), batch_output_th_np, rtol=1e-4, atol=1e-4
    )


def test_cov_sigma_N(inputs):
    noise_var = 0.3
    signal_var = 0.1
    single_output_np = []
    single_output_th = []

    for cov in inputs["utils_covs_np"]:
        single_output_np.append(
            cov_sigma(cov, noise_var=noise_var, signal_var=signal_var)
        )
    for cov in inputs["utils_covs_th"]:
        single_output_th.append(
            cov_sigma(cov, noise_var=noise_var, signal_var=signal_var)
        )

    single_output_np = np.array(single_output_np)
    single_output_th = torch.stack(single_output_th, dim=0)

    batch_output_np = cov_sigma_N(
        inputs["utils_covs_np"], noise_var=noise_var, signal_var=signal_var
    )
    batch_output_th = cov_sigma_N(
        inputs["utils_covs_th"], noise_var=noise_var, signal_var=signal_var
    )

    assert_allclose(single_output_np, batch_output_np, rtol=1e-4, atol=1e-4)
    assert_close(single_output_th, batch_output_th, rtol=1e-4, atol=1e-4)


def test_cov_trace_norm_sigma_N(inputs):
    noise_var = 0.1
    single_output_np = []
    single_output_th = []

    for cov in inputs["utils_covs_np"]:
        single_output_np.append(cov_sigma(trace_norm(cov), noise_var=noise_var))
    for cov in inputs["utils_covs_th"]:
        single_output_th.append(cov_sigma(trace_norm(cov), noise_var=noise_var))

    single_output_np = np.array(single_output_np)
    single_output_th = torch.stack(single_output_th, dim=0)

    batch_output_np = cov_trace_norm_sigma_N(
        inputs["utils_covs_np"], noise_var=noise_var
    )
    batch_output_th = cov_trace_norm_sigma_N(
        inputs["utils_covs_th"], noise_var=noise_var
    )

    assert_allclose(single_output_np, batch_output_np, rtol=1e-4, atol=1e-4)
    assert_close(single_output_th, batch_output_th, rtol=1e-4, atol=1e-4)


def test_check_cov_normalized(inputs):
    non_normalized_np = inputs["utils_covs_np"][0]
    non_normalized_th = inputs["utils_covs_th"][0]
    normalized_np = trace_norm(non_normalized_np)
    normalized_th = trace_norm(non_normalized_th)

    assert check_cov_normalized(non_normalized_np) == False
    assert check_cov_normalized(non_normalized_th) == False
    assert check_cov_normalized(normalized_np) == True
    assert check_cov_normalized(normalized_th) == True


def test_check_cov_symmetry(inputs):
    symmetric_np = (inputs["utils_covs_np"][5] + inputs["utils_covs_np"][5].T) / 2
    symmetric_th = (inputs["utils_covs_th"][5] + inputs["utils_covs_th"][5].T) / 2
    non_sym_np = symmetric_np + np.triu(symmetric_np)
    non_sym_th = symmetric_th + torch.triu(symmetric_th)

    assert check_cov_symmetry(symmetric_np)
    assert check_cov_symmetry(symmetric_th)
    assert not check_cov_symmetry(non_sym_np)
    assert not check_cov_symmetry(non_sym_th)


def test_check_input_format(inputs):
    np_input = inputs["utils_covs_np"][10]
    th_input = inputs["utils_covs_th"][10]
    dict_input = {}
    dict_input["inputs"] = (np_input, th_input)

    assert check_input_format(np_input) is np
    assert check_input_format(th_input) is torch
    with pytest.raises(
        TypeError, match="Input must be a Numpy array or a PyTorch tensor."
    ):
        check_input_format(dict_input)


def test_check_and_change_input_format(inputs):
    # single input numpy and torch arrays with shape (50, 50)
    with pytest.raises(
        ValueError, match="Covariance tensor has to have more than 1 matrix."
    ):
        out_list, N, module = check_and_change_input_format(inputs["utils_covs_np"][12])
    with pytest.raises(
        ValueError, match="Covariance tensor has to have more than 1 matrix."
    ):
        out_list, N, module = check_and_change_input_format(inputs["utils_covs_th"][12])

    # single input numpy and torch arrays with shape (1, 50, 50)
    with pytest.raises(
        ValueError, match="Covariance tensor has to have more than 1 matrix."
    ):
        out_list, N, module = check_and_change_input_format(
            np.reshape(inputs["utils_covs_np"][12], (1, 50, 50))
        )
    with pytest.raises(
        ValueError, match="Covariance tensor has to have more than 1 matrix."
    ):
        out_list, N, module = check_and_change_input_format(
            torch.reshape(inputs["utils_covs_th"][12], (1, 50, 50))
        )

    # single input numpy and torch arrays with shape (20, 45, 50)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected shape (N, dim, dim), but got (20, 45, 50): last two dimensions must be equal."
        ),
    ):
        out_list, N, module = check_and_change_input_format(
            np.reshape(inputs["utils_covs_np"], (20, 50, 50))[:, :45, :]
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected shape (N, dim, dim), but got (20, 45, 50): last two dimensions must be equal."
        ),
    ):
        out_list, N, module = check_and_change_input_format(
            torch.reshape(inputs["utils_covs_th"], (20, 50, 50))[:, :45, :]
        )

    # batch input np and torch arrays
    out_list, N, module = check_and_change_input_format(inputs["utils_covs_np"])
    assert isinstance(out_list, list)
    assert N == 20
    assert module is np

    out_list, N, module = check_and_change_input_format(inputs["utils_covs_th"])
    assert isinstance(out_list, list)
    assert N == 20
    assert module is torch

    # list input for np and torch
    out_list, N, module = check_and_change_input_format(list(inputs["utils_covs_np"]))
    assert isinstance(out_list, list)
    assert N == 20
    assert module is np

    out_list, N, module = check_and_change_input_format(list(inputs["utils_covs_th"]))
    assert isinstance(out_list, list)
    assert N == 20
    assert module is torch

    # tuple input for np and torch
    out_list, N, module = check_and_change_input_format(tuple(inputs["utils_covs_np"]))
    assert isinstance(out_list, list)
    assert N == 20
    assert module is np

    out_list, N, module = check_and_change_input_format(tuple(inputs["utils_covs_th"]))
    assert isinstance(out_list, list)
    assert N == 20
    assert module is torch

    # empty list/tuple
    with pytest.raises(ValueError, match="Input list is empty."):
        out_list, N, module = check_and_change_input_format([])
    with pytest.raises(ValueError, match="Input list is empty."):
        out_list, N, module = check_and_change_input_format(())

    # one element list/tuple
    with pytest.raises(
        ValueError, match="Covariance list has to have more than 1 matrix."
    ):
        out_list, N, module = check_and_change_input_format(
            [inputs["utils_covs_np"][0]]
        )
    with pytest.raises(
        ValueError, match="Covariance list has to have more than 1 matrix."
    ):
        out_list, N, module = check_and_change_input_format(
            (inputs["utils_covs_np"][0],)
        )

    # non-defined type
    dict_inp = {}
    dict_inp["input"] = inputs["utils_covs_np"][0]
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Input must be a tuple/list of (dim, dim) matrices or an array/tensor of shape (N, dim, dim) with N>1."
        ),
    ):
        out_list, N, module = check_and_change_input_format(dict_inp)
