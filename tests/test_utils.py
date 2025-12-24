import numpy as np
import torch
import pathlib
import os
import glob
from BayesCompare import cov_utils
from numpy.testing import assert_allclose
from torch.testing import assert_close


def read_input():

    home_path = pathlib.Path.home()
    sample_path = os.path.join(home_path, "Documents/BayesCompare/tests/sample_data")
    input_args = {}
    for input_file in ["utils_covs_np", "utils_covs_th"]:
        filename = glob.glob(os.path.join(sample_path, f"test_{input_file}*"))[0]
        file_ext = pathlib.Path(filename).suffix

        if file_ext == ".npy":
            sample_file = np.load(filename)
        elif file_ext == ".pt":
            sample_file = torch.load(filename)
        else:
            return NotImplementedError("Input file type not implemented for loading")

        input_args[input_file] = sample_file

    return input_args


def test_trace_norm_N():

    input_args = read_input()

    single_output_np = []
    single_output_th = []
    for cov in input_args["utils_covs_np"]:
        single_output_np.append(cov_utils.trace_norm(cov))
    for cov in input_args["utils_covs_th"]:
        single_output_th.append(cov_utils.trace_norm(cov))

    single_output_np = np.array(single_output_np)
    single_output_th = torch.stack(single_output_th, dim=0)

    batch_output_np = cov_utils.trace_norm_N(input_args["utils_covs_np"])
    batch_output_th = cov_utils.trace_norm_N(input_args["utils_covs_th"])

    idx = np.random.randint(len(batch_output_np))

    assert cov_utils.check_cov_normalized(single_output_np[idx, :, :])
    assert cov_utils.check_cov_normalized(single_output_th[idx, :, :])
    assert cov_utils.check_cov_normalized(batch_output_np[idx, :, :])
    assert cov_utils.check_cov_normalized(batch_output_th[idx, :, :])

    assert_allclose(single_output_np, batch_output_np, rtol=1e-4, atol=1e-4)
    assert_close(single_output_th, batch_output_th, rtol=1e-4, atol=1e-4)

    print("trace_norm_N test successful!")


def test_cov_sigma_N():

    input_args = read_input()

    noise_var = 0.3
    signal_var = 0.1
    single_output_np = []
    single_output_th = []
    for cov in input_args["utils_covs_np"]:
        single_output_np.append(
            cov_utils.cov_sigma(cov, noise_var=noise_var, signal_var=signal_var)
        )
    for cov in input_args["utils_covs_th"]:
        single_output_th.append(
            cov_utils.cov_sigma(cov, noise_var=noise_var, signal_var=signal_var)
        )

    single_output_np = np.array(single_output_np)
    single_output_th = torch.stack(single_output_th, dim=0)

    batch_output_np = cov_utils.cov_sigma_N(
        input_args["utils_covs_np"], noise_var=noise_var, signal_var=signal_var
    )
    batch_output_th = cov_utils.cov_sigma_N(
        input_args["utils_covs_th"], noise_var=noise_var, signal_var=signal_var
    )

    assert_allclose(single_output_np, batch_output_np, rtol=1e-4, atol=1e-4)
    assert_close(single_output_th, batch_output_th, rtol=1e-4, atol=1e-4)

    print("cov_sigma_N test successful!")


def test_cov_trace_norm_sigma_N():

    input_args = read_input()

    noise_var = 0.1
    single_output_np = []
    single_output_th = []

    for cov in input_args["utils_covs_np"]:
        single_output_np.append(
            cov_utils.cov_sigma(cov_utils.trace_norm(cov), noise_var=noise_var)
        )
    for cov in input_args["utils_covs_th"]:
        single_output_th.append(
            cov_utils.cov_sigma(cov_utils.trace_norm(cov), noise_var=noise_var)
        )

    single_output_np = np.array(single_output_np)
    single_output_th = torch.stack(single_output_th, dim=0)

    batch_output_np = cov_utils.cov_trace_norm_sigma_N(
        input_args["utils_covs_np"], noise_var=noise_var
    )
    batch_output_th = cov_utils.cov_trace_norm_sigma_N(
        input_args["utils_covs_th"], noise_var=noise_var
    )

    assert_allclose(single_output_np, batch_output_np, rtol=1e-4, atol=1e-4)
    assert_close(single_output_th, batch_output_th, rtol=1e-4, atol=1e-4)

    print("cov_trace_norm_sigma_N test successful!")


test_trace_norm_N()
test_cov_sigma_N()
test_cov_trace_norm_sigma_N()
