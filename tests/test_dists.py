import numpy as np
import torch
import pathlib
import os
import glob
from BayesCompare import distances
from numpy.testing import assert_allclose
from torch.testing import assert_close
from pathlib import Path
import pytest


def read_input(dir):

    file_ext = pathlib.Path(dir).suffix

    if file_ext == ".npy":
        sample_file = np.load(dir)
    elif file_ext == ".pt":
        sample_file = torch.load(dir)
    else:
        return NotImplementedError("Input file type not implemented for loading")

    return sample_file


def load_inputs():
    home_path = Path.home()
    sample_path = os.path.join(home_path, "Documents/BayesCompare/tests/sample_data")
    input_cases_np = {}
    input_cases_th = {}
    for input_file in [
        "same",
        "similar",
        "symmetric",
        "nonsymmetric",
        "slightly_nonsymmetric",
        "psd",
        "almost_psd",
        "large_scale",
        "small_scale",
        "large_and_small_scale",
    ]:
        filename_np = glob.glob(
            os.path.join(sample_path, f"test_input_dist_{input_file}_np*")
        )[0]
        input_cases_np[input_file] = read_input(filename_np)

        filename_th = glob.glob(
            os.path.join(sample_path, f"test_input_dist_{input_file}_th*")
        )[0]
        input_cases_th[input_file] = read_input(filename_th)

    return input_cases_np, input_cases_th


# Case 1: Same covariance matrix given as input: d(A, A)
def test_same_input(input_np, input_th, meas_name_to_test):

    output_np = []
    output_th = []

    for input_np, input_th in zip(input_np, input_th):
        output_np.append(
            distances.measure_dist(
                [input_np[0], input_np[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [input_th[0], input_th[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, np.zeros_like(output_np), rtol=1e-10, atol=1e-10)
    assert_allclose(output_th, torch.zeros_like(output_th), rtol=1e-10, atol=1e-10)

    print("Test 1: Same input test is successful!")


# Case 2: Very close covairance matrices given as input: d(A, A + ε)
def test_similar_input(input_np, input_th, meas_name_to_test):

    output_np = []
    output_th = []

    for input_np, input_th in zip(input_np, input_th):
        output_np.append(
            distances.measure_dist(
                [input_np[0], input_np[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [input_th[0], input_th[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)
    # atol = 1e-2 as we have ε = 0.05 in the input
    assert_allclose(output_np, np.zeros_like(output_np), rtol=1e-8, atol=1e-2)
    assert_close(output_th, torch.zeros_like(output_th), rtol=1e-8, atol=1e-2)
    assert_allclose(output_np, output_th.numpy(), rtol=1e-10, atol=1e-10)
    assert_close(
        torch.from_numpy(output_np.astype(np.float32)),
        output_th,
        rtol=1e-10,
        atol=1e-10,
    )

    print("Test 2: Similar input test is successful!")


# Case 3: Symmetry of distance check: d(A, B) vs d(B, A)
def test_symmetric_inputs(input_np, input_th, meas_name_to_test):

    output_np = []
    output_th = []

    for input_np, input_th in zip(input_np, input_th):
        output_np.append(
            distances.measure_dist(
                [input_np[0], input_np[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )
        output_np.append(
            distances.measure_dist(
                [input_np[1], input_np[0]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [input_th[0], input_th[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [input_th[1], input_th[0]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np[0::2], output_np[1::2], rtol=1e-8, atol=1e-10)
    assert_close(output_th[0::2], output_th[1::2], rtol=1e-8, atol=1e-10)
    assert_allclose(output_np, output_th.numpy(), rtol=1e-7, atol=1e-6)
    assert_close(
        torch.from_numpy(output_np.astype(np.float32)),
        output_th,
        rtol=1e-10,
        atol=1e-10,
    )

    print("Test 3: Symmetric input test is successful!")


# Case 4: Non-symmetric inputs: A != A.T
def test_nonsymmetric_input(input_np, input_th, meas_name_to_test):

    for input_np, input_th in zip(input_np, input_th):
        with pytest.raises(ValueError):
            distances.measure_dist(
                [input_np[0], input_np[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        with pytest.raises(ValueError):
            distances.measure_dist(
                [input_th[0], input_th[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )

    print("Test 4: Non-symmetric input test is successful!")


# Case 5: Slightly non-symmetric inputs: A ~ A.T
def test_slightly_nonsymmetric_input(input_np, input_th, meas_name_to_test):

    idx = 0

    for input_np, input_th in zip(input_np, input_th):

        if idx != 4:
            with pytest.raises(ValueError):
                distances.measure_dist(
                    [input_np[0], input_np[1]],
                    meas_name=meas_name_to_test,
                    show_progress=False,
                )
            with pytest.raises(ValueError):
                distances.measure_dist(
                    [input_th[0], input_th[1]],
                    meas_name=meas_name_to_test,
                    show_progress=False,
                )
            idx += 1

        else:
            output_np = distances.measure_dist(
                [input_np[0], input_np[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
            output_th = distances.measure_dist(
                [input_th[0], input_th[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
            assert_allclose(output_np, output_th.numpy(), rtol=1e-7, atol=1e-6)
            assert_close(
                torch.from_numpy(output_np.astype(np.float32)),
                output_th,
                rtol=1e-10,
                atol=1e-10,
            )

    print("Test 5: Slightly non-symmetric input test is successful!")


# Case 6: PSD inputs (one eigenval = 0) and Case 7: Almost PSD inputs (one eigenval ~ 0)
def test_psd_input(input_np, input_th, meas_name_to_test, case_num):

    output_np = []
    output_th = []

    for input_np, input_th in zip(input_np, input_th):
        output_np.append(
            distances.measure_dist(
                [input_np[0], input_np[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [input_th[0], input_th[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    # what to check about outputs for the psd tests?
    assert_allclose(output_np, output_th.numpy(), rtol=1e-6, atol=1e-7)
    assert_close(
        torch.from_numpy(output_np.astype(np.float32)), output_th, rtol=1e-6, atol=1e-7
    )

    print(f"Test {case_num}: PSD inputs test is successful!")


# Case 8: Large scale inputs & Case 9: Small scale inputs & Case 10: One large scale, one small scale input
def test_different_scale_inputs(input_np, input_th, meas_name_to_test, case_num):
    output_np = []
    output_th = []

    for input_np, input_th in zip(input_np, input_th):
        output_np.append(
            distances.measure_dist(
                [input_np[0], input_np[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [input_th[0], input_th[1]],
                meas_name=meas_name_to_test,
                show_progress=False,
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    # what to check about outputs for the scale tests?
    assert_allclose(output_np, output_th.numpy(), rtol=1e-6, atol=1e-7)
    assert_close(
        torch.from_numpy(output_np.astype(np.float32)), output_th, rtol=1e-6, atol=1e-7
    )

    print(f"Test {case_num}: Different scale inputs test is successful!")


def call_all_tests(meas_name):

    input_cases_np, input_cases_th = load_inputs()

    # Case 1: Same covariance matrix given as input: d(A, A)
    test_same_input(input_cases_np["same"], input_cases_th["same"], meas_name)

    # Case 2: Very close covairance matrices given as input: d(A, A + ε)
    test_similar_input(input_cases_np["similar"], input_cases_th["similar"], meas_name)

    # Case 3: Symmetry of distance check: d(A, B) vs d(B, A)
    test_symmetric_inputs(
        input_cases_np["symmetric"], input_cases_th["symmetric"], meas_name
    )

    # Case 4: Non-symmetric inputs: A != A.T
    test_nonsymmetric_input(
        input_cases_np["nonsymmetric"], input_cases_th["nonsymmetric"], meas_name
    )

    # Case 5: Slightly non-symmetric inputs: A ~ A.T
    test_slightly_nonsymmetric_input(
        input_cases_np["slightly_nonsymmetric"],
        input_cases_th["slightly_nonsymmetric"],
        meas_name,
    )

    # Case 6: PSD inputs (one eigenval = 0)
    test_psd_input(input_cases_np["psd"], input_cases_th["psd"], meas_name, 6)

    # Case 7: Almost PSD inputs (one eigenval ~ 0)
    test_psd_input(
        input_cases_np["almost_psd"], input_cases_th["almost_psd"], meas_name, 7
    )

    # Case 8: Large scale inputs
    test_different_scale_inputs(
        input_cases_np["large_scale"], input_cases_th["large_scale"], meas_name, 8
    )

    # Case 9: Small scale inputs
    test_different_scale_inputs(
        input_cases_np["small_scale"], input_cases_th["small_scale"], meas_name, 9
    )

    # Case 10: One large scale, one small scale input
    test_different_scale_inputs(
        input_cases_np["large_and_small_scale"],
        input_cases_th["large_and_small_scale"],
        meas_name,
        10,
    )


def call_some_tests(meas_name, test_list):

    input_cases_np, input_cases_th = load_inputs()

    test_dict = {
        1: lambda: test_same_input(
            input_cases_np["same"], input_cases_th["same"], meas_name
        ),
        2: lambda: test_similar_input(
            input_cases_np["similar"], input_cases_th["similar"], meas_name
        ),
        3: lambda: test_symmetric_inputs(
            input_cases_np["symmetric"], input_cases_th["symmetric"], meas_name
        ),
        4: lambda: test_nonsymmetric_input(
            input_cases_np["nonsymmetric"], input_cases_th["nonsymmetric"], meas_name
        ),
        5: lambda: test_slightly_nonsymmetric_input(
            input_cases_np["slightly_nonsymmetric"],
            input_cases_th["slightly_nonsymmetric"],
            meas_name,
        ),
        6: lambda: test_psd_input(
            input_cases_np["psd"], input_cases_th["psd"], meas_name, 6
        ),
        7: lambda: test_psd_input(
            input_cases_np["almost_psd"], input_cases_th["almost_psd"], meas_name, 7
        ),
        8: lambda: test_different_scale_inputs(
            input_cases_np["small_scale"], input_cases_th["small_scale"], meas_name, 8
        ),
        9: lambda: test_different_scale_inputs(
            input_cases_np["small_scale"], input_cases_th["small_scale"], meas_name, 9
        ),
        10: lambda: test_different_scale_inputs(
            input_cases_np["large_and_small_scale"],
            input_cases_th["large_and_small_scale"],
            meas_name,
            10,
        ),
    }

    for test in test_list:
        test_dict[test]()


# Testing mahalanobis without any mean vector is actually not meaningful as the distance is directly equal to zero in this case (without any computations).
# call_all_tests("mahalanobis")

# call_all_tests("bhattacharyya")

# call_all_tests("wasserstein")

# call_all_tests("JSD")

# call_all_tests("TVD")

# call_all_tests("hellinger")

# KL divergence do not satisfy symmetry property so it doesn't make sense to test it on that.
# call_some_tests("KL div", [1, 2, 4, 5, 6, 7, 8, 9, 10])
