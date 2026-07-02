"""
Tests for package.encoding_utils.voxel_reliability and its helper functions
_check_consistent_repetitions and noise_estimation.

Run with: pytest test_encoding_utils.py -v
"""

import pytest

import numpy as np

from types import SimpleNamespace
from numpy.testing import assert_almost_equal
from scipy.stats import beta

from BayesCompare.encoding_utils import (
    voxel_reliability,
    _check_consistent_repetitions,
    noise_estimation,
    _neg_loglik,
    sample_noise_values,
)


@pytest.fixture
def balanced_stim_list():
    # 3 stimuli, 3 reps each, grouped consecutively
    return [1, 1, 1, 2, 2, 2, 3, 3, 3]


@pytest.fixture
def imbalanced_stim_list():
    # 3 stimuli, reps of 2, 3, 1
    return [1, 1, 2, 2, 2, 3]


@pytest.fixture
def balanced_case(balanced_stim_list):
    """Single-voxel data for a balanced design (3 stimuli, 3 reps each),
    plus its hand-derived expected noise variance, shared by every test
    that needs a known-correct balanced example.

    Manual derivation:
        groups: [1,3,5]    mean=3  diff=[-2,0,2] -> sq sum 8
                [10,10,10] mean=10 diff=[0,0,0]  -> sq sum 0
                [2,4,6]    mean=4  diff=[-2,0,2] -> sq sum 8
        sum_sq = 16, sum_of_reps = 9 trials - 3 images = 6
        sigma_noise = 16 / 6
    """
    data = np.array([1, 3, 5, 10, 10, 10, 2, 4, 6], dtype=float)
    split_param = 3  # n_groups, as produced internally for the balanced case
    expected_noise = 16 / 6

    return SimpleNamespace(
        stim_list=balanced_stim_list,
        data=data,
        split_param=split_param,
        expected_noise=expected_noise,
    )


@pytest.fixture
def imbalanced_case(imbalanced_stim_list):
    """Single-voxel data for an imbalanced design (reps of 2, 3, 1), plus
    its hand-derived expected noise variance, shared by every test that
    needs a known-correct imbalanced example.

    Manual derivation:
        groups: [5,7]        mean=6  diff=[-1,1]  -> sq sum 2
                [10,12,14]   mean=12 diff=[-2,0,2] -> sq sum 8
                [100]        (single rep, excluded)
        sum_sq = 10, sum_of_reps = 6 trials - 3 images = 3
        sigma_noise = 10 / 3
    """
    data = np.array([5, 7, 10, 12, 14, 100], dtype=float)
    split_param = np.array([2, 5])  # cumsum(counts)[:-1], as produced internally
    expected_noise = 10 / 3

    return SimpleNamespace(
        stim_list=imbalanced_stim_list,
        data=data,
        split_param=split_param,
        expected_noise=expected_noise,
    )


@pytest.fixture
def small_x():
    return np.array([0.2, 0.5, 0.8])


@pytest.fixture
def noise_and_total_var():
    """Single fixture; tests can parametrize sizes if needed."""
    np.random.seed(0)
    total_var = np.random.uniform(1.0, 2.0, size=50)
    noise_var = np.random.uniform(0.1, 0.9, size=50) * total_var
    return noise_var, total_var


## Tests for _check_consistent_repetitions


def test_check_consistent_repetitions_balanced_returns_true(balanced_stim_list):
    consistent, counts = _check_consistent_repetitions(balanced_stim_list)
    assert consistent
    np.testing.assert_array_equal(counts, [3, 3, 3])


def test_check_consistent_repetitions_imbalanced_returns_false(imbalanced_stim_list):
    consistent, counts = _check_consistent_repetitions(imbalanced_stim_list)
    assert not consistent
    np.testing.assert_array_equal(counts, [2, 3, 1])


def test_check_consistent_repetitions_all_single_repetitions_is_consistent():
    # every stim appears exactly once -> counts are all equal (to 1)
    stim_list = [1, 2, 3, 4]
    consistent, counts = _check_consistent_repetitions(stim_list)
    assert consistent
    np.testing.assert_array_equal(counts, [1, 1, 1, 1])


def test_check_consistent_repetitions_single_stimulus_group():
    stim_list = [7, 7, 7, 7]
    consistent, counts = _check_consistent_repetitions(stim_list)
    assert consistent
    np.testing.assert_array_equal(counts, [4])


def test_check_consistent_repetitions_non_consecutive_repeats_are_separate_groups():
    # As stated in the docstring, it is asumed that all equal IDs are already
    # grouped in stim_list. If they are separated they are treated as different
    # groups
    stim_list = [1, 1, 2, 1, 1]
    consistent, counts = _check_consistent_repetitions(stim_list)
    # groups: (1,1) (2) (1,1) -> counts [2, 1, 2]
    np.testing.assert_array_equal(counts, [2, 1, 2])
    assert not consistent


def test_check_consistent_repetitions_empty_list_raises():
    # Check that passing an empty list results in an IndexError
    with pytest.raises(IndexError):
        _check_consistent_repetitions([])


# Tests for noise_estimation


def test_noise_estimation_balanced_matches_manual_calculation(balanced_case):
    result = noise_estimation(balanced_case.data, balanced_case.split_param)
    assert_almost_equal(result, balanced_case.expected_noise)


def test_noise_estimation_imbalanced_matches_manual_calculation(imbalanced_case):
    result = noise_estimation(imbalanced_case.data, imbalanced_case.split_param)
    assert_almost_equal(result, imbalanced_case.expected_noise)


def test_noise_estimation_zero_noise_when_repetitions_identical():
    data = np.array([5.0, 5.0, 5.0, 9.0, 9.0, 9.0])
    result = noise_estimation(data, 2)
    assert_almost_equal(result, 0.0)


def test_noise_estimation_single_repetition_group_contributes_zero_variance():
    # Test that stim with 1 rep do not contribute to numerator but still
    # contributes to denominator
    data = np.array([1.0, 3.0, 100.0])  # groups: [1,3], [100]
    split_param = np.array([2])
    result = noise_estimation(data, split_param)
    # group1 diff=[-1,1] -> sq sum 2; sum_of_reps = 3 - 2 = 1
    expected = 2 / 1
    assert_almost_equal(result, expected)


def test_noise_estimation_no_repetitions_anywhere_raises():
    # If no stimuli has repetitions, then the function should raise a ValueError
    data = np.array([1.0, 2.0, 3.0, 4.0])
    split_param = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        noise_estimation(data, split_param)


## Tests for voxel_reliability


def test_voxel_reliability_output_shapes(balanced_stim_list):
    n_voxels, n_stim = 5, len(balanced_stim_list)
    np.random.seed(0)
    voxel_data = np.random.normal(size=(n_voxels, n_stim))

    reliability, sigma_noise, sigma_tot = voxel_reliability(
        voxel_data, balanced_stim_list, n_jobs=1
    )
    assert reliability.shape == (n_voxels,)
    assert sigma_noise.shape == (n_voxels,)
    assert sigma_tot.shape == (n_voxels,)


def test_voxel_reliability_balanced_matches_manual_calculation(balanced_case):
    voxel_data = balanced_case.data.reshape(1, -1)

    reliability, sigma_noise, sigma_tot = voxel_reliability(
        voxel_data, balanced_case.stim_list, n_jobs=1
    )

    expected_tot = np.var(balanced_case.data, ddof=1)
    expected_reliability = 1 - balanced_case.expected_noise / expected_tot

    assert_almost_equal(sigma_noise[0], balanced_case.expected_noise)
    assert_almost_equal(sigma_tot[0], expected_tot)
    assert_almost_equal(reliability[0], expected_reliability)


def test_voxel_reliability_imbalanced_matches_manual_calculation(imbalanced_case):
    voxel_data = imbalanced_case.data.reshape(1, -1)

    reliability, sigma_noise, sigma_tot = voxel_reliability(
        voxel_data, imbalanced_case.stim_list, n_jobs=1
    )

    expected_tot = np.var(imbalanced_case.data, ddof=1)
    expected_reliability = 1 - imbalanced_case.expected_noise / expected_tot

    assert_almost_equal(sigma_noise[0], imbalanced_case.expected_noise)
    assert_almost_equal(sigma_tot[0], expected_tot)
    assert_almost_equal(reliability[0], expected_reliability)


def test_voxel_reliability_multiple_voxels_independent(imbalanced_case):
    # Add a second voxel to test multiple voxel data
    other_data = np.array([1, 3, 5, 10, 15, 50], dtype=float)
    voxel_data = np.vstack([imbalanced_case.data, other_data])

    reliability, sigma_noise, sigma_tot = voxel_reliability(
        voxel_data, imbalanced_case.stim_list, n_jobs=1
    )

    for i, data in enumerate(voxel_data):
        expected_noise = noise_estimation(data, imbalanced_case.split_param)
        expected_tot = np.var(data, ddof=1)
        assert_almost_equal(sigma_noise[i], expected_noise)
        assert_almost_equal(sigma_tot[i], expected_tot)
        assert_almost_equal(reliability[i], 1 - expected_noise / expected_tot)


def test_voxel_reliability_perfectly_reliable_voxel_has_reliability_near_one(
    balanced_stim_list,
):
    # No variability should produce a reliability of 1 and a noise variance of 0
    voxel_data = np.array(
        [
            [1, 1, 1, 5, 5, 5, 9, 9, 9],
        ],
        dtype=float,
    )

    reliability, sigma_noise, _ = voxel_reliability(
        voxel_data, balanced_stim_list, n_jobs=1
    )
    assert_almost_equal(sigma_noise[0], 0.0)
    assert_almost_equal(reliability[0], 1.0)


def test_voxel_reliability_n_jobs_parallel_matches_serial(imbalanced_stim_list):
    # Make sure there is no difference by using parallelization
    np.random.seed(42)
    voxel_data = np.random.normal(size=(4, len(imbalanced_stim_list)))

    serial = voxel_reliability(voxel_data, imbalanced_stim_list, n_jobs=1)
    parallel = voxel_reliability(voxel_data, imbalanced_stim_list, n_jobs=-1)

    for s, p in zip(serial, parallel):
        assert_almost_equal(s, p)


def test_voxel_reliability_sensible_output(balanced_stim_list):
    # Make sure that reliability is between 0 and 1, and that
    # total variance is higher than noise variance
    voxel_data = np.array(
        [
            [1, 2, 3, 10, 11, 12, 20, 21, 22],
        ],
        dtype=float,
    )
    reliability, sigma_noise, sigma_tot = voxel_reliability(
        voxel_data, balanced_stim_list, n_jobs=1
    )
    assert 0 <= reliability[0] <= 1
    assert sigma_tot[0] > sigma_noise[0]


def test_voxel_reliability_zero_total_variance_produces_runtime_warning(
    balanced_stim_list,
):
    # A voxel constant across stimuli should raise a RunTimeWarning due to
    # division by 0 and have 0 total variance
    voxel_data = np.array(
        [
            [5.0] * 9,
        ]
    )
    with pytest.warns(RuntimeWarning):
        _, _, sigma_tot = voxel_reliability(voxel_data, balanced_stim_list, n_jobs=1)
    assert sigma_tot[0] == 0.0


## Tests for neg_loglik


def test_neg_loglik_positive_a_matches_manual_computation(small_x):
    a = 2.0
    expected = -(len(small_x) * np.log(a) + (a - 1) * np.sum(np.log(small_x)))
    result = _neg_loglik(a, small_x)
    assert np.isclose(result, expected)


@pytest.mark.parametrize("a", [0.0, -1.0, -0.5])
def test_neg_loglik_non_positive_a_returns_inf(a, small_x):
    result = _neg_loglik(a, small_x)
    assert result == np.inf


## Tests for sample_noise_values


@pytest.mark.parametrize("n_vals", [1, 5, 10])
def test_sample_noise_values_mle_shape_and_bounds(noise_and_total_var, n_vals):
    noise_var, total_var = noise_and_total_var

    noise_vals = sample_noise_values(
        noise_var=noise_var,
        total_var=total_var,
        n_vals=n_vals,
        method="mle",
        return_params=False,
    )

    assert noise_vals.shape == (n_vals,)
    assert np.all(noise_vals > 0.0)
    assert np.all(noise_vals < 1.0)


def test_sample_noise_values_mle_returns_params(noise_and_total_var):
    noise_var, total_var = noise_and_total_var
    n_vals = 5

    noise_vals, a, b = sample_noise_values(
        noise_var=noise_var,
        total_var=total_var,
        n_vals=n_vals,
        method="mle",
        return_params=True,
    )

    assert noise_vals.shape == (n_vals,)
    assert a > 0
    assert np.isclose(b, 1.0)


@pytest.mark.parametrize(
    "a_fixed,b_fixed,n_vals",
    [
        (2.0, 3.0, 3),
        (1.5, 1.0, 5),
        (2.5, 1.5, 7),
    ],
)
def test_sample_noise_values_fixed_uses_given_parameters(
    noise_and_total_var, a_fixed, b_fixed, n_vals
):
    noise_var, total_var = noise_and_total_var

    noise_vals, a, b = sample_noise_values(
        noise_var=noise_var,
        total_var=total_var,
        n_vals=n_vals,
        method="fixed",
        a=a_fixed,
        b=b_fixed,
        return_params=True,
    )

    assert noise_vals.shape == (n_vals,)
    assert np.isclose(a, a_fixed)
    assert np.isclose(b, b_fixed)

    n_bins = 2 * n_vals
    y_vals = np.linspace(1 / n_bins, (n_bins - 1) / n_bins, n_vals)
    expected_noise_vals = beta.ppf(y_vals, a_fixed, b_fixed)

    assert np.allclose(noise_vals, expected_noise_vals)


@pytest.mark.parametrize(
    "a_val,b_val",
    [
        (None, 1.0),
        (1.0, None),
        (None, None),
    ],
)
def test_sample_noise_values_fixed_missing_params_raises_value_error(
    noise_and_total_var, a_val, b_val
):
    noise_var, total_var = noise_and_total_var

    with pytest.raises(ValueError):
        sample_noise_values(
            noise_var=noise_var,
            total_var=total_var,
            n_vals=5,
            method="fixed",
            a=a_val,
            b=b_val,
        )


@pytest.mark.parametrize("invalid_method", ["unknown", "other", "MLE "])
def test_sample_noise_values_invalid_method_raises_not_implemented_error(
    noise_and_total_var, invalid_method
):
    noise_var, total_var = noise_and_total_var

    # Any method except 'mle' and 'fixed' should raise an error
    with pytest.raises(NotImplementedError):
        sample_noise_values(
            noise_var=noise_var,
            total_var=total_var,
            n_vals=5,
            method=invalid_method,
        )
