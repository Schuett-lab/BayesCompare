import numpy as np

from itertools import groupby
from joblib import Parallel, delayed
from typing import Optional, Tuple

from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import beta


def _check_consistent_repetitions(stim_list: NDArray) -> Tuple[bool, NDArray]:
    """Check if all elements of the list are repeated the same number of times"""

    counts = np.array([len(list(group)) for _, group in groupby(stim_list)])
    constant = (counts[0] == counts).all()
    return constant, counts


def noise_estimation(data: NDArray, split_param: NDArray | int) -> float:
    """
    Estimate the variance of each voxel attributed to the noise. Expressed as
    the mean of the variances of each image across their repetitions.
    """
    # Divides array into sub_arrays for each stim
    split_data = np.split(data, split_param)

    # Differences with the mean (images with 1 repetition do not contribute)
    diff = np.concatenate(
        [split - np.mean(split) for split in split_data if len(split) > 1]
    )

    # Outer sum and division by total reps
    sum_of_reps = len(data) - len(split_data)  # n_trials - n_imgs
    sigma_noise = np.sum(np.dot(diff, diff)) / sum_of_reps

    return sigma_noise


def voxel_reliability(
    voxel_data: NDArray, stim_list: NDArray, n_jobs: int = -1
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Calculate voxel reliability by estimating noise variance across repetitions
    of the same stimuli.

    Note: This function assumes that the elements of stim_list are grouped by
    stimulus (i.e. all occurrences of the same stimulus are consecutive)

    Parameters
    ----------

    voxel_data: NDArray, shape (n_voxels, n_stim)
        The data array, with each row representing the activation profile of
        a single voxel and each colum representing the activation patterns of
        a single presentation of stimuli

    stim_list: NDArray, shape (n_stim,)
        Array containing the stim_id of each column of voxel_data

    Returns
    -------

    reliability: np.array, shape (n_voxels,)
        Reliability score for each voxel

    sigma_noise: np.array, shape (n_voxels,)
        Estimated noise variance for each voxel

    sigma_tot: np.array, shape (n_voxels,)
        Total estimated variance for each voxel

    """

    consistent_reps, counts = _check_consistent_repetitions(stim_list)
    if consistent_reps:
        split_param = counts.shape[0]
    else:
        split_param = np.cumsum(counts)[:-1]
    sigma_noise = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(noise_estimation)(voxel, split_param) for voxel in voxel_data
    )
    sigma_noise = np.array(sigma_noise)

    sigma_tot = np.var(voxel_data, ddof=1, axis=1)
    reliability = 1 - (sigma_noise / sigma_tot)

    return reliability, sigma_noise, sigma_tot


def _neg_loglik(a, x):
    if a <= 0:
        return np.inf
    return -(len(x) * np.log(a) + (a - 1) * np.sum(np.log(x)))


def sample_noise_values(
    noise_var: NDArray,
    total_var: NDArray,
    n_vals: int = 10,
    method: str = "mle",
    a: Optional[float] = None,
    b: Optional[float] = None,
    return_params: bool = False,
) -> Tuple[NDArray, float, float] | NDArray:
    """
    Obtain noise values to use on the analysis. A beta-distribution is q fitted
    to the participant's SNR and values are sampled from the inverse CDF in
    order to sample more densely where the noise distribution's likelihood is
    higher
    """
    noise_ratio = noise_var / total_var
    # Fix b=1 and estimate a via maximum likelihood estimation
    if method == "mle":
        optimizer = minimize(
            _neg_loglik, x0=np.array([1.0]), args=(noise_ratio,), bounds=[(1e-6, None)]
        )
        a = optimizer.x[0]
        b = 1.0
    elif method == "fixed":
        if a is None or b is None:
            raise ValueError(
                f"Method {method} requires parameters a and b to be specified"
            )
    else:
        raise NotImplementedError(f"Invalid method {method}")

    # Sample from the CDF and obtain the corresponding values in the X-axis
    n_bins = 2 * n_vals
    y_vals = np.linspace(1 / n_bins, (n_bins - 1) / n_bins, n_vals)
    noise_vals = beta.ppf(y_vals, a, b)

    if return_params:
        return noise_vals, a, b
    else:
        return noise_vals
