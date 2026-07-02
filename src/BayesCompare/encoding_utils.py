import numpy as np

from itertools import groupby
from joblib import Parallel, delayed


def _check_consistent_repetitions(stim_list):
    """Check if all elements of the list are repeated the same number of times"""

    counts = np.array([len(list(group)) for _, group in groupby(stim_list)])
    constant = (counts[0] == counts).all()
    return constant, counts


def noise_estimation(data, split_param):
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


def voxel_reliability(voxel_data, stim_list, n_jobs=-1):
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
