"""
Module for camputing distances between prior predictive distributions (BayesCompare measures)
or existing representational similarity measures in the literature computed with second moment matrix in parallel.
Authors: Sezan Oral
"""

import numpy as np
import torch
import tqdm

from joblib import Parallel, delayed
import multiprocessing as mp

from BayesCompare.dist_utils import (
    simplify_string,
)
from .cov_utils import (
    cov_trace_norm_sigma_N,
    check_cov_symmetry,
    check_and_change_input_format,
    trace_norm_N,
    cov_sigma_N,
)
from .measure_distance_utils import select_measure, load_covs, check_saved_hdf, writer

from numpy.typing import NDArray
from typing import Optional, Sequence


def measure_dist(
    covs: NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor],
    mean: Optional[NDArray | torch.Tensor | Sequence[int]] = None,
    meas_name: str = "TVD",
    noise_var: Optional[float] = None,
    b: float = 0.0,
    normalize: bool = True,
    samples_jsd_tvd: int = 10000,
    lmbd: Optional[float] = None,
    k: Optional[int] = None,
    generator: Optional[np.random.Generator | torch.Generator] = None,
    show_progress: bool = True,
) -> NDArray | torch.Tensor:
    """
    Compute a symmetric pairwise distance matrix from the list of covariance matrices and optionally mean vectors.

    It expects covariances matrices to be symmetric and positive semi-definite.
    It trace-normalizes the covariance matrices to have trace equals to N (dimension of the square matrix) by default and
    converts covariance arrays into a list of square matrices.

    Parameters
    ----------
    covs : NDArray or torch.Tensor or a list/tuple of either of these types
        A NumPy array or PyTorch tensor with shape (N, dim, dim) or a list/tuple of N matrices with shape (dim, dim).
    mean : array-like, optional
        Mean vactor of shape (N, dim).
    meas_name : str, optional
        Name of the distance/divergence measure to use. This name is resolved via `select_measure(meas_name)`. Default is "TVD".
    noise_var : float, optional
        Additive noise variance to be added to the covariance matrices.
        If None, and a `b` value is provided, then noise_var is computed from the number of images (dim)
        used to obtain the cov matrix and the parameter `b` using the formula
        noise_var = (dim * b) / (1 + (dim * b)).
        It overwrites b if both noise_var and b is provided. Default is None.
    b : float, optional
        Scalar used to compute a default `noise_var` when `noise_var` is None. Default is 0.
    normalize : bool, optional
        Flag for selecting to apply trace normalization or not. Defaults to True (normalization is applied by default).
    samples_jsd_tvd : integer, optional
        Number of samples used for computing JSD and TVD measures. Defaults to 10000.
    lmbd : float, optional
        Lambda parameter for the GULP distance.
    k : integer, optional
        k parameter for determining the k-nearest neighbors for computing Jaccard similarity.
    generator: np.random.Generator or torch.Generator, optional
        A Generator object for the randomization for generating samples for JSD and TVD computations. Default is None.
    show_progress : bool, optional
        Boolean to turn the tqdm progress bars on (True) or off (False). Default is on (True).

    Returns
    -------
    dist : numpy.ndarray or torch.Tensor
        A symmetric 2-D array of shape (N, N) containing pairwise distances covariance inputs.
        The diagonal elements are zero.
        Only the upper triangle (j > i) is computed explicitly and mirrored to the
        lower triangle.

    Examples
    --------
    >>> # Given a list of covariance matrices `cov_list`
    >>> dist_matrix = measure_dist(cov_list, meas_name="TVD")
    """
    # normalize and add noise
    if normalize and (b != 0 or noise_var):
        if noise_var == None:  # if noise_var was not given but b is given
            dim = covs[0].shape[0]  # number of images used for obtaining one cov matrix
            noise_var = dim * b / (1 + (dim * b))

        covs = cov_trace_norm_sigma_N(covs, noise_var=noise_var)

    # normalize but don't add noise
    elif normalize and not (b) and not (noise_var):
        covs = trace_norm_N(covs)

    # don't normalize but add noise
    elif not (normalize) and (b != 0 or noise_var):
        if noise_var == None:  # if noise_var was not given but b is given
            dim = covs[0].shape[0]  # number of images used for obtaining one cov matrix
            noise_var = dim * b / (1 + (dim * b))

        covs = cov_sigma_N(covs, noise_var=noise_var)

    covs, N, module = check_and_change_input_format(covs)

    idx = np.random.randint(len(covs))
    symmetric = check_cov_symmetry(covs[idx])
    if not symmetric:
        raise ValueError(
            f"Covariance matrices should be symmetric! The covariance matrix at index {idx} violates this condition."
        )

    meas_name = simplify_string(meas_name)
    measure = select_measure(covs[0], meas_name, module=module)

    dist = module.zeros((N, N))

    progress_bar = tqdm.tqdm(total=int((N * (N - 1)) / 2), disable=not show_progress)

    for i, ci in enumerate(covs):
        for j, cj in enumerate(covs):
            if j > i:
                if "tvd" in meas_name or "jsd" in meas_name:
                    dist[i, j] = measure(
                        ci, cj, num_samples=samples_jsd_tvd, gen=generator
                    )  # not using mean, for a generalized code mean should be provided
                elif "gulp" in meas_name:
                    dist[i, j] = measure(ci, cj, lmbd=lmbd)
                elif "jaccard" in meas_name:
                    dist[i, j] = measure(ci, cj, k=k)
                else:
                    dist[i, j] = measure(
                        ci, cj
                    )  # not using mean, for a generalized code mean should be provided

                dist[j, i] = dist[i, j]
                progress_bar.update(1)

    return dist


def measure_dist_parallel(
    covs_dir: str,
    output_dir: str,
    mean: Optional[str] = None,
    meas_name: str | Sequence[str] = ["TVD"],
    noise_var: Optional[float] = None,
    b: float = 0.0,
    normalize: bool = True,
    samples_jsd_tvd: int = 10000,
    lmbd: Optional[float] = None,
    k: Optional[int] = None,
    generator: Optional[np.random.Generator | torch.Generator] = None,
    num_workers: int = mp.cpu_count() - 1,
) -> None:
    """
    Compute pairwise distances between covariance matrices in parallel and save results to disk.

    This function loads covariance matrices from a directory, selects one or more distance
    measures, and computes the requested pairwise distances in parallel. Results are written
    incrementally to an HDF file by a dedicated writer process. Previously computed distances can be
    reused: when an output HDF already contains the requested entries, computation for those
    pairs is skipped.

    Parameters
    ----------
    covs_dir : str
        Path to the directory (or file) containing the covariance matrices to load. The
        helper function `load_covs` is used to read covariances and associated filenames.
    output_dir : str
        Directory where computed distance matrices (HDF files) will be stored. The helper
        `check_saved_hdf` is used to determine an output filename and which pair indices
        still need to be computed.
    mean : str, optional
        Path to the directory (or file) containing the mean vectors to be load.
    meas_name : str or sequence of str, optional
        Name of the distance measure to compute (e.g. "TVD") or a list/tuple of measure
        names. Each name is resolved to a callable via `dist.select_measure`. Default is
        "TVD".
    noise_var : float, optional
        Additive noise variance to be added to the covariance matrices.
        If None, and a `b` value is provided, then noise_var is computed from the number of images (dim)
        used to obtain the cov matrix and the parameter `b` using the formula
        noise_var = (dim * b) / (1 + (dim * b)).
        It overwrites b if both is provided. Default is None.
    b : float, optional
        Scalar used to compute a default `noise_var` when `noise_var` is None. Default is 0.
    normalize : bool, optional
        Flag for selecting to apply trace normalization or not. Defaults to True (normalization is applied by default).
    samples_jsd_tvd : integer, optional
        Number of samples used for computing JSD and TVD measures. Defaults to 10000.
    lmbd : float, optional
        Lambda parameter for the GULP distance.
    k : integer, optional
        k parameter for determining the k-nearest neighbors for computing Jaccard similarity.
    generator: np.random.Generator or torch.Generator, optional
        A Generator object for the randomization for generating samples for JSD and TVD computations. Default is None.
    num_workers : int, optional
        Number of worker processes for parallel computation. By default this is set to
        (number of CPUs - 1). Must be >= 1. The function uses joblib.Parallel with the
        "loky" backend to dispatch work.

    Returns
    -------
    None
        Results are saved to HDF files in `output_dir`. The function has no return value.

    Raises
    ------
    AssertionError
        If the loaded covariance matrices are not trace-normalized. The check is performed
        by `check_normalization`.

    Notes
    -----
    - If `check_saved_hdf` determines that all requested distances are already present,
      no computation is performed.

    Examples
    --------
    Basic usage:
    >>> measure_dist_parallel("/path/to/covs.pkl", "/path/to/output", meas_name="TVD")
    Compute several measures:
    >>> measure_dist_parallel("/covs.npy", "/out", meas_name=["TVD", "JSD"], num_workers=8)
    """
    # check if a single string or a list of strings is given in the meas_name
    if isinstance(meas_name, str):
        meas_name = [meas_name]

    covs, covs_filename = load_covs(covs_dir)

    # normalize and add noise
    if normalize and (b != 0 or noise_var):
        if noise_var == None:  # if noise_var was not given but b is given
            dim = covs[0].shape[0]  # number of images used for obtaining one cov matrix
            noise_var = dim * b / (1 + (dim * b))

        covs = cov_trace_norm_sigma_N(covs, noise_var=noise_var)

    # normalize but don't add noise
    elif normalize and not (b) and not (noise_var):
        covs = trace_norm_N(covs)

    # don't normalize but add noise
    elif not (normalize) and (b != 0 or noise_var):
        if noise_var == None:  # if noise_var was not given but b is given
            dim = covs[0].shape[0]  # number of images used for obtaining one cov matrix
            noise_var = dim * b / (1 + (dim * b))

        covs = cov_sigma_N(covs, noise_var=noise_var)

    N = len(covs)
    idx = np.random.randint(N)
    symmetric = check_cov_symmetry(covs[idx])
    if not symmetric:
        raise ValueError(
            f"Covariance matrices should be symmetric! The covariance matrix at index {idx} violates this condition."
        )

    for name in meas_name:
        measure = select_measure(covs[0], name)
        indices, output_filename = check_saved_hdf(output_dir, N, covs_filename, name)

        if len(indices) == 0:
            print("Distance already calculated")
        else:
            with mp.Manager() as manager:

                output_queue = manager.Queue(2 * num_workers)

                writer_proc = mp.Process(
                    target=writer, args=(output_filename, output_queue, len(indices))
                )
                writer_proc.start()

                def pairwise_dist(i, j):
                    if "tvd" in name or "jsd" in name:
                        val = measure(
                            covs[i], covs[j], num_samples=samples_jsd_tvd, gen=generator
                        )
                    elif "gulp" in name:
                        val = measure(
                            covs[i],
                            covs[j],
                            lmbd=lmbd,
                        )
                    elif "jaccard" in meas_name:
                        val = measure(
                            covs[i],
                            covs[j],
                            k=k,
                        )
                    else:
                        val = measure(covs[i], covs[j])
                    output_queue.put((i, j, val))

                Parallel(n_jobs=num_workers, backend="loky", verbose=0)(
                    delayed(pairwise_dist)(i, j) for i, j in indices
                )

                output_queue.put(None)
                writer_proc.join()
