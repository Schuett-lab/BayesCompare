"""
Module for camputing distances between prior predictive distributions (BayesCompare measures)
or existing representational similarity measures in the literature computed with second moment matrix in parallel.
Authors: Sezan Oral
"""

import numpy as np
import torch

import h5py
import pickle
import tqdm
import os

from joblib import Parallel, delayed
import multiprocessing as mp

from multiprocessing.queues import Queue
from typing import Optional, Sequence
from numpy.typing import NDArray

from BayesCompare.distances import select_measure
from BayesCompare.dist_utils import simplify_string
from BayesCompare.cov_utils import (
    cov_trace_norm_sigma_N,
    trace_norm_N,
    check_cov_symmetry,
    cov_sigma_N,
)


def check_saved_hdf(
    hdf_dir: str, N: int, covs_filename: str, measure_name: str
) -> tuple[Sequence[tuple[int, int]], str]:
    """
    Checks if the distance matrix for the given measure and covariance filename already exists in the specified directory.
    If it exists, it identifies which pairwise distances are still missing (NaN) and returns their indices.
    If it does not exist, it creates a new HDF5 file with the appropriate structure and returns the indices for all pairwise
    distances that need to be computed.
    """
    # Check whether the folder exists
    if not os.path.exists(hdf_dir):
        os.makedirs(hdf_dir)

    print(f"Now computing {measure_name}")

    # convention for the name of the distance HDF5 files is: dist_<covs_list_filename>_<measure_name>.hdf5
    measure_name = simplify_string(measure_name)
    hdf_filename = (
        os.path.join(hdf_dir, "")
        + "dist_"
        + covs_filename
        + "_"
        + measure_name
        + ".hdf5"
    )

    if os.path.exists(hdf_filename):
        with h5py.File(hdf_filename, "r") as f:
            dist = f["dist"][...]
            tril_idx = np.tril_indices(dist.shape[0], k=-1)
            nan_mask = np.isnan(dist[tril_idx])
            indices = [
                [int(i), int(j)]
                for i, j in zip(tril_idx[0][nan_mask], tril_idx[1][nan_mask])
            ]
    else:
        with h5py.File(hdf_filename, "w") as f:
            init_mtx = np.empty((N, N))
            init_mtx[:] = np.nan
            np.fill_diagonal(init_mtx, 0)
            dist_dset = f.create_dataset("dist", shape=(N, N), data=init_mtx)
            indices = [(i, j) for j in range(N) for i in range(j + 1, N)]
            f.flush()

    return indices, hdf_filename


def writer(file_dir: str, que: Queue, total_num_ops: int) -> None:
    """
    Writer process that listens to a queue for computed distance values and writes them to an HDF5 file.
    It updates a progress bar as it writes the values.
    """
    progress_bar = tqdm.tqdm(total=int(total_num_ops))

    with h5py.File(file_dir, "r+") as f:
        res_dset = f["dist"]
        while 1:
            item = que.get()
            if item is None:
                break

            res_dset[item[0], item[1]] = item[2]
            res_dset[item[1], item[0]] = item[2]
            f.flush()
            progress_bar.update(1)


def load_covs(full_filename: str) -> tuple[NDArray | torch.Tensor, str]:
    """
    Loads covariance matrices from a specified file. Supports .pkl, .pickle, .np, .npy, and .npz formats.
    Returns the loaded covariance matrices and the base filename (without extension) for use in naming output files.
    """
    _, ext = os.path.splitext(full_filename)
    pckl_exts = {".pkl", ".pickle"}
    numpy_exts = {".np", ".npy", ".npz"}

    if ext in pckl_exts:
        filename_ext = os.path.basename(full_filename)
        filename, ext = os.path.splitext(filename_ext)

        with open(full_filename, "rb") as f:
            covs_names = pickle.load(f)

        if isinstance(covs_names, list) and isinstance(covs_names[0], dict):
            covs = []
            for cov_dict in covs_names:
                covs.append(list(cov_dict.values()))
            covs = np.stack(covs)
            covs = covs.reshape(
                covs.shape[0] * covs.shape[1], covs.shape[2], covs.shape[3]
            )

        if isinstance(covs_names, np.ndarray):
            covs = covs_names

    elif ext in numpy_exts:
        filename_ext = os.path.basename(full_filename)
        filename, ext = os.path.splitext(filename_ext)
        covs = np.load(full_filename)

    return covs, filename


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
