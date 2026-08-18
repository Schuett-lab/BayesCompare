import os
import numpy as np
import torch
import h5py
import tqdm
import pickle

from .distances import REGISTRY, SIMILARITIES
from .cov_utils import (
    check_input_format,
    cov_trace_norm_sigma_N,
    trace_norm_N,
    cov_sigma_N,
)
from .dist_utils import simplify_string

from multiprocessing.queues import Queue
from typing import Optional, Callable, Sequence
from numpy.typing import NDArray
from types import ModuleType


def select_measure(
    cov_mtx: NDArray | torch.Tensor, meas_name: str, module: Optional[ModuleType] = None
) -> Callable:
    """
    Select and return the appropriate distance measure function based on input covariances type (NumPy array or PyTorch tensor) and measure name.

    Parameters
    ----------
    cov_mtx : np.ndarray or torch.Tensor
        The covariance matrix for which to select a measure. Used to infer the module
        type if not explicitly provided, and to determine device placement (CPU/GPU)
        for PyTorch tensors.
    meas_name : str
        The name of the distance measure to use. Case-insensitive. Supported measures are listed in `DISTANCES` dictionary below.
    module : {np, torch}, optional
        The module type indicating whether to use NumPy or PyTorch implementations.
        If None (default), the module type is inferred from cov_mtx using check_input_format.

    Returns
    -------
    measure : callable
        A function that computes the selected distance measure between two covariance
        matrices. For stochastic measures (TVD, JSD), returns a functools.partial object
        with the appropriate random generator pre-configured.

    Raises
    ------
    NotImplementedError
        If the metric name is not valid for the given module type, or if the covariance
        matrix is neither a NumPy array nor a PyTorch tensor.
    """

    if module == None:
        module = check_input_format(cov_mtx)

    meas_name = simplify_string(meas_name)

    if module == np:
        try:
            measure = REGISTRY["numpy"][meas_name]
        except KeyError:
            raise NotImplementedError(
                "Given metric name is not valid for Numpy array covariances."
            )
    elif module == torch:
        try:
            measure = REGISTRY["torch"][meas_name]
        except KeyError:
            raise NotImplementedError(
                "Given metric name is not valid for Tensor tensor covariances."
            )
    else:
        raise NotImplementedError(
            "Covariance matrices must be either a torch tensor or a numpy array."
        )

    return measure


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
            if measure_name in SIMILARITIES:
                np.fill_diagonal(init_mtx, 1)
            else:
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
            covs_loaded = pickle.load(f)

        if isinstance(covs_loaded, list) and isinstance(covs_loaded[0], dict):
            covs = []
            for cov_dict in covs_loaded:
                covs.append(list(cov_dict.values()))

            covs = np.stack(covs)
            covs = covs.reshape(
                covs.shape[0] * covs.shape[1], covs.shape[2], covs.shape[3]
            )

        elif isinstance(covs_loaded, np.ndarray):
            covs = covs_loaded

    elif ext in numpy_exts:
        filename_ext = os.path.basename(full_filename)
        filename, ext = os.path.splitext(filename_ext)

        covs = np.load(full_filename)

    return covs, filename


def preprocess_input_covs(
    covs: NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor],
    noise_var: Optional[float] = None,
    b: float = 0.01,
    normalize: bool = True,
):
    """
    Applies specified preprocessing steps to covariances:
    trace normalization, and noise addition using either noise_var or b
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

    return covs


def get_preprocessing_params(meas_name, b, noise_var, normalize):
    """
    Converts required preprocessing parameters into lists.
    """
    if isinstance(meas_name, str):
        meas_name = [meas_name]

        if isinstance(b, float):
            b_list = [b]
        else:
            raise TypeError(
                f"When single measure name is given, `b` can only be a float! Given b type is {type(b)}"
            )

        if isinstance(noise_var, float):
            noise_var_list = [noise_var]
        elif noise_var is None:
            noise_var_list = None
        else:
            raise TypeError(
                f"When single measure name is given, `noise_var` can only be a float! Given noise_var type is {type(noise_var)}"
            )

        if isinstance(normalize, bool):
            normalize_list = [normalize]
        else:
            raise TypeError(
                f"When single measure name is given, `normalize` can only be a bool! Given normalize type is {type(normalize)}"
            )

    elif isinstance(meas_name, list):

        if isinstance(b, float):
            b_list = [b] * len(meas_name)
        elif isinstance(b, list):
            if isinstance(b[0], float):
                b_list = b
            else:
                raise TypeError(
                    f"`b` can either be a float or a list of floats! Given b type is {type(b)}"
                )
        else:
            raise TypeError(
                f"`b` can either be a float or a list of floats! Given b type is {type(b)}"
            )

        if isinstance(noise_var, float):
            noise_var_list = [noise_var] * len(meas_name)
        elif isinstance(noise_var, list):
            if isinstance(noise_var[0], float):
                noise_var_list = noise_var
            else:
                raise TypeError(
                    f"`noise_var` can either be a float or a list of floats! Given noise_var type is {type(noise_var)}"
                )
        elif noise_var is None:
            noise_var_list = None
        else:
            raise TypeError(
                f"`noise_var` can either be a float or a list of floats! Given noise_var type is {type(noise_var)}"
            )

        if isinstance(normalize, bool):
            normalize_list = [normalize] * len(meas_name)
        elif isinstance(normalize, list):
            if isinstance(normalize[0], bool):
                normalize_list = normalize
            else:
                raise TypeError(
                    f"`normalize` can either be a bool or a list of bools! Given normalize type is {type(normalize)}"
                )
        else:
            raise TypeError(
                f"`normalize` can either be a bool or a list of bools! Given normalize type is {type(normalize)}"
            )

    else:
        raise TypeError(
            f"Measure name cannot be {type(meas_name)}. It can either be a string or a list of strings."
        )

    return meas_name, b_list, noise_var_list, normalize_list
