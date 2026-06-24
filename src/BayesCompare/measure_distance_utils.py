import os
import numpy as np
import torch
import h5py
import tqdm
import pickle

from .distances import REGISTRY
from .cov_utils import check_input_format
from .dist_utils import simplify_string

from typing import Optional, Callable
from numpy.typing import NDArray
from types import ModuleType


def select_measure(
    cov_mtx: NDArray | torch.Tensor, meas_name: str, module: Optional[ModuleType] = None
) -> Callable:
    """
    Select and return the appropriate distance measure function based on input covariances and measure name.

    Selects a distance/similarity measure function that is compatible with
    the given covariance matrix type (NumPy array or PyTorch tensor) and the specified
    metric name.

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


def check_saved_hdf(hdf_dir, N, covs_name, measure_name):
    # Check whether the folder exists
    if not os.path.exists(hdf_dir):
        os.makedirs(hdf_dir)

    print(f"Now computing {measure_name}")
    # convention for the name of the distance HDF5 files is: dist_<covs_list_filename>_<measure_name>.hdf5
    measure_name = simplify_string(measure_name)
    hdf_filename = (
        os.path.join(hdf_dir, "") + "dist_" + covs_name + "_" + measure_name + ".hdf5"
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


def writer(file_dir, que, total_num_ops):

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


def load_covs(full_filename):

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
