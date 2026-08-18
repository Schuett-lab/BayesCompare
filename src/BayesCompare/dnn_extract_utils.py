import torch
import numpy as np
import warnings
import torchlens as tl
import os
import h5py

from numpy.typing import NDArray
from typing import Optional, List


def check_act_dims(
    act: NDArray | torch.Tensor, N: int, module_name: str
) -> NDArray | torch.Tensor | None:
    """
    Checks and reshapes activation tensor dimensions to ensure N (number of images) is the first dimension.

    If N is the first dimension, the tensor itself is returned.
    If N is not the first dimension, the tensor is permuted to move it to the first position.
    If N is not found in any dimension, a warning is issued.

    Parameters
    ----------
    act : torch.Tensor or np.ndarray
        Activation tensor whose dimensions need to be checked and potentially reordered.
    N : int
        Number of images used for obtaining the covariance matrix. This dimension should exist in act.
    module_name : str
        Indicator of the activation type. It is either 'torch' or 'numpy'

    Returns
    -------
    activations : torch.Tensor or np.ndarray or None
        Activation tensor with N as the first dimension, or None if N is not found
        in any dimension of the input tensor.

    Raises
    ------
    UserWarning
        If N is not found in any dimension of the activation tensor.
    """
    shape = list(act.shape)

    if shape[0] == N:
        activations = act

    elif N in shape:
        n_dim = shape.index(N)  # find which dimension equals n
        perm = [n_dim] + [i for i in range(len(shape)) if i != n_dim]

        if module_name == "numpy":
            activations = act.transpose(perm)
        elif module_name == "torch":
            activations = act.permute(perm)

    elif N not in shape:
        warnings.warn("This layer does not have a number of images dimension.")
        return None

    return activations


def get_cov(
    activations: NDArray | torch.Tensor,
    N: Optional[int] = None,
    detach: bool = False,
) -> NDArray | torch.Tensor:
    """
    Computes the covariance matrix of the given flattened activations.

    Parameters
    ----------
    activations : torch.Tensor or np.ndarray
        The activations from which to compute the covariance matrix. The first dimension is assumed to be the number of images unless N is provided.
    N : int, optional, default None
        The number of images used for obtaining the covariance matrix.
        If provided, the function checks that the first dimension of activations matches N.
        If it does not match, it reorders dimensions accordingly.
    detach : bool, default False
        Flag for detaching activations from the network. By default, it keeps the activations (and hence the covariances) attached to the model.
        Considered only if the activations are torch tensors.

    Returns
    -------
    cov_matrix : torch.Tensor or np.ndarray
        Covariance matrix computed from the activations.

    Raises
    ------
    NotImplementedError
        If activations is neither a torch tensor nor a numpy array.
    """
    if torch.is_tensor(activations):
        module = torch
        module_name = "torch"
        if detach:
            activations = activations.detach().clone()
    elif isinstance(activations, np.ndarray):
        module = np
        module_name = "numpy"
    else:
        raise NotImplementedError(
            "Activations must be either a torch tensor or a numpy array."
        )

    # if the number of images is provided as an input check whether the first dimension of activations is equal to the number of images used
    if N != None:
        activations = check_act_dims(activations, N, module_name)

    act = module.reshape(activations, [activations.shape[0], -1])
    return module.matmul(act, act.T)


def create_covs_dict(
    trace_obj: tl.Trace,
    layer_names: List[str],
    compute_covs: bool,
    flatten_acts: bool,
    save_raw_output: bool,
) -> dict:
    """
    Creates a dict of covariances/activations from the trace object
    """
    covs_dict = {}
    for layer_name in layer_names:
        if compute_covs:
            covs_dict[layer_name] = trace_obj[layer_name].transformed_out
        else:
            if flatten_acts:
                covs_dict[layer_name] = trace_obj[layer_name].tensor.reshape(
                    trace_obj[layer_name].tensor.shape[0], -1
                )
            else:
                covs_dict[layer_name] = trace_obj[layer_name].tensor
    if save_raw_output:
        covs_dict["output_1"] = trace_obj["output_1"].tensor
    return covs_dict


def get_model_device_dtype(model: torch.nn.Module):
    """Finds on which device a model is and with which datatype a model works"""
    try:
        p = next(model.parameters())
        device = p.device
        dtype = p.dtype if p.dtype.is_floating_point else torch.float32
    except StopIteration:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


def make_mock_input(
    model: torch.nn.Module,
    input_shape: tuple[int, ...] | None = None,
    batch_size: int = 1,
) -> torch.Tensor:
    device, dtype = get_model_device_dtype(model)
    """ Creates mock input given a torch.nn network"""
    if input_shape is not None:
        return torch.randn(*input_shape, device=device, dtype=dtype)

    # Try to infer from the first recognizable layer
    for module in model.modules():
        if module is model:
            continue

        if isinstance(module, torch.nn.Linear):
            return torch.randn(
                batch_size, module.in_features, device=device, dtype=dtype
            )

        if isinstance(module, torch.nn.Conv1d):
            return torch.randn(
                batch_size, module.in_channels, 224, device=device, dtype=dtype
            )

        if isinstance(module, torch.nn.Conv2d):
            return torch.randn(
                batch_size, module.in_channels, 224, 224, device=device, dtype=dtype
            )

        if isinstance(module, torch.nn.Conv3d):
            return torch.randn(
                batch_size, module.in_channels, 16, 224, 224, device=device, dtype=dtype
            )

    raise ValueError(
        "Could not infer a valid input shape. Please provide input_shape explicitly."
    )


def check_hdf_exists_save_acts(
    hdf5_dir, activations, num_inputs, dset_name, first_creation
):
    """
    Checks if the HDF5 file exists in the given directory
    """
    if first_creation:
        with h5py.File(hdf5_dir, "a") as f:
            if dset_name not in f:
                f.create_dataset(
                    dset_name,
                    shape=(activations.shape[0], *activations.shape[1:]),
                    maxshape=(num_inputs, *activations.shape[1:]),
                    data=activations,
                    chunks=True,
                )
            else:
                raise FileExistsError(
                    f"Dataset {dset_name!r} already exists in HDF5 file {hdf5_dir!r}. "
                    "Please remove the existing dataset, choose a different dataset name, "
                    "or use a different output file."
                )
    elif not first_creation and os.path.exists(hdf5_dir):
        with h5py.File(hdf5_dir, "a") as f:
            dset = f[dset_name]
            prev_len = len(dset)
            dset.resize(
                (
                    prev_len + activations.shape[0],
                    *activations.shape[1:],
                )
            )
            dset[-activations.shape[0] :] = activations
