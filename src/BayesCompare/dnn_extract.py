import torch
import numpy as np
import torchlens as tl
from functools import partial
from .dnn_extract_utils import get_cov, create_covs_dict

import h5py
import tqdm
import pickle
import os

from typing import List
from numpy.typing import NDArray


def get_layer_names(
    model: torch.nn.Module, random_seed: int = 42, eval_mode: bool = True
) -> List[str]:
    """
    Retrieves the names of all layers in a DNN model, visible to TorchLens under torch.inference_mode().

    Parameters
    ----------
    model : torch.nn.Module)
        The DNN model from which to extract layer names.
    random_seed : int, default 42
        Fixed RNG seed for reproducibility with stochastic models.
    eval_mode : bool, default True
        Model activations/covariances are obtained when the model is in eval mode (by default, True).

    Returns
    -------
    all_layers : list of strings
        A list of layer names in the model.
    """
    mock_input = torch.rand(1, 3, 224, 224)
    if eval_mode:
        model.eval()

    with torch.inference_mode():
        all_layers_obj = tl.trace(
            model, mock_input, layers_to_save="none", random_seed=random_seed
        )
        all_layer_names = list(all_layers_obj.layer_dict_main_keys.keys())

    return all_layer_names


def cov_extractor_grad(
    model: torch.nn.Module,
    inputs: NDArray | torch.Tensor,
    layer_list: str | List[str],
    random_seed: int = 42,
    compute_covs: bool = True,
) -> dict:
    """
    Extracts covariance matrices from specified layers of a DNN model given input data.

    Parameters
    ----------
    model : torch.nn.Module
        The DNN model from which to extract covariances.
    inputs : numpy array or a torch tensor
        An input tensor or array of images to extract covariances from. First dimension is expected to be the number of images.
    layer_list : str or a list of strings
        A single layer name or a list of layer names for which to compute the covariance matrices.
    random_seed : int, default 42
        Fixed RNG seed for reproducibility with stochastic models.
    compute_covs : bool, default True
        Flag for specifiying whether to get covariance (True) or activations (False) from the model.
        Defaults to True, meaning that covariances will be returned.

    Returns
    -------
    covs : dict
        A dictionary where keys are layer names and values are the corresponding covariance/activation matrices.

    Raises
    ------
    UserWarning
        If a specified layer does not have a dimension matching the number of input images.
    """
    if type(layer_list) == str:
        layer_list = [layer_list]

    N = inputs.shape[0]
    postfunc = partial(get_cov, N=N) if compute_covs else None

    if compute_covs:
        trace = tl.trace(
            model,
            inputs,
            layers_to_save=layer_list,
            out_transform=postfunc,
            random_seed=random_seed,
            backward_ready="True",
            save_raw_outs="False",
        )
    else:
        trace = tl.trace(
            model,
            inputs,
            layers_to_save=layer_list,
            random_seed=random_seed,
            backward_ready="True",
        )

    return create_covs_dict(trace, layer_list, compute_covs)


def act_extractor_batch(
    model: torch.nn.Module,
    inputs: NDArray | torch.Tensor,
    layer_list: str | List[str],
    out_filename: str,
    out_dir: str,
    batch_size: int = 10,
    layer_by_layer: bool = False,
    random_seed: int = 42,
    eval_mode: bool = True,
):
    """
    Extracts and saves activations from specified layers of a DNN model given input data to the disk as HDF5 files.

    Parameters
    ----------
    model : torch.nn.Module
        The DNN model from which to extract covariances.
    inputs : numpy array or a torch tensor
        An input tensor or array of images to extract covariances from. First dimension is expected to be the number of images.
    layer_list : str or a list of strings
        A single layer name or a list of layer names for which to compute the covariance matrices.
    out_filename : str
        Filename with which the activation files are saved as.
        If layer_by_layer is `False`, activations files are saved as `activations_{out_filename}.hdf5`.
        If layer_by_layer is `True`, activations files are saved as `activations_{out_filename}_{layer_name}.hdf5`.
    out_dir : str
        Directory where the output activation HDF5 files will be saved.
    batch_size : int, default 10
        Size of the batches that will be used to get covariances/batches.
    layer_by_layer : bool, default False
        Flag for indicating to compute activations layer by layer and then to save separate layer files.
        False is for saving all layer activations into the same HDF5 file.
    random_seed : int, default 42
        Fixed RNG seed for reproducibility with stochastic models.
    eval_mode : bool, default True
        Model activations/covariances are obtained when the model is in eval mode (by default, True).
    """
    if type(layer_list) == str:
        layer_list = [layer_list]

    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)

    if layer_by_layer:
        layer_out_list = [
            f"{out_dir}/activations_{out_filename}_{layer_name}.hdf5"
            for layer_name in layer_list
        ]

        for layer_out_file in layer_out_list:
            with h5py.File(layer_out_file, "w") as f:
                f.create_dataset(
                    "activations",
                    shape=(0, 0),
                    maxshape=(inputs.shape[0], None),
                    dtype="f",
                )
            print(
                f"\nCreated empty hdf5 file {layer_out_file} (max shape: inputs.shape[0] = {inputs.shape[0]})"
            )
    else:
        with h5py.File(f"{out_dir}/activations_{out_filename}.hdf5", "w") as f:
            for layer_name in layer_list:
                f.create_dataset(
                    "activations_" + layer_name,
                    shape=(0, 0),
                    maxshape=(inputs.shape[0], None),
                    dtype="f",
                )
            print(
                f"\nCreated empty hdf5 file 'activations_{out_filename}.hdf5' (max shape: inputs.shape[0] = {inputs.shape[0]})"
            )

    if eval_mode:
        model.eval()

    with torch.inference_mode():
        if layer_by_layer:
            for k, layer_name in enumerate(layer_list):
                for batch in tqdm.tqdm(
                    torch.split(inputs, batch_size),
                    desc=f"Batches - Activation Extraction - Layer {layer_name}",
                    position=1,
                ):
                    activations = tl.batched_extract(
                        model=model,
                        stimuli=batch,
                        layers=layer_name,
                        batch_size=batch_size,
                    )
                    activations[layer_name] = (
                        torch.reshape(
                            activations[layer_name],
                            [activations[layer_name].shape[0], -1],
                        )
                        .detach()
                        .cpu()
                    )
                    with h5py.File(layer_out_list[k], "a") as f:
                        dset = f["activations"]
                        prev_len = len(dset)
                        dset.resize(
                            (
                                prev_len + activations[layer_name].shape[0],
                                activations[layer_name].shape[1],
                            )
                        )
                        dset[-activations[layer_name].shape[0] :] = activations[
                            layer_name
                        ]
                print(
                    f"Activations for layer {layer_name} are saved at {layer_out_list[k]}."
                )
            print(f"All activations are saved to HDF5 files at the directory {out_dir}")
        else:
            for batch in tqdm.tqdm(
                torch.split(inputs, batch_size),
                desc=f"Batches - Activation Extraction - All Layers",
                position=1,
            ):
                activations = tl.batched_extract(
                    model=model,
                    stimuli=batch,
                    layers=layer_list,
                    batch_size=batch_size,
                )
                for layer_name in layer_list:
                    activations[layer_name] = (
                        torch.reshape(
                            activations[layer_name],
                            [activations[layer_name].shape[0], -1],
                        )
                        .detach()
                        .cpu()
                    )
                    with h5py.File(
                        f"{out_dir}/activations_{out_filename}.hdf5", "a"
                    ) as f:
                        dset = f["activations_" + layer_name]
                        prev_len = len(dset)
                        dset.resize(
                            (
                                prev_len + activations[layer_name].shape[0],
                                activations[layer_name].shape[1],
                            )
                        )
                        dset[-activations[layer_name].shape[0] :] = activations[
                            layer_name
                        ]
            print(
                f"All activations are saved to '{out_dir}/activations_{out_filename}.hdf5'"
            )


def compute_covs_from_act_files(
    layer_list: str | List[str],
    out_filename: str,
    out_dir: str,
    layer_by_layer: bool,
):
    """
    Reads from the saved activation files, then computes and saves the covariances to the disk as pickle files.

    Parameters
    ----------
    layer_list : str or a list of strings
        A single layer name or a list of layer names for which to compute the covariance matrices.
    out_filename : str
        Filename with which the covariance files are saved as.
        If layer_by_layer is `False`, covariance files are saved as `covs_{out_filename}.pkl`.
        If layer_by_layer is `True`, covariance files are saved as `covs_{out_filename}_{layer_name}.pkl`.
    out_dir : str
        Directory where the output covariance files will be saved.
    layer_by_layer : bool, default False
        Flag for indicating to compute covariances layer by layer and then to save separate layer files.
        False is for saving all layer covariances to the same pkl file as a dict where the keys are layer names.
    """
    if layer_by_layer:
        layer_out_list = [
            f"{out_dir}/activations_{out_filename}_{layer_name}.hdf5"
            for layer_name in layer_list
        ]
        for k, activations_filename in enumerate(layer_out_list):
            layer_name = layer_list[k]

            with h5py.File(activations_filename) as f:
                activations = f["activations"][...]
                layer_cov = get_cov(activations)

            cov_out_filename = f"{out_dir}/covs_{out_filename}_{layer_name}.pkl"
            with open(cov_out_filename, "wb") as f:
                pickle.dump(layer_cov, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\nSaved covariance for layer {layer_name} at {cov_out_filename}.")
    else:
        covs_dict = {}
        for k, layer_name in enumerate(layer_list):
            with h5py.File(f"{out_dir}/activations_{out_filename}.hdf5", "r") as f:
                activations = f["activations_" + layer_name][...]
                layer_cov = get_cov(activations)
            covs_dict[layer_name] = layer_cov

        cov_out_filename = f"{out_dir}/covs_{out_filename}.pkl"
        with open(cov_out_filename, "wb") as f:
            pickle.dump(covs_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved covariance for layer {layer_name} at {cov_out_filename}.")


def cov_extractor_batch(
    model: torch.nn.Module,
    inputs: NDArray | torch.Tensor,
    layer_list: str | List[str],
    out_filename: str,
    out_dir: str,
    batch_size: int = 10,
    layer_by_layer: bool = False,
    random_seed: int = 42,
    compute_covs: bool = True,
    delete_act_files: bool = True,
    eval_mode: bool = True,
):
    """
    Extracts and saves covariances/activations from specified layers of a DNN model given input data to the disk as HDF5 (if activations) or pickle (if covariances) files.

    Parameters
    ----------
    model : torch.nn.Module
        The DNN model from which to extract covariances.
    inputs : numpy array or a torch tensor
        An input tensor or array of images to extract covariances from. First dimension is expected to be the number of images.
    layer_list : str or a list of strings
        A single layer name or a list of layer names for which to compute the covariance matrices.
    out_filename : str
        Filename with which the covariance/activations files are saved as.
        If layer_by_layer is `False`, activations files are saved as `activations_{out_filename}.hdf5` and covariance files are saved as `covs_{out_filename}.pkl`.
        If layer_by_layer is `True`, activations files are saved as `activations_{out_filename}_{layer_name}.hdf5` and covariance files are saved as `covs_{out_filename}_{layer_name}.pkl`.
    out_dir : str
        Directory where the output covariance/activation files will be saved.
    batch_size : int, default 10
        Size of the batches that will be used to get covariances/batches.
    layer_by_layer : bool, default False
        Flag for indicating to compute covariances/activations layer by layer and then to save separate layer files.
        False is for saving all layer activations into the same HDF5 file or all layer covariances to the same pkl file as a dict where the keys are layer names.
    random_seed : int, default 42
        Fixed RNG seed for reproducibility with stochastic models.
    compute_covs : bool, default True
        Flag for specifiying whether to get covariance (True) or activations (False) from the model.
        Defaults to True, meaning that covariances will be saved.
    delete_act_files : bool, default True
        Flag for specifying whether to delete activation files from the disk.
        Ignored when compute_covs is False (when only the activations are saved from the model).
        When both `delete_act_files" and `compute_covs` is True, only the covariance files are saved to the disk.
        When `delete_act_files" is False and `compute_covs` is True, both activation files and covariance files are saved to the disk.
    eval_mode : bool, default True
        Model activations/covariances are obtained when the model is in eval mode (by default, True).

    Raises:
    ----------
    UserWarning
        If a specified layer does not have a dimension matching the number of input images.
    """
    act_extractor_batch(
        model=model,
        inputs=inputs,
        layer_list=layer_list,
        out_filename=out_filename,
        out_dir=out_dir,
        batch_size=batch_size,
        layer_by_layer=layer_by_layer,
        random_seed=random_seed,
        eval_mode=eval_mode,
    )
    if compute_covs:
        compute_covs_from_act_files(
            layer_list=layer_list,
            out_filename=out_filename,
            out_dir=out_dir,
            layer_by_layer=layer_by_layer,
        )
        if delete_act_files:
            if layer_by_layer:
                layer_out_list = [
                    f"{out_dir}/activations_{out_filename}_{layer_name}.hdf5"
                    for layer_name in layer_list
                ]
            else:
                layer_out_list = [f"{out_dir}/activations_{out_filename}.hdf5"]
            for layer_activation_file in layer_out_list:
                os.remove(layer_activation_file)
