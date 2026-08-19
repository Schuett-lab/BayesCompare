import torch
import numpy as np
import torchlens as tl
from functools import partial
from .dnn_extract_utils import (
    get_cov,
    create_covs_dict,
    make_mock_input,
    check_hdf_exists_save_acts,
)

import h5py
import tqdm
import pickle
import os

from typing import List, Optional
from numpy.typing import NDArray


def get_layer_names(
    model: torch.nn.Module,
    mock_input: Optional[torch.Tensor] = None,
    random_seed: int = 42,
    eval_mode: bool = True,
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
    if mock_input is None:
        mock_input = make_mock_input(model)

    if eval_mode:
        model.eval()

    with torch.inference_mode():
        all_layers_obj = tl.trace(
            model, mock_input, layers_to_save="none", random_seed=random_seed
        )
        all_layer_names = list(all_layers_obj.layer_dict_main_keys.keys())

    return all_layer_names


def cov_extractor(
    model: torch.nn.Module,
    inputs: NDArray | torch.Tensor,
    layer_list: str | List[str],
    random_seed: int = 42,
    compute_covs: bool = True,
    gradient: bool = False,
    eval_mode: bool = True,
    inference_mode: bool = True,
    save_network_output: bool = False,
    flatten_acts: bool = False,
) -> dict:
    """
    Extracts covariance matrices from specified layers of a DNN model given input data, keeping the acts/covs connected to the graph.

    Depending on the configuration flags, the function can either:
    - run the model with gradients enabled (for further backpropagation), or
    - run it in evaluation / inference mode without tracking gradients.

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
        If True, compute and return covariance matrices of the saved layer
        outputs. If False, return the raw activations instead.
    gradient : bool, default False
        If True, run the model in a gradient enabled setting. Obtained activations/covariances are attached to the graph
        for further backpropagation. This forces `eval_mode=False` and `inference_mode=False` regardless of their input values.
        If False, the function uses `eval_mode` and `inference_mode` as provided.
    eval_mode : bool, default True
        Whether to put the model into evaluation mode via `model.eval()`. This argument is ignored (internally forced to False) when
        `gradient=True`.
    inference_mode : bool, defalt True
        Whether to run the forward pass inside `torch.inference_mode()` for improved performance and disabled gradient tracking.
        This argument is ignored (internally forced to False) when `gradient=True`.
    save_network_output: bool, default False
        Whether to save the output of the network.
    flatten_acts : bool, default False
        When True, extracted activations are flattened to have dimensions (N, C*H*W) instead of (N, C, H, W).
        Considered only when `compute_covs=False`.

    Returns
    -------
    covs : dict
        A dictionary where keys are layer names and values are the corresponding covariance matrices (if `compute_covs=True`)
        or activations (if `compute_covs=False`).

    Raises
    ------
    UserWarning
        If a specified layer does not have a dimension matching the number of input images.
    """
    if type(layer_list) == str:
        layer_list = [layer_list]

    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)

    N = inputs.shape[0]

    if gradient:
        eval_mode = False
        inference_mode = False
        backward_ready = True
        out_transform = partial(get_cov, N=N)
    else:
        backward_ready = False
        if eval_mode or inference_mode:
            out_transform = partial(get_cov, N=N, detach=True)
        else:
            out_transform = partial(get_cov, N=N)

    if compute_covs:
        save_raw_outs = False
    else:
        out_transform = None
        save_raw_outs = True

    if save_network_output:
        save_raw_outs = True

    if eval_mode:
        model.eval()

    trace_kwargs = dict(
        model=model,
        input_args=inputs,
        layers_to_save=layer_list,
        out_transform=out_transform,
        random_seed=random_seed,
        backward_ready=backward_ready,
        save_raw_outs=save_raw_outs,
        save_raw_output=save_network_output,
    )

    if inference_mode:
        with torch.inference_mode():
            trace = tl.trace(**trace_kwargs)
    else:
        trace = tl.trace(**trace_kwargs)

    return create_covs_dict(
        trace, layer_list, compute_covs, flatten_acts, save_network_output
    )


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
    flatten_acts: bool = False,
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
    flatten_acts : bool, default False
        When True, extracted activations are flattened to have dimensions (N, C*H*W) instead of (N, C, H, W).
    """
    if type(layer_list) == str:
        layer_list = [layer_list]

    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)

    if eval_mode:
        model.eval()

    with torch.inference_mode():
        if layer_by_layer:
            layer_out_list = [
                f"{out_dir}/activations_{out_filename}_{layer_name}.hdf5"
                for layer_name in layer_list
            ]

            for k, layer_name in enumerate(layer_list):
                for batch_idx, batch in enumerate(
                    tqdm.tqdm(
                        torch.split(inputs, batch_size),
                        desc=f"Batches - Activation Extraction - Layer {layer_name}",
                        position=1,
                    )
                ):
                    activations = tl.batched_extract(
                        model=model,
                        stimuli=batch,
                        layers=layer_name,
                        batch_size=batch_size,
                    )

                    if flatten_acts:
                        acts = (
                            torch.reshape(
                                activations[layer_name],
                                [activations[layer_name].shape[0], -1],
                            )
                            .detach()
                            .cpu()
                        )
                    else:
                        acts = activations[layer_name].detach().cpu()

                    check_hdf_exists_save_acts(
                        layer_out_list[k],
                        acts,
                        num_inputs=inputs.shape[0],
                        dset_name="activations",
                        first_creation=True if batch_idx == 0 else False,
                    )

                print(
                    f"Activations for layer {layer_name} are saved at {layer_out_list[k]}."
                )
            print(f"All activations are saved to HDF5 files at the directory {out_dir}")

        else:
            hdf_filename = f"{out_dir}/activations_{out_filename}.hdf5"

            for batch_idx, batch in enumerate(
                tqdm.tqdm(
                    torch.split(inputs, batch_size),
                    desc=f"Batches - Activation Extraction - All Layers",
                    position=1,
                )
            ):
                activations = tl.batched_extract(
                    model=model,
                    stimuli=batch,
                    layers=layer_list,
                    batch_size=batch_size,
                )

                for layer_name in layer_list:
                    if flatten_acts:
                        acts = (
                            torch.reshape(
                                activations[layer_name],
                                [activations[layer_name].shape[0], -1],
                            )
                            .detach()
                            .cpu()
                        )
                    else:
                        acts = activations[layer_name].detach().cpu()

                    check_hdf_exists_save_acts(
                        hdf_filename,
                        acts,
                        num_inputs=inputs.shape[0],
                        dset_name="activations_" + layer_name,
                        first_creation=True if batch_idx == 0 else False,
                    )
            print(
                f"All activations are saved to '{out_dir}/activations_{out_filename}.hdf5'"
            )


def compute_covs_from_act_files(
    layer_list: str | List[str],
    out_filename: str,
    out_dir: str,
    layer_by_layer: bool,
    delete_act_files: bool,
    flatten_acts: bool,
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
    delete_act_files : bool, default True
        Flag for specifying whether to delete activation files from the disk.
    flatten_acts : bool
        When True, extracted activations are flattened to have dimensions (N, C*H*W) instead of (N, C, H, W).
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

                if flatten_acts:
                    layer_cov = get_cov(activations)
                else:
                    layer_cov = get_cov(activations.reshape(activations.shape[0], -1))

            cov_out_filename = f"{out_dir}/covs_{out_filename}_{layer_name}.pkl"
            with open(cov_out_filename, "wb") as f:
                pickle.dump(layer_cov, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"\nSaved covariance for layer {layer_name} at {cov_out_filename}.")

            if delete_act_files:
                os.remove(activations_filename)
                print(
                    f"\nDeleted the activation file for layer {layer_name} at {activations_filename}."
                )
    else:
        covs_dict = {}
        for k, layer_name in enumerate(layer_list):
            with h5py.File(f"{out_dir}/activations_{out_filename}.hdf5", "r") as f:
                activations = f["activations_" + layer_name][...]

                if flatten_acts:
                    layer_cov = get_cov(activations)
                else:
                    layer_cov = get_cov(activations.reshape(activations.shape[0], -1))

            covs_dict[layer_name] = layer_cov

        cov_out_filename = f"{out_dir}/covs_{out_filename}.pkl"
        with open(cov_out_filename, "wb") as f:
            pickle.dump(covs_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nSaved covariance for layer {layer_name} at {cov_out_filename}.")

        if delete_act_files:
            os.remove(f"{out_dir}/activations_{out_filename}.hdf5")
            print(
                f"\nDeleted the activation file at {out_dir}/activations_{out_filename}.hdf5."
            )


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
    flatten_acts: bool = False,
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
    flatten_acts : bool, default False
        When True, extracted activations are flattened to have dimensions (N, C*H*W) instead of (N, C, H, W).
        Considered only when `compute_covs=False`.

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
        flatten_acts=flatten_acts,
    )
    if compute_covs:
        compute_covs_from_act_files(
            layer_list=layer_list,
            out_filename=out_filename,
            out_dir=out_dir,
            layer_by_layer=layer_by_layer,
            delete_act_files=delete_act_files,
            flatten_acts=flatten_acts,
        )
