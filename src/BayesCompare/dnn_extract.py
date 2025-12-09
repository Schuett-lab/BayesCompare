import torch
import numpy as np
from typing import List, Optional, Union
import warnings
import torch
from functools import partial
import torchlens as tl
import tqdm


def check_act_dims(act, N):
    """
    Checks and reshapes activation tensor dimensions to ensure N is the first dimension.
    This function verifies that the activation tensor contains a dimension equal to N, the
    number of images used for obtaining the covariance matrix. If N is not the first dimension, the
    tensor is permuted to move it to the first position. If N is not found in any dimension,
    a warning is issued. If N is the first dimension the tensor itself is returned.
    Args:
        act (torch.Tensor): Activation tensor whose dimensions need to be checked and potentially reordered.
        N (int): Number of images used for obtaining the covariance matrix. This dimension should exist in act.
    Returns:
        activations (torch.Tensor or None): The activation tensor with N as the first dimension, or None if N is not found
        in any dimension of the input tensor.
    Raises:
        UserWarning: If N is not found in any dimension of the activation tensor.
    """

    shape = list(act.shape)

    if shape[0] == N:
        activations = act

    elif N in shape:
        n_dim = shape.index(N)  # find which dimension equals n
        perm = [n_dim] + [i for i in range(len(shape)) if i != n_dim]
        activations = act.permute(perm)

    elif N not in shape:
        warnings.warn("This layer does not have a number of images dimension")
        return None

    return activations


def get_cov(activations: Union[torch.Tensor, np.ndarray], N=None):
    """
    Computes the covariance matrix of the given activations.
    Args:
        activations (torch.Tensor or np.ndarray): The activations from which to compute the covariance matrix.
            The first dimension is assumed to be the number of images unless N is provided.
        N (Optional[int]): The number of images used for obtaining the covariance matrix. If provided,
            the function checks that the first dimension of activations matches N. Then, if it does not match,
            it reorders dimensions accordingly.
    Returns:
        cov_matrix (torch.Tensor or np.ndarray): The computed covariance matrix of the activations.
    Raises:
        NotImplementedError: If activations is neither a torch tensor nor a numpy array.
    """
    # check if the first dimension of activations is equal to the number of images used
    # if the number of images is provided as an input
    if N != None:
        activations = check_act_dims(activations, N)

    if torch.is_tensor(activations):
        module = torch
        # x = activations.detach().clone() # we dont want the covs to be detached from the graph because we would like to use them for training
    elif isinstance(activations, np.ndarray):
        module = np
    else:  # Also we can catch bad arguments (not mandatory)
        raise NotImplementedError(
            "Activations must be either a torch tensor or a numpy array."
        )

    act = module.reshape(activations, [activations.shape[0], -1])
    x = act - module.mean(act, 1, keepdims=True)
    return module.matmul(x, x.T)


def _compare_wanted_output(output_layers: List[str], wanted_layers: List[str]):
    """
    Compares the layers saved in the model history object with the
    originally requested layers to ensure all wanted layers are included
    and no additional layers are inserted.
    Args:
        output_layers (List[str]): List of layer names saved in the model history object.
        wanted_layers (List[str]): List of originally requested layer names.
    Returns:
        (List[str]): Filtered list of layer names that are both in the output_layers and wanted_layers.
    """
    wanted_set = set(wanted_layers)
    return [x for x in output_layers if x in wanted_set]


def get_layer_names(model: torch.nn.Module, get_graph: Optional[str] = "none"):
    """
    Retrieves the names of all layers in a DNN model, visible to TorchLens under torch.inference_mode().

    Args:
        model (torch.nn.Module): The DNN model from which to extract layer names.
        get_graph (Optional[str]): Visualization option for torchlens log_forward_pass.
            Defaults to "none". Can be "unrolled" or "rolled" to visualize the model graph.
    Returns:
        all_layers (List[str]): A list of layer names in the model.
    """

    with torch.inference_mode():

        mock_input = torch.rand(1, 3, 224, 224)

        model_history = tl.log_forward_pass(
            model,
            mock_input,
            layers_to_save=None,
            vis_opt=get_graph,
            detach_saved_tensors=True,
        )

        all_layers = list(model_history.layer_dict_main_keys.keys())

        return all_layers


def cov_extractor(
    model: torch.nn.Module,
    layer_list: Union[List[str], str],
    inputs: Union[torch.Tensor, np.ndarray],
):
    """
    Extracts covariance matrices from specified layers of a DNN model given input data.

    Args:
        model (torch.nn.Module): The DNN model from which to extract covariances.
        layer_list (Union[List[str], str]): A list of layer names or a single layer name
            for which to compute the covariance matrices.
        inputs (Union[torch.Tensor, np.array]): An input tensor or array of images to
            extract covariances from. First dimension is assumed to be the number of images.

    Returns:
        covs (dict): A dictionary where keys are layer names and values are the corresponding
            covariance matrices.
    Raises:
        UserWarning: If a specified layer does not have a dimension matching the number of input images.
    """

    if type(layer_list) == str:
        layer_list = [layer_list]

    covs = {}

    N = inputs.shape[0]

    get_cov_n = partial(get_cov, N=N)

    model.eval()

    with torch.inference_mode():

        print("Covariance computation has started. This may take a while.")

        model_history = tl.log_forward_pass(
            model,
            inputs,
            layers_to_save=layer_list[0],
            vis_opt="none",
            activation_postfunc=get_cov_n,
            detach_saved_tensors=True,
        )

        covs[layer_list[0]] = model_history[layer_list[0]].tensor_contents

        if len(layer_list) > 1:

            print(
                "Model history object is created, now covariances for selected layers will be extracted."
            )

            model_history.save_new_activations(
                model, inputs, layers_to_save=layer_list[1:]
            )

            if len(model_history.layers_with_saved_activations) != len(layer_list) - 1:

                proper_saved_layers_list = _compare_wanted_output(
                    model_history.layers_with_saved_activations, layer_list
                )

                for layer in tqdm.tqdm(
                    proper_saved_layers_list,
                    initial=1,
                    total=len(layer_list),
                    desc="Requested Layer Activations",
                ):
                    covs[layer] = model_history[layer].tensor_contents

            else:
                for layer in tqdm.tqdm(
                    model_history.layers_with_saved_activations,
                    initial=1,
                    total=len(layer_list),
                    desc="Requested Layer Activations",
                ):
                    covs[layer] = model_history[layer].tensor_contents

    return covs
