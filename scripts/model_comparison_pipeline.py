import os
import re
import PIL
import json
import tqdm
import pickle
import argparse
import torch
import torchvision
import numpy as np
import BayesCompare as bc

from pathlib import Path
from system_dirs import get_dirs


# Helper functions


def load_config(config_path):
    """
    Loads the configuration file, which is a JSON file containing parameter values for model training.
    Parses the command-line argument '--config_dir' to get the directory containing the config file.

    Parameters
    ----------
    config_path (str): Directory of the configuration JSON file.

    Returns
    -------
    config (dict): Configuration parameters loaded from the JSON file.
    """
    with open(config_path) as file:
        print("Training is done with the configuration file: " + str(config_path))
        config = json.load(file)

    return config


def load_trained_model(model_name, model_dir, device="cpu"):
    """
    Loads the pretrained models from their directories.

    Parameters
    ----------
    model_name (str): Model name as used for creating an instance of that model in PyTorch Vision Models.
    model_dir (str): Directory of the checkpoint model whose weights will be used.
    device (str): 'cuda' or 'cpu' indicating which device model will be. By default it is 'cpu'

    Returns
    -------
    model (nn.Module): Loaded pretrained model in eval mode.
    """

    snapshot = torch.load(model_dir, map_location=torch.device(device))
    model = torchvision.models.get_model(model_name, weights=None).to(
        torch.device(device)
    )
    if model_name == "resnet50" and ("333" in model_dir or "128" in model_dir):
        model.load_state_dict(snapshot["model_state"])
    else:
        model.load_state_dict(snapshot["model"])

    # should it be in eval mode? It should be discussed!
    return model.eval()


def get_model_weights(model_weights_name):
    """
    Gets the pre-trained model weights. This weights instance can be used
    both for loading the pretrained weights as well as transforms for the input images.

    Parameters
    ----------
    model_weights_name (str): Model weights name as used for the PyTorch Vision Models.

    Returns
    -------
    weighs (torchvision.models._api.WeightsEnum): Pretrained torchvision model weights (e.g., ResNet, EfficientNet,
        Swin-T, ViT).
    """

    if model_weights_name == "ResNet50_IMAGENET1K_V1":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1

    elif model_weights_name == "ResNet50_IMAGENET1K_V2":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2

    elif model_weights_name == "ViT_B_16_IMAGENET1K_V1":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1

    elif model_weights_name == "Swin_T_IMAGENET1K_V1":
        weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1

    elif model_weights_name == "Swin_V2_S_IMAGENET1K_V1":
        weights = torchvision.models.Swin_V2_S_Weights.IMAGENET1K_V1

    elif model_weights_name == "ConvNeXt_Tiny_IMAGENET1K_V1":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1

    elif model_weights_name == "EfficientNet_V2_S_IMAGENET1K_V1":
        weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1

    return weights


def get_normed_cov_files(covs_dir, cov_filename):
    """
    Gets the list of filenames of normalized covariance files. Those are the
    ones which has "bval" in their filenames.

    Parameters
    ----------
    covs_dir (str): Directory of the folder in which covariance files are stored.

    Returns
    -------
    normed_cov_filenames (list[str]): A list of normalized covariance's filenames.
    """

    dir_path = Path(covs_dir)

    # covs_<anything>_bval_<number>_<number>.npy
    pattern = re.compile(rf"^{cov_filename}_bval_([0-9]+)_([0-9]+)\.npy$")

    normed_cov_filenames = []

    for file in dir_path.iterdir():

        if not file.is_file():
            continue

        match = pattern.match(file.name)

        if match:
            normed_cov_filenames.append(file.name)

    return normed_cov_filenames


# Process functions

DIRS = get_dirs()


def load_ims(model_weights_name, num_ims):
    """
    Loads the images from which the covariance matrices will be calculated.

    Parameters
    ----------
    model_weights_name (str): Model weights name as used for the PyTorch Vision Models.
    num_ims (int): Number of images that will be used for covariance calculation.

    Returns
    -------
    act_ims (list[torch.Tensor]): A list of transformed images.
    """

    im_folder = os.path.join(DIRS["input_images"])
    file_names = os.listdir(im_folder)

    N = num_ims
    ims = [PIL.Image.open(os.path.join(im_folder, f_name)) for f_name in file_names[:N]]

    weights = get_model_weights(model_weights_name)

    transforms = weights.transforms()

    transformed_ims = [transforms(im.convert("RGB")) for im in ims]
    act_ims = torch.stack(transformed_ims)

    return act_ims


def load_models(model_args):
    """
    Loads the pretrained models (both self-trained and PyTorch pretrained).

    Parameters
    ----------
    model_args (dict): A dictionary consisting the parameters for the model name of the model to be
    loaded and its trained checkpoint directories.

    Returns
    -------
    models (list[nn.Module]): A list containing the loaded models in order.
    """

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device is set as {device}")

    models = []

    for i, model_filename in enumerate(model_args["checkpoint_dirs"]):
        model_dir = os.path.join(
            DIRS["checkpoint_main_path"],
            model_args["model_name"],
            "checkpoints",
            "seed_" + model_args["seeds"][i],
            model_filename,
        )
        models.append(load_trained_model(model_args["model_name"], model_dir, device))

    weights = get_model_weights(model_args["model_weights_name"])

    models.append(
        torchvision.models.get_model(model_args["model_name"], weights=weights).to(
            torch.device(device)
        )
    )

    return models


def get_covs_from_models(models_list, input_ims, model_args):
    """
    Computes the covariance matrices from each model given the given model list and input images.

    Parameters
    ----------
    models (list[nn.Module]): A list containing the pretrained and loaded models in order.
    input_ims (list[torch.Tensor]): A list of transformed images.
    model_args (dict): A dictionary consisting the parameters for the wanted layers and filename of the covariance matrices to be saved.
    """

    with open(
        os.path.join(DIRS["configs_path"], model_args["wanted_layers_dir"]),
        "r",
        encoding="utf-8",
    ) as f:
        wanted_layers = json.load(f)

    models_covs_list = []

    # For each model, get the cov_dict for all wanted layers and collect them into the models_covs_list
    for model in tqdm.tqdm(models_list, desc="Models", position=1):

        cov_dict = bc.cov_extractor(model, wanted_layers, input_ims)

        models_covs_list.append(cov_dict)

    # Save the models_covs_list which have all the wanted activations for all models
    cov_full_filename = model_args["covs_filename"] + ".pkl"

    with open(os.path.join(DIRS["result_path"], cov_full_filename), "wb") as f:
        pickle.dump(models_covs_list, f)


def normalize_covs(model_args):
    """
    Loads the saved covariance matrices and trace-normalizes them with adding noise.
    Saves them with the addition of noise level in the original filename.

    Parameters
    ----------
    model_args (dict): A dictionary consisting the parameters for the noise levels,
    number of images used for obtaining the covariance matrices and filename of the covariance matrices to be saved.
    """

    cov_full_filename = model_args["covs_filename"] + ".pkl"

    with open(os.path.join(DIRS["result_path"], cov_full_filename), "rb") as f:
        covs_dicts = pickle.load(f)

    covs = []

    for cov_dict in covs_dicts:
        covs.append(list(cov_dict.values()))

    covs = np.stack(covs)
    covs = covs.reshape(covs.shape[0] * covs.shape[1], covs.shape[2], covs.shape[3])

    for noise_b in tqdm.tqdm(model_args["noise_bs"], desc="Noise_Levels", position=1):

        normed_cov_full_filename = (
            model_args["covs_filename"]
            + "_bval_"
            + str(noise_b[0])
            + "_"
            + str(noise_b[1])
            + ".npy"
        )

        b = noise_b[0] / noise_b[1]
        noise_level = model_args["num_ims"] * b / (1 + (model_args["num_ims"] * b))

        print(
            f"Normalized covariance file {normed_cov_full_filename} has the noise level = {noise_level}"
        )

        normalized_covs = bc.cov_utils.cov_trace_norm_sigma_N(
            covs, noise_var=noise_level
        )

        np.save(
            os.path.join(DIRS["result_path"], normed_cov_full_filename), normalized_covs
        )


def compute_dists(model_args):
    """
    Computes the distances whose names are passed with `model_args` from the saved normalized covariance matrices.

    model_args (dict): A dictionary consisting the parameters for measure names.
    """

    # Get the list of covs with bvals in their names. Those are the ones that were normalized.
    covs_filename_list = get_normed_cov_files(
        DIRS["result_path"], model_args["covs_filename"]
    )

    # In a for loop iterate through them to compute distances out of them.
    for cov_filename in tqdm.tqdm(covs_filename_list, desc="Noise Levels", position=1):
        print(f"Computations for the cov file {cov_filename} has started.")
        bc.measure_dist_parallel(
            covs_dir=os.path.join(DIRS["result_path"], cov_filename),
            output_dir=os.path.join(DIRS["result_path"]),
            meas_name=model_args["measures"],
        )


def main(model_args):

    # try to get these keys from the configuration dict.
    # If not specified in config file, they will be True by default and computed.
    get_covs = model_args.get("get_covs", True)
    norm_covs = model_args.get("norm_covs", True)
    comp_dists = model_args.get("comp_dists", True)

    if get_covs == True:

        ims = load_ims(model_args["model_weights_name"], model_args["num_ims"])

        models_list = load_models(model_args)

        # directly saves the computed covs, so it doesn't return anything.
        get_covs_from_models(models_list, ims, model_args)

    if norm_covs == True:

        # directly saves the normalized covs, so it doesn't return anything.
        normalize_covs(model_args)

    if comp_dists == True:

        # directly saves the dists, so it doesn't return anything.
        compute_dists(model_args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config_dir)

    main(config)
