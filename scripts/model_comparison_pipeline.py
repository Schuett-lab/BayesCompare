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


DIRS = get_dirs()

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
        print("Analysis is done with the configuration file: " + str(config_path))
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


def get_normed_cov_files(covs_dir, cov_filename, imseed, N, bvals):
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

    normed_cov_filenames = []

    for bval in bvals:
        # covs_<something>_imseed_<number>_N_<number>_bval_<number>.npy
        pattern = re.compile(rf"^{cov_filename}_imseed_{imseed}_N_{N}_bval_{bval}.npy$")

        for file in dir_path.iterdir():

            if not file.is_file():
                continue

            match = pattern.match(file.name)

            if match:
                normed_cov_filenames.append(file.name)

    return normed_cov_filenames


def get_seed_from_filename(filename):

    pattern = re.compile(r"image_filenames_seed_(\d+)_N_\d+\.txt$")
    m = pattern.match(filename)
    if m:
        seed = int(m.group(1))
        return seed
    else:
        raise ValueError(f"Invalid filename format: {filename}")


def save_im_filenames(selected_files, seed, N, im_folder):

    output_txt_filename = "image_filenames_seed_" + str(seed) + "_N_" + str(N) + ".txt"

    output_txt_path = os.path.join(im_folder, "im_filename_files", output_txt_filename)

    if Path(output_txt_path).exists():
        raise FileExistsError(f"{output_txt_path} already exists")

    print(f"Image filename list is saved at {output_txt_path}")
    with open(output_txt_path, "w") as f:
        for name in selected_files:
            f.write(name + "\n")


# Process functions


def load_ims(model_args):
    """
    Loads the images from which the covariance matrices will be calculated.

    Parameters
    ----------
    model_args (dict): A dictionary consisting the parameters for input image filename directory/seed and number of images.

    Returns
    -------
    act_ims (list[torch.Tensor]): A list of transformed images.
    """

    ims_filename_file = model_args.get("ims_filename_file", None)

    im_folder = os.path.join(DIRS["input_images"])

    N = model_args["num_ims"]

    if ims_filename_file == None:

        seed = model_args.get(
            "ims_seed", np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0]
        )

        print(f"Seed for the images is {seed}")

        rng = np.random.default_rng(seed)

        all_file_names = os.listdir(im_folder + "/mscoco/")

        selected_files = rng.choice(all_file_names, size=N, replace=False)

        save_im_filenames(selected_files, seed, N, im_folder)

    else:

        with open(os.path.join(im_folder, "im_filename_files", ims_filename_file)) as f:
            selected_files = [line.strip() for line in f]

        seed = get_seed_from_filename(ims_filename_file)

    ims = [
        PIL.Image.open(os.path.join(im_folder, "mscoco", f_name))
        for f_name in selected_files
    ]

    weights = get_model_weights(model_args["model_weights_name"])

    transforms = weights.transforms()

    transformed_ims = [transforms(im.convert("RGB")) for im in ims]
    act_ims = torch.stack(transformed_ims)

    return act_ims, seed


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


def get_covs_from_models(models_list, input_ims, im_seed, model_args):
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
    cov_full_filename = (
        model_args["covs_filename"]
        + "_imseed_"
        + str(im_seed)
        + "_N_"
        + str(model_args["num_ims"])
        + ".pkl"
    )

    print(f"Covariance filename is {cov_full_filename}")

    with open(
        os.path.join(
            DIRS["result_path"], "covs", model_args["model_name"], cov_full_filename
        ),
        "wb",
    ) as f:
        pickle.dump(models_covs_list, f)


def normalize_covs(model_args, im_seed):
    """
    Loads the saved covariance matrices and trace-normalizes them with adding noise.
    Saves them with the addition of noise level in the original filename.

    Parameters
    ----------
    model_args (dict): A dictionary consisting the parameters for the noise levels,
    number of images used for obtaining the covariance matrices and filename of the covariance matrices to be saved.
    """

    cov_full_filename = (
        model_args["covs_filename"]
        + "_imseed_"
        + str(im_seed)
        + "_N_"
        + str(model_args["num_ims"])
    )

    with open(
        os.path.join(
            DIRS["result_path"],
            "covs",
            model_args["model_name"],
            cov_full_filename + ".pkl",
        ),
        "rb",
    ) as f:
        covs_dicts = pickle.load(f)

    covs = []

    for cov_dict in covs_dicts:
        covs.append(list(cov_dict.values()))

    covs = np.stack(covs)
    covs = covs.reshape(covs.shape[0] * covs.shape[1], covs.shape[2], covs.shape[3])

    for noise_b in tqdm.tqdm(
        model_args["noise_bs"], desc="Noise_Levels - Normalize Covs", position=1
    ):

        normed_cov_full_filename = cov_full_filename + "_bval_" + str(noise_b) + ".npy"

        file_path = os.path.join(
            DIRS["result_path"],
            "covs",
            model_args["model_name"],
            normed_cov_full_filename,
        )

        if Path(file_path).exists():
            print(f"File at {file_path} already exists.")

        else:
            b = 1 / noise_b
            noise_level = model_args["num_ims"] * b / (1 + (model_args["num_ims"] * b))

            print(
                f"Normalized covariance file {normed_cov_full_filename} has the noise level = {noise_level}"
            )

            normalized_covs = bc.cov_utils.cov_trace_norm_sigma_N(
                covs, noise_var=noise_level
            )

            np.save(file_path, normalized_covs)


def compute_dists(model_args, imseed):
    """
    Computes the distances whose names are passed with `model_args` from the saved normalized covariance matrices.

    model_args (dict): A dictionary consisting the parameters for measure names.
    """

    # Get the list of covs with bvals in their names. Those are the ones that were normalized.
    # (covs_dir, cov_filename, imseed, N)
    covs_filename_list = get_normed_cov_files(
        os.path.join(DIRS["result_path"], "covs", model_args["model_name"]),
        model_args["covs_filename"],
        imseed,
        model_args["num_ims"],
        model_args["noise_bs"],
    )

    # In a for loop iterate through them to compute distances out of them.
    for cov_filename in tqdm.tqdm(
        covs_filename_list, desc="Noise Levels - Dists", position=1
    ):
        print(f"Computations for the cov file {cov_filename} has started.")
        bc.measure_dist_parallel(
            covs_dir=os.path.join(
                DIRS["result_path"], "covs", model_args["model_name"], cov_filename
            ),
            output_dir=os.path.join(
                DIRS["result_path"], "dists", model_args["model_name"]
            ),
            meas_name=model_args["measures"],
        )


def main(model_args):

    # try to get these keys from the configuration dict.
    # If not specified in config file, they will be True by default and computed.
    ims_seed = None
    get_covs = model_args.get("get_covs", True)
    norm_covs = model_args.get("norm_covs", True)
    comp_dists = model_args.get("comp_dists", True)

    if get_covs == True:

        ims, ims_seed = load_ims(model_args)

        models_list = load_models(model_args)

        # directly saves the computed covs, so it doesn't return anything.
        get_covs_from_models(models_list, ims, ims_seed, model_args)

    if norm_covs == True:

        # directly saves the normalized covs, so it doesn't return anything.
        if ims_seed == None:
            ims_seed = model_args["ims_seed"]

        normalize_covs(model_args, ims_seed)

    if comp_dists == True:

        # directly saves the dists, so it doesn't return anything.
        if ims_seed == None:
            ims_seed = model_args["ims_seed"]

        compute_dists(model_args, ims_seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config_dir)

    main(config)
