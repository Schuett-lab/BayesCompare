import os
import PIL
import torch
import torchvision
import BayesCompare as bc
from pathlib import Path

# Load the MS COCO dataset
home_path = Path.home()
im_folder = os.path.join(
    home_path, "Documents/BayesCompare/images/unlabeled2017"
)  # Put your path to COCO images here
file_names = os.listdir(im_folder)

# number of images to use for covariance computation
N = 20

# get the first N images from the image folder
ims = [PIL.Image.open(os.path.join(im_folder, f_name)) for f_name in file_names[:N]]

# apply the transforms of the model
transforms = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
transformed_ims = [transforms(im.convert("RGB")) for im in ims]
x_input = torch.stack(transformed_ims)

# Load the model
model = torchvision.models.get_model(
    "vit_b_16", weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
)


# First, get all the layer names of the model. If you want to obtain the graph of the model as well, you may use
# the input parameter get_graph="unrolled" or get_graph="rolled" in the function call.
all_layer_names = bc.get_layer_names(model)

# Then select a subset of layers to compute the covs from.
# try only for 1 layer
selected_layers1 = all_layer_names[0]

cov_dict1 = bc.cov_extractor(model, selected_layers1, x_input)

# try with a list of layers
selected_layers2 = all_layer_names[120:143]

cov_dict2 = bc.cov_extractor(model, selected_layers2, x_input)
