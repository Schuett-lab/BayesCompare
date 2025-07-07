import os
import numpy as np
import torch
import torchvision
import PIL
import BayesCompare
import tqdm

device = "cpu"

# layers to select:

select = [1, 2, 4, 5, 7, 9, 11, 12, 15, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
al_select = [1, 2, 4, 5, 7, 9, 11, 12, 15, 18, 19]
# al_sel = [1, 2, 4, 5, 7, 9, 11, 12]  # , 15, 18]
# res_sel = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


# loading images
im_folder = 'images/unlabeled2017'
file_names = os.listdir(im_folder)

N = 1000
ims = [PIL.Image.open(
    os.path.join(im_folder, f_name))
    for f_name in file_names[:N]
]

preprocess = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
x_input = torch.stack([preprocess(im.convert('RGB')) for im in ims]).to(device)


def ReLU_inplace_to_False(module):
    for layer in module._modules.values():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)


# take an imagenet trained alexnet
al = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
activation = [None] * 20
ReLU_inplace_to_False(al)


def get_activation(n):
    def hook(model, input, output):
        activation[n] = output.detach()
    return hook


for i in range(13):
    al.features[i].register_forward_hook(get_activation(i))
for i in range(7):
    al.classifier[i].register_forward_hook(get_activation(i+13))

al.eval()
al.to(device)
out = al(x_input)
# compute covariance matrices
covs = [BayesCompare.get_cov(activation[i_act]).detach().cpu().numpy()
        for i_act in tqdm.tqdm(al_select)]


# resnet computations
resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
ReLU_inplace_to_False(resnet)
activation_r = [None] * 14


def get_activation_r(n):
    def hook(model, input, output):
        activation_r[n] = output.detach()
    return hook


k = 0
for i_layer in resnet.children():
    if isinstance(i_layer, torch.nn.Sequential):
        for j_layer in i_layer.children():
            j_layer.register_forward_hook(get_activation_r(k))
            k += 1
    else:
        i_layer.register_forward_hook(get_activation_r(k))
        k += 1

# input preprocessing is the same as for alexnet! We reuse x_input
resnet.eval()
resnet.to(device)
out = resnet(x_input)
# compute covariance matrices & transform to numpy arrays
covs.extend(
    [BayesCompare.get_cov(act).detach().cpu().numpy()
     for act in tqdm.tqdm(activation_r)])

np.save("covs_1000.npy", np.stack(covs))
