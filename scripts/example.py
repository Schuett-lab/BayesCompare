import os
import numpy as np
import torch
import torchvision
import PIL
import BayesCompare
import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

device = "cpu"

# loading images
im_folder = 'images/unlabeled2017'
file_names = os.listdir(im_folder)

N = 200
ims = [PIL.Image.open(  # type: ignore
    os.path.join(im_folder, f_name))
    for f_name in file_names[:N]
]

covs = []


# take an imagenet trained alexnet
preprocess = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
x_input = torch.stack([preprocess(im.convert('RGB')) for im in ims]).to(device)
covs.append(BayesCompare.get_cov(x_input).cpu().detach().numpy())


def ReLU_inplace_to_False(module):
    for layer in module._modules.values():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)


# layers to select:
al_select = [1, 2, 4, 5, 7, 9, 11, 12, 15, 18, 19]
al = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
activation = [None] * 20
ReLU_inplace_to_False(al)


def get_activation(n):
    def hook(model, input, output):
        activation[n] = output.cpu().detach().numpy()
    return hook


for i in range(13):
    al.features[i].register_forward_hook(get_activation(i))
for i in range(7):
    al.classifier[i].register_forward_hook(get_activation(i+13))

al.eval()
al.to(device)
out = al(x_input)
# compute covariance matrices
covs.extend([
    BayesCompare.get_cov(activation[i_act])
    for i_act in tqdm.tqdm(al_select)])


# resnet computations
resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
ReLU_inplace_to_False(resnet)
activation = [None] * 14


k = 0
for i_layer in resnet.children():
    if isinstance(i_layer, torch.nn.Sequential):
        for j_layer in i_layer.children():
            j_layer.register_forward_hook(get_activation(k))
            k += 1
    else:
        i_layer.register_forward_hook(get_activation(k))
        k += 1

# input preprocessing is the same as for alexnet! We reuse x_input
resnet.eval()
resnet.to(device)
out = resnet(x_input)
# compute covariance matrices
covs.extend(
    [BayesCompare.get_cov(act)
     for act in tqdm.tqdm(activation)])


weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
preprocess = weights.transforms()
vit = torchvision.models.vit_b_16(weights=weights)
vit.eval()
vit.to(device)

n_l = len(vit.encoder.layers)

activation = [None] * n_l


def get_activation(n):
    def hook(model, input, output):
        activation[n] = output.detach()
        print(n)
    return hook


for i, l in enumerate(vit.encoder.layers):
    l.register_forward_hook(get_activation(i))

im_tensor = torch.stack([preprocess(im) for im in ims]).to(device)
covs.append(BayesCompare.get_cov(im_tensor).cpu().detach().numpy())
out = vit(im_tensor)
# compute covariance matrices
covs_single = [
    BayesCompare.get_cov(act).cpu().detach().numpy()
    for act in activation]
covs.extend(covs_single)

np.save("covs_example.npy", np.stack(covs))


dist = np.zeros((len(covs), len(covs)))
for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
    for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
        if j > i:
            dist[i, j] = BayesCompare.jsd_normal_sig(ci, cj, 10000, eye_w=2/3)
            dist[j, i] = dist[i, j]

np.save("dist_example.npy", dist)

dist = np.load("dist_example.npy")

mds = MDS(dissimilarity="precomputed")
mds.fit(dist)

x = mds.embedding_
x0 = x[1:12]
x1 = x[12:26]
x2 = x[27:39]

plt.figure()
plt.plot(x0[:, 0], x0[:, 1], '.-', linewidth=2, markersize=10, color="#fa5750")  # color="#d6000c")
plt.plot(x1[:, 0], x1[:, 1], '.-', linewidth=2, markersize=10, color="#dbb32d")  # color="#c49700")
plt.plot(x2[:, 0], x2[:, 1], '.-', linewidth=2, markersize=10, color="#4695f7")  # color="#0064e4")
plt.arrow(
    x[0, 0], x[0, 1], x0[0, 0] - x[0, 0], x0[0, 1] - x[0, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.arrow(
    x[0, 0], x[0, 1], x1[0, 0] - x[0, 0], x1[0, 1] - x[0, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.arrow(
    x[26, 0], x[26, 1], x2[0, 0] - x[26, 0], x2[0, 1] - x[26, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.axis("equal")
ax = plt.gca()
ax.set_axis_off()
plt.savefig("figures/example_mds.svg")
plt.savefig("figures/example_mds.pdf")

order = [0, 26, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 27, 28, 29, 30,
         31, 32, 33, 34, 35, 36, 37, 38]
plt.figure()
plt.imshow(np.sqrt(dist[order][:, order]), "bone", vmin=0, vmax=1)
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("figures/example_dist.svg")
plt.savefig("figures/example_dist.pdf")

plt.close('all')
