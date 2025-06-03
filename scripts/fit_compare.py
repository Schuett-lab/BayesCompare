import os
import numpy as np
from scipy.special import logsumexp
from matplotlib import pyplot as plt
import PIL
import torchvision
import torch
import tqdm
import BayesCompare
from BayesCompare import inference, evidence, inference_cov


device = "cpu"
al_select = [1, 2, 4, 5, 7, 9, 11, 12, 15, 18, 19]
im_folder = "/Users/heiko.schutt/code/predseg/coco/images/unlabeled2017"
file_names = os.listdir(im_folder)

i_voxel = 0

N_train = 500
N_test = 500

# get a voxel to predict:
places_fs = np.load(
    "/Users/heiko.schutt/Algonouts/subj01/roi_masks/lh.floc-places_fsaverage_space.npy"
)
mask_fs = np.array(places_fs == 2, dtype=bool)
idx_fs = np.where(mask_fs)[0]

places_c = np.load(
    "/Users/heiko.schutt/Algonouts/subj01/roi_masks/lh.floc-places_challenge_space.npy"
)
mask_c = np.array(places_c == 2, dtype=bool)
idx_c = np.where(mask_c)[0]

d_train = np.load(
    "/Users/heiko.schutt/Algonouts/subj01/training_split/training_fmri/lh_training_fmri.npy"
)
d_train = d_train[:, idx_c]

# choose the N_train+N_test first images
im_idx = np.array([int(fn[6:10]) for fn in file_names[: (N_train + N_test)]])

d_train_ims = d_train[im_idx[:N_train], :]
d_val_ims = d_train[im_idx[N_train : (N_train + N_test)], :]

# get Alexnet Kernel matrix
ims = [
    PIL.Image.open(os.path.join(im_folder, f_name))  # type: ignore
    for f_name in file_names[: (N_train + N_test)]
]


def ReLU_inplace_to_False(module):
    for layer in module._modules.values():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)


# take an imagenet trained alexnet
al = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
activation = [None]
ReLU_inplace_to_False(al)


def get_activation(n):
    def hook(model, input, output):
        activation[n] = output.detach()

    return hook


activation = [None] * 20
ReLU_inplace_to_False(al)

for i in range(13):
    al.features[i].register_forward_hook(get_activation(i))
for i in range(7):
    al.classifier[i].register_forward_hook(get_activation(i + 13))

preprocess = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()

x_input = torch.stack([preprocess(im.convert("RGB")) for im in ims])
al.eval()
out = al(x_input)
# compute covariance matrix

covs = [BayesCompare.get_cov(activation[i_act]) for i_act in tqdm.tqdm(al_select)]


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
covs.extend([BayesCompare.get_cov(act).detach().cpu().numpy() for act in tqdm.tqdm(activation_r)])

np.save("covs_algo_1000.npy", np.stack(covs))


# loading covariance estimates here is possible!

covs = np.load("covs_algo_1000.npy")

# Normalize covariances:
alpha = 0.9

covs_norm = [
    (cov / np.trace(cov) * (N_test + N_train)) * np.var(d_train_ims[:, i_voxel]) for cov in covs
]

y_mus = [None] * len(covs_norm)
y_sigma = [None] * len(covs_norm)

for i, cov in enumerate(covs_norm):
    y_mus[i], y_sigma[i] = inference(cov, d_train_ims[:, i_voxel].flatten(), alpha=alpha)

dist = np.zeros((len(covs_norm), len(covs_norm)))

for i in tqdm.trange(len(covs_norm)):
    for j in range(len(covs_norm)):
        if j > i:
            dist[i, j] = BayesCompare.tvd_normal_general(
                y_mus[i],
                y_mus[j],
                (1 - alpha) * y_sigma[i] + alpha * np.eye(N_test) * np.var(d_train_ims[:, i_voxel]),  # type: ignore
                (1 - alpha) * y_sigma[j] + alpha * np.eye(N_test) * np.var(d_train_ims[:, i_voxel]),
            )  # type: ignore
            dist[j, i] = dist[i, j]


dist_orig = np.zeros((len(covs_norm), len(covs_norm)))

for i in tqdm.trange(len(covs_norm)):
    for j in range(len(covs_norm)):
        if j > i:
            dist_orig[i, j] = BayesCompare.tvd_normal_general(
                np.zeros_like(y_mus[i]),
                np.zeros_like(y_mus[i]),
                (1 - alpha) * covs_norm[i][N_train:, N_train:]
                + alpha * np.eye(N_test) * np.var(d_train_ims[:, i_voxel]),
                (1 - alpha) * covs_norm[j][N_train:, N_train:]
                + alpha * np.eye(N_test) * np.var(d_train_ims[:, i_voxel]),
            )
            dist_orig[j, i] = dist_orig[i, j]

np.save("dist_algo_1000.npy", dist)
np.save("dist_orig_algo_1000.npy", dist_orig)

dist = np.load("dist_algo_1000.npy")
dist_orig = np.load("dist_orig_algo_1000.npy")

plt.imshow(dist, cmap="bone", vmax=1, vmin=0)
plt.colorbar()
plt.show()


ddist = dist - dist_orig
clim = np.max(np.abs(ddist))
plt.imshow(ddist, vmin=-clim, vmax=clim, cmap="RdBu")
plt.colorbar()
plt.savefig("figures/fit_ddist.svg")
plt.savefig("figures/fit_ddist.pdf")


plt.plot(dist_orig.flatten(), dist.flatten(), "k.")
plt.plot([0, 1], [0, 1], "k--")
plt.axis("square")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("TVD distance prior")
plt.ylabel("TVD distance posterior")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.savefig("figures/fit_dists.svg")
plt.savefig("figures/fit_dists.pdf")

# caculating evidence new
a_vals = np.linspace(-5, 0, 10)
a_vals = np.exp(a_vals) / (1 + np.exp(a_vals))
model_evidences = []
for i_voxel in tqdm.trange(d_train_ims.shape[1]):
    model_evidence = []
    for cov in tqdm.tqdm(covs):
        m_ev = []
        for a in a_vals:
            var_vox = np.var(d_train_ims[:, i_voxel])
            cov_norm = var_vox * (
                a * cov / np.trace(cov) * (N_test + N_train) + (1 - a) * np.eye(cov.shape[0])
            )
            m_ev.append(evidence(cov_norm, d_train_ims[:, i_voxel], sigma_e=0))
        model_evidence.append(np.array(m_ev).flatten())
    model_evidences.append(np.stack(model_evidence))

m_ev = np.stack(model_evidences)
np.save("m_ev_algo_a_1000.npy", m_ev)

# caculating evidence
model_evidences = []
for i_voxel in tqdm.trange(d_train_ims.shape[1]):
    model_evidence = [evidence(cov, d_train_ims[:, i_voxel]) for cov in covs_norm]
    model_evidences.append(np.stack(model_evidence).flatten())

m_ev = np.stack(model_evidences)

np.save("m_ev_algo_1000.npy", m_ev)
m_ev = np.load("m_ev_algo_1000.npy")
m_post = m_ev - logsumexp(m_ev, 1, keepdims=True)

m_ev = np.load("m_ev_algo_a_1000.npy")
m_post = logsumexp(m_ev, 2)
m_post = m_post - logsumexp(m_post, 1, keepdims=True)


# plotting posterior(s) over models
plt.axes((0.1, 0.1, 0.8, 0.55))
plt.boxplot(m_post, whis=[0, 100])
plt.plot([11.5, 11.5], [-10, 0], "k--")
plt.xlim(0, 26)
plt.ylim(-10, 0)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

m_win, counts = np.unique(np.argmax(m_post, 1), return_counts=True)
plt.axes((0.1, 0.7, 0.8, 0.2))
plt.bar(m_win + 1, counts, color="k")
plt.xlim(0, 26)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xticks([])

plt.savefig("figures/m_posterior.pdf")
plt.savefig("figures/m_posterior.svg")

plt.close("all")


### illustration of conditioned model prediction

covs = np.load("covs_algo_1000.npy")
m_ev = np.load("m_ev_algo_a_1000.npy")

# find the most informative voxel and best model for it:
i_voxel = np.argmax(np.var(logsumexp(m_ev, 2), 1))
i_model = np.argmax(logsumexp(m_ev[i_voxel], 1))

alpha = 1 - a_vals[np.argmax(m_ev[i_voxel, i_model])]

# Normalize covariances:
covs_norm = [
    (cov / np.trace(cov) * (N_test + N_train)) * np.var(d_train_ims[:, i_voxel]) for cov in covs
]

cov = covs_norm[i_model]

y_mu, y_sigma = inference_cov(cov, d_train_ims[:, i_voxel].flatten(), alpha=alpha)

scale = np.max(np.abs(cov))
plt.imshow(cov, vmin=-scale, vmax=scale, cmap="RdBu")
plt.colorbar()
plt.savefig("cov_prior.pdf")

y_sigma_prior = cov[N_train:, N_train:]
scale_prior = np.max(np.abs(y_sigma_prior))
scale_post = np.max(np.abs(y_sigma))
scale = np.max([scale_prior, scale_post])
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(y_sigma_prior, vmin=-scale, vmax=scale, cmap="RdBu")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(y_sigma, vmin=-scale, vmax=scale, cmap="RdBu")
plt.colorbar()
plt.savefig("covariances.pdf")

plt.clf()
plt.plot(y_mu, "k.", markersize=10)
plt.errorbar(range(len(y_mu)), y_mu, color="k", yerr=np.sqrt(np.diag(y_sigma)))
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
plt.xticks([])
plt.savefig("y_pred.pdf")


idx_sort = np.argsort(y_mu)
plt.clf()
plt.errorbar(
    range(len(y_mu)),
    y_mu[idx_sort],
    color=[0.5, 0.5, 0.5],
    yerr=np.sqrt(np.diag(y_sigma))[idx_sort],
)
plt.plot(y_mu[idx_sort], "k.", markersize=10)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks([])
plt.xlabel("Stimulus [sorted]")
plt.savefig("../figures/y_pred_sorted.svg")
plt.savefig("../figures/y_pred_sorted.pdf")
