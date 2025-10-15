import numpy as np
import os
import PIL
import BayesCompare
import matplotlib.pyplot as plt
import tqdm
from sklearn.manifold import MDS


eye_w = 9/10

covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000_resnet50_trial.npy")

dist = np.zeros((len(covs), len(covs)))
for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
    for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
        if j > i:
            dist[i, j] = BayesCompare.jsd_normal_sig(ci, cj, 10000, eye_w=eye_w)
            dist[j, i] = dist[i, j]
            
np.save("dist_example_resnet.npy", dist)


order = range(0, 14*3)
values = np.sqrt(dist[order][:, order])
plt.figure()
plt.imshow(np.sqrt(dist[order][:, order]), "bone", vmin=0, vmax=1)
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("figures/example_dist_resnet_eye_"+str(eye_w)+".svg")
plt.savefig("figures/example_dist_resnet_eye_"+str(eye_w)+".pdf")

dist = np.load("dist_example_resnet.npy")

mds = MDS(dissimilarity="precomputed")
mds.fit(dist)

x = mds.embedding_
x0 = x[1:14]
x1 = x[15:28]
x2 = x[29:42]

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
    x[0, 0], x[0, 1], x2[0, 0] - x[0, 0], x2[0, 1] - x[0, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.axis("equal")
ax = plt.gca()
ax.set_axis_off()
plt.savefig("figures/example_mds_resnet.svg")
plt.savefig("figures/example_mds_resnet.pdf")