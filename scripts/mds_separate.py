import numpy as np
from sklearn.manifold import MDS
from matplotlib import pyplot as plt

dist_example = np.load("dist_example.npy")

mds = MDS(dissimilarity="precomputed")

d = dist_example[1:11, 1:11]
mds.fit(d)
x_alex = mds.embedding_

d = dist_example[12:26, 12:26]
mds.fit(d)
x = mds.embedding_

d = dist_example[27:, 27:]
mds.fit(d)
x_vit = mds.embedding_

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(x_alex[:, 0], x_alex[:, 1], '.-', linewidth=2, markersize=10, color="#fa5750")  # color="#d6000c")
plt.axis("equal")
plt.subplot(1, 3, 2)
plt.plot(x[:, 0], x[:, 1], '.-', linewidth=2, markersize=10, color="#dbb32d")  # color="#c49700")
plt.axis("equal")
plt.subplot(1, 3, 3)
plt.plot(x_vit[:, 0], x_vit[:, 1], '.-', linewidth=2, markersize=10, color="#4695f7")  # color="#0064e4")
plt.axis("equal")
plt.savefig("figures/mds_separate.pdf")
plt.savefig("figures/mds_separate.svg")

plt.close('all')
