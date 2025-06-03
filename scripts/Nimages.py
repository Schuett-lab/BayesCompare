import numpy as np
import BayesDist
import tqdm
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from scipy.stats import spearmanr

# analysis for number of images

covs = np.load("covs_1000.npy")

n_rep = 100
Ns = [25, 50, 100]

gen = np.random.Generator(np.random.SFC64(5))

dist = np.zeros((6, len(Ns), n_rep, len(covs), len(covs)))
for i_rep in tqdm.trange(n_rep, position=0):
    for i_N, N in tqdm.tqdm(enumerate(Ns), total=len(Ns), position=1, leave=False):
        for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs), position=2, leave=False):
            for j, cj in enumerate(covs):
                if j > i:
                    eye_w = N / 200 / (1 + (N / 200))
                    idx = gen.choice(1000, N, replace=False)
                    dist[0, i_N, i_rep, i, j] = BayesDist.jsd_normal_sig(
                        ci[idx][:, idx], cj[idx][:, idx], 10000, eye_w=eye_w
                    )
                    dist[1, i_N, i_rep, i, j] = BayesDist.tvd_normal_sig(
                        ci[idx][:, idx], cj[idx][:, idx], 10000, eye_w=eye_w
                    )
                    dist[2, i_N, i_rep, i, j] = BayesDist.others.cka(
                        ci[idx][:, idx], cj[idx][:, idx]
                    )
                    dist[3, i_N, i_rep, i, j] = BayesDist.others.rsa_corr(
                        ci[idx][:, idx], cj[idx][:, idx]
                    )
                    dist[4, i_N, i_rep, i, j] = BayesDist.others.rsa_cos(
                        ci[idx][:, idx], cj[idx][:, idx]
                    )
                    dist[5, i_N, i_rep, i, j] = BayesDist.others.rsa_acos(
                        ci[idx][:, idx], cj[idx][:, idx]
                    )
                    for l in range(6):
                        dist[l, i_N, i_rep, j, i] = dist[l, i_N, i_rep, i, j]

np.save("dist_rep_all.npy", dist)

# same distances for different eye_w
gen = np.random.Generator(np.random.SFC64(5))

dist = np.zeros((4, len(Ns), len(covs), len(covs)))
for i_N, N in tqdm.tqdm(enumerate(Ns), total=len(Ns), position=1, leave=False):
    idx = gen.choice(1000, N, replace=False)
    for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs), position=2, leave=False):
        for j, cj in enumerate(covs):
            if j > i:
                eye_w = N / 200 / (1 + (N / 200))
                eye_w_2 = 5 * N / 200 / (1 + 5 * (N / 200))
                dist[0, i_N, i, j] = BayesDist.jsd_normal_sig(
                    ci[idx][:, idx], cj[idx][:, idx], 10000, eye_w=eye_w
                )
                dist[1, i_N, i, j] = BayesDist.tvd_normal_sig(
                    ci[idx][:, idx], cj[idx][:, idx], 10000, eye_w=eye_w
                )
                dist[2, i_N, i, j] = BayesDist.jsd_normal_sig(
                    ci[idx][:, idx], cj[idx][:, idx], 10000, eye_w=eye_w_2
                )
                dist[3, i_N, i, j] = BayesDist.tvd_normal_sig(
                    ci[idx][:, idx], cj[idx][:, idx], 10000, eye_w=eye_w_2
                )
                for l in range(4):
                    dist[l, i_N, j, i] = dist[l, i_N, i, j]

np.save("dist_5x.npy", dist)


dist = np.load("dist_rep_all.npy")

# transform to distance-like
# dist[0] = np.sqrt(dist[0])
dist[4] = np.arccos(dist[2])
dist[2] = 1 - dist[2]
dist[3] = 1 - dist[3]
for i in range(dist.shape[3]):
    dist[:, :, :, i, i] = 0


idx = np.triu_indices(dist.shape[3], 1)
dist_v = dist[:, :, :, idx[0], idx[1]]
m = np.mean(dist_v, 2)
v = np.var(dist_v, 2)


plt.figure(figsize=(15, 4))
for i in range(len(m)):
    ax = plt.subplot(2, len(m), i + 1)
    plt.plot(m[i, 2], m[1, 2], "k.")
    c = np.corrcoef(m[1, 2], m[i, 2])[0, 1]
    s = (1 - c**2) / np.sqrt(m.shape[2] - 3)
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(str(round(c, 3)) + " \xb1 " + str(round(s, 3)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.subplot(2, len(m), len(m) + i + 1)
    plt.imshow(np.mean(dist[i, 2], 0), cmap="bone")
    plt.xticks([0, 10, 11, 24], [""] * 4)
    plt.yticks([0, 10, 11, 24], [""] * 4)

# plt.savefig("figures/dist_compare_2.pdf")
plt.savefig("figures/dist_compare_2.svg")

plt.figure()
plt.plot(m[1, 0].flatten(), np.sqrt(v)[1, 0].flatten(), ".", color="#0000ff")
plt.plot(m[1, 1].flatten(), np.sqrt(v)[1, 1].flatten(), ".", color="#5555ff")
plt.plot(m[1, 2].flatten(), np.sqrt(v)[1, 2].flatten(), ".", color="#9999ff")
plt.show()


# approximate transformation function
plt.rc("xtick", labelsize=18)
plt.rc("ytick", labelsize=18)
comp = [2, 3]
x_t = np.sort(m[comp[0], 2])
y_t = np.sort(m[comp[1], 2])
x_i = rankdata(m[comp[0], 2]) / m.shape[2]
y_i = rankdata(m[comp[1], 2]) / m.shape[2]

plt.figure(figsize=(10, 10))
ax = plt.subplot(2, 2, 1)
plt.plot(x_i, y_i, "k.")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("RSA arccos [percentiles]", fontsize=24)
plt.ylabel("linear CKA [perc.]", fontsize=24)

ax = plt.subplot(2, 2, 2)
plt.plot(m[comp[0], 2], m[comp[1], 2], "k.")
plt.plot(x_t, y_t, linewidth=2, color="r")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("RSA arccos", fontsize=24)
plt.ylabel("linear CKA", fontsize=24)


comp = [2, 4]
x_t = np.sort(m[comp[0], 2])
y_t = np.sort(m[comp[1], 2])
x_i = rankdata(m[comp[0], 2]) / m.shape[2]
y_i = rankdata(m[comp[1], 2]) / m.shape[2]
ax = plt.subplot(2, 2, 3)
plt.plot(x_i, y_i, "k.")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("RSA arccos [percentiles]", fontsize=24)
plt.ylabel("RSA correlation [perc.]", fontsize=24)

ax = plt.subplot(2, 2, 4)
plt.plot(m[comp[0], 2], m[comp[1], 2], "k.")
plt.plot(x_t, y_t, linewidth=2, color="r")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xlabel("RSA arccos", fontsize=24)
plt.ylabel("RSA correlation", fontsize=24)

plt.savefig("figures/transform.pdf")

# plotting for checks on change of signal to noise ratio

dist = np.load("dist_5x.npy")

for i in range(dist.shape[3]):
    dist[:, :, i, i] = 0


idx = np.triu_indices(dist.shape[3], 1)
dist_v = dist[:, :, idx[0], idx[1]]
m = dist_v

# TVD
plt.figure(figsize=(15, 4))
indices = [[1, 2], [1, 1], [1, 0], [3, 2], [3, 1], [3, 0]]
for i in range(len(indices)):
    ax = plt.subplot(2, len(indices), i + 1)
    m_plot = m[indices[i][0], indices[i][1]]
    plt.plot(m_plot, m[1, 2], "k.")
    # c = np.corrcoef(m[1, 2], m_plot)[0, 1]
    c = spearmanr(m[1, 2], m_plot).correlation
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(str(round(c, 3)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.subplot(2, len(indices), len(indices) + i + 1)
    plt.imshow(dist[indices[i][0], indices[i][1]], cmap="bone")
    plt.xticks([0, 10, 11, 24], [""] * 4)
    plt.yticks([0, 10, 11, 24], [""] * 4)
plt.show()

# JSD
plt.figure(figsize=(15, 4))
indices = [[0, 2], [0, 1], [0, 0], [2, 2], [2, 1], [2, 0]]
for i in range(len(indices)):
    ax = plt.subplot(2, len(indices), i + 1)
    m_plot = m[indices[i][0], indices[i][1]]
    plt.plot(m_plot, m[0, 2], "k.")
    # c = np.corrcoef(m[1, 2], m_plot)[0, 1]
    c = spearmanr(m[0, 2], m_plot).correlation
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(str(round(c, 3)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.subplot(2, len(indices), len(indices) + i + 1)
    plt.imshow(dist[indices[i][0], indices[i][1]], cmap="bone")
    plt.xticks([0, 10, 11, 24], [""] * 4)
    plt.yticks([0, 10, 11, 24], [""] * 4)
plt.show()
