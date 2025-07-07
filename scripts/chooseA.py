import BayesCompare
import numpy as np
import tqdm
import matplotlib.pyplot as plt

Ns = [25, 50, 100, 200, 400, 800]
ws = [0.001, 0.01, 0.05, 0.1, round(0.2 / 1.2, 3), round(0.5 / 1.5, 3), round(1 / 2, 3),
      round(2 / 3, 3), round(4 / 5, 3), round(8 / 9, 3), round(16 / 17, 3), round(32 / 33, 3)]

Ns = np.array(np.round(np.exp(np.linspace(np.log(25), np.log(800), 21))), dtype=int)
ws = np.flip(np.round(1/(1+2**(np.linspace(-3.5, 3.5, 29))), 3))

covs = np.load("covs_1000.npy")

N = 100

dist = np.zeros((len(covs), len(covs)))
for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
    for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
        if j > i:
            dist[i, j] = BayesCompare.jsd_normal_sig(
                covs[i][:N, :N], covs[j][:N, :N], 10000, eye_w=0.5)
            dist[j, i] = dist[i, j]


# based on this choose 20, 17 as our target distance which is medium strength at 0.38


distNw = np.zeros((len(Ns), len(ws)))
for j, w in tqdm.tqdm(enumerate(ws), total=len(ws), position=0):
    for i, N in tqdm.tqdm(enumerate(Ns), total=len(Ns), position=1):
        distNw[i, j] = BayesCompare.jsd_normal_sig(covs[17][:N, :N], covs[22][:N, :N], 10000, eye_w=w)

plt.figure()
plt.imshow(distNw, cmap="bone")
plt.colorbar()
plt.xticks(np.arange(len(ws)), labels=ws)
plt.yticks(np.arange(len(Ns)), labels=Ns)
plt.savefig("figures/choose_a.pdf")
# plt.show()


distNw2 = np.zeros((len(Ns), len(ws)))
for j, w in tqdm.tqdm(enumerate(ws), total=len(ws), position=0):
    for i, N in tqdm.tqdm(enumerate(Ns), total=len(Ns), position=1):
        distNw2[i, j] = BayesCompare.jsd_normal_sig(covs[18][:N, :N], covs[19][:N, :N], 10000, eye_w=w)

plt.figure()
plt.imshow(distNw2, cmap="bone")
plt.colorbar()
plt.xticks(np.arange(len(ws)), labels=ws)
plt.yticks(np.arange(len(Ns)), labels=Ns)
plt.savefig("figures/choose_a2.pdf")
# plt.show()
plt.close('all')
