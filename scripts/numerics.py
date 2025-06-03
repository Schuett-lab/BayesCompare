import numpy as np
import tqdm
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.linalg.blas import dtrmm as mm  # type: ignore

covs = np.load("covs_1000.npy")

gen = np.random.default_rng(42)

# analyse variances
N = 100


def trace_norm(sigma, eye_w=0.001):
    """normalization of covariance matrices
    & computation of cholesky decomposition

    returns a cholesky factor for sigma after normalization
    """
    A = np.linalg.cholesky(
        (1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)
        + eye_w * np.eye(sigma.shape[0]))
    return A


def tvd_normal_var(sigma1, sigma2, N=10000, eye_w=0.001, gen=gen):
    """
    total variation distance for two normal distributions with
    means mu1 & mu2
    and variances sigma1 & sigma2

    does not check any conditions, but normalizes variances to trace = 1
    """
    k = sigma1.shape[0]
    A1 = trace_norm(sigma1, eye_w)
    A2 = trace_norm(sigma2, eye_w)
    logdet1 = np.sum(np.log(np.diag(A1)))
    logdet2 = np.sum(np.log(np.diag(A2)))
    # generate random samples from each distribution
    x10 = gen.standard_normal(size=(k, N))
    x1 = mm(1, A1, x10, lower=1)
    x20 = gen.standard_normal(size=(k, N))
    x2 = mm(1, A2, x20, lower=1)
    # compute densities for each
    p1 = - np.sum(x10 ** 2, 0) / 2 - logdet1
    delta21 = scipy.linalg.solve_triangular(A1, x2, lower=True)
    p2 = - np.sum(delta21 ** 2, 0) / 2 - logdet1
    delta12 = scipy.linalg.solve_triangular(A2, x1, lower=True)
    q1 = - np.sum(delta12 ** 2, 0) / 2 - logdet2
    q2 = - np.sum(x20 ** 2, 0) / 2 - logdet2
    f1 = np.maximum(1 - np.exp(q1-p1), 0)
    f2 = np.maximum(1 - np.exp(p2-q2), 0)
    tvd = (np.mean(f1) + np.mean(f2)) / 2
    var = (np.var(f1) + np.var(f2)) / 4
    return tvd, var


dist = np.zeros((len(covs), len(covs)))
var = np.zeros((len(covs), len(covs)))
for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
    for j, cj in enumerate(covs):
        if j > i:
            dist[i, j], var[i, j] = tvd_normal_var(
                covs[i][:N][:, :N], covs[j][:N][:, :N],
                10000, eye_w=(N / 100) / (1 + (N / 100)))
            var[j, i] = var[i, j]
            dist[j, i] = dist[i, j]

dist_tvd = dist[np.triu_indices(len(covs), 1)]
var_tvd = var[np.triu_indices(len(covs), 1)]


def jsd_normal_var(sigma1, sigma2, N=10000, eye_w=0.0, gen=gen):
    """
    JSD for two normal distributions with
    zero mean
    and variances sigma1 & sigma2

    This function normalizes the sigmas to determinant 1
    before computing the jsd.
    """
    k = sigma1.shape[0]
    A1 = trace_norm(sigma1, eye_w)
    A2 = trace_norm(sigma2, eye_w)
    logdet1 = np.sum(np.log(np.diag(A1)))
    logdet2 = np.sum(np.log(np.diag(A2)))
    # generate random samples from each distribution
    x10 = gen.standard_normal(size=(k, N))
    x1 = mm(1, A1, x10, lower=1)
    x20 = gen.standard_normal(size=(k, N))
    x2 = mm(1, A2, x20, lower=1)
    # compute densities for each
    p1 = - np.sum(x10 ** 2, 0) / 2 - logdet1
    delta21 = scipy.linalg.solve_triangular(A1, x2, lower=True)
    p2 = - np.sum(delta21 ** 2, 0) / 2 - logdet1
    delta12 = scipy.linalg.solve_triangular(A2, x1, lower=True)
    q1 = - np.sum(delta12 ** 2, 0) / 2 - logdet2
    q2 = - np.sum(x20 ** 2, 0) / 2 - logdet2

    # log (P) - log (P + Q)
    term1 = p1 - np.logaddexp(p1, q1)
    term2 = q2 - np.logaddexp(p2, q2)

    jsd = 1 + (np.mean(term1) + np.mean(term2)) / 2 / np.log(2)
    var = (np.var(term1) + np.var(term2)) / 4 / np.log(2) / np.log(2)
    return jsd, var


dist = np.zeros((len(covs), len(covs)))
var = np.zeros((len(covs), len(covs)))
for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs), position=0):
    for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
        if j > i:
            dist[i, j], var[i, j] = jsd_normal_var(
                covs[i][:N][:, :N], covs[j][:N][:, :N],
                10000, eye_w=(N / 100) / (1 + (N / 100)))
            var[j, i] = var[i, j]
            dist[j, i] = dist[i, j]


dist_jsd = dist[np.triu_indices(len(covs), 1)]
var_jsd = var[np.triu_indices(len(covs), 1)]


np.save("dist_tvd.npy", dist_tvd)
np.save("var_tvd.npy", var_tvd)
np.save("dist_jsd.npy", dist_jsd)
np.save("var_jsd.npy", var_jsd)

# loading & plotting

dist_tvd = np.load("dist_tvd.npy")
var_tvd = np.load("var_tvd.npy")
dist_jsd = np.load("dist_jsd.npy")
var_jsd = np.load("var_jsd.npy")

plt.figure(figsize=(15, 4))
ax = plt.subplot(1, 3, 1)
plt.plot(dist_tvd, var_tvd, 'k.')
plt.title("TVD")
plt.ylim(bottom=0)
plt.xlim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax = plt.subplot(1, 3, 2)
plt.plot(dist_jsd, var_jsd, 'k.')
plt.title("JSD")
plt.ylim(bottom=0)
plt.xlim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax = plt.subplot(1, 3, 3)
plt.plot(np.sqrt(dist_jsd), var_jsd / dist_jsd / 4, 'k.')
plt.title("sqrt JSD")
plt.ylim(bottom=0)
plt.xlim(0, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("figures/numerics.svg")

print("TVD")
print(np.max(var_tvd))
print(np.sqrt(np.max(var_tvd)) / 100)

print("JSD")
print(np.max(var_jsd))
print(np.sqrt(np.max(var_jsd)) / 100)
