"""other metrics"""

import numpy as np
import torch
import scipy.stats as stats
from BayesCompare.cov_utils import cov_sigma, trace_norm
from BayesCompare.dist_utils import (
    check_small_negative,
    check_slight_greater_than_one,
    check_small_negative_eigenval,
)


def _cka_np(K1, K2):
    """centred kernel alignment"""
    K1_centered = double_centering_np(K1)
    K2_centered = double_centering_np(K2)

    return (
        np.sum(K1_centered * K2_centered)
        / np.sqrt(np.sum(K1_centered * K1_centered))
        / np.sqrt(np.sum(K2_centered * K2_centered))
    )


def _cka_torch(K1, K2):
    """centred kernel alignment"""
    K1_centered = double_centering_torch(K1)
    K2_centered = double_centering_torch(K2)

    return (
        torch.sum(K1_centered * K2_centered)
        / torch.sqrt(torch.sum(K1_centered * K1_centered))
        / torch.sqrt(torch.sum(K2_centered * K2_centered))
    )


def _rsa_corr_np(K1, K2):
    """RSA with euclidean distances, correlation similarity"""
    # conversion to distances
    d1 = _cov_to_euc_dist_np(K1)
    d2 = _cov_to_euc_dist_np(K2)
    idx = np.triu_indices(d1.shape[0], 1)
    d1[idx] -= np.mean(d1[idx])
    d2[idx] -= np.mean(d2[idx])
    corr_res = (
        np.sum(d1[idx] * d2[idx])
        / np.sqrt(np.sum(d1[idx] * d1[idx]))
        / np.sqrt(np.sum(d2[idx] * d2[idx]))
    )
    return check_slight_greater_than_one(corr_res)


def _rsa_corr_torch(K1, K2):
    """RSA with euclidean distances, correlation similarity"""
    # conversion to distances
    d1 = _cov_to_euc_dist_torch(K1)
    d2 = _cov_to_euc_dist_torch(K2)
    idx = torch.triu_indices(d1.shape[0], d1.shape[0], 1)
    triu_d1 = d1[idx[0], idx[1]]
    triu_d1 = triu_d1 - triu_d1.mean()
    triu_d2 = d2[idx[0], idx[1]]
    triu_d2 = triu_d2 - triu_d2.mean()
    corr_res = (
        torch.sum(triu_d1 * triu_d2)
        / torch.sqrt(torch.sum(triu_d1 * triu_d1))
        / torch.sqrt(torch.sum(triu_d2 * triu_d2))
    )
    return check_slight_greater_than_one(corr_res)


def _rsa_rank_spearman_np(K1, K2):
    """RSA with euclidean distances, Spearman's rank correlation similarity"""
    d1 = _cov_to_euc_dist_np(K1)
    d2 = _cov_to_euc_dist_np(K2)
    idx = np.triu_indices(d1.shape[0], 1)
    ranked_d1 = stats.rankdata(d1[idx], "average")
    ranked_d2 = stats.rankdata(d2[idx], "average")
    ranked_d1 = ranked_d1 - np.mean(ranked_d1)
    ranked_d2 = ranked_d2 - np.mean(ranked_d2)
    n = ranked_d1.shape[0]
    rho_a = np.sum(ranked_d1 * ranked_d2) / (n**3 - n) * 12
    return rho_a


def _rsa_rank_spearman_torch(K1, K2):
    """RSA with euclidean distances, Spearman's rank correlation similarity"""
    d1 = _cov_to_euc_dist_torch(K1)
    d2 = _cov_to_euc_dist_torch(K2)
    idx = torch.triu_indices(d1.shape[0], d1.shape[0], 1)
    triu_d1 = d1[idx[0], idx[1]]
    triu_d2 = d2[idx[0], idx[1]]
    ranked_d1 = torch.Tensor(stats.rankdata(triu_d1, "average"))
    ranked_d2 = torch.Tensor(stats.rankdata(triu_d2, "average"))
    ranked_d1 = ranked_d1 - torch.mean(ranked_d1)
    ranked_d2 = ranked_d2 - torch.mean(ranked_d2)
    n = ranked_d1.shape[0]
    rho_a = torch.sum(ranked_d1 * ranked_d2) / (n**3 - n) * 12
    return rho_a


def _rsa_cos_np(K1, K2):
    """RSA with euclidean distances, cosine similarity"""
    # conversion to distances
    d1 = _cov_to_euc_dist_np(K1)
    d2 = _cov_to_euc_dist_np(K2)
    idx = np.triu_indices(d1.shape[0], 1)
    cos_res = (
        np.sum(d1[idx] * d2[idx])
        / np.sqrt(np.sum(d1[idx] * d1[idx]))
        / np.sqrt(np.sum(d2[idx] * d2[idx]))
    )
    return check_slight_greater_than_one(cos_res)


def _rsa_cos_torch(K1, K2):
    """RSA with euclidean distances, cosine similarity"""
    # conversion to distances
    d1 = _cov_to_euc_dist_torch(K1)
    d2 = _cov_to_euc_dist_torch(K2)
    idx = torch.triu_indices(d1.shape[0], d1.shape[0], 1)
    triu_d1 = d1[idx[0], idx[1]]
    triu_d2 = d2[idx[0], idx[1]]
    cos_res = (
        torch.sum(triu_d1 * triu_d2)
        / torch.sqrt(torch.sum(triu_d1 * triu_d1))
        / torch.sqrt(torch.sum(triu_d2 * triu_d2))
    )
    return check_slight_greater_than_one(cos_res)


def _rsa_acos_np(K1, K2):
    """RSA with euclidean distances, arc-cosine similarity"""
    return np.arccos(_rsa_cos_np(K1, K2))


def _rsa_acos_torch(K1, K2):
    """RSA with euclidean distances, arc-cosine similarity"""
    return torch.arccos(_rsa_cos_torch(K1, K2))


def _kernel_gulp_np(K1, K2, lmbd):
    """Kernel-based GULP"""
    k1_normed = trace_norm(K1)
    k2_normed = trace_norm(K2)
    k1_lmbd_inv = np.linalg.inv(cov_sigma(k1_normed, noise_var=lmbd, signal_var=1))
    k2_lmbd_inv = np.linalg.inv(cov_sigma(k2_normed, noise_var=lmbd, signal_var=1))
    k1_term = k1_lmbd_inv @ k1_normed @ k1_lmbd_inv @ k1_normed
    k2_term = k2_lmbd_inv @ k2_normed @ k2_lmbd_inv @ k2_normed
    k12_term = k1_lmbd_inv @ k1_normed @ k2_lmbd_inv @ k2_normed
    d_sq = np.trace(k1_term) + np.trace(k2_term) - 2 * np.trace(k12_term)

    d_sq = check_small_negative(d_sq)

    return d_sq**0.5


def _kernel_gulp_torch(K1, K2, lmbd):
    """Kernel-based GULP"""
    k1_normed = trace_norm(K1)
    k2_normed = trace_norm(K2)
    k1_lmbd_inv = torch.linalg.inv(cov_sigma(k1_normed, noise_var=lmbd, signal_var=1))
    k2_lmbd_inv = torch.linalg.inv(cov_sigma(k2_normed, noise_var=lmbd, signal_var=1))
    k1_term = k1_lmbd_inv @ k1_normed @ k1_lmbd_inv @ k1_normed
    k2_term = k2_lmbd_inv @ k2_normed @ k2_lmbd_inv @ k2_normed
    k12_term = k1_lmbd_inv @ k1_normed @ k2_lmbd_inv @ k2_normed
    d_sq = torch.trace(k1_term) + torch.trace(k2_term) - 2 * torch.trace(k12_term)

    d_sq = check_small_negative(d_sq)

    return d_sq**0.5


def _dist_corr_np(K1, K2):
    """Distance Correlation"""
    K1_centered = double_centering_np(K1)
    K2_centered = double_centering_np(K2)

    return np.sqrt(
        _dCov2(K1_centered, K2_centered)
        / np.sqrt(_dCov2(K1_centered, K1_centered) * _dCov2(K2_centered, K2_centered))
    )


def _dist_corr_torch(K1, K2):
    """Distance Correlation"""
    K1_centered = double_centering_torch(K1)
    K2_centered = double_centering_torch(K2)

    return torch.sqrt(
        _dCov2(K1_centered, K2_centered)
        / torch.sqrt(
            _dCov2(K1_centered, K1_centered) * _dCov2(K2_centered, K2_centered)
        )
    )


def _jaccard_np(K1, K2, k):
    """Jaccard Similarity"""
    d1 = _cov_to_cos_sim_np(K1)
    d2 = _cov_to_cos_sim_np(K2)

    n1 = _knn_from_dist_np(d1, k, asc_or_desc="desc")
    n2 = _knn_from_dist_np(d2, k, asc_or_desc="desc")

    n = K1.shape[0]
    j = np.empty(n, dtype=float)

    for i in range(n):
        s1 = set(n1[i])
        s2 = set(n2[i])
        inter = len(s1 & s2)
        union = len(s1 | s2)
        j[i] = inter / union

    return j.mean()


def _jaccard_torch(K1, K2, k):
    """Jaccard Similarity"""
    d1 = _cov_to_cos_sim_torch(K1)
    d2 = _cov_to_cos_sim_torch(K2)

    n1 = _knn_from_dist_torch(d1, k, asc_or_desc="desc")
    n2 = _knn_from_dist_torch(d2, k, asc_or_desc="desc")

    n = K1.shape[0]
    j = torch.empty(n, dtype=float)

    for i in range(n):
        s1 = set({int(x) for x in n1[i]})
        s2 = set({int(x) for x in n2[i]})
        inter = len(s1 & s2)
        union = len(s1 | s2)
        j[i] = inter / union

    return j.mean()


def _procrustes_np(K1, K2):

    sigma1 = double_centering_np(K1)
    sigma2 = double_centering_np(K2)

    E_sig1, V_sig1 = np.linalg.eigh(sigma1)
    E_sig1 = check_small_negative_eigenval(E_sig1)
    sig1_sqrt = (V_sig1 * np.sqrt(E_sig1)) @ V_sig1.T

    sig12 = sig1_sqrt @ sigma2 @ sig1_sqrt
    E_sig12, V_sig12 = np.linalg.eigh(sig12)
    E_sig12 = check_small_negative_eigenval(E_sig12)
    sig1_sig2_sqrt = (V_sig12 * np.sqrt(E_sig12)) @ V_sig12.T

    d_sq = np.trace(sigma1 + sigma2 - 2 * (sig1_sig2_sqrt))

    d_sq = check_small_negative(d_sq)

    return d_sq**0.5


def _procrustes_torch(K1, K2):

    if type(K1) != torch.Tensor:
        K1 = torch.tensor(K1)
    if type(K2) != torch.Tensor:
        K2 = torch.tensor(K2)

    K1 = K1.to(torch.float64)
    K2 = K2.to(torch.float64)
    sigma1 = double_centering_torch(K1)
    sigma2 = double_centering_torch(K2)

    E_sig1, V_sig1 = torch.linalg.eigh(sigma1)
    E_sig1 = check_small_negative_eigenval(E_sig1)
    sig1_sqrt = (V_sig1 * torch.sqrt(E_sig1)) @ V_sig1.T

    sig12 = sig1_sqrt @ sigma2 @ sig1_sqrt
    E_sig12, V_sig12 = torch.linalg.eigh(sig12)
    E_sig12 = check_small_negative_eigenval(E_sig12)
    sig1_sig2_sqrt = (V_sig12 * torch.sqrt(E_sig12)) @ V_sig12.T

    d_sq = torch.trace(sigma1 + sigma2 - 2 * (sig1_sig2_sqrt))

    d_sq = check_small_negative(d_sq)

    return d_sq**0.5


## Helper Functions


def _dCov2(A, B) -> float:
    return (A * B).mean()


def _cov_to_euc_dist_np(M):
    """
    Function for obtaining squared euclidean distance from covariance matrix.
    """
    diag = np.diag(M)
    d = np.expand_dims(diag, 0) + np.expand_dims(diag, 1) - 2 * M

    return d


def _cov_to_euc_dist_torch(M):
    """
    Function for obtaining squared euclidean distance from covariance matrix.
    """
    diag = torch.diag(M)
    d = diag.unsqueeze(0) + diag.unsqueeze(1) - 2 * M

    return d


def _cov_to_cos_sim_np(M):
    """
    Function for obtaining cosine similarity from covariance matrix.
    """
    diag = np.diag(M)
    denom = np.sqrt(diag)[None, :] * np.sqrt(diag)[:, None]  # outer product
    cos_sim = M / denom
    np.fill_diagonal(cos_sim, 1.0)

    return cos_sim


def _cov_to_cos_sim_torch(M):
    """
    Function for obtaining cosine similarity from covariance matrix.
    """
    diag = torch.diagonal(M)
    denom = torch.sqrt(diag)[None, :] * torch.sqrt(diag)[:, None]
    cos_sim = M / denom
    cos_sim.fill_diagonal_(1.0)

    return cos_sim


def double_centering_np(M):
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def double_centering_torch(M):
    return M - M.mean(dim=1, keepdim=True) - M.mean(dim=0, keepdim=True) + M.mean()


def _knn_from_dist_np(dist, k, asc_or_desc):
    if asc_or_desc == "asc":
        order = np.argsort(dist, axis=1)
    elif asc_or_desc == "desc":
        order = np.argsort(-dist, axis=1)
    return order[:, 1 : k + 1]


def _knn_from_dist_torch(dist, k, asc_or_desc):
    if asc_or_desc == "asc":
        order = torch.argsort(dist, dim=1, descending=False)
    elif asc_or_desc == "desc":
        order = torch.argsort(dist, dim=1, descending=True)
    return order[:, 1 : k + 1]
