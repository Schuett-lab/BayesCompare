"""other metrics"""

import numpy as np
import torch
import scipy.stats as stats
from BayesCompare.cov_utils import cov_sigma, trace_norm
from BayesCompare.dist_utils import check_small_negative, check_slight_greater_than_one


def _cka_np(K1, K2):
    """centred kernel alignment"""
    # centering
    K1_centered = (
        K1 - K1.mean(axis=1, keepdims=True) - K1.mean(axis=0, keepdims=True) + K1.mean()
    )
    K2_centered = (
        K2 - K2.mean(axis=1, keepdims=True) - K2.mean(axis=0, keepdims=True) + K2.mean()
    )

    return (
        np.sum(K1_centered * K2_centered)
        / np.sqrt(np.sum(K1_centered * K1_centered))
        / np.sqrt(np.sum(K2_centered * K2_centered))
    )


def _cka_torch(K1, K2):
    """centred kernel alignment"""
    # centering
    K1_centered = (
        K1 - K1.mean(dim=1, keepdim=True) - K1.mean(dim=0, keepdim=True) + K1.mean()
    )
    K2_centered = (
        K2 - K2.mean(dim=1, keepdim=True) - K2.mean(dim=0, keepdim=True) + K2.mean()
    )
    return (
        torch.sum(K1_centered * K2_centered)
        / torch.sqrt(torch.sum(K1_centered * K1_centered))
        / torch.sqrt(torch.sum(K2_centered * K2_centered))
    )


def _rsa_corr_np(K1, K2):
    """euclidean distances, correlation similarity"""
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
    """euclidean distances, correlation similarity"""
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
    """euclidean distances, Spearman's rank correlation similarity"""
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
    """euclidean distances, Spearman's rank correlation similarity"""
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
    """euclidean distances, cosine similarity"""
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
    """euclidean distances, cosine similarity"""
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
    """euclidean distances, arc-cosine similarity"""
    return np.arccos(_rsa_cos_np(K1, K2))


def _rsa_acos_torch(K1, K2):
    """euclidean distances, arc-cosine similarity"""
    return torch.arccos(_rsa_cos_torch(K1, K2))


def _kernel_gulp_np(K1, K2, lmbd):
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
    K1_centered = (
        K1 - K1.mean(axis=1, keepdims=True) - K1.mean(axis=0, keepdims=True) + K1.mean()
    )
    K2_centered = (
        K2 - K2.mean(axis=1, keepdims=True) - K2.mean(axis=0, keepdims=True) + K2.mean()
    )

    return np.sqrt(
        _dCov2(K1_centered, K2_centered)
        / np.sqrt(_dCov2(K1_centered, K1_centered) * _dCov2(K2_centered, K2_centered))
    )


def _dist_corr_torch(K1, K2):
    K1_centered = (
        K1 - K1.mean(axis=1, keepdims=True) - K1.mean(axis=0, keepdims=True) + K1.mean()
    )
    K2_centered = (
        K2 - K2.mean(axis=1, keepdims=True) - K2.mean(axis=0, keepdims=True) + K2.mean()
    )

    return torch.sqrt(
        _dCov2(K1_centered, K2_centered)
        / torch.sqrt(
            _dCov2(K1_centered, K1_centered) * _dCov2(K2_centered, K2_centered)
        )
    )


def _jaccard_np(K1, K2, k):
    d1 = _cov_to_euc_dist_np(K1)
    d2 = _cov_to_euc_dist_np(K2)

    n1 = _knn_indices_from_euc_dist_np(d1, k)
    n2 = _knn_indices_from_euc_dist_np(d2, k)

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
    d1 = _cov_to_euc_dist_torch(K1)
    d2 = _cov_to_euc_dist_torch(K2)

    n1 = _knn_indices_from_euc_dist_torch(d1, k)
    n2 = _knn_indices_from_euc_dist_torch(d2, k)

    n = K1.shape[0]
    j = torch.empty(n, dtype=float)

    for i in range(n):
        s1 = set({int(x) for x in n1[i]})
        s2 = set({int(x) for x in n2[i]})
        inter = len(s1 & s2)
        union = len(s1 | s2)
        j[i] = inter / union

    return j.mean()


## Helper Functions


def _dCov2(A, B) -> float:
    return (1 / (A.shape[0])) * (A * B).mean()


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


def _knn_indices_from_euc_dist_np(euc_dist, k):
    order = np.argsort(euc_dist, axis=1)
    return order[:, 1 : k + 1]


def _knn_indices_from_euc_dist_torch(euc_dist, k):
    order = torch.argsort(euc_dist, axis=1)
    return order[:, 1 : k + 1]
