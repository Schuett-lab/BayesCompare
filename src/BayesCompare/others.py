"""other metrics"""

import numpy as np
import torch
import scipy.stats as stats


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
    d1 = _rsa_euclidean_dist_np(K1)
    d2 = _rsa_euclidean_dist_np(K2)
    idx = np.triu_indices(d1.shape[0], 1)
    d1[idx] -= np.mean(d1[idx])
    d2[idx] -= np.mean(d2[idx])
    return (
        np.sum(d1[idx] * d2[idx])
        / np.sqrt(np.sum(d1[idx] * d1[idx]))
        / np.sqrt(np.sum(d2[idx] * d2[idx]))
    )


def _rsa_corr_torch(K1, K2):
    """euclidean distances, correlation similarity"""
    # conversion to distances
    d1 = _rsa_euclidean_dist_torch(K1)
    d2 = _rsa_euclidean_dist_torch(K2)
    idx = torch.triu_indices(d1.shape[0], 1)
    d1[idx] -= torch.mean(d1[idx])
    d2[idx] -= torch.mean(d2[idx])
    return (
        torch.sum(d1[idx] * d2[idx])
        / torch.sqrt(torch.sum(d1[idx] * d1[idx]))
        / torch.sqrt(torch.sum(d2[idx] * d2[idx]))
    )


def _rsa_rank_spearman_np(K1, K2):
    """euclidean distances, Spearman's rank correlation similarity"""
    d1 = _rsa_euclidean_dist_np(K1)
    d2 = _rsa_euclidean_dist_np(K2)
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
    d1 = _rsa_euclidean_dist_torch(K1)
    d2 = _rsa_euclidean_dist_torch(K2)
    idx = torch.triu_indices(d1.shape[0], 1)
    ranked_d1 = torch.Tensor(stats.rankdata(d1[idx], "average"))
    ranked_d2 = torch.Tensor(stats.rankdata(d2[idx], "average"))
    ranked_d1 = ranked_d1 - torch.mean(ranked_d1)
    ranked_d2 = ranked_d2 - torch.mean(ranked_d2)
    n = ranked_d1.shape[0]
    rho_a = torch.sum(ranked_d1 * ranked_d2) / (n**3 - n) * 12
    return rho_a


def _rsa_cos_np(K1, K2):
    """euclidean distances, cosine similarity"""
    # conversion to distances
    d1 = _rsa_euclidean_dist_np(K1)
    d2 = _rsa_euclidean_dist_np(K2)
    idx = np.triu_indices(d1.shape[0], 1)
    return (
        np.sum(d1[idx] * d2[idx])
        / np.sqrt(np.sum(d1[idx] * d1[idx]))
        / np.sqrt(np.sum(d2[idx] * d2[idx]))
    )


def _rsa_cos_torch(K1, K2):
    """euclidean distances, cosine similarity"""
    # conversion to distances
    d1 = _rsa_euclidean_dist_torch(K1)
    d2 = _rsa_euclidean_dist_torch(K2)
    idx = torch.triu_indices(d1.shape[0], 1)
    return (
        torch.sum(d1[idx] * d2[idx])
        / torch.sqrt(torch.sum(d1[idx] * d1[idx]))
        / torch.sqrt(torch.sum(d2[idx] * d2[idx]))
    )


def _rsa_acos_np(K1, K2):
    """euclidean distances, arc-cosine similarity"""
    # conversion to distances
    d1 = _rsa_euclidean_dist_np(K1)
    d2 = _rsa_euclidean_dist_np(K2)
    idx = np.triu_indices(d1.shape[0], 1)
    return np.arccos(
        np.sum(d1[idx] * d2[idx])
        / np.sqrt(np.sum(d1[idx] * d1[idx]))
        / np.sqrt(np.sum(d2[idx] * d2[idx]))
    )


def _rsa_acos_torch(K1, K2):
    """euclidean distances, arc-cosine similarity"""
    # conversion to distances
    d1 = _rsa_euclidean_dist_torch(K1)
    d2 = _rsa_euclidean_dist_torch(K2)
    idx = torch.triu_indices(d1.shape[0], 1)
    return torch.arccos(
        torch.sum(d1[idx] * d2[idx])
        / torch.sqrt(torch.sum(d1[idx] * d1[idx]))
        / torch.sqrt(torch.sum(d2[idx] * d2[idx]))
    )


def _rsa_euclidean_dist_np(M):
    diag = np.diag(M)
    d = np.expand_dims(diag, 0) + np.expand_dims(diag, 1) - 2 * M

    return d


def _rsa_euclidean_dist_torch(M):
    diag = torch.diag(M)
    d = diag.unsqueeze(0) + diag.unsqueeze(1) - 2 * M

    return d
