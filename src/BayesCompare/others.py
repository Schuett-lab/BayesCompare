"""other metrics"""

import numpy as np
import torch
from scipy.stats import spearmanr
from typing import Sequence, Union, Optional
from .distances import simplify_string
from .cov_utils import (
    check_cov_normalized,
    cov_trace_norm_sigma_N,
    check_cov_symmetry,
    check_and_change_input_format,
    check_input_format,
)
import tqdm


def _cka_np(K1, K2):
    """centred kernel alignment"""
    # centering
    K1 = K1 - np.mean(K1, 0, keepdims=True)
    K2 = K2 - np.mean(K2, 0, keepdims=True)
    return np.sum(K1 * K2) / np.sqrt(np.sum(K1 * K1)) / np.sqrt(np.sum(K2 * K2))


def _cka_torch(K1, K2):
    """centred kernel alignment"""
    # centering
    K1 = K1 - torch.mean(K1, 0, keepdim=True)
    K2 = K2 - torch.mean(K2, 0, keepdim=True)
    return (
        torch.sum(K1 * K2)
        / torch.sqrt(torch.sum(K1 * K1))
        / torch.sqrt(torch.sum(K2 * K2))
    )


def _rsa_corr_np(K1, K2):
    """euclidean distances, correlation similarity"""
    # conversion to distances
    d1, d2 = _rsa_euclidean_dist_np(K1, K2)
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
    d1, d2 = _rsa_euclidean_dist_torch(K1, K2)
    idx = torch.triu_indices(d1.shape[0], 1)
    d1[idx] -= torch.mean(d1[idx])
    d2[idx] -= torch.mean(d2[idx])
    return (
        torch.sum(d1[idx] * d2[idx])
        / torch.sqrt(np.sum(d1[idx] * d1[idx]))
        / torch.sqrt(np.sum(d2[idx] * d2[idx]))
    )


def _rsa_rank_spearman_np(K1, K2):
    """euclidean distances, Spearman's rank correlation similarity"""
    d1, d2 = _rsa_euclidean_dist_np(K1, K2)
    idx = np.triu_indices(d1.shape[0], 1)
    rho, p = spearmanr(d1[idx], d2[idx])
    return rho


def _rsa_rank_spearman_torch(K1, K2):
    """euclidean distances, Spearman's rank correlation similarity"""
    d1, d2 = _rsa_euclidean_dist_torch(K1, K2)
    idx = torch.triu_indices(d1.shape[0], 1)
    rho, p = spearmanr(d1[idx], d2[idx])
    return rho


def _rsa_cos_np(K1, K2):
    """euclidean distances, cosine similarity"""
    # conversion to distances
    d1, d2 = _rsa_euclidean_dist_np(K1, K2)
    idx = np.triu_indices(d1.shape[0], 1)
    return (
        np.sum(d1[idx] * d2[idx])
        / np.sqrt(np.sum(d1[idx] * d1[idx]))
        / np.sqrt(np.sum(d2[idx] * d2[idx]))
    )


def _rsa_cos_torch(K1, K2):
    """euclidean distances, cosine similarity"""
    # conversion to distances
    d1, d2 = _rsa_euclidean_dist_torch(K1, K2)
    idx = torch.triu_indices(d1.shape[0], 1)
    return (
        torch.sum(d1[idx] * d2[idx])
        / torch.sqrt(torch.sum(d1[idx] * d1[idx]))
        / torch.sqrt(torch.sum(d2[idx] * d2[idx]))
    )


def _rsa_acos_np(K1, K2):
    """euclidean distances, arc-cosine similarity"""
    # conversion to distances
    d1, d2 = _rsa_euclidean_dist_np(K1, K2)
    idx = np.triu_indices(d1.shape[0], 1)
    return np.arccos(
        np.sum(d1[idx] * d2[idx])
        / np.sqrt(np.sum(d1[idx] * d1[idx]))
        / np.sqrt(np.sum(d2[idx] * d2[idx]))
    )


def _rsa_acos_torch(K1, K2):
    """euclidean distances, arc-cosine similarity"""
    # conversion to distances
    d1, d2 = _rsa_euclidean_dist_torch(K1, K2)
    idx = torch.triu_indices(d1.shape[0], 1)
    return torch.arccos(
        torch.sum(d1[idx] * d2[idx])
        / torch.sqrt(torch.sum(d1[idx] * d1[idx]))
        / torch.sqrt(torch.sum(d2[idx] * d2[idx]))
    )


def _rsa_euclidean_dist_np(K1, K2):
    diag1 = np.diag(K1)
    d1 = np.expand_dims(diag1, 0) + np.expand_dims(diag1, 1) - 2 * K1
    diag2 = np.diag(K2)
    d2 = np.expand_dims(diag2, 0) + np.expand_dims(diag2, 1) - 2 * K2

    return d1, d2


def _rsa_euclidean_dist_torch(K1, K2):
    diag1 = torch.diag(K1)
    d1 = torch.Tensor.expand(diag1, 0) + torch.Tensor.expand(diag1, 1) - 2 * K1
    diag2 = torch.diag(K2)
    d2 = torch.Tensor.expand(diag2, 0) + torch.Tensor.expand(diag2, 1) - 2 * K2

    return d1, d2


def comp_other_metrics(
    covs: Union[Sequence[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
    meas_name: str = "RSA",
    noise_var: Optional[float] = None,
    b: float = 1 / 100,
):

    covs, N, module = check_and_change_input_format(covs)

    if noise_var == None:
        dim = covs[0].shape[0]  # number of images used for obtaining one cov matrix
        noise_var = dim * b / (1 + (dim * b))

    idx = np.random.randint(len(covs))
    normalized = check_cov_normalized(covs[idx])

    if not normalized:
        covs = cov_trace_norm_sigma_N(covs, noise_var=noise_var)

    # is it okay to check the symmetry of only one randomly chosen matrix or should I check all matrices in covs?
    symmetric = check_cov_symmetry(covs[idx])

    if not symmetric:
        raise ValueError(
            f"Covariance matrices should be symmetric! The covariance matrix at index {idx} violates this condition."
        )

    measure = select_other_metric(covs[0], meas_name, module=module)

    dist = module.zeros((N, N))

    progress_bar = tqdm.tqdm(total=int((N * (N - 1)) / 2))

    for i, ci in enumerate(covs):

        for j, cj in enumerate(covs):

            if j > i:

                dist[i, j] = measure(ci, cj)

                dist[j, i] = dist[i, j]

                progress_bar.update(1)

    return dist


def select_other_metric(cov_mtx, meas_name: str = "RSA", module=None):

    if module == None:
        module = check_input_format(cov_mtx)

    meas_name = simplify_string(meas_name)

    if module == np:

        if "rsa" in meas_name:
            if "arccos" in meas_name:
                measure = _rsa_acos_np
            elif "cos" in meas_name:
                measure = _rsa_cos_np
            elif "corr" in meas_name:
                measure = _rsa_corr_np
            elif "rank" in meas_name:
                measure = _rsa_rank_spearman_np

        elif "cka" in meas_name:
            measure = _cka_np

        else:
            raise NotImplementedError(
                "Given metric name is not valid for Numpy array covariances."
            )

    elif module == torch:

        if "rsa" in meas_name:
            if "arccos" in meas_name:
                measure = _rsa_acos_torch
            elif "cos" in meas_name:
                measure = _rsa_cos_torch
            elif "corr" in meas_name:
                measure = _rsa_corr_torch
            elif "rank" in meas_name:
                measure = _rsa_rank_spearman_torch

        elif "cka" in meas_name:
            measure = _cka_torch

        else:
            raise NotImplementedError(
                "Given metric name is not valid for Tensor tensor covariances."
            )

    else:
        raise NotImplementedError(
            "Covariance matrices must be either a torch tensor or a numpy array."
        )

    return measure
