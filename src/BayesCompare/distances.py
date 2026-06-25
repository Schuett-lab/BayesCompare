"""
Module for camputing distances between prior predictive distributions (BayesCompare measures)
or existing representational similarity measures in the literature computed with second moment matrix.
Authors: Sezan Oral, Heiko Schütt
"""

import numpy as np
import torch
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
from scipy.linalg import cho_factor, cho_solve

from typing import Optional
from numpy.typing import NDArray

from .others import (
    _cka_np,
    _cka_torch,
    _rsa_corr_np,
    _rsa_corr_torch,
    _rsa_rank_spearman_np,
    _rsa_rank_spearman_torch,
    _rsa_cos_np,
    _rsa_cos_torch,
    _rsa_acos_np,
    _rsa_acos_torch,
    _kernel_gulp_np,
    _kernel_gulp_torch,
    _dist_corr_np,
    _dist_corr_torch,
    _jaccard_np,
    _jaccard_torch,
    _procrustes_np,
    _procrustes_torch,
    _normalized_bures_similarity_np,
    _normalized_bures_similarity_torch,
)

from .dist_utils import (
    check_small_negative,
    check_array,
)

### Metrics


def _wasserstein_numpy(
    sigma1: NDArray, sigma2: NDArray, mu1: NDArray = None, mu2: NDArray = None
) -> float:
    """
    Computes the Wasserstein distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Peyré and Cuturi (2019).
    """
    if mu1 is None:
        if mu2 is None:
            means_term = 0
        else:
            np.sum(mu2**2)
    elif mu2 is None:
        means_term = np.sum(mu1**2)
    else:
        means_term = np.sum((mu1 - mu2) ** 2)

    sig1_sqrt = scipy.linalg.sqrtm(sigma1)
    sig1_sig2_sqrt = scipy.linalg.sqrtm(sig1_sqrt @ sigma2 @ sig1_sqrt)

    tr_term = sigma1 + sigma2 - 2 * (sig1_sig2_sqrt)
    d_sq = means_term + np.trace(tr_term)
    d_sq = check_small_negative(d_sq)

    return check_array(d_sq**0.5)


def _wasserstein_torch(
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    mu1: torch.Tensor = None,
    mu2: torch.Tensor = None,
) -> float:
    """
    Computes the Wasserstein distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Peyré and Cuturi (2019).
    """
    if mu1 is None:
        if mu2 is None:
            means_term = 0
        else:
            torch.sum(mu2**2)
    elif mu2 is None:
        means_term = torch.sum(mu1**2)
    else:
        means_term = torch.sum((mu1 - mu2) ** 2)

    if type(sigma1) != torch.Tensor:
        sigma1 = torch.tensor(sigma1)
    if type(sigma2) != torch.Tensor:
        sigma2 = torch.tensor(sigma2)

    E_sig1, V_sig1 = torch.linalg.eigh(sigma1)
    sig1_sqrt = (V_sig1 * torch.sqrt(E_sig1)) @ V_sig1.T

    sig12 = sig1_sqrt @ sigma2 @ sig1_sqrt
    E_sig12, V_sig12 = torch.linalg.eigh(sig12)
    sig1_sig2_sqrt = (V_sig12 * torch.sqrt(E_sig12)) @ V_sig12.T

    tr_term = sigma1 + sigma2 - 2 * (sig1_sig2_sqrt)
    d_sq = means_term + torch.trace(tr_term)
    d_sq = check_small_negative(d_sq)

    return check_array(d_sq**0.5)


def _hellinger_numpy(
    sigma1: NDArray, sigma2: NDArray, mu1: NDArray = None, mu2: NDArray = None
) -> float:
    """
    Computes the Hellinger (Jeffreys-Matusita) distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Pardo (2018).
    """
    d_B = _bhattacharyya_numpy(sigma1, sigma2, mu1, mu2)
    d_sq = 1 - np.exp(-d_B)
    d_sq = check_small_negative(d_sq)

    return check_array(d_sq**0.5)


def _hellinger_torch(
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    mu1: torch.Tensor = None,
    mu2: torch.Tensor = None,
) -> float:
    """
    Computes the Hellinger (Jeffreys-Matusita) distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Pardo (2018).
    """
    d_B = _bhattacharyya_torch(sigma1, sigma2, mu1, mu2)
    d_sq = 1 - torch.exp(-d_B)
    d_sq = check_small_negative(d_sq)

    return check_array(d_sq**0.5)


def _mahalanobis_numpy(
    sigma1: NDArray, sigma2: NDArray, mu1: NDArray = None, mu2: NDArray = None
) -> float:  # not the true mahalanobis definition
    """
    Computes the Mahalanobis "term" between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Equal to zero if both means are None.
    The original definition in Mahalanobis requires two Gaussians to have thte same covariance matrix. Therefore, this is not the true Mahalanobis distance, but rather a generalization of it.
    Implementation based on Stackoverflow answer: https://stats.stackexchange.com/q/106352
    """
    if mu1 is None and mu2 is None:
        d_sq = np.array([0.0])
    else:
        delta_mu = np.subtract(mu1, mu2)
        d_sq = np.inner(
            delta_mu,
            np.matmul((np.linalg.inv(np.divide(sigma1 + sigma2, 2))), delta_mu),
        )
        d_sq = check_small_negative(d_sq)

    return check_array(d_sq**0.5)


def _mahalanobis_torch(
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    mu1: torch.Tensor = None,
    mu2: torch.Tensor = None,
) -> float:  # not the true mahalanobis definition
    """
    Computes the Mahalanobis "term" between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Equal to zero if both means are None.
    The original definition in Mahalanobis requires two Gaussians to have thte same covariance matrix. Therefore, this is not the true Mahalanobis distance, but rather a generalization of it.
    Implementation based on Stackoverflow answer: https://stats.stackexchange.com/q/106352
    """
    if mu1 is None and mu2 is None:
        d_sq = torch.Tensor([0.0])
    else:
        delta_mu = torch.subtract(mu1, mu2)
        d_sq = torch.inner(
            delta_mu,
            torch.matmul(
                (torch.linalg.inv(torch.divide(sigma1 + sigma2, 2))), delta_mu
            ),
        )
        d_sq = check_small_negative(d_sq)

    return check_array(d_sq**0.5)


def _jsd_numpy(
    sigma1: NDArray,
    sigma2: NDArray,
    mu1: NDArray = None,
    mu2: NDArray = None,
    num_samples: int = 10000,
    gen: Optional[np.random.Generator] = None,
) -> float:
    """
    Computes the Jensen-Shannon distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Endres and Schindelin (2003).
    """
    if gen == None:
        gen = np.random.Generator(np.random.SFC64())

    if mu1 is None and mu2 is None:
        k = sigma1.shape[0]
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        # generate random samples from each distribution
        x10 = gen.standard_normal(size=(k, num_samples))
        x1 = mm(1, A1, x10, lower=1)
        x20 = gen.standard_normal(size=(k, num_samples))
        x2 = mm(1, A2, x20, lower=1)
        # compute densities for each
        p1 = -np.sum(x10**2, 0) / 2 - logdet1
        delta21 = scipy.linalg.solve_triangular(A1, x2, lower=True)
        p2 = -np.sum(delta21**2, 0) / 2 - logdet1
        delta12 = scipy.linalg.solve_triangular(A2, x1, lower=True)
        q1 = -np.sum(delta12**2, 0) / 2 - logdet2
        q2 = -np.sum(x20**2, 0) / 2 - logdet2
        # log (P) - log (P + Q)
        term1 = p1 - np.logaddexp(p1, q1)
        term2 = q2 - np.logaddexp(p2, q2)

        jsd = 1 + (np.mean(term1) + np.mean(term2)) / 2 / np.log(2)

    else:
        k = len(mu1)
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        # Ainv1 = np.linalg.inv(A1)
        # Ainv2 = np.linalg.inv(A2)
        Ainv1 = scipy.linalg.solve_triangular(A1, np.eye(k), lower=True)
        Ainv2 = scipy.linalg.solve_triangular(A2, np.eye(k), lower=True)
        # generate random samples from each distribution
        x1 = np.expand_dims(mu1, 1) + A1 @ gen.standard_normal(size=(k, num_samples))
        x2 = np.expand_dims(mu2, 1) + A2 @ gen.standard_normal(size=(k, num_samples))
        # compute densities for each
        # removed factor 2 from these as it cancels
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        delta11 = Ainv1 @ (x1 - np.expand_dims(mu1, 1))
        p1 = -np.sum(delta11**2, 0) / 2 - logdet1
        delta21 = Ainv1 @ (x2 - np.expand_dims(mu1, 1))
        p2 = -np.sum(delta21**2, 0) / 2 - logdet1
        delta12 = Ainv2 @ (x1 - np.expand_dims(mu2, 1))
        q1 = -np.sum(delta12**2, 0) / 2 - logdet2
        delta22 = Ainv2 @ (x2 - np.expand_dims(mu2, 1))
        q2 = -np.sum(delta22**2, 0) / 2 - logdet2
        # log (P) - log (P + Q)
        term1 = p1 - np.logaddexp(p1, q1)
        term2 = q2 - np.logaddexp(p2, q2)

        jsd = 1 + (np.mean(term1) + np.mean(term2)) / 2 / np.log(2)

    return max(0.0, jsd)


def _jsd_torch(
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    mu1: torch.Tensor = None,
    mu2: torch.Tensor = None,
    num_samples: int = 10000,
    gen: Optional[torch.Generator] = None,
) -> float:
    """
    Computes the Jensen-Shannon distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Endres and Schindelin (2003).
    """
    if gen == None:
        gen = torch.Generator(device=sigma1.device)

    if type(sigma1) != torch.Tensor:
        sigma1 = torch.tensor(sigma1)
    if type(sigma2) != torch.Tensor:
        sigma2 = torch.tensor(sigma2)

    if mu1 is None and mu2 is None:
        k = sigma1.shape[0]
        A1 = torch.linalg.cholesky(sigma1)
        A2 = torch.linalg.cholesky(sigma2)
        logdet1 = torch.sum(torch.log(torch.diag(A1)))
        logdet2 = torch.sum(torch.log(torch.diag(A2)))
        # generate random samples from each distribution
        x10 = torch.randn((k, num_samples), generator=gen)
        x1 = torch.Tensor(mm(1, A1, x10, lower=1))
        x20 = torch.randn((k, num_samples), generator=gen)
        x2 = torch.Tensor(mm(1, A2, x20, lower=1))
        # compute densities for each
        p1 = -torch.sum(x10**2, 0) / 2 - logdet1
        delta21 = torch.linalg.solve_triangular(A1, x2, upper=False)
        p2 = -torch.sum(delta21**2, 0) / 2 - logdet1
        delta12 = torch.linalg.solve_triangular(A2, x1, upper=False)
        q1 = -torch.sum(delta12**2, 0) / 2 - logdet2
        q2 = -torch.sum(x20**2, 0) / 2 - logdet2
        # log (P) - log (P + Q)
        term1 = p1 - torch.logaddexp(p1, q1)
        term2 = q2 - torch.logaddexp(p2, q2)

        jsd = 1 + (torch.mean(term1) + torch.mean(term2)) / 2 / np.log(2)

    else:
        k = len(mu1)
        A1 = torch.linalg.cholesky(sigma1)
        A2 = torch.linalg.cholesky(sigma2)
        Ainv1 = torch.linalg.solve_triangular(A1, torch.eye(k), lower=True)
        Ainv2 = torch.linalg.solve_triangular(A2, torch.eye(k), lower=True)
        # generate random samples from each distribution
        x1 = torch.Tensor.expand(mu1, 1) + A1 @ torch.randn(
            (k, num_samples), generator=gen
        )
        x2 = torch.Tensor.expand(mu2, 1) + A2 @ torch.randn(
            (k, num_samples), generator=gen
        )
        # compute densities for each
        # removed factor 2 from these as it cancels
        logdet1 = torch.sum(torch.log(torch.diag(A1)))
        logdet2 = torch.sum(torch.log(torch.diag(A2)))
        delta11 = Ainv1 @ (x1 - torch.Tensor.expand(mu1, 1))
        p1 = -torch.sum(delta11**2, 0) / 2 - logdet1
        delta21 = Ainv1 @ (x2 - torch.Tensor.expand(mu1, 1))
        p2 = -torch.sum(delta21**2, 0) / 2 - logdet1
        delta12 = Ainv2 @ (x1 - torch.Tensor.expand(mu2, 1))
        q1 = -torch.sum(delta12**2, 0) / 2 - logdet2
        delta22 = Ainv2 @ (x2 - torch.Tensor.expand(mu2, 1))
        q2 = -torch.sum(delta22**2, 0) / 2 - logdet2
        # log (P) - log (P + Q)
        term1 = p1 - torch.logaddexp(p1, q1)
        term2 = q2 - torch.logaddexp(p2, q2)

        jsd = 1 + (torch.mean(term1) + torch.mean(term2)) / 2 / torch.log(2)

    return max(0.0, jsd)


def _tvd_numpy(
    sigma1: NDArray,
    sigma2: NDArray,
    mu1: NDArray = None,
    mu2: NDArray = None,
    num_samples: int = 10000,
    gen: Optional[np.random.Generator] = None,
) -> float:
    """
    Computes the Total Variation Distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    """
    if gen == None:
        gen = np.random.Generator(np.random.SFC64())

    if mu1 is not None and mu2 is not None:
        k = len(mu1)
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        Ainv1 = scipy.linalg.solve_triangular(A1, np.eye(k), lower=True)
        Ainv2 = scipy.linalg.solve_triangular(A2, np.eye(k), lower=True)
        # generate random samples from each distribution
        x1 = np.expand_dims(mu1, 1) + A1 @ gen.standard_normal(size=(k, num_samples))
        x2 = np.expand_dims(mu2, 1) + A2 @ gen.standard_normal(size=(k, num_samples))
        # compute densities for each
        # removed factor 2 from these as it cancels
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        delta11 = Ainv1 @ (x1 - np.expand_dims(mu1, 1))
        p1 = -np.sum(delta11**2, 0) / 2 - logdet1
        delta21 = Ainv1 @ (x2 - np.expand_dims(mu1, 1))
        p2 = -np.sum(delta21**2, 0) / 2 - logdet1
        delta12 = Ainv2 @ (x1 - np.expand_dims(mu2, 1))
        q1 = -np.sum(delta12**2, 0) / 2 - logdet2
        delta22 = Ainv2 @ (x2 - np.expand_dims(mu2, 1))
        q2 = -np.sum(delta22**2, 0) / 2 - logdet2
        f1 = np.maximum(1 - np.exp(q1 - p1), 0)
        f2 = np.maximum(1 - np.exp(p2 - q2), 0)
        tvd = (np.mean(f1) + np.mean(f2)) / 2

    else:
        k = sigma1.shape[0]
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        # generate random samples from each distribution
        x10 = gen.standard_normal(size=(k, num_samples))
        x1 = mm(1, A1, x10, lower=1)
        x20 = gen.standard_normal(size=(k, num_samples))
        x2 = mm(1, A2, x20, lower=1)
        # compute densities for each
        p1 = -np.sum(x10**2, 0) / 2 - logdet1
        delta21 = scipy.linalg.solve_triangular(A1, x2, lower=True)
        p2 = -np.sum(delta21**2, 0) / 2 - logdet1
        delta12 = scipy.linalg.solve_triangular(A2, x1, lower=True)
        q1 = -np.sum(delta12**2, 0) / 2 - logdet2
        q2 = -np.sum(x20**2, 0) / 2 - logdet2
        f1 = np.maximum(1 - np.exp(q1 - p1), 0)
        f2 = np.maximum(1 - np.exp(p2 - q2), 0)
        tvd = (np.mean(f1) + np.mean(f2)) / 2

    return max(0.0, tvd)


def _tvd_torch(
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    mu1: torch.Tensor = None,
    mu2: torch.Tensor = None,
    num_samples: int = 10000,
    gen: Optional[torch.Generator] = None,
) -> float:
    """
    Computes the Total Variation Distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    """
    if gen == None:
        gen = torch.Generator(device=sigma1.device)

    if type(sigma1) != torch.Tensor:
        sigma1 = torch.tensor(sigma1)
    if type(sigma2) != torch.Tensor:
        sigma2 = torch.tensor(sigma2)

    if mu1 is not None and mu2 is not None:
        k = len(mu1)
        A1 = torch.linalg.cholesky(sigma1)
        A2 = torch.linalg.cholesky(sigma2)
        Ainv1 = torch.linalg.solve_triangular(A1, torch.eye(k), upper=False)
        Ainv2 = torch.linalg.solve_triangular(A2, torch.eye(k), upper=False)
        # generate random samples from each distribution
        x1 = torch.Tensor.expand(mu1, 1) + A1 @ torch.randn(
            (k, num_samples), generator=gen
        )
        x2 = torch.Tensor.expand(mu2, 1) + A2 @ torch.randn(
            (k, num_samples), generator=gen
        )
        # compute densities for each
        # removed factor 2 from these as it cancels
        logdet1 = torch.sum(torch.log(torch.diag(A1)))
        logdet2 = torch.sum(torch.log(torch.diag(A2)))
        delta11 = Ainv1 @ (x1 - torch.Tensor.expand(mu1, 1))
        p1 = -torch.sum(delta11**2, 0) / 2 - logdet1
        delta21 = Ainv1 @ (x2 - torch.Tensor.expand(mu1, 1))
        p2 = -torch.sum(delta21**2, 0) / 2 - logdet1
        delta12 = Ainv2 @ (x1 - torch.Tensor.expand(mu2, 1))
        q1 = -torch.sum(delta12**2, 0) / 2 - logdet2
        delta22 = Ainv2 @ (x2 - torch.Tensor.expand(mu2, 1))
        q2 = -torch.sum(delta22**2, 0) / 2 - logdet2
        f1 = max(1 - torch.exp(q1 - p1), 0)
        f2 = max(1 - torch.exp(p2 - q2), 0)
        tvd = (torch.mean(f1) + torch.mean(f2)) / 2

    else:
        k = sigma1.shape[0]
        A1 = torch.linalg.cholesky(sigma1)
        A2 = torch.linalg.cholesky(sigma2)
        logdet1 = torch.sum(torch.log(torch.diag(A1)))
        logdet2 = torch.sum(torch.log(torch.diag(A2)))
        # generate random samples from each distribution
        x10 = torch.randn((k, num_samples), generator=gen)
        x1 = torch.Tensor(mm(1, A1, x10, lower=1))
        x20 = torch.randn((k, num_samples), generator=gen)
        x2 = torch.Tensor(mm(1, A2, x20, lower=1))
        # compute densities for each
        p1 = -torch.sum(x10**2, 0) / 2 - logdet1
        delta21 = torch.linalg.solve_triangular(A1, x2, upper=False)
        p2 = -torch.sum(delta21**2, 0) / 2 - logdet1
        delta12 = torch.linalg.solve_triangular(A2, x1, upper=False)
        q1 = -torch.sum(delta12**2, 0) / 2 - logdet2
        q2 = -torch.sum(x20**2, 0) / 2 - logdet2
        f1 = torch.max(1 - torch.exp(q1 - p1), torch.zeros_like(q1))
        f2 = torch.max(1 - torch.exp(p2 - q2), torch.zeros_like(p2))
        tvd = (torch.mean(f1) + torch.mean(f2)) / 2

    return max(0.0, tvd)


## Divergences


def _KL_div_numpy(
    sigma1: NDArray,
    sigma2: NDArray,
    mu1: NDArray = None,
    mu2: NDArray = None,
) -> float | NDArray | torch.Tensor:
    """
    Computes the symmetric Kullback-Leibler divergence between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Kullback and Leibler (1951).
    """
    c2, lower2 = cho_factor(sigma2)
    c1, lower1 = cho_factor(sigma1)
    tr_21 = np.trace(cho_solve((c2, lower2), sigma1))
    tr_12 = np.trace(cho_solve((c1, lower1), sigma2))

    if mu1 == None and mu2 == None:
        mean_term = 0
    else:
        delta_mu = np.subtract(mu1, mu2)
        mean_term = np.inner(
            delta_mu,
            np.matmul(np.linalg.inv(sigma1) + np.linalg.inv(sigma2), delta_mu),
        )
        mean_term = 0.25 * mean_term

    d = check_small_negative(0.25 * (tr_21 + tr_12) - 0.5 * sigma1.shape[0] + mean_term)
    return d


def _KL_div_torch(
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    mu1: torch.Tensor = None,
    mu2: torch.Tensor = None,
) -> float | NDArray | torch.Tensor:
    """
    Computes the symmetric Kullback-Leibler divergence between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Kullback and Leibler (1951).
    """
    c2 = torch.linalg.cholesky(sigma2)
    c1 = torch.linalg.cholesky(sigma1)
    tr_21 = torch.trace(torch.cholesky_solve(sigma1, c2))
    tr_12 = torch.trace(torch.cholesky_solve(sigma2, c1))

    if mu1 == None and mu2 == None:
        mean_term = 0
    else:
        delta_mu = torch.subtract(mu1, mu2)
        mean_term = torch.inner(
            delta_mu,
            torch.matmul(
                (torch.linalg.inv(sigma1) + torch.linalg.inv(sigma2), delta_mu)
            ),
        )
        mean_term = 0.25 * mean_term

    d = check_small_negative(0.25 * (tr_21 + tr_12) - 0.5 * sigma1.shape[0] + mean_term)
    return d


def _bhattacharyya_numpy(
    sigma1: NDArray,
    sigma2: NDArray,
    mu1: NDArray = None,
    mu2: NDArray = None,
) -> float:
    """
    Computes the Bhattacharyya distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Kashyap (2019).
    """
    means_term = _mahalanobis_numpy(sigma1, sigma2, mu1, mu2)
    log_term = np.linalg.slogdet(np.divide(sigma1 + sigma2, 2))[1] - 0.5 * (
        np.linalg.slogdet(sigma1)[1] + np.linalg.slogdet(sigma2)[1]
    )  # these may also require float64 casting

    d = (1 / 8) * means_term + (1 / 2) * log_term
    d = check_small_negative(d)
    return check_array(d)


def _bhattacharyya_torch(
    sigma1: torch.Tensor,
    sigma2: torch.Tensor,
    mu1: torch.Tensor = None,
    mu2: torch.Tensor = None,
) -> float:
    """
    Computes the Bhattacharyya distance between two multivariate normal distributions with means mu1, mu2 and covariances sigma1, sigma2.
    Implementation based on Kashyap (2019).
    """
    means_term = _mahalanobis_torch(sigma1, sigma2, mu1, mu2)
    log_term = torch.linalg.slogdet(torch.divide(sigma1 + sigma2, 2))[1] - 0.5 * (
        torch.linalg.slogdet(sigma1)[1] + torch.linalg.slogdet(sigma2)[1]
    )  # these may also require float64 casting

    d = (1 / 8) * means_term + (1 / 2) * log_term
    d = check_small_negative(d)
    return check_array(d)


DISTANCES = {
    "ours": [
        "wasserstein",
        "hellinger",
        "tvd",
        "totalvariation",
        "totalvariationdistance",
        "jsd",
        "jensenshannon",
        "jensenshannondivergence",
        "kldiv",
        "kldivergence",
        "bhattacharyya",
        "mahalanobis",
    ],
    "others": [
        "cka",
        "rsa_arccos",
        "rsa_cos",
        "rsa_corr",
        "rsa_rank",
        "rsaarccos",
        "rsacos",
        "rsacorr",
        "rsarank",
        "gulp",
        "kernelgulp",
        "distcorr",
        "distancecorrelation",
        "jaccard",
        "procrustes",
        "normalized_bures_similarity",
        "normalizedburessimilarity",
        "normburessim",
        "nbs",
    ],
}

REGISTRY = {
    "numpy": {
        "wasserstein": _wasserstein_numpy,
        "hellinger": _hellinger_numpy,
        "tvd": _tvd_numpy,
        "totalvariation": _tvd_numpy,
        "totalvariationdistance": _tvd_numpy,
        "jsd": _jsd_numpy,
        "jensenshannon": _jsd_numpy,
        "jensenshannondivergence": _jsd_numpy,
        "kldiv": _KL_div_numpy,
        "kldivergence": _KL_div_numpy,
        "bhattacharyya": _bhattacharyya_numpy,
        "mahalanobis": _mahalanobis_numpy,
        "cka": _cka_np,
        "rsaarccos": _rsa_acos_np,
        "rsacos": _rsa_cos_np,
        "rsacorr": _rsa_corr_np,
        "rsarank": _rsa_rank_spearman_np,
        "gulp": _kernel_gulp_np,
        "kernelgulp": _kernel_gulp_np,
        "distcorr": _dist_corr_np,
        "distancecorrelation": _dist_corr_np,
        "jaccard": _jaccard_np,
        "procrustes": _procrustes_np,
        "normalizedburessimilarity": _normalized_bures_similarity_np,
        "normburessim": _normalized_bures_similarity_np,
        "nbs": _normalized_bures_similarity_np,
    },
    "torch": {
        "wasserstein": _wasserstein_torch,
        "hellinger": _hellinger_torch,
        "tvd": _tvd_torch,
        "totalvariation": _tvd_torch,
        "totalvariationdistance": _tvd_torch,
        "jsd": _jsd_torch,
        "jensenshannon": _jsd_torch,
        "jensenshannondivergence": _jsd_torch,
        "kldiv": _KL_div_torch,
        "kldivergence": _KL_div_torch,
        "bhattacharyya": _bhattacharyya_torch,
        "mahalanobis": _mahalanobis_torch,
        "cka": _cka_torch,
        "rsaarccos": _rsa_acos_torch,
        "rsacos": _rsa_cos_torch,
        "rsacorr": _rsa_corr_torch,
        "rsarank": _rsa_rank_spearman_torch,
        "gulp": _kernel_gulp_torch,
        "kernelgulp": _kernel_gulp_torch,
        "distcorr": _dist_corr_torch,
        "distancecorrelation": _dist_corr_torch,
        "jaccard": _jaccard_torch,
        "procrustes": _procrustes_torch,
        "normalizedburessimilarity": _normalized_bures_similarity_torch,
        "normburessim": _normalized_bures_similarity_torch,
        "nbs": _normalized_bures_similarity_torch,
    },
}
