import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm
import re
import torch
import functools
from scipy.linalg import cho_factor, cho_solve
from typing import Sequence, Union, Optional

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
)
from .cov_utils import (
    check_cov_normalized,
    cov_trace_norm_sigma_N,
    check_cov_symmetry,
    check_and_change_input_format,
    check_input_format,
)


### Metrics


def _wasserstein_numpy(sigma1, sigma2, mu1=None, mu2=None):

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

    return d_sq**0.5


def _wasserstein_torch(sigma1, sigma2, mu1=None, mu2=None):

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

    return d_sq**0.5


def _hellinger_numpy(sigma1, sigma2, mu1=None, mu2=None):

    d_B = _bhattacharyya_numpy(sigma1, sigma2, mu1, mu2)

    d_sq = 1 - np.exp(-d_B)

    d_sq = check_small_negative(d_sq)

    return d_sq**0.5


def _hellinger_torch(sigma1, sigma2, mu1=None, mu2=None):

    d_B = _bhattacharyya_torch(sigma1, sigma2, mu1, mu2)

    d_sq = 1 - torch.exp(-d_B)

    d_sq = check_small_negative(d_sq)

    return d_sq**0.5


def _mahalanobis_numpy(
    sigma1, sigma2, mu1=None, mu2=None
):  # not the true mahalanobis definition

    if mu1 is None and mu2 is None:
        d_sq = np.array([0.0])
    else:
        delta_mu = np.subtract(mu1, mu2)
        d_sq = np.inner(
            delta_mu,
            np.matmul((np.linalg.inv(np.divide(sigma1 + sigma2, 2))), delta_mu),
        )
        d_sq = check_small_negative(d_sq)

    return d_sq**0.5


def _mahalanobis_torch(
    sigma1, sigma2, mu1=None, mu2=None
):  # not the true mahalanobis definition

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

    return d_sq**0.5


gen = np.random.Generator(np.random.SFC64(42))
gen_torch_cpu = torch.Generator(device="cpu").manual_seed(42)

if torch.cuda.is_available():
    gen_torch_cuda = torch.Generator(device="cuda").manual_seed(42)


def _jsd_numpy(sigma1, sigma2, mu1=None, mu2=None, num_samples=10000, gen=None):

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

    return max(0, jsd)


def _jsd_torch(sigma1, sigma2, mu1=None, mu2=None, num_samples=10000, gen=None):

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

    return max(0, jsd)


def _tvd_numpy(sigma1, sigma2, mu1=None, mu2=None, num_samples=10000, gen=None):

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

    return max(0, tvd)


def _tvd_torch(sigma1, sigma2, mu1=None, mu2=None, num_samples=10000, gen=None):

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

    return max(0, tvd)


## Divergences


def _KL_div_numpy(sigma1, sigma2, mu1=0, mu2=0):

    c2, lower2 = cho_factor(sigma2)
    c1, lower1 = cho_factor(sigma1)

    tr_21 = np.trace(cho_solve((c2, lower2), sigma1))
    tr_12 = np.trace(cho_solve((c1, lower1), sigma2))

    if mu1 == 0 and mu2 == 0:

        mean_term = 0

    else:

        delta_mu = np.subtract(mu1, mu2)
        mean_term = np.inner(
            delta_mu,
            np.matmul(np.linalg.inv(sigma1) + np.linalg.inv(sigma2), delta_mu),
        )
        mean_term = 0.25 * mean_term

    return 0.25 * (tr_21 + tr_12) - 0.5 * sigma1.shape[0] + mean_term


def _KL_div_torch(sigma1, sigma2, mu1=0, mu2=0):

    c2 = torch.linalg.cholesky(sigma2)
    c1 = torch.linalg.cholesky(sigma1)

    tr_21 = torch.trace(torch.cholesky_solve(sigma1, c2))
    tr_12 = torch.trace(torch.cholesky_solve(sigma2, c1))

    if mu1 == 0 and mu2 == 0:

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

    return 0.25 * (tr_21 + tr_12) - 0.5 * sigma1.shape[0] + mean_term


def _bhattacharyya_numpy(sigma1, sigma2, mu1=None, mu2=None):

    means_term = _mahalanobis_numpy(sigma1, sigma2, mu1, mu2)
    log_term = np.linalg.slogdet(np.divide(sigma1 + sigma2, 2))[1] - 0.5 * (
        np.linalg.slogdet(sigma1)[1] + np.linalg.slogdet(sigma2)[1]
    )  # these may also require float64 casting

    d = (1 / 8) * means_term + (1 / 2) * log_term

    return d


def _bhattacharyya_torch(sigma1, sigma2, mu1=None, mu2=None):

    means_term = _mahalanobis_torch(sigma1, sigma2, mu1, mu2)
    log_term = torch.linalg.slogdet(torch.divide(sigma1 + sigma2, 2))[1] - 0.5 * (
        torch.linalg.slogdet(sigma1)[1] + torch.linalg.slogdet(sigma2)[1]
    )  # these may also require float64 casting

    d = (1 / 8) * means_term + (1 / 2) * log_term

    return d


## Distance function caller


## no check points, no parallelization, single measure only and torch compatable (after correcting select measure for torch comp measures too)
def measure_dist(
    covs: Union[Sequence[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
    mean: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
    meas_name: str = "TVD",
    noise_var: Optional[float] = None,
    b: float = 1 / 100,
    samples_jsd_tvd=10000,
    show_progress: Optional[bool] = True,
):
    """
    Compute a symmetric pairwise distance matrix from the list of covariances.

    This function takes a sequence of covariance matrices, applies a trace-normalization
    step to each (via `trace_norm` with an `eye_w` regularizer), selects a distance/divergence
    function by name (via `select_measure`), and computes the upper-triangular pairwise distances. The
    result is returned as a symmetric NumPy array of shape (N, N), where N is the
    number of input covariance matrices.

    Parameters
    ----------
    covs : Sequence[array-like]
        Iterable of covariance matrices (e.g., NumPy arrays).
    mean : array-like, optional
        Mean parameter.
    meas_name : str, optional
        Name of the distance/divergence measure to use. This name is resolved via
        `select_measure(meas_name)`. Default is "TVD".
    noise_var : float, optional
        Noise variance to be applied in the normalization if the matrices are not already normalized.
        If None, noise_var is computed from the number of images (dim) used to obtain the cov matrix and
        the parameter `b` using the formula
        noise_var = (dim * b) / (1 + (dim * b)). Default is None.
    b : float, optional
        Scalar used to compute a default `noise_var` when `noise_var` is None. Default is
        1/100.
    samples_jsd_tvd : integer, optional
        Number of samples used for computing JSD and TVD measures. Defaults to 10000.
    show_progress : bool, optional
        Boolean to turn the tqdm progress bars on (True) or off (False). Default is on (True).

    Returns
    -------
    dist : numpy.ndarray or torch.Tensor
        A symmetric 2-D array of shape (N, N) containing pairwise distances
        between trace-normalized and noise added covariance inputs. The diagonal elements are zero.
        Only the upper triangle (j > i) is computed explicitly and mirrored to the
        lower triangle.

    Examples
    --------
    >>> # Given a list of covariance matrices `cov_list`
    >>> dist_matrix = measure_dist(cov_list, meas_name="TVD")
    """

    covs, N, module = check_and_change_input_format(covs)

    if noise_var == None:
        dim = covs[0].shape[0]  # number of images used for obtaining one cov matrix
        noise_var = dim * b / (1 + (dim * b))

    meas_name = simplify_string(meas_name)

    idx = np.random.randint(len(covs))
    normalized = check_cov_normalized(covs[idx])

    if not normalized and meas_name in DISTANCES["ours"]:
        covs = cov_trace_norm_sigma_N(covs, noise_var=noise_var)

    # is it okay to check the symmetry of only one randomly chosen matrix or should I check all matrices in covs?
    symmetric = check_cov_symmetry(covs[idx])

    if not symmetric:
        raise ValueError(
            f"Covariance matrices should be symmetric! The covariance matrix at index {idx} violates this condition."
        )

    measure = select_measure(covs[0], meas_name, module=module)

    dist = module.zeros((N, N))

    progress_bar = tqdm.tqdm(total=int((N * (N - 1)) / 2), disable=not show_progress)

    for i, ci in enumerate(covs):

        for j, cj in enumerate(covs):

            if j > i:

                if "tvd" in meas_name or "jsd" in meas_name:
                    dist[i, j] = measure(
                        ci, cj, num_samples=samples_jsd_tvd
                    )  # not using mean, for a generalized code mean should be provided
                else:
                    dist[i, j] = measure(
                        ci, cj
                    )  # not using mean, for a generalized code mean should be provided

                dist[j, i] = dist[i, j]

                progress_bar.update(1)

    return dist


## Helper functions


def simplify_string(s: str) -> str:
    """
    - convert to lowercase
    - remove underscores, dashes, and spaces
    """
    return re.sub(r"[ _-]+", "", s.lower())


def select_measure(cov_mtx, meas_name, module=None):
    """
    Select and return the appropriate distance measure function based on input parameters.

    This function selects a distance/similarity measure function that is compatible with
    the given covariance matrix type (NumPy array or PyTorch tensor) and the specified
    metric name.

    Parameters
    ----------
    cov_mtx : np.ndarray or torch.Tensor
        The covariance matrix for which to select a measure. Used to infer the module
        type if not explicitly provided, and to determine device placement (CPU/GPU)
        for PyTorch tensors.
    meas_name : str
        The name of the distance measure to use. Case-insensitive. Supported measures
        include: "wasserstein", "hellinger", "tvd", "jsd", "kl divergence", "bhattacharyya",
        and "mahalanobis".
    module : {np, torch}, optional
        The module type indicating whether to use NumPy or PyTorch implementations.
        If None (default), the module type is inferred from cov_mtx using check_input_format.

    Returns
    -------
    measure : callable
        A function that computes the selected distance measure between two covariance
        matrices. For stochastic measures (TVD, JSD), returns a functools.partial object
        with the appropriate random generator pre-configured.

    Raises
    ------
    NotImplementedError
        If the metric name is not valid for the given module type, or if the covariance
        matrix is neither a NumPy array nor a PyTorch tensor.
    """

    if module == None:
        module = check_input_format(cov_mtx)

    meas_name = simplify_string(meas_name)

    if module == np:

        try:
            measure = REGISTRY["numpy"][meas_name]

        except KeyError:
            raise NotImplementedError(
                "Given metric name is not valid for Numpy array covariances."
            )

    elif module == torch:

        try:
            if cov_mtx.is_cuda and ("tvd" in meas_name or "jsd" in meas_name):

                measure = REGISTRY["torch"][meas_name + "cuda"]

            elif not (cov_mtx.is_cuda) and ("tvd" in meas_name or "jsd" in meas_name):

                measure = REGISTRY["torch"][meas_name + "cpu"]

            else:
                measure = REGISTRY["torch"][meas_name]

        except KeyError:
            raise NotImplementedError(
                "Given metric name is not valid for Tensor tensor covariances."
            )

    else:
        raise NotImplementedError(
            "Covariance matrices must be either a torch tensor or a numpy array."
        )

    return measure


def check_small_negative(d_sq):

    if d_sq < 0 and d_sq > -1e-7:
        d_sq = 0

    return d_sq


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
    "others": ["cka", "rsa_arccos", "rsa_cos", "rsa_corr", "rsa_rank"],
}

REGISTRY = {
    "numpy": {
        "wasserstein": _wasserstein_numpy,
        "hellinger": _hellinger_numpy,
        "tvd": functools.partial(_tvd_numpy, gen=gen),
        "totalvariation": functools.partial(_tvd_numpy, gen=gen),
        "totalvariationdistance": functools.partial(_tvd_numpy, gen=gen),
        "jsd": functools.partial(_jsd_numpy, gen=gen),
        "jensenshannon": functools.partial(_jsd_numpy, gen=gen),
        "jensenshannondivergence": functools.partial(_jsd_numpy, gen=gen),
        "kldiv": _KL_div_numpy,
        "kldivergence": _KL_div_numpy,
        "bhattacharyya": _bhattacharyya_numpy,
        "mahalanobis": _mahalanobis_numpy,
        "cka": _cka_np,
        "rsaarccos": _rsa_acos_np,
        "rsacos": _rsa_cos_np,
        "rsacorr": _rsa_corr_np,
        "rsarank": _rsa_rank_spearman_np,
    },
    "torch": {
        "wasserstein": _wasserstein_torch,
        "hellinger": _hellinger_torch,
        "tvdcuda": functools.partial(_tvd_torch, gen=gen_torch_cuda),
        "totalvariationcuda": functools.partial(_tvd_torch, gen=gen_torch_cuda),
        "totalvariationdistancecuda": functools.partial(_tvd_torch, gen=gen_torch_cuda),
        "tvdcpu": functools.partial(_tvd_torch, gen=gen_torch_cpu),
        "totalvariationcpu": functools.partial(_tvd_torch, gen=gen_torch_cpu),
        "totalvariationdistancecpu": functools.partial(_tvd_torch, gen=gen_torch_cpu),
        "jsdcuda": functools.partial(_jsd_torch, gen=gen_torch_cuda),
        "jensenshannoncuda": functools.partial(_jsd_torch, gen=gen_torch_cuda),
        "jensenshannondivergencecuda": functools.partial(
            _jsd_torch, gen=gen_torch_cuda
        ),
        "jsdcpu": functools.partial(_jsd_torch, gen=gen_torch_cpu),
        "jensenshannoncpu": functools.partial(_jsd_torch, gen=gen_torch_cpu),
        "jensenshannondivergencecpu": functools.partial(_jsd_torch, gen=gen_torch_cpu),
        "kldiv": _KL_div_torch,
        "kldivergence": _KL_div_torch,
        "bhattacharyya": _bhattacharyya_torch,
        "mahalanobis": _mahalanobis_torch,
        "cka": _cka_torch,
        "rsaarccos": _rsa_acos_torch,
        "rsacos": _rsa_cos_torch,
        "rsacorr": _rsa_corr_torch,
        "rsarank": _rsa_rank_spearman_torch,
    },
}
