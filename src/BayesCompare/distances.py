import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm
import torch
import functools
import warnings
from typing import Sequence, Union, Optional
from .cov_utils import (
    check_cov_normalized,
    cov_trace_norm_sigma_N,
    check_cov_symmetry,
    check_and_change_input_format,
    check_input_format,
)

## Metrics


def _wasserstein_numpy(sigma1, sigma2, mu1=None, mu2=None):

    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        means_term = np.linalg.norm(mu1 - mu2, 2) ** 2
    else:
        means_term = 0

    sig1_sqrt = scipy.linalg.sqrtm(sigma1)
    sig1_sig2_sqrt = scipy.linalg.sqrtm(sig1_sqrt @ sigma2 @ sig1_sqrt)
    tr_term = sigma1 + sigma2 - 2 * (sig1_sig2_sqrt)
    d_sq = means_term + np.trace(tr_term)

    d_sq = prevent_negative_square(d_sq, "Wasserstein")

    return d_sq**0.5


def _wasserstein_torch(sigma1, sigma2, mu1=None, mu2=None):

    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        means_term = torch.linalg.norm(mu1 - mu2, 2) ** 2
    else:
        means_term = 0

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

    d_sq = prevent_negative_square(d_sq, "Wasserstein")

    return d_sq**0.5


def _hellinger_numpy(sigma1, sigma2, mu1=None, mu2=None):

    d_B = _bhattacharyya_numpy(sigma1, sigma2, mu1, mu2)

    d_sq = 2 * (1 - np.exp(d_B))

    d_sq = prevent_negative_square(d_sq, "Hellinger")

    return d_sq**0.5


def _hellinger_torch(sigma1, sigma2, mu1=None, mu2=None):

    d_B = _bhattacharyya_torch(sigma1, sigma2, mu1, mu2)

    d_sq = 2 * (1 - torch.exp(d_B))

    d_sq = prevent_negative_square(d_sq, "Hellinger")

    return d_sq**0.5


def _mahalanobis_numpy(
    sigma1, sigma2, mu1=None, mu2=None
):  # not the true mahalanobis definition

    if mu1 is None and mu2 is None:
        d = np.array([0.0])
    else:
        delta_mu = np.subtract(mu1, mu2)
        d_sq = np.inner(
            delta_mu,
            np.matmul((np.linalg.inv(np.divide(sigma1 + sigma2, 2))), delta_mu),
        )
        d_sq = prevent_negative_square(d_sq, "Mahalanobis")
        d = d_sq**0.5

    return d


def _mahalanobis_torch(
    sigma1, sigma2, mu1=None, mu2=None
):  # not the true mahalanobis definition

    if mu1 is None and mu2 is None:
        d = torch.Tensor([0.0])
    else:
        delta_mu = torch.subtract(mu1, mu2)
        d_sq = torch.inner(
            delta_mu,
            torch.matmul(
                (torch.linalg.inv(torch.divide(sigma1 + sigma2, 2))), delta_mu
            ),
        )
        d_sq = prevent_negative_square(d_sq, "Mahalanobis")
        d = d_sq**0.5

    return d


gen = np.random.Generator(np.random.SFC64(42))

if torch.cuda.is_available():
    gen_torch_cuda = torch.Generator(device="cuda").manual_seed(42)
    # else: # I am commenting this out as I think if cuda is available both seeds should be initiated. But I will think better on it.
    gen_torch_cpu = torch.Generator(device="cpu").manual_seed(42)


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


def _KL_div_numpy(sigma1, sigma2, mu1=None, mu2=None):

    # Handling of mean term feels a bit sloppy. Can be improved in the future
    if mu1 is None:
        mu1 = np.zeros(sigma1.shape[0])

    if mu2 is None:
        mu2 = np.zeros(sigma2.shape[0])

    delta_mu = np.subtract(mu2, mu1)

    inv_s2 = np.linalg.inv(sigma2)

    if (delta_mu < 1e-15).all():  # only this condition is tested
        mean_term = 0

    else:  # this condition was not tested
        mean_term = np.transpose(delta_mu) @ inv_s2 @ delta_mu

    tr_term = np.trace(inv_s2 @ sigma1)

    log_term = np.linalg.slogdet(sigma1)[1] - np.linalg.slogdet(sigma2)[1]

    d = (1 / 2) * (mean_term + tr_term - log_term - sigma1.shape[0])

    return d


def _KL_div_torch(sigma1, sigma2, mu1=None, mu2=None):

    # Handling of mean term feels a bit sloppy. Can be improved in the future
    if mu1 is None:
        mu1 = torch.zeros(sigma1.shape[0])

    if mu2 is None:
        mu2 = torch.zeros(sigma2.shape[0])

    delta_mu = torch.subtract(mu2, mu1)

    inv_s2 = torch.linalg.inv(sigma2)

    if (delta_mu < 1e-15).all():  # only this condition is tested
        mean_term = 0

    else:  # this condition was not tested
        mean_term = torch.transpose(delta_mu) @ inv_s2 @ delta_mu

    tr_term = torch.trace(inv_s2 @ sigma1)

    log_term = torch.linalg.slogdet(sigma1)[1] - torch.linalg.slogdet(sigma2)[1]

    d = (1 / 2) * (mean_term + tr_term - log_term - sigma1.shape[0])

    return d


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
    alpha: Optional[float] = None,
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
        Iterable of covariance matrices (e.g., NumPy arrays). Each element is
        passed to `trace_norm` before the pairwise distance is computed.
    mean : array-like, optional
        Mean parameter.
    meas_name : str, optional
        Name of the distance/divergence measure to use. This name is resolved via
        `select_measure(meas_name)`. Default is "TVD".
    alpha : float, optional
        Weight applied inside `trace_norm` as the `eye_w` argument. If None,
        alpha is computed from the number of input covariances `N = len(covs)` and
        the parameter `b` using the formula
        alpha = (N * b) / (1 + (N * b)). Default is None.
    b : float, optional
        Scalar used to compute a default `alpha` when `alpha` is None. Default is
        1/100.
    samples_jsd_tvd : integer, optional
        Number of samples used for computing JSD and TVD measures. Defaults to 10000.
    show_progress : bool, optional
        Boolean to turn the tqdm progress bars on (True) or off (False). Default is on (True).

    Returns
    -------
    dist : numpy.ndarray or torch.Tensor
        A symmetric 2-D array of shape (N, N) containing pairwise distances
        between trace-normalized covariance inputs. The diagonal elements are zero.
        Only the upper triangle (j > i) is computed explicitly and mirrored to the
        lower triangle.

    Examples
    --------
    >>> # Given a list of covariance matrices `cov_list`
    >>> dist_matrix = measure_dist(cov_list, meas_name="TVD")
    """
    # I am turning covs into a list or matrices rather than using it as torch/numpy array. I don't know if it is a good thing to do.
    # I am doing it inside `check_and_change_input_format` as the whole `measure_dist` function is written to work with list of matrices.
    # But normally functions such as `cov_trace_norm_sigma_N` accept 3D torch/np arrays and work more efficiently that way.
    covs, N, module = check_and_change_input_format(covs)

    if alpha == None:
        alpha = N * b / (1 + (N * b))

    idx = np.random.randint(len(covs))
    normalized = check_cov_normalized(covs[idx])

    if not normalized:
        covs = cov_trace_norm_sigma_N(covs, noise_var=alpha)

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

                if (
                    measure == _jsd_numpy
                    or measure == _tvd_numpy
                    or measure == _jsd_torch
                    or measure == _tvd_torch
                ):
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


def select_measure(cov_mtx, meas_name, module=None):

    if module == None:
        module = check_input_format(cov_mtx)

    meas_name = meas_name.strip().casefold()

    if module == np:

        if meas_name == "wasserstein":
            measure = _wasserstein_numpy

        elif meas_name == "hellinger":
            measure = _hellinger_numpy

        elif meas_name == "tvd":
            measure = functools.partial(_tvd_numpy, gen=gen)

        elif meas_name == "jsd":
            measure = functools.partial(_jsd_numpy, gen=gen)

        elif meas_name == "kl div" or meas_name == "kl divergence":
            measure = _KL_div_numpy

        elif meas_name == "bhattacharyya":
            measure = _bhattacharyya_numpy

        elif meas_name == "mahalanobis":
            measure = _mahalanobis_numpy

        else:
            raise NotImplementedError(
                "Given metric name is not valid for Numpy array covariances."
            )

    elif module == torch:

        if meas_name == "wasserstein":
            measure = _wasserstein_torch

        elif meas_name == "hellinger":
            measure = _hellinger_torch

        elif meas_name == "tvd":
            # if cov mtx is on GPU, provide the TVD with the CUDA seeded generator, or else with the CPU seeded generator
            if cov_mtx.is_cuda:
                measure = functools.partial(_tvd_torch, gen=gen_torch_cuda)
            else:
                measure = functools.partial(_tvd_torch, gen=gen_torch_cpu)

        elif meas_name == "jsd":
            # if cov mtx is on GPU, provide the JSD with the CUDA seeded generator, or else with the CPU seeded generator
            if cov_mtx.is_cuda:
                measure = functools.partial(_jsd_torch, gen=gen_torch_cuda)
            else:
                measure = functools.partial(_jsd_torch, gen=gen_torch_cpu)

        elif meas_name == "kl div" or meas_name == "kl divergence":
            measure = _KL_div_torch

        elif meas_name == "bhattacharyya":
            measure = _bhattacharyya_torch

        elif meas_name == "mahalanobis":
            measure = _mahalanobis_torch

        else:
            raise NotImplementedError(
                "Given metric name is not valid for Tensor tensor covariances."
            )

    else:
        raise NotImplementedError(
            "Covariance matrices must be either a torch tensor or a numpy array."
        )
    return measure


def prevent_negative_square(d_sq, dist_name):

    if d_sq < 0 and d_sq > -1e-7:
        d_sq = 0

    elif d_sq < -1e-7:
        d_sq = 0
        warnings.warn(f"{dist_name} distance cannot be negative. Value is: {d_sq}")

    return d_sq
