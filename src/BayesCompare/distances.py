import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm
import torch

## Metrics


def wasserstein(sigma1, sigma2, mu1=None, mu2=None):

    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        means_term = np.linalg.norm(mu1 - mu2, 2) ** 2
    else:
        means_term = 0

    sig1_sqrt = scipy.linalg.sqrtm(sigma1)
    sig1_sig2_sqrt = scipy.linalg.sqrtm(sig1_sqrt @ sigma2 @ sig1_sqrt)
    tr_term = sigma1 + sigma2 - 2 * (sig1_sig2_sqrt)
    d_sq = means_term + np.trace(tr_term)

    if d_sq < 0 and d_sq > -1e-7:
        d_sq = 0

    elif d_sq < -1e-7:
        raise ValueError(f"Wasserstein distance cannot be negative. Value is: {d_sq}")

    return d_sq**0.5


def wasserstein_torch_comp(sigma1, sigma2, mu1=None, mu2=None):

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

    if d_sq < 0 and d_sq > -1e-7:
        d_sq = 0

    elif d_sq < -1e-7:
        raise ValueError(f"Wasserstein distance cannot be negative. Value is: {d_sq}")

    return d_sq**0.5


def hellinger(sigma1, sigma2, mu1=None, mu2=None):

    d_B = bhattacharyya(sigma1, sigma2, mu1, mu2)

    d = np.sqrt(2 * (1 - np.exp(-d_B)))

    return d


def mahalanobis(
    sigma1, sigma2, mu1=None, mu2=None
):  # not the true mahalanobis definition

    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        delta_mu = np.subtract(mu1, mu2)
        d = np.inner(delta_mu, np.matmul((np.linalg.inv(sigma1 + sigma2)), delta_mu))
    else:
        d = np.array(0)

    return d


gen = np.random.Generator(np.random.SFC64(42))

if torch.cuda.is_available():
    gen_torch = torch.Generator(device="cuda").manual_seed(42)
else:
    gen_torch = torch.Generator(device="cpu").manual_seed(42)


def jsd(sigma1, sigma2, mu1=None, mu2=None, N=10000, gen=gen):

    if mu1 is None and mu2 is None:
        k = sigma1.shape[0]
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        # generate random samples from each distribution
        x10 = gen.standard_normal(size=(k, N))
        x1 = mm(1, A1, x10, lower=1)
        x20 = gen.standard_normal(size=(k, N))
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
        x1 = np.expand_dims(mu1, 1) + A1 @ gen.standard_normal(size=(k, N))
        x2 = np.expand_dims(mu2, 1) + A2 @ gen.standard_normal(size=(k, N))
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


def jsd_torch_comp(sigma1, sigma2, mu1=None, mu2=None, N=10000, gen=gen_torch):

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
        x10 = torch.randn((k, N), generator=gen)
        x1 = torch.Tensor(mm(1, A1, x10, lower=1))
        x20 = torch.randn((k, N), generator=gen)
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
        x1 = torch.Tensor.expand(mu1, 1) + A1 @ torch.randn((k, N), generator=gen)
        x2 = torch.Tensor.expand(mu2, 1) + A2 @ torch.randn((k, N), generator=gen)
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


def tvd(sigma1, sigma2, mu1=None, mu2=None, N=10000, gen=gen):

    if mu1 is not None and mu2 is not None:
        k = len(mu1)
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        Ainv1 = scipy.linalg.solve_triangular(A1, np.eye(k), lower=True)
        Ainv2 = scipy.linalg.solve_triangular(A2, np.eye(k), lower=True)
        # generate random samples from each distribution
        x1 = np.expand_dims(mu1, 1) + A1 @ gen.standard_normal(size=(k, N))
        x2 = np.expand_dims(mu2, 1) + A2 @ gen.standard_normal(size=(k, N))
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
        x10 = gen.standard_normal(size=(k, N))
        x1 = mm(1, A1, x10, lower=1)
        x20 = gen.standard_normal(size=(k, N))
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


def tvd_torch_comp(sigma1, sigma2, mu1=None, mu2=None, N=10000, gen=gen_torch):

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
        x1 = torch.Tensor.expand(mu1, 1) + A1 @ torch.randn((k, N), generator=gen)
        x2 = torch.Tensor.expand(mu2, 1) + A2 @ torch.randn((k, N), generator=gen)
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
        x10 = torch.randn((k, N), generator=gen)
        x1 = torch.Tensor(mm(1, A1, x10, lower=1))
        x20 = torch.randn((k, N), generator=gen)
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


def KL_div(sigma1, sigma2, mu1=None, mu2=None):

    if mu1 is None:
        mu1 = np.zeros(sigma1.shape[0])

    if mu2 is None:
        mu2 = np.zeros(sigma2.shape[0])

    delta_mu = np.subtract(mu2, mu1)

    inv_s2 = np.linalg.inv(sigma2)

    if (delta_mu < 1e-50).all():  # only this condition is tested
        mean_term = 0

    else:  # this condition was not tested
        mean_term = np.transpose(delta_mu) @ inv_s2 @ delta_mu

    tr_term = np.trace(inv_s2 @ sigma1)

    log_term = np.linalg.slogdet(sigma1)[1] - np.linalg.slogdet(sigma2)[1]

    d = (1 / 2) * (mean_term + tr_term - log_term - sigma1.shape[0])

    return d


def bhattacharyya(sigma1, sigma2, mu1=None, mu2=None):

    means_term = mahalanobis(sigma1, sigma2, mu1, mu2)
    log_term = np.linalg.slogdet(np.divide(sigma1 + sigma2, 2))[1] - 0.5 * (
        np.linalg.slogdet(sigma1)[1] + np.linalg.slogdet(sigma2)[1]
    )  # these may also require float64 casting

    d = 1 / 8 * means_term + 1 / 2 * log_term

    return d


## Distance function caller


## no check points, no parallelization, single measure only and torch compatable (after correcting select measure for torch comp measures too)
def measure_dist(covs, mean=None, meas_name="TVD", alpha=None, b=1 / 100):
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

    Returns
    -------
    dist : numpy.ndarray
        A symmetric 2-D array of shape (N, N) containing pairwise distances
        between trace-normalized covariance inputs. The diagonal elements are zero.
        Only the upper triangle (j > i) is computed explicitly and mirrored to the
        lower triangle.

    Examples
    --------
    >>> # Given a list of covariance matrices `cov_list`
    >>> dist_matrix = measure_dist(cov_list, meas_name="TVD")
    """

    N = len(covs)

    if alpha == None:
        alpha = N * b / (1 + (N * b))

    if isinstance(covs[0], np.ndarray):
        measure = select_measure(meas_name)

    elif isinstance(covs[0], torch.Tensor):
        measure = select_measure_torch(meas_name)

    dist = np.zeros((N, N))

    progress_bar = tqdm.tqdm(total=int((N * (N - 1)) / 2))

    for i, ci in enumerate(covs):

        sig1 = trace_norm(ci, eye_w=alpha)

        for j, cj in enumerate(covs):

            if j > i:

                sig2 = trace_norm(cj, eye_w=alpha)

                if measure == jsd or measure == tvd:
                    dist[i, j] = measure(
                        sig1, sig2, N=10000
                    )  # not using mean, for a generalized code mean should be provided
                else:
                    dist[i, j] = measure(
                        sig1, sig2
                    )  # not using mean, for a generalized code mean should be provided

                dist[j, i] = dist[i, j]

                progress_bar.update(1)

    return dist


## Helper functions


def trace_norm(sigma, eye_w=0.001):

    if eye_w == 0:
        A = sigma

    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (
            eye_w * np.eye(sigma.shape[0])
        )

    return A


def select_measure(meas_name):

    if meas_name == "wasserstein":
        measure = wasserstein

    elif meas_name == "hellinger":
        measure = hellinger

    elif meas_name == "TVD":
        measure = tvd

    elif meas_name == "JSD":
        measure = jsd

    elif meas_name == "KL_div":
        measure = KL_div

    elif meas_name == "bhattacharyya":
        measure = bhattacharyya

    return measure


def select_measure_torch(meas_name):

    if meas_name == "wasserstein":
        measure = wasserstein_torch_comp

    elif meas_name == "TVD":
        measure = tvd_torch_comp

    elif meas_name == "JSD":
        measure = jsd_torch_comp

    return measure
