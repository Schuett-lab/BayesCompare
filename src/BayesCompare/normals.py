import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm

gen = np.random.Generator(np.random.SFC64(42))


def jsd_normal_general(mu1, mu2, sigma1, sigma2, N=10000, gen=gen):
    """
    JSD for two normal distributions with
    means mu1 & mu2
    and variances sigma1 & sigma2

    does not check any conditions
    """
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
    p1 = - np.sum(delta11 ** 2, 0) / 2 - logdet1
    delta21 = Ainv1 @ (x2 - np.expand_dims(mu1, 1))
    p2 = - np.sum(delta21 ** 2, 0) / 2 - logdet1
    delta12 = Ainv2 @ (x1 - np.expand_dims(mu2, 1))
    q1 = - np.sum(delta12 ** 2, 0) / 2 - logdet2
    delta22 = Ainv2 @ (x2 - np.expand_dims(mu2, 1))
    q2 = - np.sum(delta22 ** 2, 0) / 2 - logdet2

    # log (P) - log (P + Q)
    term1 = p1 - np.logaddexp(p1, q1)
    term2 = q2 - np.logaddexp(p2, q2)

    jsd = 1 + (np.mean(term1) + np.mean(term2)) / 2 / np.log(2)
    return jsd


def jsd_normal_sig(sigma1, sigma2, N=10000, eye_w=0.0, gen=gen):
    """
    JSD for two normal distributions with
    zero mean
    and variances sigma1 & sigma2

    This function normalizes the sigmas to trace 1
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
    return jsd


def tvd_normal_general(mu1, mu2, sigma1, sigma2, N=10000, gen=gen):
    """
    total variation distance for two normal distributions with
    means mu1 & mu2
    and variances sigma1 & sigma2

    does not check any conditions
    """
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
    p1 = - np.sum(delta11 ** 2, 0) / 2 - logdet1
    delta21 = Ainv1 @ (x2 - np.expand_dims(mu1, 1))
    p2 = - np.sum(delta21 ** 2, 0) / 2 - logdet1
    delta12 = Ainv2 @ (x1 - np.expand_dims(mu2, 1))
    q1 = - np.sum(delta12 ** 2, 0) / 2 - logdet2
    delta22 = Ainv2 @ (x2 - np.expand_dims(mu2, 1))
    q2 = - np.sum(delta22 ** 2, 0) / 2 - logdet2
    f1 = np.maximum(1 - np.exp(q1-p1), 0)
    f2 = np.maximum(1 - np.exp(p2-q2), 0)
    tvd = (np.mean(f1) + np.mean(f2)) / 2
    return tvd


def tvd_normal_sig(sigma1, sigma2, N=10000, eye_w=0.001, gen=gen):
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
    return tvd


def det_norm(sigma, eye_w):
    A = np.linalg.cholesky(sigma)
    # normalize to determinant 1
    logdet = np.mean(np.log(np.diag(A)))
    # add multiple of identity to sigmas
    A = np.linalg.cholesky(
        (1 - eye_w) * sigma / np.exp(2 * logdet) + eye_w * np.eye(sigma.shape[0]))
    # normalize to determinant 1 again because det is not additive
    logdet = np.mean(np.log(np.diag(A)))
    A /= np.exp(logdet)
    return A


def trace_norm(sigma, eye_w=0.001):
    """normalization of covariance matrices
    & computation of cholesky decomposition

    returns a cholesky factor for sigma after normalization
    """
    A = np.linalg.cholesky(
        (1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)
        + eye_w * np.eye(sigma.shape[0]))
    return A
