from functools import total_ordering
import itertools
import numpy as np

from joblib import Parallel, delayed
from scipy.special import logsumexp
from typing import Tuple, Optional
from scipy.stats import epps_singleton_2samp
from tqdm import tqdm

from numpy.typing import NDArray

from BayesCompare.cov_utils import cov_sigma


def evidence(
    cov: NDArray,
    y: NDArray,
    mu: Optional[NDArray | float] = None,
    cov_inv: Optional[NDArray] = None,
    slogdet: Optional[np.linalg._linalg.SlogdetResult] = None,
) -> float | NDArray:
    """
    Get the log-likelihood that a given model produces the activations observed
    from a certain observed measure (voxel or model)

    Parameters
    ----------

    cov: NDArray, shape (n_stim, n_stim)
        Normalized covariance matrix coming from the model X (i.e. X^TX).
        It describes the covariance of the different experimental conditions
        with respect to the different measurement channels

    y: NDArray, shape (n_stim,) or (n_voxels, n_stim)
        Activation profile of a single measurement channel for each experimental
        condition. In the case of fMRI, it represents the activations of a
        single voxel across the space of experimental conditions or stimuli

    mu: float, NDArray or None, default None
        Mean activation from the corresponding measurement channel. It must match
        the shape of y: float for a single voxel, otherwise array of shape (n_voxels,)

    cov_inv: NDArray or None, default None
        The inner inverse of the covariance matrix. Certain variations of the
        analysis make it more efficient to compute the inverse beforehand. If
        provided, it is used instead of cov

    slogdet: np.linalg._linalg.SlogdetResult or None, default None
        Signed log-determinant of the inner inverse matrix. Pre-computed using
        np.linalg.slogdet. See the numpy documentation for more details

    Returns
    -------

    loglik: float or NDArray, shape (n_voxels,)
        Log likelihood that y is produced by the model corresponding to cov

    """
    N = y.shape[-1]
    inner_inv = np.linalg.inv(cov[:N, :N]) if cov_inv is None else cov_inv
    logdet = np.linalg.slogdet(inner_inv) if slogdet is None else slogdet

    Y = np.atleast_2d(y)  # (n_voxels, n_stim) or (1, n_stim)
    if mu is not None:
        mu = np.asarray(mu)

        # Handle mu shape for correct subtraction
        if y.ndim == 1:  # Single voxel case
            if mu.ndim != 0:
                raise ValueError("Single voxel evidence: mu must be a scalar")
        else:  # Multiple voxel case
            if mu.ndim != 1:
                raise ValueError(
                    f"Multi-voxel evidence: mu must be a vector of length {y.shape[0]}"
                )
            mu = mu[:, None]

        Y = Y - mu

    ss = np.einsum("vi,vi->v", Y @ inner_inv, Y)  # y.T @ inner_inv @ y per voxel

    loglik = logdet.logabsdet / 2 - ss / 2 - N / 2 * np.log(2 * np.pi)

    return loglik


def loglik_score(
    norm_cov: NDArray,
    activations: NDArray,
    noise_var: NDArray,
    signal_var: Optional[NDArray] = None,
    img_weights: Optional[NDArray] = None,
    noise_per_voxel: bool = True,
    n_jobs: int = -1,
) -> NDArray:
    """
    Estimate the log-likelihood of the mean activation of a list of measurement
    channels across a number of potential models.

    Parameters
    ----------

    norm_cov: list[np.array], shape (n_stim, n_stim)
        List containing the covariance matrices corresponding to the different
        models. The covariance matrices must be normalized so they trace is
        equal to N, with N being the number of stimuli. Each covariance matrix
        has shape (N, N)

    activations: np.array, shape (n_channels, n_stim)
        Mean activation for a list of measurement channels in response to a list
        of stimuli

    noise_var: np.array, shape (n_channels,)
        Estimated variance attributed to noise for each measurement channel
        across all stimuli

    signal_var: np.array, shape (n_channels,) or None, default None
        Estimated variance attributed to signal for each measurement channel
        across all stimuli. If not given, it is infered from noise_var

    img_weights: np.array, shape (n_stim) or None, default None
        Array containing how many presentations of each image are in the neural
        data. This is used to weight the noise variance for each individual entry
        of the covariance matrix

    noise_per_voxel: bool, default True
        Controls the noise and signal variances are treated. If True, the function
        expects noise (and optionally signal) variances to match the length of
        activations, computing the log-likelihood of each voxel with its corresponding
        noise and signal variances. If False, all voxels will be computed with each
        value of the lists, resulting in len(noise_var) likelihoods per voxel. This
        latter method is, in general, much faster to compute.

    n_jobs: int, default = -1
        Parameter passed to joblib for parallelization

    Returns
    -------

    loglik_score: np.array, shape (n_channels,) or (n_channels, n_noise)
        Log-likelihood score for each measurement channel, and optionally for each
        signal/noise variance.

    """
    n_voxels = activations.shape[0]
    n_noise = noise_var.shape[0]

    # Input validation
    if signal_var is not None:
        if signal_var.shape[0] != n_noise:
            raise ValueError(
                f"Signal var has len {signal_var.shape[0]} but noise var has "
                f"shape {n_noise}"
            )
    else:
        signal_var = np.full(shape=n_noise, fill_value=None)

    if noise_per_voxel and n_noise != n_voxels:
        raise ValueError(
            "Parameter `noise_per_voxel` requires the shapes of `noise_var` and "
            f"`activations` to match. Got {n_noise} and {n_voxels}"
        )

    def noise_loop(activations, norm_cov, sig_var, eps_var, img_weights=None):
        """Run evidence for all voxels with the same signal/noise variance"""
        N = activations.shape[-1]
        s_cov = cov_sigma(
            norm_cov, signal_var=sig_var, noise_var=eps_var, img_weights=img_weights
        )
        cov_inv = np.linalg.inv(s_cov[:N, :N])
        slogdet = np.linalg.slogdet(cov_inv)
        ev = evidence(None, activations, cov_inv=cov_inv, slogdet=slogdet)

        return ev

    def voxel_loop(y, norm_cov, sig_var, eps_var, img_weights=None):
        """Run evidence on one voxel and its matching signal/noise variance"""
        s_cov = cov_sigma(
            norm_cov,
            signal_var=sig_var,
            noise_var=eps_var,
            img_weights=img_weights,
        )
        ev = evidence(s_cov, y)

        return ev.flatten()

    if not noise_per_voxel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(noise_loop)(activations, norm_cov, sig_var, eps_var, img_weights)
            for sig_var, eps_var in tqdm(
                zip(signal_var, noise_var), desc="Noise values"
            )
        )
        loglik_score = np.stack(results, axis=1)  # shape (n_voxels, n_noise)
    else:

        results = Parallel(n_jobs=n_jobs)(
            delayed(voxel_loop)(y, norm_cov, sig_var, eps_var, img_weights)
            for y, sig_var, eps_var in tqdm(
                zip(activations, signal_var, noise_var), desc="Voxels"
            )
        )
        loglik_score = np.concatenate(results)  # shape (n_voxels,)

    return loglik_score


def log_posterior(
    loglik_array: NDArray, target_dims: Optional[Tuple] | Optional[int] = None
) -> NDArray:
    """
    Obtain the posterior probabilities that a given model produces the activations
    observed from a certain measure (voxel or model)

    Parameters
    ----------

    loglik_array: NDArray, shape (n_channels, ...)
        The first dimension represents the measurement channels, with the remaining
        dimensions representing the candidate models and optionally the different
        noise values used (in the 3D case)

    target_dims: Tuple, int or None, default None
        Dimensions over which to compute the posterior. If None, it defaults to
        all dimensions except the first (n_channels). The posterior is intended
        to be computed over models and noise values. Use this parameter if you
        have extra dimensions representing other variables (e.g. subjects)

    Returns
    -------

    post_array: NDArray
        Each element represents the posterior probability of each of the candidate
        models producing the observed activation in each channel. Preserves input
        shape
    """
    if target_dims is None:
        target_dims = tuple(range(1, loglik_array.ndim))
    post_array = loglik_array - logsumexp(loglik_array, axis=target_dims, keepdims=True)

    return post_array
