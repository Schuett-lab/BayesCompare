"""
Module with helper functions to manipulate covariance matrices
Authors: Juan Jesús Torre Tresols, Sezan Oral, Heiko Schütt
"""

import torch
import numpy as np

from numpy.typing import NDArray


def trace_norm(cov: NDArray | torch.Tensor) -> NDArray | torch.Tensor:
    """Normalize the covariance to have trace equal to its shape"""

    cov_norm = cov * cov.shape[0] / np.trace(cov)

    return cov_norm


def cov_sigma(
    cov: NDArray | torch.Tensor, noise_var: float, signal_var: float | None = None
) -> NDArray | torch.Tensor:
    """
    Multiply the (normalized) covariance matrix by the estimated variance of
    the signal, and add the estimated variance of the noise, so the resulting
    matrix represents the estimated total variance of the predictive distribution

    Parameters
    ----------

    cov: NDArray or torch.Tensor, shape (n_stim, n_stim)
        Covariance matrix

    noise_var: float
        Estimate of the variance attributed to noise

    signal_var: float or None, default None
        Estimate of the variance attributed to the data. If not passed, it is
        assumed to be (1 - noise_var)

    Returns
    -------

    cov_sigma: NDArray or torch.Tensor
        Modified covariance matrix that represents the variance of the
        predictive distribution of y
    """
    if not signal_var:
        signal_var = 1 - noise_var

    cov_sigma = signal_var * cov + noise_var * np.eye(cov.shape[0])

    return cov_sigma


def check_cov_normalized(cov: NDArray | torch.Tensor, tolerance=1e-4) -> bool:
    """
    Check if the given covariance is trace normalized
    """

    trace_cov = cov.trace()

    return abs(trace_cov - len(cov)) < tolerance
