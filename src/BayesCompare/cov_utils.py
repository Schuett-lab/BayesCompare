"""
Module with helper functions to manipulate covariance matrices
Authors: Juan Jesús Torre Tresols, Sezan Oral, Heiko Schütt
"""

import torch
import numpy as np

from numpy.typing import NDArray


def trace_norm(cov: NDArray | torch.Tensor) -> NDArray | torch.Tensor:
    """Normalize the covariance to have trace equal to its shape"""

    module = check_input_format(cov)

    cov_norm = cov * cov.shape[0] / module.trace(cov)

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

    module = check_input_format(cov)

    cov_sigma = signal_var * cov + noise_var * module.eye(cov.shape[0])

    return cov_sigma


def cov_sigma_N(
    covs: torch.Tensor, noise_var: float, signal_var: float | None = None
) -> torch.Tensor:

    if not signal_var:
        signal_var = 1 - noise_var

    module = check_input_format(covs)

    cov_sigma = signal_var * covs + (noise_var * module.eye(covs.shape[-1])[None, ...])

    return cov_sigma


def trace_norm_N(covs: torch.Tensor) -> torch.Tensor:

    if check_input_format(covs) == np:
        denominator = covs.diagonal(offset=0, axis1=-1, axis2=-2).sum(-1)[:, None, None]
    elif check_input_format(covs) == torch:
        denominator = covs.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)[:, None, None]

    cov_norm = covs * covs.shape[-1] / denominator

    return cov_norm


def cov_trace_norm_sigma_N(
    covs: torch.Tensor, noise_var: float, signal_var: float | None = None
) -> torch.Tensor:

    normed_covs = cov_sigma_N(
        trace_norm_N(covs), noise_var=noise_var, signal_var=signal_var
    )

    return normed_covs


def check_cov_normalized(cov: NDArray | torch.Tensor, tolerance=1e-4) -> bool:
    """
    Check if the given covariance is trace normalized
    """

    trace_cov = cov.trace()

    return abs(trace_cov - len(cov)) < tolerance


def check_cov_symmetry(cov_mtx, rtol=1e-05, atol=1e-08):

    module = check_input_format(cov_mtx)
    return module.allclose(cov_mtx, cov_mtx.T, rtol=rtol, atol=atol)


def check_input_format(input):
    """
    Determines the appropriate numerical library module for the input data type.

    Parameters
    ----------
    input : torch.Tensor or np.ndarray
        The input data to check.

    Returns
    -------
    module : type
        torch if input is a torch.Tensor, np if input is a np.ndarray.
    """
    if isinstance(input, torch.Tensor):
        return torch
    elif isinstance(input, np.ndarray):
        return np
    raise TypeError("Input must be a Numpy array of a PyTorch tensor.")


def check_and_change_input_format(input):
    """
    Normalize matrix input into an iterable of square matrices.

    Accepts either a list/tuple of square matrices with shape ``(dim, dim)``,
    or a NumPy array / PyTorch tensor with shape ``(N, dim, dim)``. A single
    square matrix with shape ``(dim, dim)`` is treated as ``N = 1``.

    Parameters
    ----------
    input : list of ndarray or list of torch.Tensor or ndarray or torch.Tensor
        Input matrices in one of the supported formats.

    Returns
    -------
    modified_input : iterable
        Iterable yielding ``N`` matrices of shape ``(dim, dim)``.

    N : int
        Number of matrices.

    module:type
        torch if input is a torch.Tensor, np if input is a np.ndarray or the respective type if a list of either of these types.

    Raises
    ------
    ValueError
        If the input is empty or contains non-square matrices.

    TypeError
        If the input type or dimensionality is unsupported.
    """

    if isinstance(input, (list, tuple)):
        if len(input) == 0:
            raise ValueError("Input list is empty.")
        module = check_input_format(input[0])
        return input, len(input), module

    if hasattr(input, "ndim"):
        if input.ndim == 2:
            d1, d2 = input.shape
            if d1 != d2:
                raise ValueError(
                    f"Expected square matrix (dim, dim), got {input.shape}"
                )
            module = check_input_format(input)
            return (input,), 1, module

        if input.ndim == 3:
            N, d1, d2 = input.shape
            if d1 != d2:
                raise ValueError(
                    f"Expected shape (N, dim, dim), but got {input.shape}: "
                    "last two dimensions must be equal."
                )
            module = check_input_format(input)
            return (input[i] for i in range(N)), N, module

    raise TypeError(
        "Input must be a list of (dim, dim) matrices "
        "or an array/tensor of shape (N, dim, dim)."
    )
