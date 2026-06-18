"""
Module with helper functions to manipulate covariance matrices
Authors: Juan Jesús Torre Tresols, Sezan Oral, Heiko Schütt
"""

import torch
import numpy as np

from typing import Sequence, Optional
from types import ModuleType
from numpy.typing import NDArray


def trace_norm(cov: NDArray | torch.Tensor) -> NDArray | torch.Tensor:
    """Normalize the covariance to have trace equal to its shape"""
    module = check_input_format(cov)
    return cov * cov.shape[0] / module.trace(cov)


def cov_sigma(
    cov: NDArray | torch.Tensor,
    noise_var: float,
    signal_var: float | None = None,
    img_weights: Optional[NDArray] = None,
) -> NDArray | torch.Tensor:
    """
    Multiply the (normalized) covariance matrix by the estimated variance of
    the signal, and add the estimated variance of the noise, so the resulting
    matrix represents the estimated total variance of the predictive distribution.

    Parameters
    ----------
    cov : NDArray or torch.Tensor, shape (n_stim, n_stim)
        Covariance matrix
    noise_var : float
        Estimate of the variance attributed to noise
    signal_var : float or None, default None
        Estimate of the variance attributed to the data. If not passed, it is
        assumed to be (1 - noise_var)
    img_weights : NDArray or None, shape (n_stim,), default None
        Array with weights to reduce noise variance added to each individual image
        of the covariance matrix. This is useful when images have been presented
        more than once, which is the case in comparisons with neural data.


    Returns
    -------
    cov_sigma : NDArray or torch.Tensor, shape (n_stim, n_stim)
        Modified covariance matrix that represents the variance of the
        predictive distribution of y
    """
    if not signal_var:
        signal_var = 1 - noise_var

    module = check_input_format(cov)

    if img_weights is not None:
        cov_sigma = signal_var * cov + noise_var * np.diag(1 / img_weights)
    else:
        cov_sigma = signal_var * cov + noise_var * module.eye(cov.shape[0])

    return cov_sigma


def cov_sigma_N(
    covs: NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor],
    noise_var: float,
    signal_var: Optional[float] = None,
) -> NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor]:
    """
    Serialized version of cov_sigma for multiple covariance matrices.

    Parameters
    ----------
    covs : NDArray or torch.Tensor or a list/tuple of either of these types
        A NumPy array or PyTorch tensor with shape (N, dim, dim) or a list/tuple of cov. matrices with shape (dim, dim).
    noise_var : float
        Estimate of the variance attributed to noise
    signal_var : float or None, default None
        Estimate of the variance attributed to the data. If not passed, it is
        assumed to be (1 - noise_var)

    Returns
    -------
    cov_sigma_out : NDArray or torch.Tensor or a list/tuple of either of these types
        The modified covariance matrix or a list/tuple of modified covariance matrices
        with added noise and signal variance.
    """
    if not signal_var:
        signal_var = 1 - noise_var

    if isinstance(covs, (list, tuple)):
        cov_sigma_out = []
        for cov in covs:
            cov_sigma_out.append(
                cov_sigma(cov, noise_var=noise_var, signal_var=signal_var)
            )

    else:
        module = check_input_format(covs)

        cov_sigma_out = signal_var * covs + (
            noise_var * module.eye(covs.shape[-1])[None, ...]
        )

    return cov_sigma_out


def trace_norm_N(
    covs: NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor],
) -> NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor]:
    """
    Normalize N covariance matrices to have trace equal to its shape.
    Serialized version of trace_norm for multiple covariance matrices.

    Parameters
    ----------
    covs : NDArray or torch.Tensor or a list/tuple of either of these types
        A NumPy array or PyTorch tensor with shape (N, dim, dim) or a list/tuple of cov. matrices with shape (dim, dim).
        If a sequence (list/tuple), each element is independently normalized.

    Returns
    -------
    cov_norm : NDArray or torch.Tensor or a list/tuple of either of these types
        Trace-normalized covariance matrices with the same type and shape as input.
    """
    if isinstance(covs, (list, tuple)):
        cov_norm = []
        for cov in covs:
            cov_norm.append(trace_norm(cov))
    else:
        module = check_input_format(covs)

        if module == np:
            denominator = covs.diagonal(offset=0, axis1=-1, axis2=-2).sum(-1)[
                :, None, None
            ]
        elif module == torch:
            denominator = covs.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)[
                :, None, None
            ]

        cov_norm = covs * covs.shape[-1] / denominator

    return cov_norm


def cov_trace_norm_sigma_N(
    covs: NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor],
    noise_var: float,
    signal_var: Optional[float] = None,
) -> NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor]:
    """
    Combination of trace_norm_N and cov_sigma_N.

    Parameters
    ----------
    covs : NDArray or torch.Tensor or a list/tuple of either of these types
        A NumPy array or PyTorch tensor with shape (N, dim, dim) or a list/tuple of cov. matrices with shape (dim, dim).
        If a sequence (list/tuple), each element is independently normalized.
    noise_var : float
        Estimate of the variance attributed to noise
    signal_var : float or None, default None
        Estimate of the variance attributed to the data. If not passed, it is
        assumed to be (1 - noise_var)

    Returns
    -------
    normed_sigma_covs : NDArray or torch.Tensor or a list/tuple of either of these types
        Trace-normalized and noise/signal variance added covariance matrices with the same type and shape as input.
    """

    normed_sigma_covs = cov_sigma_N(
        trace_norm_N(covs), noise_var=noise_var, signal_var=signal_var
    )
    return normed_sigma_covs


def check_cov_normalized(cov: NDArray | torch.Tensor, tolerance: float = 1e-4) -> bool:
    """
    Check if the given covariance matrix is trace normalized within a tolerance point.
    """
    trace_cov = cov.trace()
    return abs(trace_cov - len(cov)) < tolerance


def check_cov_symmetry(
    cov_mtx: NDArray | torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    """
    Check if the given covariance matrix is trace symmetric up to a tolerance point.
    """
    module = check_input_format(cov_mtx)
    return module.allclose(cov_mtx, cov_mtx.T, rtol=rtol, atol=atol)


def check_input_format(input: NDArray | torch.Tensor) -> ModuleType:
    """
    Determines the appropriate numerical library module for the input data type.

    Parameters
    ----------
    input : torch.Tensor or np.ndarray
        The input data to check.

    Returns
    -------
    module : ModuleType
        torch if input is a torch.Tensor, numpy if input is a np.ndarray.

    Raises
    ------
    TypeError
        If the input type is not supported (not a torch.Tensor or np.ndarray).
    """
    if isinstance(input, torch.Tensor):
        return torch
    elif isinstance(input, np.ndarray):
        return np
    raise TypeError("Input must be a Numpy array or a PyTorch tensor.")


def check_and_change_input_format(
    input: NDArray | torch.Tensor | Sequence[NDArray | torch.Tensor],
) -> tuple[Sequence[float], int, ModuleType]:
    """
    Changes given input matrix or tuple/list of matrices into a list of matrices.

    Parameters
    ----------
    input : NDArray or torch.Tensor or a list/tuple of either of these types
        A NumPy array or PyTorch tensor with shape (N, dim, dim) or a list/tuple of N matrices with shape (dim, dim).

    Returns
    -------
    modified_input : iterable
        Iterable yielding a list of length N, consisting of matrices of shape (dim, dim).
    N : int
        Number of matrices.
    module : type
        torch if input is a torch.Tensor, np if input is a np.ndarray or the respective type if a list/tuple of either of these types.

    Raises
    ------
    ValueError
        If the input is empty or contains only one square matrix.
        If the input is contains non-square matrices.
    TypeError
        If the input type or dimensionality is unsupported.
    """
    if isinstance(input, (list, tuple)):
        if len(input) == 0:
            raise ValueError("Input list is empty.")
        elif len(input) == 1:
            raise ValueError("Covariance list has to have more than 1 matrix.")
        module = check_input_format(input[0])
        return list(input), len(input), module

    elif hasattr(input, "ndim"):
        if input.ndim == 2:
            raise ValueError("Covariance tensor has to have more than 1 matrix.")
        elif input.ndim == 3:
            N, d1, d2 = input.shape
            if N == 1:
                raise ValueError("Covariance tensor has to have more than 1 matrix.")
            if d1 != d2:
                raise ValueError(
                    f"Expected shape (N, dim, dim), but got {input.shape}: "
                    "last two dimensions must be equal."
                )
            module = check_input_format(input)
            return [input[i] for i in range(N)], N, module

    raise TypeError(
        "Input must be a tuple/list of (dim, dim) matrices "
        "or an array/tensor of shape (N, dim, dim) with N>1."
    )
