"""
Module with helper functions to manipulate covariance matrices
Authors: Juan Jesús Torre Tresols, Sezan Oral, Heiko Schütt
"""

import torch
import numpy as np

from typing import Sequence, Union, Optional
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
    covs: Union[Sequence[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
    noise_var: float,
    signal_var: Optional[float] = None,
) -> Union[Sequence[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor]:

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
    covs: Union[Sequence[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
) -> Union[Sequence[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor]:

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
    covs: Union[Sequence[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
    noise_var: float,
    signal_var: Optional[float] = None,
) -> torch.Tensor:

    normed_sigma_covs = cov_sigma_N(
        trace_norm_N(covs), noise_var=noise_var, signal_var=signal_var
    )

    return normed_sigma_covs


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
    raise TypeError("Input must be a Numpy array or a PyTorch tensor.")


def check_and_change_input_format(
    input: Union[Sequence[Union[np.ndarray, torch.Tensor]], np.ndarray, torch.Tensor],
):
    """
    Changes given input matrix or tuple/list of matrices into a list of matrices.

    Accepts either a list/tuple of square matrices with shape ``(dim, dim)``,
    or a NumPy array / PyTorch tensor with shape ``(N, dim, dim)``.

    Parameters
    ----------
    input : list/tuple of np.ndarray or list of torch.Tensor or np.ndarray or torch.Tensor
        Input matrices in one of the supported formats.

    Returns
    -------
    modified_input : iterable
        Iterable yielding a list of length ``N``, consisting of matrices of shape ``(dim, dim)``.

    N : int
        Number of matrices.

    module:type
        torch if input is a torch.Tensor, np if input is a np.ndarray or the respective type if a list of either of these types.

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
