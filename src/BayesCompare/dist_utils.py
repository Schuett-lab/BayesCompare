"""
Module with helper functions to check certain properties of the distance outputs and covariance matrices
Authors: Sezan Oral
"""

import re
import torch
import numpy as np

from typing import Sequence
from numpy.typing import NDArray


def simplify_string(s: str) -> str:
    """
    Simplifies a string by:
    - convert to lowercase
    - remove underscores, dashes, and spaces
    """
    return re.sub(r"[ _-]+", "", s.lower())


def check_array(
    dist_output: NDArray | torch.Tensor,
) -> NDArray | torch.Tensor:
    """If the distance output is a 1 dimensional array with only one element,
    return the element itself. Otherwise, return the original output.

    Parameters
    ----------
    dist_output : NDArray or torch.Tensor
        The output distance, which can be a scalar or a 1D array with one element.

    Returns
    -------
    dist_output : NDArray or torch.Tensor
        The value inside the 1D array if the input is a 1D array with one element, otherwise the original input.

    Raises
    ------
    ValueError
        If the input is an array with more than one element or has more than one dimensions.
    """
    if isinstance(dist_output, np.ndarray):
        if dist_output.shape[0] == 1 and len(dist_output.shape) == 1:
            return dist_output.item()
        else:
            raise ValueError(
                "Distance output is a numpy array with more than one element."
            )

    elif isinstance(dist_output, torch.Tensor):
        if dist_output.ndim == 0:
            return dist_output
        elif dist_output.shape[0] == 1 and len(dist_output.shape) == 1:
            return dist_output.item()
        else:
            raise ValueError(
                "Distance output is a torch tensor with more than one element."
            )
    else:
        return dist_output


def check_small_negative(
    d: int | float | NDArray | torch.Tensor, epsilon: float = 1e-7
) -> float | NDArray | torch.Tensor:
    """
    Sets very small and negative distances (tolerance set with epsilon), due to numeric errors, to zero.

    Parameters
    ----------
    d : integer or float or NDArray or torch.Tensor
        The output distance, which can be a scalar or a 1D array with one element.
    epsilon : float, default 1e-7
        Tolerance for how small a negative distance can be to be set to zero.
        If the distance is smaller than epsilon, it is set to zero. Otherwise, it is returned as is.

    Returns
    -------
    d: float or NDArray or torch.Tensor
        The input distance, but if it is a small negative value (smaller than epsilon), it is set to zero.
    """
    # Python / NumPy scalar
    if isinstance(d, (int, float, np.floating)):
        return 0.0 if (-epsilon < d < 0) else d

    # NumPy array
    if isinstance(d, np.ndarray):
        if -epsilon < d[0] < 0:
            d[0] = 0.0
        return d

    # Torch tensor
    if isinstance(d, torch.Tensor):
        if d.ndim == 0:
            if -epsilon < d < 0:
                d[()] = 0.0
        else:
            if -epsilon < d[0] < 0:
                d[0] = 0.0

        return d

    return d


def check_small_negative_eigenval(
    eigenvals: NDArray | torch.Tensor | Sequence[float],
    tolerance: float = 1e-3,
) -> NDArray | torch.Tensor | Sequence[float]:
    """
    Sweeps all eigenvalues and sets very small negative eigenvalues to 0 (since they are due to numeric errors).

    Parameters
    ----------
    eigenvals : NDArray or torch.Tensor or a list/tuple of floats
        The eigenvalue vector to check.
    tolerance : float, default 1e-3
        Tolerance for how small a negative eigenvalue can be to be set to zero.
        If the eigenvalue is smaller than epsilon, and negative, it is set to zero.

    Returns
    -------
    eigenvals : NDArray or torch.Tensor or a list/tuple of floats
        Manipulated eigenvalue vector, where small negative eigenvalues (smaller than the negative of the tolerance) are set to zero.

    Raises
    ------
    ValueError
        If the eigenvalue vector has a large negative eigenvalue (smaller than the negative of the tolerance).
    """
    for idx, eigenval in enumerate(eigenvals):
        if eigenval < 0 and abs(eigenval) <= tolerance:
            eigenvals[idx] = 0
        elif eigenval < 0 and abs(eigenval) > tolerance:
            raise ValueError(f"Large negative eigenvalue ({eigenval}) is found!")

    return eigenvals


def check_cos_output(
    cos_val: int | float | NDArray | torch.Tensor, epsilon: float = 1e-7
) -> float | NDArray | torch.Tensor:
    """
    Checks if a cosine output is slightly greater one and if so sets it to 1.

    Parameters
    ----------
    cos_val : integer or float or NDArray or torch.Tensor
        The output cosine value.
    epsilon : float, default 1e-7
        Tolerance for how much greater a value than 1 can be set back to 1.

    Returns
    -------
    cos_val : float or NDArray or torch.Tensor
        The input cosine value, but if it is slightly greater than 1 (greater than 1 but smaller than 1 + epsilon), it is set to 1.
    """
    # Python / NumPy scalar
    if isinstance(cos_val, (int, float, np.floating)):
        if abs(cos_val) > 1:
            return np.sign(cos_val) * 1.0 if (abs(cos_val) - 1 <= epsilon) else cos_val

    # NumPy array
    if isinstance(cos_val, np.ndarray):
        if abs(cos_val) > 1:
            if abs(cos_val[0]) - 1 <= epsilon:
                cos_val[0] = np.sign(cos_val[0]) * 1.0
        return cos_val

    # Torch tensor
    if isinstance(cos_val, torch.Tensor):
        if cos_val.ndim == 0:
            if abs(cos_val) > 1 and abs(cos_val) - 1 <= epsilon:
                cos_val[()] = torch.sign(cos_val[()]) * 1.0
        else:
            if abs(cos_val[0]) > 1 and abs(cos_val[0]) - 1 <= epsilon:
                cos_val[0] = torch.sign(cos_val[0]) * 1.0
        return cos_val

    return cos_val
