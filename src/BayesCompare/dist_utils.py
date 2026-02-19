import re
import torch
import numpy as np


def simplify_string(s: str) -> str:
    """
    - convert to lowercase
    - remove underscores, dashes, and spaces
    """
    return re.sub(r"[ _-]+", "", s.lower())


def check_small_negative(d):

    epsilon = 1e-7

    # Python / NumPy scalar
    if isinstance(d, (int, float, np.floating)):
        return 0.0 if (-epsilon < d < 0) else d

    # NumPy array
    if isinstance(d, np.ndarray):
        if -epsilon < d[0] < 0:
            d[0] = 0
        return d

    # Torch tensor
    if isinstance(d, torch.Tensor):

        if d.ndim == 0:
            if -epsilon < d < 0:
                d[()] = 0
        else:
            if -epsilon < d[0] < 0:
                d[0] = 0

        return d

    return d
