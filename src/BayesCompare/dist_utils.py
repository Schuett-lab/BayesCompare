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


def check_slight_greater_than_one(cos_val):
    epsilon = 1e-7

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
