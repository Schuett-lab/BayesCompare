import torch
import numpy as np


def get_cov(activations):
    """computes the covariance matrix for a set of DNN activations

    This is the first step for calculating differences between
    predictive distributions, because random zero-mean weights will
    reproduce the covariance of the activations.

    This assumes the first dimension of the activations tensor is the stimulus
    dimension.
    """
    if torch.is_tensor(activations):
        x = activations.detach().clone()
        x = torch.reshape(x, [x.shape[0], -1])
        x -= torch.mean(x, 1, keepdim=True)
        return torch.matmul(x, x.T)
    else:
        x = np.reshape(activations, [activations.shape[0], -1])
        x -= np.mean(x, 1, keepdims=True)
        return np.matmul(x, x.T)
