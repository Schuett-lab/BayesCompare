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
        activations = torch.reshape(activations, [activations.shape[0], -1])
        activations -= torch.mean(activations, 1, keepdim=True)
        return torch.matmul(activations, activations.T)
    else:
        activations = np.reshape(activations, [activations.shape[0], -1])
        activations -= np.mean(activations, 1, keepdims=True)
        return np.matmul(activations, activations.T)
