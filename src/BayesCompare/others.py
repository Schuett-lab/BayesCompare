"""other metrics"""
import numpy as np


def cka(K1, K2):
    """centred kernel alignment"""
    # centering
    K1 = K1 - np.mean(K1, 0, keepdims=True)
    K2 = K2 - np.mean(K2, 0, keepdims=True)
    return np.sum(K1*K2) / np.sqrt(np.sum(K1*K1)) / np.sqrt(np.sum(K2*K2))


def rsa_corr(K1, K2):
    """euclidean distances, correlation similarity"""
    # conversion to distances
    diag1 = np.diag(K1)
    d1 = np.expand_dims(diag1, 0) + np.expand_dims(diag1, 1) - 2 * K1
    diag2 = np.diag(K2)
    d2 = np.expand_dims(diag2, 0) + np.expand_dims(diag2, 1) - 2 * K2
    idx = np.triu_indices(d1.shape[0], 1)
    d1[idx] -= np.mean(d1[idx])
    d2[idx] -= np.mean(d2[idx])
    return (np.sum(d1[idx] * d2[idx])
            / np.sqrt(np.sum(d1[idx]*d1[idx]))
            / np.sqrt(np.sum(d2[idx] * d2[idx])))


def rsa_cos(K1, K2):
    """euclidean distances, cosine similarity"""
    # conversion to distances
    diag1 = np.diag(K1)
    d1 = np.expand_dims(diag1, 0) + np.expand_dims(diag1, 1) - 2 * K1
    diag2 = np.diag(K2)
    d2 = np.expand_dims(diag2, 0) + np.expand_dims(diag2, 1) - 2 * K2
    idx = np.triu_indices(d1.shape[0], 1)
    return (np.sum(d1[idx] * d2[idx])
            / np.sqrt(np.sum(d1[idx] * d1[idx]))
            / np.sqrt(np.sum(d2[idx] * d2[idx])))


def rsa_acos(K1, K2):
    """euclidean distances, arc-cosine similarity"""
    # conversion to distances
    diag1 = np.diag(K1)
    d1 = np.expand_dims(diag1, 0) + np.expand_dims(diag1, 1) - 2 * K1
    diag2 = np.diag(K2)
    d2 = np.expand_dims(diag2, 0) + np.expand_dims(diag2, 1) - 2 * K2
    idx = np.triu_indices(d1.shape[0], 1)
    return (np.arccos(np.sum(d1[idx] * d2[idx])
            / np.sqrt(np.sum(d1[idx] * d1[idx]))
            / np.sqrt(np.sum(d2[idx] * d2[idx]))))
