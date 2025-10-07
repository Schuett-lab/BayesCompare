import numpy as np
from numpy.typing import NDArray
from typing import Optional
from scipy.linalg import pinv


def inference(cov, y_train, alpha: float = 0.1):
    """calculates bayesian inference for the 0 mean Gaussian prediction
    with covariance cov and training data y_train.

    input:
        cov (2d array): covariance across *all* relevant stimuli
            the first len(y_train) are assumed to correspond to
            the observations y_train. The rest we make predictions for.
            This should be the covariances after scaling,
            but before adding noise
        y_train(1d array): data to fit to.
        alpha (float): additional noise variance to add , default = 0
            if the covariances are not invertible, this is necessary!

    output:
        y_mu (vector): The mean prediction for the test stimuli, i.e. all
            stimuli that are not training stimuli
        y_sigma (2d array): covariance between test stimulus predictions
            This does not include any noise covariance.
    """
    N = len(y_train)
    N_pred = cov.shape[0] - N
    # inner_inv = np.linalg.inv(sigma_e * np.eye(N) + cov[:N, :N])
    inner_inv = np.linalg.inv(alpha / (1 - alpha) * np.eye(N) + cov[:N, :N])
    # this is wrong for sigma_e = 0
    # y_mu = np.matmul(
    #    cov[N:(N+N_pred), :N] - np.matmul(
    #        cov[N:(N+N_pred), :N],
    #        np.matmul(inner_inv, cov[:N, :N])),
    #    y_train) / sigma_e
    y_mu = (
        (
            cov[N : (N + N_pred), :N]
            - cov[N : (N + N_pred), :N] @ inner_inv @ cov[:N, :N]
        )
        * (1 - alpha)
        / alpha
        @ y_train
    )
    # covariance is ok, checked!
    # this is the covariance without noise
    y_sigma_post = cov[N : (N + N_pred), N : (N + N_pred)] - np.matmul(
        np.matmul(cov[N : (N + N_pred), :N], inner_inv), cov[:N, N : (N + N_pred)]
    )
    return y_mu, y_sigma_post


def inference_cov(cov, y_train, alpha: float = 0):
    """calculates bayesian inference for the 0 mean Gaussian prediction
    with covariance cov and training data y_train.

    input:
        cov (2d array): covariance across *all* relevant stimuli
            the first len(y_train) are assumed to correspond to
            the observations y_train. The rest we make predictions for.
            This should be the covariances after scaling,
            but before adding noise
        y_train(1d array): data to fit to.
        alpha (float): additional noise variance to add , default = 0
            The used covariance is (1-alpha) * cov + alpha * I * trace(cov)
            if the covariances are not invertible, this is necessary!

    output:
        y_mu (vector): The mean prediction for the test stimuli, i.e. all
            stimuli that are not training stimuli
        y_sigma (2d array): covariance between test stimulus predictions.
    """
    N = len(y_train)
    N_pred = cov.shape[0] - N
    k11 = cov[:N, :N]
    k21 = cov[N : (N + N_pred), :N]
    k22 = cov[N : (N + N_pred), N : (N + N_pred)]
    if alpha < 1 and alpha > 0:
        inner_inv = np.linalg.inv(alpha / (1 - alpha) * np.eye(N) + k11)
        y_mu = (k21 - k21 @ inner_inv @ k11) * (1 - alpha) / alpha @ y_train
        y_sigma_post = k22 - k21 @ inner_inv @ k21.T
    elif alpha == 0:  # no noise
        # murphy kernel trick (17.108)
        inner_inv = pinv(k11)
        y_mu = k21 @ inner_inv @ y_train
        y_sigma_post = k22 - k21 @ inner_inv @ k21.T
    elif alpha == 1:  # only prior
        y_mu = np.zeros(N_pred)
        y_sigma_post = k22
    else:
        raise ValueError("alpha must be between 0 and 1")
    return y_mu, y_sigma_post


def evidence(
    cov: NDArray, y: NDArray, sigma_e: float = 0.001, mu: Optional[NDArray] = None
) -> float:
    """
    Get the log-likelihood that a given model produces the activations observed
    from a certain observed measure (voxel or model)

    Parameters
    ----------

    cov: np.array, shape (n_stim, n_stim)
        Normalized covariance matrix coming from the model X (i.e. X^TX).
        It describes the covariance of the different experimental conditions
        with respect to the different measurement channels

    y: np.array, shape (n_stim,)
        Activation profile of a single measurement channel for each experimental
        condition. In the case of fMRI, it represents the activations of a
        single voxel across the space of experimental conditions or stimuli

    sigma_b: float
        Estimate of the variance attributed to the data

    sigma_e: float
        Estimate of the variance attributed to noise

    Returns
    -------

    loglik: float
        Log likelihood that y is produced by the model corresponding to cov

    """
    N = len(y)
    # precision matrix
    inner_inv = np.linalg.inv(sigma_e * np.eye(N) + cov[:N, :N])
    if mu is None:
        ss = np.expand_dims(y, 0) @ inner_inv @ np.expand_dims(y, 1)
    else:
        ss = np.expand_dims(y - mu, 0) @ inner_inv @ np.expand_dims(y - mu, 1)
    logdet = np.linalg.slogdet(inner_inv)
    loglik = logdet.logabsdet / 2 - ss / 2 - N / 2 * np.log(2 * np.pi)
    return loglik
