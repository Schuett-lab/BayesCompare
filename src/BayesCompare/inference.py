import numpy as np
from numpy.typing import NDArray
from typing import Optional


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
    inner_inv = np.linalg.inv(alpha / (1-alpha) * np.eye(N) + cov[:N, :N])
    # this is wrong for sigma_e = 0
    # y_mu = np.matmul(
    #    cov[N:(N+N_pred), :N] - np.matmul(
    #        cov[N:(N+N_pred), :N],
    #        np.matmul(inner_inv, cov[:N, :N])),
    #    y_train) / sigma_e
    y_mu = (
        cov[N:(N+N_pred), :N]
        - cov[N:(N+N_pred), :N] @ inner_inv @ cov[:N, :N]
        ) * (1 - alpha) / alpha  @ y_train
    # covariance is ok, checked!
    # this is the covariance without noise
    y_sigma_post = cov[N:(N+N_pred), N:(N+N_pred)] - np.matmul(np.matmul(
        cov[N:(N+N_pred), :N], inner_inv),
        cov[:N, N:(N+N_pred)])
    return y_mu, y_sigma_post


def evidence(cov, y_train, sigma_e: float = 0.001, mu: Optional[NDArray] =  None):
    N = len(y_train)
    # precision matrix
    inner_inv = np.linalg.inv(sigma_e * np.eye(N) + cov[:N, :N])
    if mu is None:
        ss = np.expand_dims(y_train, 0) @ inner_inv @ np.expand_dims(y_train, 1)
    else:
        ss = np.expand_dims(y_train-mu, 0) @ inner_inv @ np.expand_dims(y_train-mu, 1)
    logdet = np.linalg.slogdet(inner_inv)
    loglik = - logdet.logabsdet / 2 - ss / 2 - N / 2 * np.log(2 * np.pi)
    return loglik
