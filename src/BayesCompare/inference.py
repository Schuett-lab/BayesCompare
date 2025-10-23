import numpy as np

from joblib import Parallel, delayed
from scipy.linalg import pinv
from scipy.special import logsumexp
from tqdm import tqdm

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


def sigma_cov(cov: NDArray, sigma_b: float, sigma_e: float) -> NDArray:
    """
    Multiply the (normalized) covariance matrix by the estimated variance of
    the signal, and add the estimated variance of the noise, so the resulting
    matrix represents the estimated total variance of the predictive distribution

    Parameters
    ----------

    cov: np.array, shape (n_stim, n_stim)
        Covariance matrix

    sigma_b: float
        Estimate of the variance attributed to the data

    sigma_e: float
        Estimate of the variance attributed to noise

    Returns
    -------

    sigma_cov: NDArray
        Modified covariance matrix that represents the variance of the
        predictive distribution of y
    """
    N = cov.shape[0]

    sigma_cov = sigma_b * cov + sigma_e * np.eye(N)

    return sigma_cov


def evidence(cov: NDArray, y: NDArray, mu: Optional[NDArray] = None) -> float:
    """
    Get the log-likelihood that a given model produces the activations observed
    from a certain observed measure (voxel or model)

    Parameters
    ----------

    cov: NDArray, shape (n_stim, n_stim)
        Normalized covariance matrix coming from the model X (i.e. X^TX).
        It describes the covariance of the different experimental conditions
        with respect to the different measurement channels

    y: NDArray, shape (n_stim,)
        Activation profile of a single measurement channel for each experimental
        condition. In the case of fMRI, it represents the activations of a
        single voxel across the space of experimental conditions or stimuli

    mu: NDArray or None, default None
        Mean activation from the corresponding measurement channel

    Returns
    -------

    loglik: float
        Log likelihood that y is produced by the model corresponding to cov

    """
    N = len(y)
    inner_inv = np.linalg.inv(cov[:N, :N])  # Precision matrix
    if mu is None:
        ss = np.expand_dims(y, 0) @ inner_inv @ np.expand_dims(y, 1)
    else:
        ss = np.expand_dims(y - mu, 0) @ inner_inv @ np.expand_dims(y - mu, 1)
    logdet = np.linalg.slogdet(inner_inv)
    loglik = logdet.logabsdet / 2 - ss / 2 - N / 2 * np.log(2 * np.pi)
    return loglik


def loglik_score(
    norm_covs: list[NDArray],
    activations: NDArray,
    total_var: NDArray,
    eps_var: NDArray,
    n_jobs: int = -1,
) -> NDArray:
    """
    Estimate the log-likelihood of the mean activation of a list of measurement
    channels across a number of potential models.

    Parameters
    ----------

    norm_covs: list[np.array], len (n_models,)[shape (n_stim, n_stim)]
        List containing the covariance matrices corresponding to the different
        models. The covariance matrices must be normalized so they trace is
        equal to N, with N being the number of stimuli. Each covariance matrix
        has shape (N, N)

    activations: np.array, shape (n_channels, n_stim)
        Mean activation for a list of measurement channels in response to a list
        of stimuli

    total_var: np.array, shape (n_channels,)
        Total variance of each measurement channel across all stimuli

    eps_var: np.array, shape (n_channels,)
        Estimated variance attributed to noise for each measurement channel
        across all stimuli

    n_jobs: int, default = -1
        Parameter passed to joblib for parallelization

    Returns
    -------

    loglik_score: np.array, shape (n_channels, n_models)
        Log-likelihood score for each measurement channel and candidate model

    """
    # Handle the single voxel case for iteration to work:
    if activations.ndim == 1:
        activations = np.expand_dims(activations, axis=0)
        total_var = np.expand_dims(total_var, axis=0)
        eps_var = np.expand_dims(eps_var, axis=0)

    # Same for single model
    if not isinstance(norm_covs, list):
        norm_covs = [norm_covs]

    def voxel_loop(norm_covs, y, sigma_tot, sigma_e):
        """Run evidence on one voxel and all models"""
        sigma_b = sigma_tot - sigma_e

        model_score = []  # One voxel, all layers
        for norm_cov in norm_covs:
            s_cov = sigma_cov(norm_cov, sigma_b=sigma_b, sigma_e=sigma_e)
            ev = evidence(s_cov, y)
            model_score.append(ev.flatten())

        return np.concatenate(model_score)

    model_scores: list[NDArray] = Parallel(n_jobs=n_jobs)(
        delayed(voxel_loop)(norm_covs, y, sigma_tot, sigma_e)
        for y, sigma_tot, sigma_e in tqdm(
            zip(activations, total_var, eps_var), total=activations.shape[0]
        )
    )  # All voxels, all layers

    loglik_score = np.stack(model_scores)

    return loglik_score


def posterior(loglik_array: NDArray) -> NDArray:
    """
    Obtain the posterior probabilities that a given model produces the activations
    observed from a certain measure (voxel or model)

    Parameters
    ----------

    loglik_array: NDArray, shape (n_channels, n_models)
        Each row represents a measurement channel from a and every row represents
        one of the candidate models. Each element is the log-likelihood of each
        candidate model producing the observed activation in the corresponding
        measurement channel

    Returns
    -------

    post_array: NDArray, shape (n_channels, n_models)
        Each element represents the posterior probability of each of the candidate
        models producing the observed activation in each channel
    """

    post_array = loglik_array - logsumexp(loglik_array, axis=1, keepdims=True)

    return post_array
