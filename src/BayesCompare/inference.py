import numpy as np

from joblib import Parallel, delayed
from scipy.linalg import pinv
from scipy.special import logsumexp
from tqdm import tqdm

from numpy.typing import NDArray
from typing import Optional

from .cov_utils import cov_sigma


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
    cov: NDArray,
    y: NDArray,
    mu: Optional[NDArray] = None,
    cov_inv: Optional[NDArray] = None,
) -> float:
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

    cov_inv: NDArray or None, default None
        The inner inverse of the covariance matrix. Certain variations of the
        analysis make it more efficient to compute the inverse beforehand. If
        provided, it is used instead of cov

    Returns
    -------

    loglik: float
        Log likelihood that y is produced by the model corresponding to cov

    """
    N = len(y)
    if cov_inv is not None:
        inner_inv = cov_inv
    else:
        inner_inv = np.linalg.inv(cov[:N, :N])  # Precision matrix
    if mu is None:
        ss = np.expand_dims(y, 0) @ inner_inv @ np.expand_dims(y, 1)
    else:
        ss = np.expand_dims(y - mu, 0) @ inner_inv @ np.expand_dims(y - mu, 1)

    logdet = np.linalg.slogdet(inner_inv)
    loglik = logdet.logabsdet / 2 - ss / 2 - N / 2 * np.log(2 * np.pi)

    return loglik


def loglik_score(
    norm_cov: list[NDArray],
    activations: NDArray,
    noise_var: NDArray | float,
    signal_var: Optional[NDArray] = None,
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

    noise_var: np.array, shape (n_channels,)
        Estimated variance attributed to noise for each measurement channel
        across all stimuli

    signal_var: np.array, shape (n_channels,) or None, default None

    n_jobs: int, default = -1
        Parameter passed to joblib for parallelization

    Returns
    -------

    loglik_score: np.array, shape (n_channels, n_models)
        Log-likelihood score for each measurement channel and candidate model

    """
    # If no signal_var is passed, infer it from noise_var (which should be a single value)
    if signal_var is None:
        assert isinstance(noise_var, float)
        single_noise_value = True
    else:
        single_noise_value = False

    def voxel_loop(norm_cov, y, sig_var, eps_var, cov_inv=None):
        """Run evidence on one voxel and all models"""

        if cov_inv is not None:
            ev = evidence(norm_cov, y, cov_inv=cov_inv)
        else:
            s_cov = cov_sigma(norm_cov, signal_var=sig_var, noise_var=eps_var)
            ev = evidence(s_cov, y)

        return ev.flatten()

    if single_noise_value:
        N = len(activations[0])
        s_cov = cov_sigma(norm_cov, signal_var=signal_var, noise_var=noise_var)
        cov_inv = np.linalg.inv(s_cov[:N, :N])
        model_scores: list[NDArray] = Parallel(n_jobs=n_jobs)(
            delayed(voxel_loop)(s_cov, y, signal_var, noise_var, cov_inv)
            for y in tqdm(activations, total=activations.shape[0])
        )
    else:
        model_scores: list[NDArray] = Parallel(n_jobs=n_jobs)(
            delayed(voxel_loop)(norm_cov, y, sig_var, eps_var)
            for y, sig_var, eps_var in tqdm(
                zip(activations, signal_var, noise_var), total=activations.shape[0]
            )
        )  # All voxels, all layers

    loglik_score = np.concatenate(model_scores)

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
