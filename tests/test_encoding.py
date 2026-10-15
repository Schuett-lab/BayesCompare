"""
Tests for the evidence_score function
"""

import glob
import os
import pickle
import pathlib

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.special import logsumexp

from BayesCompare.encoding import loglik_score, log_posterior


def read_sample_file(file_path):
    file_ext = pathlib.Path(file_path).suffix
    if file_ext == ".npy":
        sample_file = np.load(file_path)
    elif file_ext == ".pkl":
        with open(file_path, "rb") as pkl_file:
            sample_file = pickle.load(pkl_file)
    else:
        return NotImplementedError("Input file type not implemented for loading")

    return sample_file


def test_evidence_score_io():
    sample_path = "tests/sample_data"
    input_args = {}
    for input_file in ["input_y", "input_totvar", "input_epsvar", "evidence_covs_norm"]:
        print(os.getcwd(), os.path.join(sample_path, f"test_{input_file}*"))
        filename = glob.glob(os.path.join(sample_path, f"test_{input_file}*"))[0]
        input_args[input_file] = read_sample_file(filename)

    signal_var = input_args["input_totvar"] - input_args["input_epsvar"]
    expected_multinoise = np.load(os.path.join(sample_path, "test_output_loglik.npy"))
    expected_singlenoise = np.load(
        os.path.join(sample_path, "test_output_loglik_singlenoise.npy")
    )

    # Per-voxel noise
    model_scores = []
    for cov in input_args["evidence_covs_norm"]:
        model_score = loglik_score(
            norm_cov=cov,
            activations=input_args["input_y"],
            signal_var=signal_var,
            noise_var=input_args["input_epsvar"],
            n_jobs=None,
        )
        model_scores.append(model_score)
    score_multinoise = np.stack(model_scores, axis=1)

    # Single noise value
    model_scores = []
    for cov in input_args["evidence_covs_norm"]:
        model_score = loglik_score(
            norm_cov=cov,
            activations=input_args["input_y"],
            noise_var=np.array([0.8]),
            n_jobs=None,
        )
        # Use logsumexp since any number of noise values other than the
        # same number of voxels generates an extra last dimension over noise
        # values
        model_scores.append(logsumexp(model_score, axis=-1))
    score_singlenoise = np.stack(model_scores, axis=1)

    assert_almost_equal(score_multinoise, expected_multinoise)
    assert_almost_equal(score_singlenoise, expected_singlenoise)


def test_loglik_score_correct_output_shape():
    sample_cov = np.eye(5)
    sample_y = np.random.uniform(size=(3, 5))
    sample_noisevar = np.random.uniform(size=(3,))

    # Test with one variance per voxel (noise_var same shape as activations)
    expected_shape = sample_y.shape
    out = loglik_score(
        norm_cov=sample_cov,
        activations=sample_y,
        noise_var=sample_noisevar,
    )
    assert out.shape[0] == expected_shape[0]

    # Test with multiple values for all voxels (noise_var different shape as activations)
    n_var = 2
    out = loglik_score(
        norm_cov=sample_cov,
        activations=sample_y,
        noise_var=sample_noisevar[:n_var],
    )
    expected_shape = (sample_y.shape[0], n_var)
    assert out.shape == expected_shape


def test_posterior():
    input = np.random.rand(50, 10, 2)  # (n_voxels, n_models, n_noise)

    logpost_2d = log_posterior(input[..., 0])  # input is (n_voxels, n_models)
    logpost_3d = log_posterior(input)

    for post_array in [logpost_2d, logpost_3d]:
        # Check that the sum over the log_posterior is equal to the number of voxels
        # (log_posterior over models and noise values makes all values for one voxel
        # add up to 1)
        result = np.sum(np.exp(post_array))
        expected = post_array.shape[0]

        assert_almost_equal(result, expected)
