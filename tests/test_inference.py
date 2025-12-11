"""
Tests for the evidence_score function
"""

import glob
import os
import pickle
import pathlib

import numpy as np
from numpy.testing import assert_almost_equal

from BayesCompare.inference import loglik_score


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
    for input_file in ["input_y", "input_totvar", "input_epsvar", "input_covs_norm"]:
        print(os.getcwd(), os.path.join(sample_path, f"test_{input_file}*"))
        filename = glob.glob(os.path.join(sample_path, f"test_{input_file}*"))[0]
        input_args[input_file] = read_sample_file(filename)

    signal_var = input_args["input_totvar"] - input_args["input_epsvar"]
    expected_multinoise = np.load(os.path.join(sample_path, "test_output_loglik.npy"))
    expected_singlenoise = np.load(
        os.path.join(sample_path, "test_output_loglik_singlenoise.npy")
    )

    # Per-voxel noise
    score_multinoise = loglik_score(
        norm_covs=input_args["input_covs_norm"],
        activations=input_args["input_y"],
        signal_var=signal_var,
        noise_var=input_args["input_epsvar"],
        n_jobs=None,
    )

    # Single noise value
    score_singlenoise = loglik_score(
        norm_covs=input_args["input_covs_norm"],
        activations=input_args["input_y"],
        noise_var=0.8,
        n_jobs=None,
    )

    assert_almost_equal(score_multinoise, expected_multinoise)
    assert_almost_equal(score_singlenoise, expected_singlenoise)
