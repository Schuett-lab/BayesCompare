import numpy as np
from pathlib import Path
import os
from BayesCompare import measure_dist_parallel
from BayesCompare.distances import measure_dist
from BayesCompare.distances import simplify_string
from numpy.testing import assert_allclose
import pytest
import h5py
import pickle

ALL_MEASURES = [
    "wasserstein",
    "hellinger",
    "tvd",
    "jsd",
    "kldiv",
    "bhattacharyya",
    "mahalanobis",
    "cka",
    "rsa_arccos",
    "rsa_cos",
    "rsa_corr",
    "rsa_rank",
    "gulp",
    "dist_corr",
    "jaccard",
    "procrustes",
    "nbs",
]

home_path = Path.home()

# number of samples for computing jsd and tvd distances.
# can also be set separate for each test or globally from here
# highly affects how long the tests take
NUM_SAMPLES = 10000


@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_meas_dist_parallel(tmp_path, meas_name):

    generator = np.random.Generator(np.random.SFC64(124))

    cov_input_file = os.path.join(
        os.getcwd(),
        "tests/sample_data/test_input_meas_dist_parallel_simplified.pkl",
    )

    output_dir = os.path.join(tmp_path, "results_meas_dist_parallel")

    if (
        meas_name == "gulp"
        or meas_name == "jaccard"
        or meas_name == "procrustes"
        or meas_name == "nbs"
    ):
        measure_dist_parallel(
            covs_dir=cov_input_file,
            output_dir=output_dir,
            meas_name=meas_name,
            normalize=False,
            lmbd=0.01,
            k=3,
        )
    else:
        measure_dist_parallel(
            covs_dir=cov_input_file,
            output_dir=output_dir,
            meas_name=meas_name,
            samples_jsd_tvd=NUM_SAMPLES,
            generator=generator,
            b=1 / 100,
        )

    hdf_filename = os.path.join(
        output_dir,
        f"dist_test_input_meas_dist_parallel_simplified_{simplify_string(meas_name)}.hdf5",
    )

    with h5py.File(hdf_filename, "r") as f:

        parallel_output = f["dist"][...]

    with open(cov_input_file, "rb") as f:

        covs_names = pickle.load(f)

        covs = covs_names

    if (
        meas_name == "gulp"
        or meas_name == "jaccard"
        or meas_name == "procrustes"
        or meas_name == "nbs"
    ):
        non_parallel_output = measure_dist(
            covs=covs,
            meas_name=meas_name,
            show_progress=False,
            normalize=False,
            k=3,
            lmbd=0.01,
        )
    else:
        non_parallel_output = measure_dist(
            covs=covs,
            meas_name=meas_name,
            samples_jsd_tvd=NUM_SAMPLES,
            show_progress=False,
            generator=generator,
            b=1 / 100,
        )

    if "jsd" in meas_name or "tvd" in meas_name:
        assert_allclose(parallel_output, non_parallel_output, rtol=0.01, atol=3.8)
    elif "procrustes" in meas_name:
        assert_allclose(parallel_output, non_parallel_output, rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(parallel_output, non_parallel_output, rtol=1e-8, atol=1e-8)
