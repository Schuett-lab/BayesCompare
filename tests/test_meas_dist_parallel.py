import numpy as np
from pathlib import Path
import os
from BayesCompare import measure_dist_parallel
from BayesCompare import distances
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
]

home_path = Path.home()

# number of samples for computing jsd and tvd distances.
# can also be set separate for each test or globally from here
# highly affects how long the tests take
NUM_SAMPLES = 10000


@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_meas_dist_parallel(meas_name):

    generator = np.random.Generator(np.random.SFC64(124))

    cov_input_dir = os.path.join(
        home_path,
        "Documents/BayesCompare/tests/sample_data/test_input_meas_dist_parallel.pkl",
    )

    measure_dist_parallel(
        covs_dir=cov_input_dir,
        output_dir=os.path.join(
            home_path, "Documents/BayesCompare/tests/results_meas_dist_parallel/"
        ),
        meas_name=meas_name,
        samples_jsd_tvd=NUM_SAMPLES,
        generator=np.random.Generator(np.random.SFC64(124)),
    )

    hdf_filename = os.path.join(
        home_path,
        "Documents/BayesCompare/tests/results_meas_dist_parallel/"
        + "dist_test_input_meas_dist_parallel_"
        + simplify_string(meas_name)
        + ".hdf5",
    )

    with h5py.File(hdf_filename, "r") as f:

        parallel_output = f["dist"][...]

    with open(cov_input_dir, "rb") as f:

        covs_names = pickle.load(f)

        covs = covs_names

    non_parallel_output = distances.measure_dist(
        covs=covs,
        meas_name=meas_name,
        samples_jsd_tvd=NUM_SAMPLES,
        show_progress=False,
        generator=np.random.Generator(np.random.SFC64(124)),
    )

    if "jsd" in meas_name or "tvd" in meas_name:
        assert_allclose(parallel_output, non_parallel_output, rtol=0.01, atol=3.8)
    else:
        assert_allclose(parallel_output, non_parallel_output, rtol=1e-10, atol=1e-10)
