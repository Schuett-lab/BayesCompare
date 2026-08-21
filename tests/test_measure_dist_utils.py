import numpy as np
import torch
import os
import pytest
import h5py
import itertools
from BayesCompare.measure_distance_utils import (
    select_measure,
    check_saved_hdf,
    load_covs,
    preprocess_input_covs,
    get_preprocessing_params,
)
from BayesCompare.dist_utils import simplify_string
from BayesCompare.distances import DISTANCES, REGISTRY


def generate_covs(
    N: int,
    dim: int,
    eig_min: float = 0.5,
    eig_max: float = 2.0,
    cov_dtype: np.dtype = np.float64,
    specific_eigs: int | None = None,
    scale: int = 1,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    covs = np.empty((N, dim, dim), dtype=cov_dtype)

    for i in range(N):
        A = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(A)

        eigs = rng.uniform(eig_min, eig_max, size=dim)

        if specific_eigs:
            eigs[0] = specific_eigs[i]

        covs[i] = scale * (Q @ np.diag(eigs) @ Q.T)

    return covs


@pytest.mark.parametrize(
    "meas_name",
    list(itertools.chain(DISTANCES["ours"], DISTANCES["others"], ["undefined"])),
)
def test_select_measure(meas_name):
    cov_np = np.zeros((3, 3))
    cov_torch = torch.Tensor((3, 3))
    cov_dict = {"cov": cov_np}
    if meas_name != "undefined":
        assert (
            select_measure(cov_np, meas_name)
            == REGISTRY["numpy"][simplify_string(meas_name)]
        )
        assert (
            select_measure(cov_torch, meas_name)
            == REGISTRY["torch"][simplify_string(meas_name)]
        )
        with pytest.raises(
            TypeError,
            match="Input must be a Numpy array or a PyTorch tensor.",
        ):
            select_measure(cov_dict, meas_name)
    else:
        with pytest.raises(
            NotImplementedError,
            match="Given metric name is not valid for Tensor tensor covariances.",
        ):
            select_measure(cov_torch, meas_name)
        with pytest.raises(
            NotImplementedError,
            match="Given metric name is not valid for Numpy array covariances.",
        ):
            select_measure(cov_np, meas_name)


@pytest.mark.parametrize(
    "meas_name",
    list(itertools.chain(DISTANCES["ours"], DISTANCES["others"])),
)
def test_check_saved_hdf_non_existing_hdf(meas_name):
    hdf_dir = "tests/sample_data"
    N = 5
    covs_filename = "testing_covs"
    hdf_file_path = (
        hdf_dir
        + "/"
        + "dist_"
        + covs_filename
        + "_"
        + simplify_string(meas_name)
        + ".hdf5"
    )
    if os.path.exists(hdf_file_path):
        os.remove(hdf_file_path)
    indices, res_hdf_dir = check_saved_hdf(hdf_dir, N, covs_filename, meas_name)
    assert os.path.exists(res_hdf_dir)
    assert res_hdf_dir == hdf_file_path
    with h5py.File(res_hdf_dir, "r") as f:
        dset = f["dist"]
        assert dset.shape == (N, N)
        assert len(indices) == N * (N - 1) / 2


@pytest.mark.parametrize(
    "meas_name",
    list(itertools.chain(DISTANCES["ours"], DISTANCES["others"])),
)
def test_check_saved_hdf_existing_hdf(meas_name):
    hdf_dir = "tests/sample_data"
    N = 5
    covs_filename = "testing_covs"
    hdf_file_path = (
        hdf_dir
        + "/"
        + "dist_"
        + covs_filename
        + "_"
        + simplify_string(meas_name)
        + ".hdf5"
    )
    if os.path.exists(hdf_file_path):
        indices, res_hdf_dir = check_saved_hdf(hdf_dir, N, covs_filename, meas_name)
        assert os.path.exists(res_hdf_dir)
        assert res_hdf_dir == hdf_file_path
        with h5py.File(res_hdf_dir, "r") as f:
            dset = f["dist"]
            assert dset.shape == (N, N)
            assert len(indices) == N * (N - 1) / 2
        if os.path.exists(hdf_file_path):
            os.remove(hdf_file_path)


@pytest.mark.parametrize(
    "input_filenames",
    [
        "test_utils_covs_np.np",
        "test_utils_covs_np.npy",
        "test_utils_covs_np.npz",
        "test_input_covs_norm.pkl",
        "test_input_covs_norm.pickle",
        "test_input_covs_norm.hdf5",
        "test_input_covs_dict.pkl",
        "test_input_covs_dict.pickle",
        "test_input_covs_np_arr.pkl",
        "test_input_covs_torch.pkl",
    ],
)
def test_load_covs(input_filenames):
    if input_filenames != "test_input_covs_norm.hdf5":
        covs, filename = load_covs("tests/sample_data/" + input_filenames)
        expected_filename, ext = os.path.splitext(input_filenames)
        if ext in [".np", ".npy", ".npz"]:
            assert covs.shape == (20, 50, 50)
        else:
            assert covs.shape == (12, 20, 20)
        assert filename == expected_filename
    else:
        with pytest.raises(
            ValueError,
            match=f"Expected one of the following extensions: .pkl, .pickle, .np, .npy, .npz, got .hdf5",
        ):
            covs, filename = load_covs("tests/sample_data/" + input_filenames)


class TestPreprocessInputCovs:
    dim = 6
    N = 3
    seed = 12
    input_np = generate_covs(N=N, dim=dim, scale=1, seed=seed)
    input_th = torch.from_numpy(input_np)
    input_list = [
        generate_covs(N=1, dim=dim, scale=1, seed=seed),
        generate_covs(N=1, dim=dim, scale=1, seed=seed),
        generate_covs(N=1, dim=dim, scale=1, seed=seed),
    ]

    @pytest.mark.parametrize("input_covs", [input_np, input_th, input_list])
    def test_default_behaviour(self, input_covs):
        output_covs = preprocess_input_covs(input_covs)
        assert type(input_covs) == type(output_covs)
        if type(input_covs) in [np.ndarray, torch.Tensor]:
            assert input_covs.shape == output_covs.shape
            for idx in range(self.N):
                assert np.trace(output_covs[idx, :, :]) == pytest.approx(
                    float(self.dim), rel=1e-5, abs=1e-8
                )
        else:
            assert len(input_covs) == len(output_covs)
            for idx in range(self.N):
                assert np.trace(output_covs[idx]) == pytest.approx(
                    float(self.dim), rel=1e-5, abs=1e-8
                )

    @pytest.mark.parametrize("input_covs", [input_np, input_th, input_list])
    def test_no_normalization(self, input_covs):
        output_covs = preprocess_input_covs(input_covs, normalize=False)
        assert type(input_covs) == type(output_covs)
        if type(input_covs) in [np.ndarray, torch.Tensor]:
            assert input_covs.shape == output_covs.shape
            for idx in range(self.N):
                assert np.trace(output_covs[idx, :, :]) == pytest.approx(
                    np.float64(6), abs=1
                )
        else:
            assert len(input_covs) == len(output_covs)
            for idx in range(self.N):
                assert np.trace(output_covs[idx]) == pytest.approx(np.float64(6), abs=1)

    @pytest.mark.parametrize("input_covs", [input_np, input_th, input_list])
    def test_normalize_without_noise(self, input_covs):
        output_covs = preprocess_input_covs(
            input_covs, normalize=True, b=0, noise_var=0
        )
        assert type(input_covs) == type(output_covs)
        if type(input_covs) in [np.ndarray, torch.Tensor]:
            assert input_covs.shape == output_covs.shape
            for idx in range(self.N):
                assert np.trace(output_covs[idx, :, :]) == pytest.approx(
                    float(self.dim), rel=1e-5, abs=1e-8
                )
        else:
            assert len(input_covs) == len(output_covs)
            for idx in range(self.N):
                assert np.trace(output_covs[idx]) == pytest.approx(
                    float(self.dim), rel=1e-5, abs=1e-8
                )


class TestGetPreprocessingParams:
    @pytest.mark.parametrize(
        "meas_name",
        list(itertools.chain(DISTANCES["ours"], DISTANCES["others"])),
    )
    def test_get_preprocessing_params_default(self, meas_name):
        meas_name_list, b_list, noise_var_list, normalize_list = (
            get_preprocessing_params(meas_name, b=0.1)
        )
        assert meas_name_list == [meas_name]
        assert b_list == [0.1]
        assert noise_var_list is None
        assert normalize_list == [True]

    def test_get_preprocessing_params_single_inputs(self):
        meas_name_list, b_list, noise_var_list, normalize_list = (
            get_preprocessing_params(
                "kldiv",
                b=0.1,
                noise_var=0.8,
                normalize=True,
            )
        )
        assert meas_name_list == ["kldiv"]
        assert b_list == [0.1]
        assert noise_var_list == [0.8]
        assert normalize_list == [True]

    def test_get_preprocessing_params_list_inputs(self):
        meas_name_list, b_list, noise_var_list, normalize_list = (
            get_preprocessing_params(
                DISTANCES["ours"],
                b=[0.1] * len(DISTANCES["ours"]),
                normalize=[True] * len(DISTANCES["ours"]),
            )
        )
        assert meas_name_list == DISTANCES["ours"]
        assert b_list == [0.1] * len(DISTANCES["ours"])
        assert noise_var_list is None
        assert normalize_list == [True] * len(DISTANCES["ours"])

        meas_name_list, b_list, noise_var_list, normalize_list = (
            get_preprocessing_params(
                DISTANCES["ours"],
                b=[0.1] * len(DISTANCES["ours"]),
                noise_var=[0.8] * len(DISTANCES["ours"]),
                normalize=[True] * len(DISTANCES["ours"]),
            )
        )
        assert meas_name_list == DISTANCES["ours"]
        assert b_list == [0.1] * len(DISTANCES["ours"])
        assert noise_var_list == [0.8] * len(DISTANCES["ours"])
        assert normalize_list == [True] * len(DISTANCES["ours"])

    def test_get_preprocessing_params_invalids(self):
        with pytest.raises(TypeError):
            get_preprocessing_params(
                "kldiv",
                b=25,
                noise_var=0.8,
                normalize=True,
            )
        with pytest.raises(TypeError):
            get_preprocessing_params(
                "kldiv",
                b=0.1,
                noise_var=30,
                normalize=True,
            )
        with pytest.raises(TypeError):
            get_preprocessing_params(
                "kldiv",
                b=0.1,
                noise_var=0.8,
                normalize=20,
            )
        with pytest.raises(TypeError):
            get_preprocessing_params(
                40,
                b=0.1,
                noise_var=0.8,
                normalize=True,
            )
        with pytest.raises(TypeError):
            get_preprocessing_params(
                "kldiv",
                b=[25],
                noise_var=0.8,
                normalize=True,
            )
        with pytest.raises(TypeError):
            get_preprocessing_params(
                "kldiv",
                b=0.1,
                noise_var=[30],
                normalize=True,
            )
        with pytest.raises(TypeError):
            get_preprocessing_params(
                "kldiv",
                b=0.1,
                noise_var=0.8,
                normalize=[20],
            )
        with pytest.raises(TypeError):
            get_preprocessing_params(
                [40],
                b=0.1,
                noise_var=0.8,
                normalize=True,
            )
