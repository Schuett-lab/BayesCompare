import torch
import numpy as np
from BayesCompare import distances
from numpy.testing import assert_allclose
from torch.testing import assert_close
import pytest

ALL_MEASURES = [
    "wasserstein",
    "hellinger",
    "tvd",
    "jsd",
    "kldiv",
    "bhattacharyya",
    "mahalanobis",
]

# number of samples for computing jsd and tvd distances.
# can also be set separate for each test or globally from here
# highly affects how long the tests take
NUM_SAMPLES = 100000


def generate_covs(
    N: int,
    dim: int,
    eig_min: float = 0.5,
    eig_max: float = 2.0,
    cov_dtype: np.dtype = np.float64,
    specific_eigs: int | None = None,
    scale: int = 1,
):
    rng = np.random.default_rng(0)
    covs = np.empty((N, dim, dim), dtype=cov_dtype)

    for i in range(N):
        A = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(A)

        eigs = rng.uniform(eig_min, eig_max, size=dim)

        if specific_eigs:
            eigs[0] = specific_eigs[i]

        covs[i] = scale * (Q @ np.diag(eigs) @ Q.T)

    return covs


@pytest.fixture()
def inputs():
    dim = 25

    # Same input test
    test_same_input_data = generate_covs(N=1, dim=dim)

    # Very close input test
    # eps = [0.05, 1e-2, 1e-3, 1e-5, 1e-8]
    eps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-8]
    A_5 = generate_covs(N=5, dim=dim)
    test_close_input_data = []
    for i, A in enumerate(A_5):
        E = np.random.randn(dim, dim)
        E = (E + E.T) / 2
        B = A + eps[i] * E
        test_close_input_data.append((A, B))

    # Symmetric input test
    B_3 = generate_covs(N=3, dim=dim)
    test_symmetric_inputs_data = [(A, B) for A, B in zip(A_5[:3], B_3)]
    for A, B in zip(A_5[:3], B_3):
        test_symmetric_inputs_data.append((B, A))

    # PSD input test
    test_psd_input_data = generate_covs(
        N=4, dim=dim, specific_eigs=[0.0, 0.0, 0.0, 0.0]
    )

    # Almost PSD input test
    test_almost_psd_input_data = generate_covs(
        N=4, dim=dim, specific_eigs=[0.001, 0.0001, 0.00001, 0.0000001]
    )

    # Large input test
    test_large_input_data = generate_covs(N=4, dim=dim, scale=1e7)

    # Small input test
    test_small_input_data = generate_covs(N=4, dim=dim, scale=1e-4)

    # One small one large input test
    test_large_and_small_input = [
        (A, B) for A, B in zip(test_large_input_data, test_small_input_data)
    ]

    return {
        "test_same_input_data": test_same_input_data,
        "test_close_input_data": test_close_input_data,
        "test_symmetric_inputs_data": test_symmetric_inputs_data,
        "test_psd_input_data": test_psd_input_data,
        "test_almost_psd_input_data": test_almost_psd_input_data,
        "test_large_input_data": test_large_input_data,
        "test_small_input_data": test_small_input_data,
        "test_large_and_small_input": test_large_and_small_input,
    }


# Same covariance matrix given as input: d(A, A)
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_same_input(inputs, meas_name):

    output_np = []
    output_th = []

    num_samples = NUM_SAMPLES

    output_np.append(
        distances.measure_dist(
            [inputs["test_same_input_data"][0], inputs["test_same_input_data"][0]],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=np.random.Generator(np.random.SFC64(124)),
        )
    )
    output_th.append(
        distances.measure_dist(
            [
                torch.from_numpy(inputs["test_same_input_data"][0]),
                torch.from_numpy(inputs["test_same_input_data"][0]),
            ],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=torch.Generator(device="cpu").manual_seed(124),
        )
    )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, np.zeros_like(output_np), rtol=1e-10, atol=1e-10)
    assert_allclose(output_th, torch.zeros_like(output_th), rtol=1e-6, atol=1e-6)

    if "jsd" in meas_name or "tvd" in meas_name or "hellinger" in meas_name:
        assert np.all(output_np <= 1)
        assert torch.all(output_th <= 1)


# Very close covariance matrices given as input: d(A, A + ε)
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_similar_input(inputs, meas_name):

    output_np = []
    output_th = []

    num_samples = NUM_SAMPLES

    for pair in inputs["test_close_input_data"]:
        output_np.append(
            distances.measure_dist(
                pair,
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
                generator=np.random.Generator(np.random.SFC64(124)),
            )
        )
        output_th.append(
            distances.measure_dist(
                [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
                generator=torch.Generator(device="cpu").manual_seed(124),
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    if "wasserstein" in meas_name or "hellinger" in meas_name:
        assert_allclose(output_np, np.zeros_like(output_np), rtol=0.1, atol=0.1)
        assert_close(output_th, torch.zeros_like(output_th), rtol=0.1, atol=0.1)
        assert_allclose(output_np, output_th.numpy(), rtol=1e-3, atol=1e-3)
    else:
        assert_allclose(output_np, np.zeros_like(output_np), rtol=0.05, atol=0.05)
        assert_close(output_th, torch.zeros_like(output_th), rtol=0.05, atol=0.05)
        assert_allclose(output_np, output_th.numpy(), rtol=1e-3, atol=1e-3)

    if "jsd" in meas_name or "tvd" in meas_name or "hellinger" in meas_name:
        assert np.all(output_np <= 1)
        assert torch.all(output_th <= 1)


# Symmetry of distance measure check: d(A, B) vs d(B, A)
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_symmetric_inputs(inputs, meas_name):

    output_np = []
    output_th = []

    num_samples = NUM_SAMPLES

    for pair in inputs["test_symmetric_inputs_data"]:

        output_np.append(
            distances.measure_dist(
                [pair[0], pair[1]],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
                generator=np.random.Generator(np.random.SFC64(124)),
            )
        )

        output_th.append(
            distances.measure_dist(
                [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
                generator=torch.Generator(device="cpu").manual_seed(124),
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    if "jsd" in meas_name or "tvd" in meas_name:
        assert_allclose(output_np[0:3], output_np[3:6], rtol=0.05, atol=0.01)
        assert_close(output_th[0:3], output_th[3:6], rtol=0.05, atol=0.01)
        assert_allclose(output_np, output_th.numpy(), rtol=0.05, atol=0.01)

    else:
        assert_allclose(output_np[0:3], output_np[3:6], rtol=1e-8, atol=1e-10)
        assert_close(output_th[0:3], output_th[3:6], rtol=1e-8, atol=1e-10)
        assert_allclose(output_np, output_th.numpy(), rtol=1e-3, atol=1e-3)

    if "jsd" in meas_name or "tvd" in meas_name or "hellinger" in meas_name:
        assert np.all(output_np <= 1)
        assert torch.all(output_th <= 1)


# PSD inputs (one eigenval = 0)
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_psd_input(inputs, meas_name):

    output_np = []
    output_th = []

    num_samples = NUM_SAMPLES

    output_np.append(
        distances.measure_dist(
            inputs["test_psd_input_data"],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=np.random.Generator(np.random.SFC64(124)),
        )
    )
    output_th.append(
        distances.measure_dist(
            [torch.from_numpy(A) for A in inputs["test_psd_input_data"]],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=torch.Generator(device="cpu").manual_seed(124),
        )
    )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, output_th.numpy(), rtol=0.05, atol=0.05)
    assert np.all(output_np >= 0), "NPArray contains negative values"
    assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    if "jsd" in meas_name or "tvd" in meas_name or "hellinger" in meas_name:
        assert np.all(output_np <= 1)
        assert torch.all(output_th <= 1)


# Almost PSD inputs (one eigenval ~ 0)
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_almost_psd_input(inputs, meas_name):

    output_np = []
    output_th = []

    num_samples = NUM_SAMPLES

    output_np.append(
        distances.measure_dist(
            inputs["test_almost_psd_input_data"],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=np.random.Generator(np.random.SFC64(124)),
        )
    )
    output_th.append(
        distances.measure_dist(
            [torch.from_numpy(A) for A in inputs["test_almost_psd_input_data"]],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=torch.Generator(device="cpu").manual_seed(124),
        )
    )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    if not ("jsd" in meas_name or "tvd" in meas_name):
        assert_allclose(output_np, output_th.numpy(), rtol=1e-2, atol=1e-2)
    assert np.all(output_np >= 0), "NPArray contains negative values"
    assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    if "jsd" in meas_name or "tvd" in meas_name or "hellinger" in meas_name:
        assert np.all(output_np <= 1)
        assert torch.all(output_th <= 1)


# Large scale input
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_large_scale_inputs(inputs, meas_name):
    output_np = []
    output_th = []

    num_samples = NUM_SAMPLES

    output_np.append(
        distances.measure_dist(
            inputs["test_large_input_data"],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=np.random.Generator(np.random.SFC64(124)),
        )
    )
    output_th.append(
        distances.measure_dist(
            [torch.from_numpy(A) for A in inputs["test_large_input_data"]],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=torch.Generator(device="cpu").manual_seed(124),
        )
    )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, output_th.numpy(), rtol=1e-2, atol=1e-2)
    assert np.all(output_np >= 0), "NPArray contains negative values"
    assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    if "jsd" in meas_name or "tvd" in meas_name or "hellinger" in meas_name:
        assert np.all(output_np <= 1)
        assert torch.all(output_th <= 1)


# Small scale input
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_small_scale_inputs(inputs, meas_name):
    output_np = []
    output_th = []

    num_samples = NUM_SAMPLES

    output_np.append(
        distances.measure_dist(
            inputs["test_small_input_data"],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=np.random.Generator(np.random.SFC64(124)),
        )
    )
    output_th.append(
        distances.measure_dist(
            [torch.from_numpy(A) for A in inputs["test_small_input_data"]],
            meas_name=meas_name,
            samples_jsd_tvd=num_samples,
            show_progress=False,
            generator=torch.Generator(device="cpu").manual_seed(124),
        )
    )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, output_th.numpy(), rtol=1e-2, atol=1e-2)
    assert np.all(output_np >= 0), "NPArray contains negative values"
    assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    if "jsd" in meas_name or "tvd" in meas_name or "hellinger" in meas_name:
        assert np.all(output_np <= 1)
        assert torch.all(output_th <= 1)


# One large and one small scale input
@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_large_and_small_scale_inputs(inputs, meas_name):
    output_np = []
    output_th = []

    num_samples = NUM_SAMPLES

    for pair in inputs["test_large_and_small_input"]:

        output_np.append(
            distances.measure_dist(
                pair,
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
                generator=np.random.Generator(np.random.SFC64(124)),
            )
        )
        output_th.append(
            distances.measure_dist(
                [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
                generator=torch.Generator(device="cpu").manual_seed(124),
            )
        )

    output_np = np.stack(output_np, axis=0)
    output_th = torch.stack(output_th, dim=0)

    assert_allclose(output_np, output_th.numpy(), rtol=1e-2, atol=1e-2)
    assert np.all(output_np >= 0), "NPArray contains negative values"
    assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    if "jsd" in meas_name or "tvd" in meas_name or "hellinger" in meas_name:
        assert np.all(output_np <= 1)
        assert torch.all(output_th <= 1)


@pytest.mark.parametrize("meas_name", ALL_MEASURES, ids=ALL_MEASURES)
def test_same_input_twice_determinism(inputs, meas_name):

    num_samples = NUM_SAMPLES

    pair = inputs["test_symmetric_inputs_data"][0]

    output_1_np = distances.measure_dist(
        [pair[0], pair[1]],
        meas_name=meas_name,
        samples_jsd_tvd=num_samples,
        show_progress=False,
        generator=np.random.Generator(np.random.SFC64(124)),
    )

    output_2_np = distances.measure_dist(
        [pair[0], pair[1]],
        meas_name=meas_name,
        samples_jsd_tvd=num_samples,
        show_progress=False,
        generator=np.random.Generator(np.random.SFC64(124)),
    )

    output_1_th = distances.measure_dist(
        [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
        meas_name=meas_name,
        samples_jsd_tvd=num_samples,
        show_progress=False,
        generator=torch.Generator(device="cpu").manual_seed(124),
    )

    output_2_th = distances.measure_dist(
        [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
        meas_name=meas_name,
        samples_jsd_tvd=num_samples,
        show_progress=False,
        generator=torch.Generator(device="cpu").manual_seed(124),
    )

    assert_allclose(output_1_np, output_2_np, rtol=1e-7, atol=1e-7)
    assert_close(output_1_th, output_2_th, rtol=1e-7, atol=1e-7)
