import torch
import numpy as np
from BayesCompare import distances
from numpy.testing import assert_allclose
from torch.testing import assert_close
import unittest
from parameterized import parameterized

ALL_MEASURES = [
    "wasserstein",
    "hellinger",
    "tvd",
    "jsd",
    "kldiv",
    "bhattacharyya",
    "mahalanobis",
]


class TestDists(unittest.TestCase):

    def generate_covs(
        self,
        N: int,
        dim: int,
        eig_min: float = 0.5,
        eig_max: float = 2.0,
        cov_dtype: np.dtype = np.float64,
        specific_eigs: int | None = None,
        scale: int = 1,
    ):

        covs = np.empty((N, dim, dim), dtype=cov_dtype)

        for i in range(N):
            A = self.rng.normal(size=(dim, dim))
            Q, _ = np.linalg.qr(A)

            eigs = self.rng.uniform(eig_min, eig_max, size=dim)

            if specific_eigs:
                eigs[0] = specific_eigs[i]

            covs[i] = scale * (Q @ np.diag(eigs) @ Q.T)

        return covs

    def setUp(self) -> None:

        self.rng = np.random.default_rng(0)

        dim = 25

        # Same input test
        self.test_same_input_data = self.generate_covs(N=1, dim=dim)

        # Very close input test
        # eps = [0.05, 1e-2, 1e-3, 1e-5, 1e-8]
        eps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-8]
        A_5 = self.generate_covs(N=5, dim=dim)
        self.test_close_input_data = []
        for i, A in enumerate(A_5):
            E = np.random.randn(dim, dim)
            E = (E + E.T) / 2
            B = A + eps[i] * E
            self.test_close_input_data.append((A, B))

        # Symmetric input test
        B_3 = self.generate_covs(N=3, dim=dim)
        self.test_symmetric_inputs_data = [(A, B) for A, B in zip(A_5[:3], B_3)]
        for A, B in zip(A_5[:3], B_3):
            self.test_symmetric_inputs_data.append((B, A))

        # PSD input test
        self.test_psd_input_data = self.generate_covs(
            N=4, dim=dim, specific_eigs=[0.0, 0.0, 0.0, 0.0]
        )

        # Almost PSD input test
        self.test_almost_psd_input_data = self.generate_covs(
            N=4, dim=dim, specific_eigs=[0.001, 0.0001, 0.00001, 0.0000001]
        )

        # Large input test
        self.test_large_input_data = self.generate_covs(N=4, dim=dim, scale=1e7)

        # Small input test
        self.test_small_input_data = self.generate_covs(N=4, dim=dim, scale=1e-4)

        # One small one large input test
        self.test_large_and_small_input = [
            (A, B)
            for A, B in zip(self.test_large_input_data, self.test_small_input_data)
        ]

    # Same covariance matrix given as input: d(A, A)
    @parameterized.expand(ALL_MEASURES)
    def test_same_input(self, meas_name):

        output_np = []
        output_th = []

        num_samples = 1000000

        output_np.append(
            distances.measure_dist(
                [self.test_same_input_data[0], self.test_same_input_data[0]],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [
                    torch.from_numpy(self.test_same_input_data[0]),
                    torch.from_numpy(self.test_same_input_data[0]),
                ],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )

        output_np = np.stack(output_np, axis=0)
        output_th = torch.stack(output_th, dim=0)

        assert_allclose(output_np, np.zeros_like(output_np), rtol=1e-10, atol=1e-10)
        assert_allclose(output_th, torch.zeros_like(output_th), rtol=1e-6, atol=1e-6)

    # Very close covariance matrices given as input: d(A, A + ε)
    @parameterized.expand(ALL_MEASURES)
    def test_similar_input(self, meas_name):

        output_np = []
        output_th = []

        num_samples = 1000000

        for pair in self.test_close_input_data:
            output_np.append(
                distances.measure_dist(
                    pair,
                    meas_name=meas_name,
                    samples_jsd_tvd=num_samples,
                    show_progress=False,
                )
            )
            output_th.append(
                distances.measure_dist(
                    [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
                    meas_name=meas_name,
                    samples_jsd_tvd=num_samples,
                    show_progress=False,
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

    # Symmetry of distance measure check: d(A, B) vs d(B, A)
    @parameterized.expand(ALL_MEASURES)
    def test_symmetric_inputs(self, meas_name):

        if "kl" in meas_name:
            self.skipTest("KL divergence is not symmetric.")

        output_np = []
        output_th = []

        num_samples = 1000000

        for pair in self.test_symmetric_inputs_data:

            output_np.append(
                distances.measure_dist(
                    [pair[0], pair[1]],
                    meas_name=meas_name,
                    samples_jsd_tvd=num_samples,
                    show_progress=False,
                )
            )

            output_th.append(
                distances.measure_dist(
                    [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
                    meas_name=meas_name,
                    samples_jsd_tvd=num_samples,
                    show_progress=False,
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

    # PSD inputs (one eigenval = 0)
    @parameterized.expand(ALL_MEASURES)
    def test_psd_input(self, meas_name):

        output_np = []
        output_th = []

        num_samples = 1000000

        output_np.append(
            distances.measure_dist(
                self.test_psd_input_data,
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [torch.from_numpy(A) for A in self.test_psd_input_data],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )

        output_np = np.stack(output_np, axis=0)
        output_th = torch.stack(output_th, dim=0)

        # what to check about outputs for the psd tests?
        assert_allclose(output_np, output_th.numpy(), rtol=0.05, atol=0.05)
        assert np.all(output_np >= 0), "NPArray contains negative values"
        assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    # Almost PSD inputs (one eigenval ~ 0)
    @parameterized.expand(ALL_MEASURES)
    def test_almost_psd_input(self, meas_name):

        output_np = []
        output_th = []

        num_samples = 1000000

        output_np.append(
            distances.measure_dist(
                self.test_almost_psd_input_data,
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [torch.from_numpy(A) for A in self.test_almost_psd_input_data],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )

        output_np = np.stack(output_np, axis=0)
        output_th = torch.stack(output_th, dim=0)

        # what to check about outputs for the psd tests?
        if not ("jsd" in meas_name or "tvd" in meas_name):
            assert_allclose(output_np, output_th.numpy(), rtol=1e-2, atol=1e-2)
        assert np.all(output_np >= 0), "NPArray contains negative values"
        assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    # Large scale input
    @parameterized.expand(ALL_MEASURES)
    def test_large_scale_inputs(self, meas_name):
        output_np = []
        output_th = []

        num_samples = 1000000

        output_np.append(
            distances.measure_dist(
                self.test_large_input_data,
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [torch.from_numpy(A) for A in self.test_large_input_data],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )

        output_np = np.stack(output_np, axis=0)
        output_th = torch.stack(output_th, dim=0)

        # what to check about outputs for the scale tests?
        assert_allclose(output_np, output_th.numpy(), rtol=1e-2, atol=1e-2)
        assert np.all(output_np >= 0), "NPArray contains negative values"
        assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    # Small scale input
    @parameterized.expand(ALL_MEASURES)
    def test_large_scale_inputs(self, meas_name):
        output_np = []
        output_th = []

        num_samples = 1000000

        output_np.append(
            distances.measure_dist(
                self.test_small_input_data,
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )
        output_th.append(
            distances.measure_dist(
                [torch.from_numpy(A) for A in self.test_small_input_data],
                meas_name=meas_name,
                samples_jsd_tvd=num_samples,
                show_progress=False,
            )
        )

        output_np = np.stack(output_np, axis=0)
        output_th = torch.stack(output_th, dim=0)

        # what to check about outputs for the scale tests?
        assert_allclose(output_np, output_th.numpy(), rtol=1e-3, atol=1e-3)
        assert np.all(output_np >= 0), "NPArray contains negative values"
        assert torch.all(output_th >= 0), "Torch Tensor contains negative values"

    # One large and one small scale input
    @parameterized.expand(ALL_MEASURES)
    def test_large_scale_inputs(self, meas_name):
        output_np = []
        output_th = []

        num_samples = 1000000

        for pair in self.test_large_and_small_input:

            output_np.append(
                distances.measure_dist(
                    pair,
                    meas_name=meas_name,
                    samples_jsd_tvd=num_samples,
                    show_progress=False,
                )
            )
            output_th.append(
                distances.measure_dist(
                    [torch.from_numpy(pair[0]), torch.from_numpy(pair[1])],
                    meas_name=meas_name,
                    samples_jsd_tvd=num_samples,
                    show_progress=False,
                )
            )

        output_np = np.stack(output_np, axis=0)
        output_th = torch.stack(output_th, dim=0)

        # what to check about outputs for the scale tests?
        assert_allclose(output_np, output_th.numpy(), rtol=1e-2, atol=1e-2)
        assert np.all(output_np >= 0), "NPArray contains negative values"
        assert torch.all(output_th >= 0), "Torch Tensor contains negative values"
