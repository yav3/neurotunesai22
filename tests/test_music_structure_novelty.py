"""Tests for music_structure.novelty_curve — SSM-based novelty segmentation."""
import pytest
import numpy as np
from models.music_structure.novelty_curve import (
    NoveltyPeak, compute_ssm, checkerboard_kernel,
    novelty_from_ssm, detect_boundaries, lag_ssm,
    recurrence_plot, segment_from_novelty, transition_matrix,
)


class TestComputeSSM:
    def test_cosine_shape(self):
        features = np.random.randn(20, 5)
        ssm = compute_ssm(features, "cosine")
        assert ssm.shape == (20, 20)

    def test_cosine_diagonal_ones(self):
        features = np.random.randn(10, 3)
        ssm = compute_ssm(features, "cosine")
        np.testing.assert_array_almost_equal(np.diag(ssm), 1.0, decimal=5)

    def test_cosine_symmetric(self):
        features = np.random.randn(15, 4)
        ssm = compute_ssm(features, "cosine")
        np.testing.assert_array_almost_equal(ssm, ssm.T)

    def test_euclidean(self):
        features = np.random.randn(10, 3)
        ssm = compute_ssm(features, "euclidean")
        assert ssm.shape == (10, 10)
        assert np.all(ssm >= 0)
        np.testing.assert_array_almost_equal(np.diag(ssm), 1.0, decimal=5)

    def test_dot_product_fallback(self):
        features = np.random.randn(8, 4)
        ssm = compute_ssm(features, "dot")
        assert ssm.shape == (8, 8)

    def test_identical_rows(self):
        row = np.array([1.0, 2.0, 3.0])
        features = np.tile(row, (5, 1))
        ssm = compute_ssm(features, "cosine")
        np.testing.assert_array_almost_equal(ssm, 1.0)


class TestCheckerboardKernel:
    def test_shape(self):
        k = checkerboard_kernel(8)
        assert k.shape == (8, 8)

    def test_symmetry(self):
        k = checkerboard_kernel(16)
        np.testing.assert_array_almost_equal(k, k.T)

    def test_sign_pattern(self):
        k = checkerboard_kernel(4)
        assert k[0, 0] > 0
        assert k[0, 3] < 0

    def test_odd_size(self):
        k = checkerboard_kernel(7)
        assert k.shape == (7, 7)


class TestNoveltyFromSSM:
    def test_shape(self):
        ssm = np.eye(100)
        n = novelty_from_ssm(ssm, kernel_size=16)
        assert len(n) == 100

    def test_non_negative(self):
        ssm = np.random.randn(50, 50)
        ssm = (ssm + ssm.T) / 2
        n = novelty_from_ssm(ssm, kernel_size=8)
        assert np.all(n >= 0)

    def test_block_diagonal_peaks(self):
        ssm = np.zeros((60, 60))
        ssm[:30, :30] = 1.0
        ssm[30:, 30:] = 1.0
        n = novelty_from_ssm(ssm, kernel_size=16)
        peak_region = n[25:35]
        assert np.max(peak_region) > np.mean(n)


class TestDetectBoundaries:
    def test_clear_peak(self):
        curve = np.zeros(50)
        curve[25] = 1.0
        peaks = detect_boundaries(curve, threshold=0.3)
        assert len(peaks) >= 1
        assert any(p.time_idx == 25 for p in peaks)

    def test_empty(self):
        assert detect_boundaries(np.array([])) == []

    def test_flat(self):
        curve = np.ones(20)
        assert detect_boundaries(curve) == []

    def test_all_zeros(self):
        assert detect_boundaries(np.zeros(30)) == []

    def test_min_segment_frames(self):
        curve = np.zeros(50)
        curve[10] = 1.0
        curve[12] = 0.9
        peaks = detect_boundaries(curve, threshold=0.3, min_segment_frames=5)
        assert len(peaks) == 1

    def test_returns_novelty_peaks(self):
        curve = np.zeros(30)
        curve[15] = 1.0
        peaks = detect_boundaries(curve, threshold=0.3)
        for p in peaks:
            assert isinstance(p, NoveltyPeak)
            assert 0.0 < p.strength <= 1.0


class TestLagSSM:
    def test_shape(self):
        ssm = np.eye(20)
        lag = lag_ssm(ssm)
        assert lag.shape[0] == 20

    def test_diagonal_entry(self):
        ssm = np.eye(10)
        lag = lag_ssm(ssm)
        for i in range(10):
            assert lag[i, 0] == pytest.approx(1.0)

    def test_max_lag(self):
        ssm = np.eye(20)
        lag = lag_ssm(ssm, max_lag=5)
        assert lag.shape == (20, 5)


class TestRecurrencePlot:
    def test_binary(self):
        features = np.random.randn(15, 3)
        rp = recurrence_plot(features, threshold=0.5)
        assert set(np.unique(rp)).issubset({0, 1})

    def test_diagonal(self):
        features = np.random.randn(10, 4)
        rp = recurrence_plot(features, threshold=0.0)
        np.testing.assert_array_equal(np.diag(rp), 1)

    def test_symmetric(self):
        features = np.random.randn(12, 3)
        rp = recurrence_plot(features)
        np.testing.assert_array_equal(rp, rp.T)


class TestSegmentFromNovelty:
    def test_returns_times(self):
        features = np.random.randn(100, 10)
        features[:50] *= 0.1
        features[50:] *= 10.0
        times = segment_from_novelty(features, sr=22050.0, hop_size=512, threshold=0.3)
        assert isinstance(times, list)
        for t in times:
            assert t >= 0

    def test_short_input(self):
        features = np.random.randn(10, 5)
        times = segment_from_novelty(features, kernel_sizes=[4])
        assert isinstance(times, list)

    def test_custom_kernel_sizes(self):
        features = np.random.randn(200, 8)
        times = segment_from_novelty(features, kernel_sizes=[16, 32])
        assert isinstance(times, list)


class TestTransitionMatrix:
    def test_basic(self):
        labels = [0, 1, 0, 1, 2]
        mat = transition_matrix(labels)
        assert mat.shape == (3, 3)
        assert mat[0, 1] == 2
        assert mat[1, 0] == 1
        assert mat[1, 2] == 1

    def test_empty(self):
        mat = transition_matrix([])
        assert mat.shape == (0, 0)

    def test_single(self):
        mat = transition_matrix([0])
        assert mat.shape == (1, 1)
        assert mat[0, 0] == 0

    def test_row_sums(self):
        labels = [0, 1, 2, 0, 1, 2, 0]
        mat = transition_matrix(labels)
        total = mat.sum()
        assert total == len(labels) - 1
