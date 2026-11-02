"""Tests for music_emotion.arousal_dynamics."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_emotion.arousal_dynamics import (
    ArousalCurve,
    ArousalSegment,
    arousal_contrast,
    compute_arousal_curve,
    loudness_to_arousal,
    onset_density_to_arousal,
    segment_arousal,
    smooth_arousal,
    spectral_centroid_to_arousal,
    tempo_to_arousal,
)


class TestLoudnessToArousal:
    def test_silence_maps_to_zero(self):
        rms = np.array([1e-10, 1e-10, 1e-10])
        result = loudness_to_arousal(rms)
        assert np.allclose(result, 0.0, atol=0.05)

    def test_loud_maps_near_one(self):
        rms = np.array([1.0, 1.0])
        result = loudness_to_arousal(rms)
        assert np.all(result >= 0.9)

    def test_monotonic(self):
        rms = np.array([0.001, 0.01, 0.1, 0.5, 1.0])
        result = loudness_to_arousal(rms)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-9

    def test_output_clipped(self):
        rms = np.array([0.0, 100.0])
        result = loudness_to_arousal(rms)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestTempoToArousal:
    def test_slow_tempo(self):
        assert tempo_to_arousal(40) == pytest.approx(0.0)

    def test_fast_tempo(self):
        assert tempo_to_arousal(200) == pytest.approx(1.0)

    def test_medium_tempo(self):
        r = tempo_to_arousal(120)
        assert 0.0 < r < 1.0

    def test_clipping(self):
        assert tempo_to_arousal(10) == pytest.approx(0.0)
        assert tempo_to_arousal(300) == pytest.approx(1.0)


class TestSpectralCentroidToArousal:
    def test_low_centroid(self):
        c = np.array([100.0])
        r = spectral_centroid_to_arousal(c)
        assert r[0] < 0.1

    def test_high_centroid(self):
        c = np.array([8000.0])
        r = spectral_centroid_to_arousal(c)
        assert r[0] == pytest.approx(1.0)


class TestOnsetDensityToArousal:
    def test_zero_density(self):
        d = np.array([0.0])
        r = onset_density_to_arousal(d)
        assert r[0] == pytest.approx(0.0)

    def test_max_density(self):
        d = np.array([15.0])
        r = onset_density_to_arousal(d)
        assert r[0] == pytest.approx(1.0)


class TestComputeArousalCurve:
    def test_basic_curve(self):
        times = np.linspace(0, 10, 100)
        rms = np.linspace(0.01, 0.5, 100)
        curve = compute_arousal_curve(times, rms=rms)
        assert isinstance(curve, ArousalCurve)
        assert len(curve.arousal) == 100
        assert 0.0 <= curve.mean_arousal <= 1.0

    def test_all_features(self):
        n = 50
        times = np.linspace(0, 5, n)
        rms = np.random.default_rng(0).uniform(0.01, 0.5, n)
        centroid = np.random.default_rng(1).uniform(200, 4000, n)
        density = np.random.default_rng(2).uniform(0, 10, n)
        tension = np.random.default_rng(3).uniform(0, 0.8, n)
        curve = compute_arousal_curve(
            times, rms=rms, spectral_centroid=centroid,
            onset_density=density, tension=tension,
        )
        assert curve.arousal.shape == (n,)
        assert np.all(curve.arousal >= 0.0)
        assert np.all(curve.arousal <= 1.0)

    def test_no_features_gives_zeros(self):
        times = np.linspace(0, 1, 10)
        curve = compute_arousal_curve(times)
        assert np.allclose(curve.arousal, 0.0)

    def test_custom_weights(self):
        times = np.linspace(0, 5, 50)
        rms = np.ones(50) * 0.3
        weights = {"loudness": 1.0, "brightness": 0.0, "density": 0.0, "tension": 0.0}
        curve = compute_arousal_curve(times, rms=rms, weights=weights)
        assert curve.mean_arousal > 0

    def test_peaks_detected(self):
        times = np.linspace(0, 10, 200)
        rms = 0.1 + 0.4 * np.sin(2 * np.pi * times / 5)
        rms = np.clip(rms, 0.01, 1.0)
        curve = compute_arousal_curve(times, rms=rms)
        assert len(curve.peak_times) >= 1


class TestSegmentArousal:
    def test_basic_segmentation(self):
        times = np.linspace(0, 10, 100)
        rms = np.linspace(0.01, 0.5, 100)
        curve = compute_arousal_curve(times, rms=rms)
        segments = segment_arousal(curve, n_segments=4)
        assert len(segments) == 4
        assert all(isinstance(s, ArousalSegment) for s in segments)

    def test_rising_trend(self):
        times = np.linspace(0, 10, 100)
        rms = np.linspace(0.01, 0.8, 100)
        curve = compute_arousal_curve(times, rms=rms)
        segments = segment_arousal(curve, n_segments=2)
        assert segments[0].trend == "rising"

    def test_short_signal(self):
        times = np.array([0.0])
        curve = ArousalCurve(
            times=times, arousal=np.array([0.5]),
            mean_arousal=0.5, std_arousal=0.0,
            peak_times=[], trough_times=[],
        )
        segments = segment_arousal(curve)
        assert segments == []


class TestArousalContrast:
    def test_constant_signal_zero_contrast(self):
        times = np.linspace(0, 10, 100)
        curve = ArousalCurve(
            times=times, arousal=np.ones(100) * 0.5,
            mean_arousal=0.5, std_arousal=0.0,
            peak_times=[], trough_times=[],
        )
        c = arousal_contrast(curve, window=10)
        assert np.allclose(c, 0.0, atol=1e-10)

    def test_step_signal_has_contrast(self):
        arousal = np.concatenate([np.zeros(50), np.ones(50)])
        times = np.linspace(0, 10, 100)
        curve = ArousalCurve(
            times=times, arousal=arousal,
            mean_arousal=0.5, std_arousal=0.5,
            peak_times=[], trough_times=[],
        )
        c = arousal_contrast(curve, window=10)
        assert np.max(c) > 0.5


class TestSmoothArousal:
    def test_smoothing_reduces_variance(self):
        rng = np.random.default_rng(42)
        noisy = 0.5 + 0.3 * rng.standard_normal(100)
        smoothed = smooth_arousal(noisy, window=5)
        assert np.std(smoothed) < np.std(noisy)

    def test_preserves_length(self):
        arr = np.random.default_rng(0).uniform(0, 1, 50)
        result = smooth_arousal(arr, window=7)
        assert len(result) == 50

    def test_short_signal_returned_unchanged(self):
        arr = np.array([0.5, 0.6])
        result = smooth_arousal(arr, window=5)
        assert np.array_equal(result, arr)
