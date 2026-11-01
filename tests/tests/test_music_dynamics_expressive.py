"""Tests for expressive dynamics analysis."""
import numpy as np
import pytest

from models.music_dynamics.expressive_dynamics import (
    dynamic_contour, detect_crescendo, detect_diminuendo, detect_subito,
    loudness_histogram, dynamic_range_score, analyze_dynamics, DynamicProfile,
)


def _ramp_up(n: int = 44100) -> np.ndarray:
    return np.linspace(0.01, 1.0, n)


def _ramp_down(n: int = 44100) -> np.ndarray:
    return np.linspace(1.0, 0.01, n)


class TestDynamicContour:
    def test_shape(self):
        env = np.ones(4096)
        contour = dynamic_contour(env, hop=512)
        assert len(contour) == 4096 // 512

    def test_empty(self):
        assert len(dynamic_contour(np.array([]))) == 0

    def test_constant(self):
        env = np.ones(4096) * 0.5
        contour = dynamic_contour(env, hop=512)
        assert np.std(contour) < 1.0


class TestCrescendo:
    def test_detects_ramp(self):
        env = _ramp_up(44100)
        cresc = detect_crescendo(env, sr=44100, min_duration=0.3, min_db_change=3.0)
        assert len(cresc) >= 1

    def test_no_crescendo_in_flat(self):
        env = np.ones(44100) * 0.5
        cresc = detect_crescendo(env, sr=44100)
        assert len(cresc) == 0


class TestDiminuendo:
    def test_detects_decay(self):
        env = _ramp_down(44100)
        dim = detect_diminuendo(env, sr=44100, min_duration=0.3, min_db_change=3.0)
        assert len(dim) >= 1

    def test_no_dim_in_flat(self):
        env = np.ones(44100) * 0.5
        dim = detect_diminuendo(env, sr=44100)
        assert len(dim) == 0


class TestSubito:
    def test_detects_jump(self):
        env = np.concatenate([np.ones(22050) * 0.01, np.ones(22050) * 1.0])
        sub = detect_subito(env, sr=44100, threshold_db=6.0)
        assert len(sub) >= 1

    def test_no_subito_in_smooth(self):
        env = np.linspace(0.5, 0.6, 44100)
        sub = detect_subito(env, sr=44100, threshold_db=10.0)
        assert len(sub) == 0


class TestLoudnessHistogram:
    def test_shape(self):
        env = np.random.default_rng(0).uniform(0.01, 1.0, 10000)
        counts, edges = loudness_histogram(env, n_bins=20)
        assert len(counts) == 20
        assert len(edges) == 21

    def test_empty(self):
        c, e = loudness_histogram(np.array([]))
        assert len(c) == 0


class TestDynamicRangeScore:
    def test_wide_range(self):
        env = np.concatenate([np.ones(5000) * 0.001, np.ones(5000) * 1.0])
        score = dynamic_range_score(env)
        assert score > 0.3

    def test_narrow_range(self):
        env = np.ones(1000) * 0.5
        score = dynamic_range_score(env)
        assert score < 0.1


class TestAnalyzeDynamics:
    def test_returns_profile(self):
        env = np.concatenate([_ramp_up(22050), _ramp_down(22050)])
        profile = analyze_dynamics(env, sr=44100)
        assert isinstance(profile, DynamicProfile)
        assert profile.dynamic_range > 0

    def test_empty(self):
        profile = analyze_dynamics(np.array([]))
        assert profile.crescendo_count == 0
