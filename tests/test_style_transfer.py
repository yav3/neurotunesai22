"""Tests for historical style transfer module."""
import numpy as np
import pytest
from models.music_historical.style_transfer import (
    PERIOD_STYLES, extract_period_features, style_distance,
    interpolate_styles, closest_period, transfer_style,
    constrain_to_vocab, scale_rhythm_complexity, apply_dynamics_range,
    add_ornamentation, period_timeline, interpolate_by_year,
    StyleFeatureVector,
)


class TestPeriodFeatures:
    def test_all_periods_have_features(self):
        for name, style in PERIOD_STYLES.items():
            f = extract_period_features(style)
            assert 0 <= f.harmonic_entropy <= 1
            assert 0 <= f.rhythm_complexity <= 1

    def test_modern_more_complex(self):
        ren = extract_period_features(PERIOD_STYLES["renaissance"])
        mod = extract_period_features(PERIOD_STYLES["modern"])
        assert mod.rhythm_complexity > ren.rhythm_complexity

    def test_roundtrip(self):
        f = extract_period_features(PERIOD_STYLES["baroque"])
        arr = f.to_array()
        f2 = StyleFeatureVector.from_array(arr)
        np.testing.assert_allclose(f.to_array(), f2.to_array())


class TestStyleDistance:
    def test_self_zero(self):
        f = extract_period_features(PERIOD_STYLES["classical"])
        assert style_distance(f, f) < 1e-10

    def test_different_positive(self):
        f1 = extract_period_features(PERIOD_STYLES["renaissance"])
        f2 = extract_period_features(PERIOD_STYLES["modern"])
        assert style_distance(f1, f2) > 0.1


class TestInterpolation:
    def test_alpha_zero(self):
        f1 = extract_period_features(PERIOD_STYLES["baroque"])
        f2 = extract_period_features(PERIOD_STYLES["classical"])
        result = interpolate_styles(f1, f2, 0.0)
        np.testing.assert_allclose(result.to_array(), f1.to_array())

    def test_alpha_one(self):
        f1 = extract_period_features(PERIOD_STYLES["baroque"])
        f2 = extract_period_features(PERIOD_STYLES["classical"])
        result = interpolate_styles(f1, f2, 1.0)
        np.testing.assert_allclose(result.to_array(), f2.to_array())

    def test_midpoint(self):
        f1 = extract_period_features(PERIOD_STYLES["baroque"])
        f2 = extract_period_features(PERIOD_STYLES["romantic"])
        mid = interpolate_styles(f1, f2, 0.5)
        d1 = style_distance(mid, f1)
        d2 = style_distance(mid, f2)
        assert abs(d1 - d2) < 0.15


class TestClosestPeriod:
    def test_exact_match(self):
        f = extract_period_features(PERIOD_STYLES["baroque"])
        assert closest_period(f) == "baroque"


class TestConstrainVocab:
    def test_major_scale(self):
        pitches = np.array([60, 61, 62, 63, 64, 65, 66])
        vocab = np.array([0, 2, 4, 5, 7, 9, 11])
        result = constrain_to_vocab(pitches, vocab)
        for p in result:
            assert int(p) % 12 in vocab


class TestTransferStyle:
    def test_basic(self):
        pitches = np.array([60, 62, 64, 65, 67, 69, 71, 72])
        durations = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        dynamics = np.array([0.5] * 8)
        result = transfer_style(pitches, durations, dynamics, "baroque")
        assert len(result.pitches) == 8
        assert 0 <= result.style_similarity <= 1

    def test_unknown_period(self):
        p = np.array([60, 64, 67])
        d = np.array([1.0, 1.0, 1.0])
        dy = np.array([0.5, 0.5, 0.5])
        result = transfer_style(p, d, dy, "nonexistent")
        assert result.style_similarity == 0.0


class TestDynamicsRange:
    def test_rescale(self):
        dyn = np.array([0.0, 0.5, 1.0])
        result = apply_dynamics_range(dyn, 0.2, 0.8)
        assert abs(result[0] - 0.2) < 1e-10
        assert abs(result[-1] - 0.8) < 1e-10


class TestOrnamentation:
    def test_adds_notes(self):
        pitches = np.array([60, 62, 64, 65, 67])
        rng = np.random.default_rng(42)
        result = add_ornamentation(pitches, density=0.8, rng=rng)
        assert len(result) >= len(pitches)


class TestTimeline:
    def test_order(self):
        tl = period_timeline()
        for i in range(len(tl) - 1):
            assert tl[i][1] <= tl[i + 1][1]

    def test_interpolate_1700(self):
        f = interpolate_by_year(1700)
        assert f.harmonic_entropy > 0

    def test_interpolate_before(self):
        f = interpolate_by_year(1200)
        ren = extract_period_features(PERIOD_STYLES["renaissance"])
        np.testing.assert_allclose(f.to_array(), ren.to_array())
