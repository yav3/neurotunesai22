"""Tests for audio thumbnailing and structure analysis."""
import numpy as np
import pytest

from models.music_ir.audio_thumbnailing import (
    compute_ssm, enhance_ssm, novelty_curve, detect_boundaries,
    repetition_score, find_thumbnail, structure_thumbnail,
    information_density, representativeness, fade_envelope,
    analyze_and_thumbnail, Thumbnail, SSMAnalysis,
)


def _repeating_features(n_frames=100, n_dims=12, n_repeats=4):
    seg_len = n_frames // n_repeats
    base = np.random.RandomState(42).randn(seg_len, n_dims)
    return np.tile(base, (n_repeats, 1))[:n_frames]


class TestSSM:
    def test_shape(self):
        F = np.random.randn(50, 10)
        ssm = compute_ssm(F)
        assert ssm.shape == (50, 50)

    def test_diagonal_ones(self):
        F = np.random.randn(30, 5)
        ssm = compute_ssm(F, metric="cosine")
        np.testing.assert_allclose(np.diag(ssm), 1.0, atol=1e-10)

    def test_symmetric(self):
        F = np.random.randn(20, 8)
        ssm = compute_ssm(F)
        np.testing.assert_allclose(ssm, ssm.T, atol=1e-10)

    def test_euclidean(self):
        F = np.random.randn(15, 4)
        ssm = compute_ssm(F, metric="euclidean")
        assert ssm.shape == (15, 15)
        assert ssm.min() >= 0.0

    def test_1d_input(self):
        F = np.random.randn(20)
        ssm = compute_ssm(F)
        assert ssm.shape == (20, 20)


class TestEnhance:
    def test_same_shape(self):
        ssm = compute_ssm(np.random.randn(30, 5))
        enhanced = enhance_ssm(ssm)
        assert enhanced.shape == ssm.shape

    def test_diagonal_preserved(self):
        ssm = compute_ssm(np.random.randn(20, 5))
        enhanced = enhance_ssm(ssm, filter_size=3)
        np.testing.assert_allclose(np.diag(enhanced), np.diag(ssm), atol=0.3)


class TestNovelty:
    def test_length(self):
        ssm = np.eye(40)
        nov = novelty_curve(ssm, kernel_size=8)
        assert len(nov) == 40

    def test_nonnegative(self):
        ssm = compute_ssm(np.random.randn(50, 10))
        nov = novelty_curve(ssm)
        assert np.all(nov >= 0)

    def test_block_diagonal_shape(self):
        ssm = np.zeros((40, 40))
        ssm[:20, :20] = 1.0
        ssm[20:, 20:] = 1.0
        nov = novelty_curve(ssm, kernel_size=8)
        assert len(nov) == 40
        assert np.all(np.isfinite(nov))


class TestBoundaries:
    def test_includes_start(self):
        nov = np.zeros(50)
        nov[20] = 0.8
        bounds = detect_boundaries(nov)
        assert bounds[0] == 0

    def test_includes_end(self):
        nov = np.zeros(50)
        bounds = detect_boundaries(nov)
        assert bounds[-1] == 49


class TestRepetition:
    def test_repeating_high_score(self):
        F = _repeating_features(80, 10, 4)
        ssm = compute_ssm(F)
        score = repetition_score(ssm, 0, 20)
        assert score > 0.3

    def test_zero_length(self):
        ssm = np.eye(10)
        assert repetition_score(ssm, 0, 0) == 0.0


class TestThumbnail:
    def test_returns_thumbnail(self):
        F = _repeating_features(60, 8, 3)
        ssm = compute_ssm(F)
        thumb = find_thumbnail(ssm, target_length=20)
        assert isinstance(thumb, Thumbnail)
        assert thumb.duration_frames == 20

    def test_structure_thumbnail(self):
        F = _repeating_features(60, 8, 3)
        ssm = compute_ssm(F)
        bounds = [0, 20, 40, 59]
        thumb = structure_thumbnail(ssm, bounds, target_length=20)
        assert isinstance(thumb, Thumbnail)
        assert thumb.method == "structure"


class TestMetrics:
    def test_information_density_nonneg(self):
        F = np.random.randn(50, 10)
        d = information_density(F, 0, 20)
        assert d >= 0.0

    def test_representativeness_range(self):
        F = np.random.randn(30, 5)
        ssm = compute_ssm(F)
        r = representativeness(ssm, 5, 10)
        assert -1.0 <= r <= 1.0


class TestFadeEnvelope:
    def test_no_fade(self):
        env = fade_envelope(100)
        np.testing.assert_allclose(env, 1.0)

    def test_fade_in(self):
        env = fade_envelope(100, fade_in=10)
        assert env[0] == pytest.approx(0.0)
        assert env[10] == pytest.approx(1.0, abs=0.2)

    def test_fade_out(self):
        env = fade_envelope(100, fade_out=10)
        assert env[-1] == pytest.approx(0.0)


class TestPipeline:
    def test_full_pipeline(self):
        F = _repeating_features(80, 10, 4)
        result = analyze_and_thumbnail(F, target_length=20)
        assert isinstance(result, SSMAnalysis)
        assert result.ssm.shape == (80, 80)
        assert len(result.novelty_curve) == 80
        assert len(result.segment_boundaries) >= 2
        assert isinstance(result.thumbnail, Thumbnail)
