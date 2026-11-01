"""Tests for expressive microtiming analysis."""
import numpy as np
import pytest

from models.music_rhythm.microtiming import (
    quantize_to_grid, make_isochronous_grid, timing_profile,
    analyze_swing, swing_ratio_to_percentage, tempo_envelope,
    transfer_microtiming, groove_similarity, agogic_accent_positions,
)


class TestGrid:
    def test_isochronous(self):
        grid = make_isochronous_grid(0, 500, 4)
        np.testing.assert_allclose(grid, [0, 500, 1000, 1500])

    def test_subdivisions(self):
        grid = make_isochronous_grid(0, 500, 2, subdivisions=2)
        np.testing.assert_allclose(grid, [0, 250, 500, 750])

    def test_quantize_exact(self):
        grid = make_isochronous_grid(0, 500, 4)
        onsets = np.array([0.0, 500.0, 1000.0, 1500.0])
        indices, devs = quantize_to_grid(onsets, grid)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])
        np.testing.assert_allclose(devs, 0.0)

    def test_quantize_with_deviation(self):
        grid = make_isochronous_grid(0, 500, 4)
        onsets = np.array([10.0, 510.0, 990.0, 1520.0])
        indices, devs = quantize_to_grid(onsets, grid)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])
        np.testing.assert_allclose(devs, [10.0, 10.0, -10.0, 20.0])


class TestTimingProfile:
    def test_zero_deviation(self):
        grid = make_isochronous_grid(0, 500, 4)
        onsets = grid.copy()
        prof = timing_profile(onsets, grid, n_positions=4)
        assert prof.mean_deviation_ms == pytest.approx(0.0)
        assert prof.std_deviation_ms == pytest.approx(0.0)

    def test_per_position(self):
        grid = make_isochronous_grid(0, 250, 8)
        devs = np.array([10, -5, 10, -5, 10, -5, 10, -5], dtype=float)
        onsets = grid + devs
        prof = timing_profile(onsets, grid, n_positions=4)
        assert prof.per_position[0] == pytest.approx(10.0)
        assert prof.per_position[1] == pytest.approx(-5.0)


class TestSwing:
    def test_straight(self):
        onsets = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750], dtype=float)
        result = analyze_swing(onsets, beat_period_ms=500)
        assert result.swing_ratio == pytest.approx(1.0, abs=0.1)

    def test_triplet_swing(self):
        pairs = []
        for beat in range(4):
            pairs.append(beat * 500.0)
            pairs.append(beat * 500.0 + 333.0)
        onsets = np.array(pairs, dtype=float)
        result = analyze_swing(onsets, beat_period_ms=500)
        assert result.swing_ratio > 1.5

    def test_percentage(self):
        assert swing_ratio_to_percentage(1.0) == pytest.approx(50.0)
        assert swing_ratio_to_percentage(2.0) == pytest.approx(200 / 3.0, abs=0.1)


class TestTempoEnvelope:
    def test_constant_tempo(self):
        beats = np.array([0, 500, 1000, 1500, 2000], dtype=float)
        env = tempo_envelope(beats)
        assert env.mean_bpm == pytest.approx(120.0)
        assert env.tempo_variance == pytest.approx(0.0)

    def test_accelerando(self):
        beats = np.array([0, 600, 1100, 1500, 1800], dtype=float)
        env = tempo_envelope(beats)
        assert env.instantaneous_bpm[-1] > env.instantaneous_bpm[0]

    def test_single_beat(self):
        env = tempo_envelope(np.array([0.0]))
        assert env.mean_bpm == 120.0


class TestTransfer:
    def test_identity(self):
        grid = make_isochronous_grid(0, 500, 4)
        transferred = transfer_microtiming(grid, grid, grid)
        np.testing.assert_allclose(transferred, grid)

    def test_adds_deviations(self):
        source_grid = make_isochronous_grid(0, 500, 4)
        source_onsets = source_grid + np.array([10, -10, 10, -10])
        target_grid = make_isochronous_grid(0, 250, 4)
        result = transfer_microtiming(source_onsets, source_grid, target_grid)
        assert not np.allclose(result, target_grid)


class TestGrooveSimilarity:
    def test_identical(self):
        grid = make_isochronous_grid(0, 250, 8)
        onsets = grid + np.array([10, -5, 10, -5, 10, -5, 10, -5], dtype=float)
        prof = timing_profile(onsets, grid, n_positions=4)
        assert groove_similarity(prof, prof) == pytest.approx(1.0)

    def test_different(self):
        grid = make_isochronous_grid(0, 250, 8)
        onsets_a = grid + np.array([10, -5, 10, -5, 10, -5, 10, -5], dtype=float)
        onsets_b = grid + np.array([-10, 5, -10, 5, -10, 5, -10, 5], dtype=float)
        pa = timing_profile(onsets_a, grid, n_positions=4)
        pb = timing_profile(onsets_b, grid, n_positions=4)
        sim = groove_similarity(pa, pb)
        assert sim < 0.0  # opposite patterns → negative cosine


class TestAgogicAccent:
    def test_finds_late_onsets(self):
        grid = make_isochronous_grid(0, 500, 4)
        onsets = np.array([0, 520, 1000, 1500], dtype=float)
        accents = agogic_accent_positions(onsets, grid, threshold_ms=15.0)
        assert 1 in accents
        assert 0 not in accents
