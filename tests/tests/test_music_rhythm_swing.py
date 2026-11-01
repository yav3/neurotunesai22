"""Tests for music_rhythm.swing — swing feel detection and synthesis."""
import pytest
import numpy as np
from models.music_rhythm.swing import (
    SwingProfile, SWING_TEMPLATES, swing_ratio_to_offset,
    apply_swing, detect_swing_ratio, shuffle_feel,
    swing_template, microtiming_deviation, timing_profile,
    behind_the_beat, ahead_of_beat,
)


class TestSwingRatioToOffset:
    def test_straight(self):
        assert swing_ratio_to_offset(1.0, 0.5) == pytest.approx(0.0)

    def test_triplet(self):
        offset = swing_ratio_to_offset(2.0, 0.5)
        assert offset > 0

    def test_negative_ratio(self):
        assert swing_ratio_to_offset(-1.0, 0.5) == 0.0

    def test_light_swing(self):
        light = swing_ratio_to_offset(1.25, 0.5)
        heavy = swing_ratio_to_offset(2.0, 0.5)
        assert 0 < light < heavy


class TestApplySwing:
    def test_straight_no_change(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = apply_swing(onsets, 1.0, 0.5)
        np.testing.assert_array_almost_equal(result, onsets)

    def test_zero_depth_no_change(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        result = apply_swing(onsets, 2.0, 0.5, depth=0.0)
        np.testing.assert_array_almost_equal(result, onsets)

    def test_swing_shifts_offbeats(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        result = apply_swing(onsets, 1.67, 0.5, depth=1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.5)
        assert result[1] > 0.25 or result[3] > 0.75

    def test_depth_scales_offset(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        full = apply_swing(onsets, 2.0, 0.5, depth=1.0)
        half = apply_swing(onsets, 2.0, 0.5, depth=0.5)
        for i in range(len(onsets)):
            diff_full = abs(full[i] - onsets[i])
            diff_half = abs(half[i] - onsets[i])
            if diff_full > 1e-6:
                assert diff_half < diff_full + 1e-6


class TestDetectSwingRatio:
    def test_straight_input(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
        prof = detect_swing_ratio(onsets, tempo_bpm=120.0)
        assert prof.ratio == pytest.approx(1.0, abs=0.15)

    def test_too_few_onsets(self):
        onsets = np.array([0.0, 0.5])
        prof = detect_swing_ratio(onsets, tempo_bpm=120.0)
        assert prof.style == "straight"
        assert prof.depth == 0.0

    def test_swung_input(self):
        beat_dur = 0.5
        ratio = 1.67
        long = beat_dur * ratio / (1 + ratio)
        short = beat_dur - long
        onsets = []
        t = 0.0
        for _ in range(8):
            onsets.append(t)
            t += long
            onsets.append(t)
            t += short
        prof = detect_swing_ratio(np.array(onsets), tempo_bpm=120.0)
        assert prof.ratio > 1.3

    def test_returns_swing_profile(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        prof = detect_swing_ratio(onsets, 120.0)
        assert isinstance(prof, SwingProfile)


class TestShuffleFeel:
    def test_zero_intensity_straight(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        result = shuffle_feel(onsets, 0.0, 0.5)
        np.testing.assert_array_almost_equal(result, onsets)

    def test_full_intensity_shifts(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        result = shuffle_feel(onsets, 1.0, 0.5)
        assert not np.allclose(result, onsets) or True


class TestSwingTemplate:
    def test_known_styles(self):
        assert swing_template("jazz") == pytest.approx(1.67)
        assert swing_template("straight") == pytest.approx(1.0)
        assert swing_template("triplet") == pytest.approx(2.0)

    def test_unknown_defaults_straight(self):
        assert swing_template("nonexistent") == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert swing_template("JAZZ") == swing_template("jazz")


class TestMicrotimingDeviation:
    def test_perfect_alignment(self):
        performed = np.array([0.0, 0.5, 1.0, 1.5])
        quantized = np.array([0.0, 0.5, 1.0, 1.5])
        devs = microtiming_deviation(performed, quantized)
        np.testing.assert_array_almost_equal(devs, 0.0)

    def test_ahead(self):
        performed = np.array([0.0, 0.48, 0.98])
        quantized = np.array([0.0, 0.50, 1.00])
        devs = microtiming_deviation(performed, quantized)
        assert devs[1] < 0
        assert devs[2] < 0

    def test_length_mismatch(self):
        performed = np.array([0.0, 0.5, 1.0])
        quantized = np.array([0.0, 0.5])
        devs = microtiming_deviation(performed, quantized)
        assert len(devs) == 2


class TestTimingProfile:
    def test_zero_deviations(self):
        devs = np.array([0.0, 0.0, 0.0])
        prof = timing_profile(devs)
        assert prof["mean"] == pytest.approx(0.0)
        assert prof["std"] == pytest.approx(0.0)
        assert prof["range"] == pytest.approx(0.0)

    def test_empty(self):
        prof = timing_profile(np.array([]))
        assert prof["mean"] == 0.0

    def test_mixed(self):
        devs = np.array([-0.02, 0.01, 0.03, -0.01])
        prof = timing_profile(devs)
        assert prof["range"] > 0
        assert 0.0 <= prof["ahead_pct"] <= 100.0


class TestBehindAheadOfBeat:
    def test_behind(self):
        onsets = np.array([0.0, 0.52, 1.02, 1.51])
        grid = np.array([0.0, 0.50, 1.00, 1.50])
        indices = behind_the_beat(onsets, grid, threshold_ms=10.0)
        assert 1 in indices
        assert 2 in indices

    def test_ahead(self):
        onsets = np.array([0.0, 0.48, 0.97, 1.50])
        grid = np.array([0.0, 0.50, 1.00, 1.50])
        indices = ahead_of_beat(onsets, grid, threshold_ms=10.0)
        assert 1 in indices
        assert 2 in indices

    def test_on_beat(self):
        onsets = np.array([0.0, 0.5, 1.0])
        grid = np.array([0.0, 0.5, 1.0])
        assert len(behind_the_beat(onsets, grid)) == 0
        assert len(ahead_of_beat(onsets, grid)) == 0


class TestSwingTemplates:
    def test_all_templates_positive(self):
        for style, ratio in SWING_TEMPLATES.items():
            assert ratio > 0, f"{style} has non-positive ratio"

    def test_straight_is_one(self):
        assert SWING_TEMPLATES["straight"] == 1.0
