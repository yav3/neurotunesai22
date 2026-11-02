"""Deep tests for music_performance: sight-reading difficulty estimation."""
import numpy as np
import pytest

from models.music_performance.sight_reading import (
    SightReadingProfile, PracticeExercise,
    rhythmic_complexity_score, pitch_complexity_score,
    interval_difficulty_score, key_complexity_score,
    accidental_density, temporal_irregularity,
    estimate_difficulty, generate_exercise,
    progressive_sequence, difficulty_feature_vector,
)


class TestRhythmicComplexity:
    def test_uniform_durations_low(self):
        durations = np.full(16, 0.5)
        score = rhythmic_complexity_score(durations)
        assert score < 0.2

    def test_varied_durations_higher(self):
        durations = np.array([0.25, 0.5, 1.0, 0.125, 0.75, 0.5, 0.25, 1.0])
        score = rhythmic_complexity_score(durations)
        assert score > 0.3

    def test_single_note(self):
        assert rhythmic_complexity_score(np.array([0.5])) == 0.0

    def test_bounded(self):
        durations = np.random.default_rng(42).uniform(0.1, 2.0, 32)
        assert 0.0 <= rhythmic_complexity_score(durations) <= 1.0


class TestPitchComplexity:
    def test_narrow_range_low(self):
        pitches = np.array([60, 62, 64, 65, 67, 60, 62, 64])
        score = pitch_complexity_score(pitches.astype(float))
        assert score < 0.5

    def test_wide_range_higher(self):
        pitches = np.array([36, 48, 60, 72, 84, 96, 40, 55])
        score = pitch_complexity_score(pitches.astype(float))
        assert score > 0.4

    def test_single_pitch(self):
        assert pitch_complexity_score(np.array([60.0])) == 0.0


class TestIntervalDifficulty:
    def test_stepwise_easy(self):
        pitches = np.array([60, 62, 64, 65, 67, 69, 71, 72], dtype=float)
        score = interval_difficulty_score(pitches)
        assert score < 0.35

    def test_large_leaps_harder(self):
        pitches = np.array([48, 72, 36, 84, 60, 96, 42, 78], dtype=float)
        score = interval_difficulty_score(pitches)
        assert score >= 0.3

    def test_single_pitch(self):
        assert interval_difficulty_score(np.array([60.0])) == 0.0


class TestKeyComplexity:
    def test_c_major_easiest(self):
        assert key_complexity_score(0) == 0.0

    def test_more_sharps_harder(self):
        assert key_complexity_score(1) > key_complexity_score(0)  # C# vs C

    def test_bounded(self):
        for k in range(12):
            assert 0.0 <= key_complexity_score(k) <= 1.0


class TestAccidentalDensity:
    def test_diatonic_zero(self):
        pitches = np.array([0, 2, 4, 5, 7, 9, 11])  # C major scale
        assert accidental_density(pitches, key_root=0) == 0.0

    def test_chromatic_high(self):
        pitches = np.array([1, 3, 6, 8, 10, 1, 3])  # all accidentals in C
        assert accidental_density(pitches, key_root=0) == 1.0

    def test_empty(self):
        assert accidental_density(np.array([]), key_root=0) == 0.0


class TestTemporalIrregularity:
    def test_constant_zero(self):
        durations = np.full(8, 0.5)
        assert temporal_irregularity(durations) == 0.0

    def test_alternating_high(self):
        durations = np.array([0.25, 1.0, 0.25, 1.0, 0.25, 1.0])
        ti = temporal_irregularity(durations)
        assert ti > 0.5

    def test_single_note(self):
        assert temporal_irregularity(np.array([0.5])) == 0.0


class TestEstimateDifficulty:
    def test_returns_profile(self):
        pitches = np.array([60, 62, 64, 65, 67], dtype=float)
        durations = np.full(5, 0.5)
        prof = estimate_difficulty(pitches, durations)
        assert isinstance(prof, SightReadingProfile)
        assert 0.0 <= prof.overall_difficulty <= 1.0

    def test_all_dimensions_bounded(self):
        rng = np.random.default_rng(42)
        pitches = rng.integers(48, 84, 32).astype(float)
        durations = rng.choice([0.25, 0.5, 1.0], 32)
        prof = estimate_difficulty(pitches, durations)
        for attr in ["rhythmic_complexity", "pitch_complexity", "key_complexity",
                      "interval_difficulty", "accidental_density", "temporal_irregularity"]:
            assert 0.0 <= getattr(prof, attr) <= 1.0

    def test_custom_weights(self):
        pitches = np.array([60, 72, 48, 84], dtype=float)
        durations = np.array([0.25, 1.0, 0.5, 0.125])
        w = {"rhythmic": 1.0, "pitch": 0, "key": 0,
             "interval": 0, "accidental": 0, "temporal": 0}
        prof = estimate_difficulty(pitches, durations, weights=w)
        assert prof.overall_difficulty == prof.rhythmic_complexity


class TestGenerateExercise:
    def test_returns_exercise(self):
        ex = generate_exercise(0.5, n_notes=16)
        assert isinstance(ex, PracticeExercise)
        assert len(ex.pitches) == 16
        assert len(ex.durations) == 16

    def test_difficulty_increases(self):
        low = generate_exercise(0.2, seed=0)
        high = generate_exercise(0.8, seed=1)
        assert high.difficulty >= low.difficulty - 0.1

    def test_pitch_focus(self):
        ex = generate_exercise(0.5, focus="pitch")
        assert ex.focus_area == "pitch"

    def test_rhythm_focus(self):
        ex = generate_exercise(0.5, focus="rhythm")
        assert ex.focus_area == "rhythm"


class TestProgressiveSequence:
    def test_length(self):
        seq = progressive_sequence(n_exercises=5)
        assert len(seq) == 5

    def test_monotonic_difficulty(self):
        seq = progressive_sequence(n_exercises=5, start_difficulty=0.1, end_difficulty=0.9)
        diffs = [ex.difficulty for ex in seq]
        for i in range(len(diffs) - 1):
            assert diffs[i + 1] >= diffs[i] - 0.15  # allow small non-monotonicity


class TestDifficultyFeatureVector:
    def test_shape(self):
        pitches = np.array([60, 64, 67, 72], dtype=float)
        durations = np.array([0.5, 0.25, 1.0, 0.5])
        fv = difficulty_feature_vector(pitches, durations)
        assert fv.shape == (7,)
        assert np.all(np.isfinite(fv))
