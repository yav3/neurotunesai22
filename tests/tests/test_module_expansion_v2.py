"""Tests for expanded music modules: tuning, rhythm traditions, exercises, progress, segmentation, patterns."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_ethnomusicology.tuning_systems import (
    TuningSystem, cents_to_ratio, ratio_to_cents, compare_tunings,
    equal_temperament, pythagorean_tuning, just_intonation,
    meantone_temperament, maqam_intervals, raga_intervals,
    pelog_scale, slendro_scale,
)
from models.music_ethnomusicology.rhythm_traditions import (
    TalaPattern, indian_tala, african_timeline, flamenco_compas,
    aksak_pattern, polyrhythm_matrix, cross_rhythm_density,
)
from models.music_education.exercise_generator import (
    DifficultyLevel, ExerciseType, Exercise, ExerciseGenerator, grade_answer,
)
from models.music_education.progress_tracker import (
    LearningProfile, SkillAttempt,
)
from models.music_analysis.structure_segmentation import (
    SegmentType, Segment, self_similarity_matrix, novelty_curve,
    detect_boundaries, label_segments, repetition_structure,
    recurrence_plot, lag_matrix,
)
from models.music_rhythm.pattern import (
    RhythmPattern, euclidean_rhythm, necklace_equivalence,
    pattern_similarity, rhythm_complexity, pattern_from_ioi, pattern_to_ioi,
)


class TestTuningSystems:
    def test_cents_ratio_roundtrip(self):
        for cents in [0, 100, 200, 700, 1200]:
            r = cents_to_ratio(cents)
            assert abs(ratio_to_cents(r) - cents) < 1e-6

    def test_equal_temperament_12(self):
        et12 = equal_temperament(12)
        assert len(et12.intervals) == 12
        assert abs(et12.intervals[0] - 1.0) < 1e-10
        fifth = et12.intervals[7]
        assert abs(ratio_to_cents(fifth) - 700.0) < 1e-6

    def test_equal_temperament_24(self):
        et24 = equal_temperament(24)
        assert len(et24.intervals) == 24
        assert abs(ratio_to_cents(et24.intervals[1]) - 50.0) < 1e-6

    def test_pythagorean(self):
        p = pythagorean_tuning()
        assert len(p.intervals) == 12
        fifth = p.intervals[7]
        assert abs(ratio_to_cents(fifth) - 701.955) < 0.01

    def test_just_intonation_5limit(self):
        ji = just_intonation(limit=5)
        assert len(ji.intervals) == 12
        assert abs(ji.intervals[4] - 5 / 4) < 1e-10
        assert abs(ji.intervals[7] - 3 / 2) < 1e-10

    def test_just_intonation_7limit(self):
        ji7 = just_intonation(limit=7)
        assert len(ji7.intervals) == 12
        assert abs(ji7.intervals[10] - 7 / 4) < 1e-10

    def test_meantone(self):
        mt = meantone_temperament(0.25)
        assert len(mt.intervals) == 12
        major_third = mt.intervals[4]
        assert abs(major_third - 5 / 4) < 0.01

    def test_maqam_rast(self):
        rast = maqam_intervals("rast")
        assert isinstance(rast, TuningSystem)
        assert len(rast.intervals) == 8
        assert abs(rast.intervals[0] - 1.0) < 1e-10
        assert abs(ratio_to_cents(rast.intervals[-1]) - 1200.0) < 1e-6

    def test_maqam_unknown_raises(self):
        with pytest.raises(ValueError):
            maqam_intervals("nonexistent_maqam")

    def test_raga_bilawal(self):
        bilawal = raga_intervals("bilawal")
        assert isinstance(bilawal, TuningSystem)
        assert len(bilawal.intervals) == 7
        assert abs(bilawal.intervals[0] - 1.0) < 1e-10

    def test_raga_unknown_raises(self):
        with pytest.raises(ValueError):
            raga_intervals("nonexistent_raga")

    def test_pelog_scale(self):
        p = pelog_scale()
        assert isinstance(p, TuningSystem)
        assert len(p.intervals) == 7

    def test_slendro_scale(self):
        s = slendro_scale()
        assert isinstance(s, TuningSystem)
        assert len(s.intervals) == 5

    def test_compare_tunings_self(self):
        et12 = equal_temperament(12)
        assert compare_tunings(et12, et12) == 0.0

    def test_compare_tunings_different(self):
        et12 = equal_temperament(12)
        ji = just_intonation(limit=5)
        diff = compare_tunings(et12, ji)
        assert diff > 0

    def test_frequencies(self):
        et12 = equal_temperament(12)
        freqs = et12.frequencies(440.0)
        assert len(freqs) == 12
        assert abs(freqs[0] - 440.0) < 1e-10

    def test_cents_property(self):
        et12 = equal_temperament(12)
        c = et12.cents
        assert len(c) == 12
        assert abs(c[0]) < 1e-10


class TestRhythmTraditions:
    def test_indian_tala_teental(self):
        t = indian_tala("teental")
        assert t.beats == 16
        assert t.tradition == "Hindustani"
        assert sum(t.subdivisions) == t.beats

    def test_indian_tala_adi(self):
        t = indian_tala("adi")
        assert t.beats == 8
        assert t.tradition == "Carnatic"

    def test_indian_tala_unknown_raises(self):
        with pytest.raises(ValueError):
            indian_tala("nonexistent")

    def test_african_standard(self):
        tl = african_timeline("standard")
        assert len(tl) == 12
        assert sum(tl) == 7

    def test_african_son_clave(self):
        tl = african_timeline("son clave")
        assert len(tl) == 16
        assert sum(tl) == 5

    def test_african_unknown_raises(self):
        with pytest.raises(ValueError):
            african_timeline("nonexistent")

    def test_flamenco_bulerias(self):
        c = flamenco_compas("bulerias")
        assert len(c) == 12

    def test_flamenco_unknown_raises(self):
        with pytest.raises(ValueError):
            flamenco_compas("nonexistent")

    def test_aksak_7_8_a(self):
        a = aksak_pattern("7/8 a")
        assert sum(a) == 7

    def test_aksak_unknown_raises(self):
        with pytest.raises(ValueError):
            aksak_pattern("nonexistent")

    def test_polyrhythm_matrix_shape(self):
        patterns = [(1, 0, 1, 0), (1, 0, 0, 1, 0, 0)]
        m = polyrhythm_matrix(patterns)
        assert m.shape == (2, 6)
        assert m[0, 0] == 1
        assert m[0, 4] == 0

    def test_polyrhythm_matrix_empty(self):
        m = polyrhythm_matrix([])
        assert m.size == 0

    def test_cross_rhythm_density_identical(self):
        a = (1, 0, 1, 0)
        assert cross_rhythm_density(a, a) == 0.0

    def test_cross_rhythm_density_disjoint(self):
        a = (1, 0, 0, 0)
        b = (0, 0, 1, 0)
        assert cross_rhythm_density(a, b) == 1.0

    def test_cross_rhythm_density_empty(self):
        assert cross_rhythm_density((), ()) == 0.0


class TestExerciseGenerator:
    def test_interval_exercise(self):
        gen = ExerciseGenerator(seed=42)
        ex = gen.interval_exercise(DifficultyLevel.BEGINNER)
        assert ex.exercise_type == ExerciseType.INTERVAL_RECOGNITION
        assert ex.difficulty == DifficultyLevel.BEGINNER
        assert ex.answer is not None

    def test_chord_exercise(self):
        gen = ExerciseGenerator(seed=42)
        ex = gen.chord_exercise(DifficultyLevel.INTERMEDIATE)
        assert ex.exercise_type == ExerciseType.CHORD_IDENTIFICATION
        assert ex.answer is not None

    def test_rhythm_exercise(self):
        gen = ExerciseGenerator(seed=42)
        ex = gen.rhythm_exercise(DifficultyLevel.ADVANCED)
        assert ex.exercise_type == ExerciseType.RHYTHM_READING

    def test_melodic_dictation(self):
        gen = ExerciseGenerator(seed=42)
        ex = gen.melodic_dictation(DifficultyLevel.BEGINNER)
        assert ex.exercise_type == ExerciseType.MELODIC_DICTATION

    def test_grade_correct(self):
        gen = ExerciseGenerator(seed=42)
        ex = gen.interval_exercise(DifficultyLevel.BEGINNER)
        correct, score, feedback = grade_answer(ex, ex.answer)
        assert correct
        assert score == 1.0

    def test_grade_wrong(self):
        gen = ExerciseGenerator(seed=42)
        ex = gen.interval_exercise(DifficultyLevel.BEGINNER)
        correct, score, feedback = grade_answer(ex, "DEFINITELY_WRONG_ANSWER_XYZ")
        assert not correct
        assert score == 0.0

    def test_time_limit_scales_with_difficulty(self):
        gen = ExerciseGenerator(seed=42)
        beginner = gen.interval_exercise(DifficultyLevel.BEGINNER)
        expert = gen.interval_exercise(DifficultyLevel.EXPERT)
        assert beginner.time_limit_seconds >= expert.time_limit_seconds

    def test_hints_provided(self):
        gen = ExerciseGenerator(seed=42)
        ex = gen.interval_exercise(DifficultyLevel.BEGINNER)
        assert isinstance(ex.hints, list)
        assert len(ex.hints) > 0

    def test_generate_batch(self):
        gen = ExerciseGenerator(seed=42)
        batch = gen.generate_batch(ExerciseType.INTERVAL_RECOGNITION, DifficultyLevel.BEGINNER, 5)
        assert len(batch) == 5
        for ex in batch:
            assert ex.exercise_type == ExerciseType.INTERVAL_RECOGNITION


class TestProgressTracker:
    def test_record_and_level(self):
        profile = LearningProfile()
        profile.record_attempt("intervals", True, 1.0)
        profile.record_attempt("intervals", True, 2.0)
        level = profile.skill_level("intervals")
        assert 0.0 <= level <= 1.0

    def test_empty_skill_level(self):
        profile = LearningProfile()
        assert profile.skill_level("unknown") == 0.0

    def test_weak_strong_areas(self):
        profile = LearningProfile()
        for i in range(5):
            profile.record_attempt("intervals", True, float(i))
            profile.record_attempt("chords", False, float(i) + 0.5)
        weak = profile.weak_areas()
        strong = profile.strong_areas()
        assert "chords" in weak
        assert "intervals" in strong

    def test_recommend_next(self):
        profile = LearningProfile()
        profile.record_attempt("intervals", True, 1.0)
        profile.record_attempt("chords", False, 2.0)
        rec = profile.recommend_next(["intervals", "chords"])
        assert rec == "chords"

    def test_recommend_next_empty_list(self):
        profile = LearningProfile()
        rec = profile.recommend_next([])
        assert rec == ""

    def test_mastery_curve(self):
        profile = LearningProfile()
        for i in range(10):
            profile.record_attempt("intervals", i > 3, float(i))
        curve = profile.mastery_curve("intervals")
        assert len(curve) == 10
        assert curve[-1] >= curve[0]

    def test_streak(self):
        profile = LearningProfile()
        profile.record_attempt("intervals", True, 1.0)
        profile.record_attempt("intervals", True, 2.0)
        profile.record_attempt("intervals", True, 3.0)
        assert profile.streak("intervals") == 3

    def test_streak_broken(self):
        profile = LearningProfile()
        profile.record_attempt("intervals", True, 1.0)
        profile.record_attempt("intervals", False, 2.0)
        profile.record_attempt("intervals", True, 3.0)
        assert profile.streak("intervals") == 1

    def test_total_practice_time(self):
        profile = LearningProfile()
        profile.record_attempt("intervals", True, 10.0)
        profile.record_attempt("intervals", True, 20.0)
        assert profile.total_practice_time() == 10.0


class TestStructureSegmentation:
    def test_ssm_cosine_identity(self):
        features = np.eye(10)
        ssm = self_similarity_matrix(features, "cosine")
        assert ssm.shape == (10, 10)
        np.testing.assert_allclose(np.diag(ssm), 1.0, atol=1e-6)

    def test_ssm_euclidean(self):
        features = np.eye(5)
        ssm = self_similarity_matrix(features, "euclidean")
        assert ssm.shape == (5, 5)
        assert ssm[0, 0] == 0.0

    def test_novelty_curve_shape(self):
        ssm = np.random.RandomState(42).randn(50, 50)
        ssm = (ssm + ssm.T) / 2
        nc = novelty_curve(ssm, kernel_size=4)
        assert nc.shape == (50,)
        assert np.all(nc >= 0)
        assert np.max(nc) <= 1.0 + 1e-10

    def test_novelty_curve_small(self):
        ssm = np.ones((3, 3))
        nc = novelty_curve(ssm, kernel_size=8)
        assert nc.shape == (3,)

    def test_detect_boundaries_basic(self):
        novelty_arr = np.zeros(30)
        novelty_arr[10] = 0.8
        novelty_arr[20] = 0.6
        bounds = detect_boundaries(novelty_arr, threshold=0.3, min_segment_frames=5)
        assert 0 in bounds
        assert 10 in bounds
        assert 20 in bounds

    def test_detect_boundaries_empty_novelty(self):
        bounds = detect_boundaries(np.zeros(10))
        assert bounds == [0]

    def test_label_segments(self):
        rng = np.random.RandomState(0)
        features = np.vstack([rng.randn(10, 8), rng.randn(10, 8) + 5,
                              rng.randn(10, 8), rng.randn(10, 8) + 5])
        boundaries = [0, 10, 20, 30]
        segs = label_segments(features, boundaries, hop_time=0.5, n_clusters=2)
        assert len(segs) == 4
        for s in segs:
            assert isinstance(s, Segment)
            assert s.duration > 0

    def test_label_segments_empty(self):
        features = np.random.randn(20, 8)
        assert label_segments(features, []) == []

    def test_repetition_structure(self):
        segs = [
            Segment(SegmentType.VERSE, 0, 10, label="A"),
            Segment(SegmentType.CHORUS, 10, 20, label="B"),
            Segment(SegmentType.VERSE, 20, 30, label="A"),
        ]
        rep = repetition_structure(segs)
        assert rep["A"] == [0, 2]
        assert rep["B"] == [1]

    def test_recurrence_plot(self):
        features = np.eye(5)
        rp = recurrence_plot(features, threshold=0.5)
        assert rp.shape == (5, 5)
        np.testing.assert_array_equal(np.diag(rp), 1.0)

    def test_lag_matrix(self):
        ssm = np.ones((5, 5))
        L = lag_matrix(ssm)
        assert L.shape == (5, 5)
        assert L[0, 0] == ssm[0, 0]

    def test_segment_duration(self):
        s = Segment(SegmentType.CHORUS, 5.0, 15.0)
        assert s.duration == 10.0


class TestRhythmPattern:
    def test_euclidean_rhythm_3_8(self):
        p = euclidean_rhythm(3, 8)
        assert p.steps == 8
        assert len(p.onsets) == 3
        assert sum(p.binary) == 3

    def test_euclidean_rhythm_4_12(self):
        p = euclidean_rhythm(4, 12)
        assert p.steps == 12
        assert len(p.onsets) == 4
        ioi = pattern_to_ioi(p)
        assert all(i == 3 for i in ioi)

    def test_euclidean_rhythm_edge_0(self):
        p = euclidean_rhythm(0, 8)
        assert len(p.onsets) == 0

    def test_euclidean_rhythm_full(self):
        p = euclidean_rhythm(8, 8)
        assert all(s == 1 for s in p.binary)

    def test_necklace_equivalence_rotation(self):
        a = RhythmPattern(8, (0, 3, 6))
        b = RhythmPattern(8, (1, 4, 7))
        assert necklace_equivalence(a, b)

    def test_necklace_equivalence_different_count(self):
        a = RhythmPattern(8, (0, 2))
        b = RhythmPattern(8, (0, 3, 6))
        assert not necklace_equivalence(a, b)

    def test_pattern_similarity_identical(self):
        p = euclidean_rhythm(3, 8)
        assert pattern_similarity(p, p) == 1.0

    def test_pattern_density(self):
        p = euclidean_rhythm(3, 8)
        assert abs(p.density - 3 / 8) < 1e-10

    def test_rhythm_complexity_simple(self):
        p = euclidean_rhythm(4, 8)
        c = rhythm_complexity(p)
        assert c >= 0.0

    def test_ioi_roundtrip(self):
        p = euclidean_rhythm(3, 8)
        ioi = pattern_to_ioi(p)
        p2 = pattern_from_ioi(ioi)
        assert p2.steps == p.steps
        assert len(p2.onsets) == len(p.onsets)

    def test_pattern_from_ioi_basic(self):
        p = pattern_from_ioi([3, 3, 2])
        assert p.steps == 8
        assert len(p.onsets) == 3
        assert p.binary[0] == 1

    def test_binary_representation(self):
        p = RhythmPattern(8, (0, 2, 5))
        assert p.binary == (1, 0, 1, 0, 0, 1, 0, 0)

    def test_ioi_wrap_around(self):
        p = RhythmPattern(8, (0, 3, 6))
        ioi = p.ioi
        assert sum(ioi) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
