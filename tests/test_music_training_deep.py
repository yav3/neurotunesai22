"""Tests for musical skill assessment and practice optimization."""
import numpy as np
import pytest

from models.music_training.skill_assessment import (
    intonation_error_cents,
    intonation_score,
    intonation_bias,
    onset_timing_error,
    rhythm_precision_score,
    rhythm_consistency,
    SightReadingFeatures,
    sight_reading_difficulty,
    PracticeItem,
    sm2_update,
    schedule_practice_session,
    PerformanceSnapshot,
    performance_trend,
    improvement_rate,
    weakness_detection,
)


class TestIntonation:
    def test_perfect_intonation(self):
        target = np.array([440.0, 880.0, 220.0])
        played = target.copy()
        errs = intonation_error_cents(played, target)
        np.testing.assert_array_almost_equal(errs, 0.0)

    def test_one_semitone_sharp(self):
        target = np.array([440.0])
        played = np.array([440.0 * 2 ** (1 / 12)])
        errs = intonation_error_cents(played, target)
        assert abs(errs[0] - 100.0) < 0.01

    def test_flat_bias(self):
        target = np.array([440.0, 440.0])
        played = np.array([430.0, 435.0])
        errs = intonation_error_cents(played, target)
        assert intonation_bias(errs) < 0

    def test_score_all_in_tolerance(self):
        errs = np.array([10.0, -20.0, 30.0])
        assert intonation_score(errs, tolerance=50.0) == 1.0

    def test_score_half_out(self):
        errs = np.array([10.0, 100.0])
        assert intonation_score(errs, tolerance=50.0) == 0.5

    def test_zero_frequency(self):
        errs = intonation_error_cents(np.array([0.0, 440.0]), np.array([440.0, 0.0]))
        assert errs[0] == 0.0
        assert errs[1] == 0.0

    def test_empty(self):
        assert intonation_score(np.array([])) == 1.0
        assert intonation_bias(np.array([])) == 0.0


class TestRhythm:
    def test_perfect_timing(self):
        grid = np.array([0.0, 0.5, 1.0, 1.5])
        played = grid.copy()
        errs = onset_timing_error(played, grid)
        np.testing.assert_array_almost_equal(errs, 0.0)

    def test_early_late(self):
        grid = np.array([0.0, 1.0])
        played = np.array([0.02, 0.98])
        errs = onset_timing_error(played, grid)
        assert errs[0] == pytest.approx(0.02)
        assert errs[1] == pytest.approx(0.02)

    def test_precision_all_tight(self):
        errs = np.array([0.01, 0.02, 0.03])
        assert rhythm_precision_score(errs, tolerance_sec=0.05) == 1.0

    def test_precision_some_late(self):
        errs = np.array([0.01, 0.1])
        assert rhythm_precision_score(errs, tolerance_sec=0.05) == 0.5

    def test_consistency_perfect(self):
        errs = np.array([0.02, 0.02, 0.02])
        assert rhythm_consistency(errs) == pytest.approx(1.0 / (1.0 + 0.0))

    def test_empty(self):
        errs = onset_timing_error(np.array([]), np.array([1.0]))
        assert len(errs) == 0
        assert rhythm_precision_score(np.array([])) == 1.0


class TestSightReading:
    def test_easy_piece(self):
        f = SightReadingFeatures(
            note_density=1.0, pitch_range_semitones=12,
            max_interval_semitones=4, accidental_count=0,
            syncopation_count=0, tempo_bpm=80, key_signature_sharps_flats=0,
        )
        d = sight_reading_difficulty(f)
        assert 0 < d < 5

    def test_hard_piece(self):
        f = SightReadingFeatures(
            note_density=4.0, pitch_range_semitones=48,
            max_interval_semitones=24, accidental_count=20,
            syncopation_count=10, tempo_bpm=200,
            key_signature_sharps_flats=7, time_signature_complexity=3.0,
        )
        d = sight_reading_difficulty(f)
        assert d >= 8

    def test_capped_at_10(self):
        f = SightReadingFeatures(
            note_density=100, pitch_range_semitones=100,
            max_interval_semitones=100, accidental_count=100,
            syncopation_count=100, tempo_bpm=500,
            key_signature_sharps_flats=100, time_signature_complexity=100,
        )
        assert sight_reading_difficulty(f) == 9.0


class TestSM2:
    def test_perfect_review_increases_interval(self):
        item = PracticeItem(item_id="piece_1", ease_factor=2.5, interval_days=6.0, repetitions=2)
        updated = sm2_update(item, quality=5, current_time=10.0)
        assert updated.interval_days > 6.0
        assert updated.repetitions == 3

    def test_failed_review_resets(self):
        item = PracticeItem(item_id="piece_1", ease_factor=2.5, interval_days=30.0, repetitions=5)
        updated = sm2_update(item, quality=1, current_time=10.0)
        assert updated.repetitions == 0
        assert updated.interval_days == 1.0

    def test_ease_floor(self):
        item = PracticeItem(item_id="x", ease_factor=1.3)
        updated = sm2_update(item, quality=0)
        assert updated.ease_factor >= 1.3

    def test_first_review(self):
        item = PracticeItem(item_id="new")
        updated = sm2_update(item, quality=4)
        assert updated.repetitions == 1
        assert updated.interval_days == 1.0

    def test_second_review(self):
        item = PracticeItem(item_id="new", repetitions=1, interval_days=1.0)
        updated = sm2_update(item, quality=4)
        assert updated.repetitions == 2
        assert updated.interval_days == 6.0


class TestScheduler:
    def test_selects_overdue(self):
        items = [
            PracticeItem(item_id="a", due=5.0),
            PracticeItem(item_id="b", due=15.0),
            PracticeItem(item_id="c", due=8.0),
        ]
        selected = schedule_practice_session(items, current_time=10.0)
        ids = [s.item_id for s in selected]
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids

    def test_most_overdue_first(self):
        items = [
            PracticeItem(item_id="a", due=5.0),
            PracticeItem(item_id="b", due=2.0),
        ]
        selected = schedule_practice_session(items, current_time=10.0)
        assert selected[0].item_id == "b"

    def test_max_items(self):
        items = [PracticeItem(item_id=str(i), due=0.0) for i in range(30)]
        selected = schedule_practice_session(items, current_time=10.0, max_items=5)
        assert len(selected) == 5


class TestPerformance:
    def test_trend_extraction(self):
        snaps = [
            PerformanceSnapshot(timestamp=0, intonation_score=0.5),
            PerformanceSnapshot(timestamp=1, intonation_score=0.7),
            PerformanceSnapshot(timestamp=2, intonation_score=0.9),
        ]
        trend = performance_trend(snaps, "intonation_score")
        assert len(trend) == 3
        assert trend[2] == 0.9

    def test_improvement_positive(self):
        scores = np.array([0.3, 0.5, 0.7, 0.9])
        rate = improvement_rate(scores)
        assert rate > 0

    def test_improvement_stagnant(self):
        scores = np.array([0.5, 0.5, 0.5])
        rate = improvement_rate(scores)
        assert abs(rate) < 1e-10

    def test_weakness_detection(self):
        snaps = [
            PerformanceSnapshot(intonation_score=0.3, rhythm_score=0.9,
                                note_accuracy=0.8, tempo_stability=0.85,
                                dynamics_range=0.4),
        ]
        weak = weakness_detection(snaps, threshold=0.6)
        assert "intonation_score" in weak
        assert "dynamics_range" in weak
        assert "rhythm_score" not in weak

    def test_no_weaknesses(self):
        snaps = [PerformanceSnapshot(intonation_score=0.9, rhythm_score=0.9,
                                     note_accuracy=0.9, tempo_stability=0.9,
                                     dynamics_range=0.9)]
        assert weakness_detection(snaps, threshold=0.6) == []

    def test_empty_snapshots(self):
        assert weakness_detection([]) == []
