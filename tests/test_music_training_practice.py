"""Tests for practice analytics."""
import numpy as np
import pytest

from models.music_training.practice_analytics import (
    tempo_stability, error_density, identify_problem_spots,
    learning_curve, spaced_repetition_schedule, analyze_practice,
    PracticeSession, PracticeReport,
)


class TestTempoStability:
    def test_constant(self):
        assert tempo_stability(np.ones(10) * 120.0) == pytest.approx(1.0)

    def test_variable(self):
        tempos = np.array([100.0, 120.0, 80.0, 140.0, 60.0])
        s = tempo_stability(tempos)
        assert 0.0 <= s < 1.0

    def test_single(self):
        assert tempo_stability(np.array([120.0])) == 1.0


class TestErrorDensity:
    def test_no_errors(self):
        density = error_density([], 100)
        assert np.all(density == 0.0)

    def test_clustered_errors(self):
        errors = list(range(10, 20))
        density = error_density(errors, 100, window=10)
        assert np.max(density) > 0.5

    def test_empty(self):
        assert len(error_density([], 0)) == 0


class TestProblemSpots:
    def test_finds_cluster(self):
        errors = list(range(20, 35))
        spots = identify_problem_spots(errors, 100, threshold=0.3)
        assert len(spots) >= 1

    def test_no_spots_when_sparse(self):
        errors = [0, 25, 50, 75]
        spots = identify_problem_spots(errors, 100, threshold=0.3)
        assert len(spots) == 0


class TestLearningCurve:
    def test_improving(self):
        accs = [0.5, 0.6, 0.7, 0.8, 0.85, 0.88]
        rate, asymptote = learning_curve(accs)
        assert rate >= 0.0
        assert asymptote >= 0.88

    def test_flat(self):
        accs = [0.9, 0.9, 0.9]
        rate, asymptote = learning_curve(accs)
        assert rate == pytest.approx(0.0, abs=0.1)

    def test_single(self):
        rate, asymptote = learning_curve([0.7])
        assert asymptote == pytest.approx(0.7)


class TestSpacedRepetition:
    def test_high_mastery_long_interval(self):
        days = spaced_repetition_schedule(0.95)
        assert days >= 7

    def test_low_mastery_short_interval(self):
        days = spaced_repetition_schedule(0.3)
        assert days == 1

    def test_decay_matters(self):
        fresh = spaced_repetition_schedule(0.8, last_practice_days_ago=0)
        stale = spaced_repetition_schedule(0.8, last_practice_days_ago=30)
        assert stale <= fresh


class TestAnalyzePractice:
    def test_basic_report(self):
        sessions = [
            PracticeSession(300, np.ones(20) * 120.0, [5, 15, 30], piece_id="bach"),
            PracticeSession(300, np.ones(20) * 118.0, [5, 30], piece_id="bach"),
            PracticeSession(300, np.ones(20) * 120.0, [30], piece_id="bach"),
        ]
        report = analyze_practice(sessions, total_notes=50)
        assert isinstance(report, PracticeReport)
        assert report.accuracy > 0.9
        assert report.estimated_mastery > 0.0

    def test_empty(self):
        report = analyze_practice([])
        assert report.accuracy == 0.0
