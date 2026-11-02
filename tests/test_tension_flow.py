"""Tests for tension flow analysis."""
import numpy as np
import pytest
from models.music_structure.tension_flow import (
    TensionPoint, Climax, TensionArc,
    harmonic_tension, dissonance_tension, register_tension,
    rhythmic_tension, dynamic_tension,
    compute_tension_curve, detect_climaxes, classify_arc, tension_arc,
    tension_similarity, tension_derivative, cumulative_tension, tension_entropy,
)


class TestHarmonicTension:
    def test_tonic_low(self):
        t = harmonic_tension(np.array([60, 64, 67]))  # C major
        assert t < 0.5

    def test_empty(self):
        assert harmonic_tension(np.array([])) == 0.0

    def test_dissonant_higher(self):
        consonant = harmonic_tension(np.array([60, 67]))  # P5
        dissonant = harmonic_tension(np.array([60, 66]))  # tritone
        assert dissonant > consonant


class TestDissonanceTension:
    def test_unison(self):
        assert dissonance_tension(np.array([0])) == 0.0

    def test_tritone_high(self):
        t = dissonance_tension(np.array([6]))
        assert t > 0.5

    def test_fifth_low(self):
        t = dissonance_tension(np.array([7]))
        assert t < 0.2


class TestRegisterTension:
    def test_middle(self):
        t = register_tension(np.array([60]))
        assert 0.4 < t < 0.6

    def test_high_more(self):
        low = register_tension(np.array([48]))
        high = register_tension(np.array([84]))
        assert high > low


class TestTensionCurve:
    def test_basic(self):
        chords = [np.array([60, 64, 67]), np.array([65, 69, 72]),
                  np.array([60, 66, 69]), np.array([60, 64, 67])]
        times = np.array([0, 1, 2, 3])
        points = compute_tension_curve(chords, times)
        assert len(points) == 4
        assert all(0 <= p.tension <= 1 for p in points)

    def test_with_dynamics(self):
        chords = [np.array([60])] * 4
        times = np.array([0, 1, 2, 3])
        dynamics = np.array([0.2, 0.5, 0.9, 0.3])
        points = compute_tension_curve(chords, times, dynamics=dynamics)
        assert points[2].tension >= points[0].tension


class TestClimaxDetection:
    def test_single_peak(self):
        points = [TensionPoint(i, t, {}) for i, t in
                  enumerate([0.2, 0.3, 0.5, 0.8, 0.5, 0.3, 0.2])]
        climaxes = detect_climaxes(points, min_prominence=0.1)
        assert len(climaxes) >= 1
        assert climaxes[0].time == 3

    def test_no_peaks(self):
        points = [TensionPoint(i, 0.5, {}) for i in range(5)]
        assert detect_climaxes(points) == []


class TestArcClassification:
    def test_arch(self):
        pts = [TensionPoint(i, t, {}) for i, t in
               enumerate([0.1, 0.3, 0.6, 0.8, 0.6, 0.3, 0.1])]
        assert classify_arc(pts) == "arch"

    def test_ascending(self):
        pts = [TensionPoint(i, t, {}) for i, t in
               enumerate([0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9])]
        assert classify_arc(pts) == "ascending"


class TestTensionMetrics:
    def test_similarity_identical(self):
        pts = [TensionPoint(i, i * 0.1, {}) for i in range(10)]
        assert tension_similarity(pts, pts) > 0.99

    def test_derivative(self):
        pts = [TensionPoint(float(i), i * 0.1, {}) for i in range(5)]
        d = tension_derivative(pts)
        assert len(d) == 4
        assert all(abs(v - 0.1) < 1e-6 for v in d)

    def test_cumulative(self):
        pts = [TensionPoint(float(i), 0.5, {}) for i in range(5)]
        c = cumulative_tension(pts)
        assert abs(c - 2.0) < 0.01

    def test_entropy_uniform(self):
        pts = [TensionPoint(i, np.random.uniform(0, 1), {}) for i in range(100)]
        e = tension_entropy(pts)
        assert e > 0


class TestTensionArc:
    def test_full_arc(self):
        pts = [TensionPoint(i, t, {}) for i, t in
               enumerate([0.2, 0.4, 0.7, 0.9, 0.6, 0.3])]
        arc = tension_arc(pts)
        assert arc.arc_type == "arch"
        assert len(arc.climaxes) >= 1
        assert arc.mean_tension > 0
