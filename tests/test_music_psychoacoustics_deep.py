"""Tests for psychoacoustics: temporal perception and auditory scene analysis."""
import numpy as np
import pytest
from models.music_psychoacoustics.temporal_perception import (
    IntegrationWindow, INTEGRATION_WINDOWS,
    weber_tempo_jnd, tempo_discrimination,
    forward_masking_threshold, backward_masking_threshold,
    gap_detection_threshold, temporal_order_judgment,
    temporal_modulation_transfer, rhythm_perception_window,
    preferred_tempo_resonance, ioi_variability, perceptual_attack_time,
)
from models.music_psychoacoustics.auditory_scene import (
    StreamSegregation, AuditoryGrouping,
)


class TestTemporalPerception:
    def test_integration_windows_keys(self):
        assert "fine_structure" in INTEGRATION_WINDOWS
        assert "pitch" in INTEGRATION_WINDOWS
        assert "loudness" in INTEGRATION_WINDOWS
        assert "rhythm" in INTEGRATION_WINDOWS

    def test_integration_window_durations(self):
        assert INTEGRATION_WINDOWS["fine_structure"].duration_ms == pytest.approx(2.5)
        assert INTEGRATION_WINDOWS["pitch"].duration_ms == pytest.approx(30.0)
        assert INTEGRATION_WINDOWS["loudness"].duration_ms == pytest.approx(200.0)
        assert INTEGRATION_WINDOWS["rhythm"].duration_ms == pytest.approx(2000.0)

    def test_weber_tempo_jnd(self):
        jnd = weber_tempo_jnd(120.0)
        assert jnd > 0.0
        assert jnd < 20.0

    def test_tempo_discrimination(self):
        d = tempo_discrimination(120.0, 124.0)
        assert isinstance(d, bool)

    def test_forward_masking_decays(self):
        t1 = forward_masking_threshold(10.0, 80.0)
        t2 = forward_masking_threshold(100.0, 80.0)
        assert t1 > t2

    def test_backward_masking(self):
        t = backward_masking_threshold(5.0, 80.0)
        assert isinstance(t, float)
        assert t >= 0.0

    def test_gap_detection(self):
        g = gap_detection_threshold(1000.0)
        assert g > 0.0
        assert g < 50.0

    def test_temporal_order_judgment(self):
        t = temporal_order_judgment(2)
        assert t > 0.0

    def test_temporal_modulation_transfer(self):
        v = temporal_modulation_transfer(4.0)
        assert 0.0 <= v <= 1.0

    def test_rhythm_perception_window(self):
        w = rhythm_perception_window(120.0)
        assert isinstance(w, tuple)
        assert w[0] > 0.0
        assert w[1] > w[0]

    def test_preferred_tempo_resonance(self):
        r = preferred_tempo_resonance(120.0)
        assert r > 0.0

    def test_ioi_variability(self):
        onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        v = ioi_variability(onsets)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_ioi_variability_irregular(self):
        onsets = np.array([0.0, 0.3, 1.0, 1.2, 2.0])
        v = ioi_variability(onsets)
        assert v > 0.0

    def test_perceptual_attack_time(self):
        signal = np.concatenate([np.zeros(100), np.linspace(0, 1, 50), np.ones(100)])
        pat = perceptual_attack_time(signal, 44100)
        assert pat >= 0.0


class TestAuditoryScene:
    def test_stream_segregation(self):
        ss = StreamSegregation()
        fb = ss.fission_semitones
        assert isinstance(fb, (int, float))
        assert fb > 0.0

    def test_segregation_probability(self):
        ss = StreamSegregation()
        p = ss.segregation_probability(12.0, 200.0)
        assert 0.0 <= p <= 1.0

    def test_proximity_score(self):
        ag = AuditoryGrouping()
        score = ag.proximity_score(0.1, 2.0)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_similarity_score(self):
        ag = AuditoryGrouping()
        a = np.array([1.0, 0.5, 0.3])
        b = np.array([1.1, 0.4, 0.35])
        score = ag.similarity_score(a, b)
        assert isinstance(score, float)
        assert score >= 0.0
