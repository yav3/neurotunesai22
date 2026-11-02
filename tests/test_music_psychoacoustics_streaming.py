"""Tests for auditory stream segregation."""
import numpy as np
import pytest

from models.music_psychoacoustics.auditory_streaming import (
    AuditoryEvent, Stream, frequency_proximity, temporal_proximity,
    timbre_proximity, coherence_matrix, fission_probability,
    segregate_streams, onset_synchrony_groups, harmonic_fusion_score,
    stream_salience, stream_continuity,
)


def _alternating_tones(n=10, f_low=200.0, f_high=800.0, ioi=100.0):
    events = []
    for i in range(n):
        f = f_low if i % 2 == 0 else f_high
        events.append(AuditoryEvent(onset_ms=i * ioi, duration_ms=50.0, frequency_hz=f))
    return events


class TestProximity:
    def test_frequency_identical(self):
        assert frequency_proximity(440.0, 440.0) == pytest.approx(1.0)

    def test_frequency_octave(self):
        p = frequency_proximity(220.0, 440.0)
        assert 0.0 < p < 1.0

    def test_frequency_far(self):
        p = frequency_proximity(100.0, 5000.0)
        assert p < 0.01

    def test_temporal_identical(self):
        assert temporal_proximity(100.0, 100.0) == pytest.approx(1.0)

    def test_temporal_far(self):
        p = temporal_proximity(0.0, 10000.0)
        assert p < 0.01

    def test_timbre_identical(self):
        assert timbre_proximity(1000.0, 1000.0) == pytest.approx(1.0)


class TestCoherenceMatrix:
    def test_shape(self):
        events = _alternating_tones(5)
        mat = coherence_matrix(events)
        assert mat.shape == (5, 5)

    def test_diagonal_ones(self):
        events = _alternating_tones(4)
        mat = coherence_matrix(events)
        np.testing.assert_allclose(np.diag(mat), 1.0)

    def test_symmetric(self):
        events = _alternating_tones(4)
        mat = coherence_matrix(events)
        np.testing.assert_allclose(mat, mat.T)


class TestFission:
    def test_large_interval_fast_tempo(self):
        p = fission_probability(12.0, 50.0)
        assert p > 0.5

    def test_small_interval_slow_tempo(self):
        p = fission_probability(1.0, 500.0)
        assert p < 0.5


class TestStreamSegregation:
    def test_unison_single_stream(self):
        events = [AuditoryEvent(i * 100.0, 50.0, 440.0) for i in range(6)]
        result = segregate_streams(events)
        assert result.n_streams == 1

    def test_alternating_two_streams(self):
        events = _alternating_tones(10, 200.0, 2000.0, 80.0)
        result = segregate_streams(events, threshold=0.5)
        assert result.n_streams >= 2

    def test_coherence_matrix_present(self):
        events = _alternating_tones(4)
        result = segregate_streams(events)
        assert result.coherence_matrix.shape[0] == 4


class TestOnsetGroups:
    def test_simultaneous(self):
        events = [
            AuditoryEvent(0.0, 50.0, 440.0),
            AuditoryEvent(5.0, 50.0, 554.0),
            AuditoryEvent(10.0, 50.0, 660.0),
            AuditoryEvent(200.0, 50.0, 330.0),
        ]
        groups = onset_synchrony_groups(events)
        assert len(groups) == 2
        assert len(groups[0]) == 3


class TestHarmonicFusion:
    def test_harmonic_series(self):
        freqs = [220, 440, 660, 880]
        score = harmonic_fusion_score(freqs)
        assert score > 0.9

    def test_inharmonic(self):
        freqs = [200, 317, 423, 571]
        score = harmonic_fusion_score(freqs)
        assert score < 0.8

    def test_single_tone(self):
        assert harmonic_fusion_score([440]) == 1.0


class TestStreamProperties:
    def test_salience_range(self):
        events = [AuditoryEvent(i * 100.0, 50.0, 440.0, amplitude=0.8) for i in range(5)]
        s = Stream(stream_id=0, events=events)
        sal = stream_salience(s)
        assert 0.0 <= sal <= 1.0

    def test_continuity_even_spacing(self):
        events = [AuditoryEvent(i * 100.0, 50.0, 440.0) for i in range(10)]
        s = Stream(stream_id=0, events=events)
        c = stream_continuity(s)
        assert c > 0.9

    def test_continuity_irregular(self):
        events = [AuditoryEvent(t, 50.0, 440.0) for t in [0, 100, 105, 800, 810]]
        s = Stream(stream_id=0, events=events)
        c = stream_continuity(s)
        assert c < 0.5

    def test_empty_stream(self):
        s = Stream(stream_id=0, events=[])
        assert stream_salience(s) == 0.0
        assert stream_continuity(s) == 1.0
