"""Tests for signal envelope analysis."""
import numpy as np
import pytest

from models.music_signal.envelope import (
    analytic_signal, instantaneous_amplitude, instantaneous_phase,
    instantaneous_frequency, envelope_follower, rms_envelope,
    am_fm_decompose, tremolo_rate, tremolo_depth,
    envelope_centroid, attack_time, decay_time,
)


def _sine(freq=440.0, sr=44100, dur=0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


class TestAnalyticSignal:
    def test_complex_output(self):
        x = _sine()
        z = analytic_signal(x)
        assert z.dtype == complex
        assert len(z) == len(x)

    def test_real_part_matches(self):
        x = _sine()
        z = analytic_signal(x)
        np.testing.assert_allclose(z.real, x, atol=1e-10)

    def test_empty(self):
        z = analytic_signal(np.array([]))
        assert len(z) == 0


class TestInstantaneousAmplitude:
    def test_pure_sine(self):
        x = _sine()
        env = instantaneous_amplitude(x)
        assert np.allclose(env[1000:-1000], 1.0, atol=0.05)

    def test_am_signal(self):
        sr = 44100
        t = np.arange(22050) / sr
        x = np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        env = instantaneous_amplitude(x)
        assert env.max() > 1.2
        assert env.min() >= 0.0


class TestInstantaneousFrequency:
    def test_pure_sine(self):
        x = _sine(440.0, 44100, 0.5)
        freq = instantaneous_frequency(x, 44100)
        mid = freq[5000:-5000]
        assert np.allclose(mid, 440.0, atol=1.0)

    def test_empty(self):
        assert len(instantaneous_frequency(np.array([1.0]), 44100)) == 0


class TestEnvelopeFollower:
    def test_tracks_amplitude(self):
        x = np.zeros(1000)
        x[200:400] = 1.0
        env = envelope_follower(x, attack_samples=5, release_samples=50)
        assert env[350] > 0.5
        assert env[0] == 0.0

    def test_empty(self):
        assert len(envelope_follower(np.array([]))) == 0


class TestRMSEnvelope:
    def test_constant_signal(self):
        x = np.ones(4096) * 0.5
        env = rms_envelope(x, window=1024, hop=512)
        np.testing.assert_allclose(env, 0.5, atol=0.01)

    def test_empty(self):
        assert len(rms_envelope(np.array([]))) == 0

    def test_short_signal(self):
        env = rms_envelope(np.array([1.0, 2.0, 3.0]), window=1024)
        assert len(env) == 1


class TestAMFMDecompose:
    def test_returns_decomposition(self):
        x = _sine()
        d = am_fm_decompose(x, 44100)
        assert len(d.amplitude) == len(x)
        assert len(d.phase) == len(x)
        assert len(d.frequency) == len(x) - 1


class TestTremolo:
    def test_rate_detection(self):
        sr = 44100
        t = np.arange(sr) / sr
        x = np.sin(2 * np.pi * 440 * t) * (1 + 0.3 * np.sin(2 * np.pi * 6 * t))
        rate = tremolo_rate(x, sr)
        assert 5.0 <= rate <= 7.0

    def test_depth_nonzero(self):
        sr = 44100
        t = np.arange(sr) / sr
        x = np.sin(2 * np.pi * 440 * t) * (1 + 0.4 * np.sin(2 * np.pi * 5 * t))
        assert tremolo_depth(x) > 0.1

    def test_no_tremolo(self):
        x = _sine()
        assert tremolo_depth(x) < 0.1


class TestEnvelopeCentroid:
    def test_centered(self):
        sr = 1000
        x = np.zeros(sr)
        x[400:600] = 1.0
        c = envelope_centroid(x, sr)
        assert 0.4 <= c <= 0.6


class TestAttackDecay:
    def test_attack_time(self):
        x = np.linspace(0, 1, 1000)
        at = attack_time(x, 1000, threshold=0.9)
        assert 0.8 <= at <= 1.0

    def test_decay_time(self):
        x = np.concatenate([np.linspace(0, 1, 500), np.linspace(1, 0, 500)])
        dt = decay_time(x, 1000, threshold=0.1)
        assert 0.3 <= dt <= 0.6
