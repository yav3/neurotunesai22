"""Tests for music analysis: tempo estimation, source separation."""
import numpy as np
import pytest

from models.music_analysis.tempo_analysis import (
    onset_strength_envelope,
    autocorrelation_tempo,
    tempo_curve,
    beat_strength,
    tempo_stability,
    pulse_clarity,
    TempoEstimate,
)
from models.music_analysis.source_separation import (
    stft,
    istft,
    harmonic_percussive_separation,
    nmf_separate,
    nmf_reconstruct,
    spectral_centroid_separation,
    wiener_filter,
)


class TestOnsetStrength:
    def test_basic_shape(self):
        sig = np.random.randn(22050)
        odf = onset_strength_envelope(sig, sr=22050, hop=512)
        expected_frames = 1 + (22050 - 2048) // 512
        assert len(odf) == expected_frames

    def test_silent_signal(self):
        sig = np.zeros(22050)
        odf = onset_strength_envelope(sig, sr=22050, hop=512)
        assert np.allclose(odf, 0.0)

    def test_impulse_peak(self):
        sig = np.zeros(22050)
        sig[11025] = 1.0
        odf = onset_strength_envelope(sig, sr=22050, hop=512)
        peak_frame = np.argmax(odf)
        peak_time = peak_frame * 512 / 22050
        assert abs(peak_time - 0.5) < 0.1


class TestAutocorrelationTempo:
    def test_120bpm_click(self):
        sr = 22050
        dur = 4.0
        sig = np.zeros(int(sr * dur))
        period = int(sr * 60.0 / 120.0)
        for i in range(0, len(sig), period):
            end = min(i + 200, len(sig))
            sig[i:end] = 1.0
        odf = onset_strength_envelope(sig, sr=sr)
        est = autocorrelation_tempo(odf, sr=sr)
        # ACF may pick half or double tempo (octave ambiguity)
        assert est.bpm > 50 and (
            abs(est.bpm - 120) < 30 or abs(est.bpm - 60) < 20 or abs(est.bpm - 240) < 40
        )

    def test_short_signal_fallback(self):
        odf = np.array([1.0, 0.5])
        est = autocorrelation_tempo(odf)
        assert est.bpm == 120.0
        assert est.confidence == 0.0


class TestTempoCurve:
    def test_constant_tempo(self):
        sr = 22050
        dur = 16.0
        sig = np.zeros(int(sr * dur))
        period = int(sr * 60.0 / 100.0)
        for i in range(0, len(sig), period):
            end = min(i + 100, len(sig))
            sig[i:end] = 1.0
        times, tempos = tempo_curve(sig, sr=sr, window_sec=8.0, step_sec=4.0)
        assert len(times) == len(tempos)
        assert len(times) >= 1


class TestBeatStrength:
    def test_basic(self):
        odf = np.array([0.0, 0.5, 1.0, 0.3, 0.8])
        beats = np.array([2, 4])
        strengths = beat_strength(odf, beats)
        assert strengths[0] == pytest.approx(1.0)
        assert strengths[1] == pytest.approx(0.8)

    def test_empty(self):
        odf = np.array([1.0, 2.0])
        beats = np.array([], dtype=int)
        strengths = beat_strength(odf, beats)
        assert len(strengths) == 0


class TestTempoStability:
    def test_perfect_stability(self):
        tempos = np.array([120.0, 120.0, 120.0])
        assert tempo_stability(tempos) == pytest.approx(1.0)

    def test_unstable(self):
        tempos = np.array([60.0, 120.0, 180.0])
        s = tempo_stability(tempos)
        assert 0.0 < s < 0.8


class TestPulseClarity:
    def test_periodic_signal(self):
        sr = 22050
        dur = 4.0
        sig = np.zeros(int(sr * dur))
        period = int(sr * 60.0 / 120.0)
        for i in range(0, len(sig), period):
            end = min(i + 100, len(sig))
            sig[i:end] = 1.0
        odf = onset_strength_envelope(sig, sr=sr)
        pc = pulse_clarity(odf, sr=sr)
        assert pc > 0.1

    def test_noise_low_clarity(self):
        odf = np.random.RandomState(0).randn(200)
        pc = pulse_clarity(odf)
        assert pc < 0.5


class TestSTFT:
    def test_roundtrip(self):
        sig = np.random.RandomState(42).randn(8000)
        S = stft(sig, n_fft=1024, hop=256)
        rec = istft(S, n_fft=1024, hop=256)
        n = min(len(sig), len(rec))
        err = np.mean((sig[:n] - rec[:n]) ** 2)
        assert err < 0.01

    def test_shape(self):
        sig = np.random.randn(4096)
        S = stft(sig, n_fft=1024, hop=256)
        assert S.shape[0] == 513


class TestHarmonicPercussive:
    def test_both_components_nonzero(self):
        rng = np.random.RandomState(0)
        sig = rng.randn(22050)
        h, p = harmonic_percussive_separation(sig, n_fft=1024, hop=256)
        assert np.sum(h ** 2) > 0
        assert np.sum(p ** 2) > 0
        assert len(h) == len(sig)
        assert len(p) == len(sig)

    def test_sine_mostly_harmonic(self):
        sr = 22050
        t = np.arange(sr) / sr
        sig = np.sin(2 * np.pi * 440 * t)
        h, p = harmonic_percussive_separation(sig, n_fft=1024, hop=256)
        assert np.sum(h ** 2) > np.sum(p ** 2)


class TestNMF:
    def test_cost_decreases(self):
        rng = np.random.RandomState(0)
        V = rng.rand(50, 30) + 0.01
        result = nmf_separate(V, n_components=3, max_iter=50)
        assert result.cost[-1] <= result.cost[0]

    def test_reconstruction(self):
        rng = np.random.RandomState(0)
        V = rng.rand(50, 30) + 0.01
        result = nmf_separate(V, n_components=5, max_iter=100)
        recon = result.W @ result.H
        rel_err = np.linalg.norm(V - recon) / np.linalg.norm(V)
        assert rel_err < 0.5

    def test_reconstruct_component(self):
        rng = np.random.RandomState(1)
        S = rng.randn(50, 30) + 1j * rng.randn(50, 30)
        mag = np.abs(S)
        res = nmf_separate(mag, n_components=2, max_iter=50)
        comp = nmf_reconstruct(S, res.W, res.H, component=0)
        assert comp.shape == S.shape


class TestSpectralCentroidSeparation:
    def test_low_high_split(self):
        sr = 22050
        t = np.arange(sr) / sr
        low = np.sin(2 * np.pi * 200 * t)
        high = np.sin(2 * np.pi * 8000 * t)
        sig = low + high
        lo, hi = spectral_centroid_separation(sig, sr=sr, cutoff_hz=2000)
        assert np.sum(lo ** 2) > 0.1
        assert np.sum(hi ** 2) > 0.1


class TestWienerFilter:
    def test_two_sources(self):
        rng = np.random.RandomState(0)
        mag1 = rng.rand(50, 30) + 0.01
        mag2 = rng.rand(50, 30) + 0.01
        S_mix = (mag1 + mag2) * np.exp(1j * rng.uniform(0, 2 * np.pi, (50, 30)))
        separated = wiener_filter([mag1, mag2], S_mix)
        assert len(separated) == 2
        assert separated[0].shape == S_mix.shape
