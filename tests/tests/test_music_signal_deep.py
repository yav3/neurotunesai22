"""Deep tests for harmonic product spectrum and multi-method pitch estimation."""
import numpy as np
import pytest

from models.music_signal.harmonic_product_spectrum import (
    harmonic_product_spectrum, autocorrelation_pitch, cepstral_pitch,
    multi_pitch_estimate, consensus_pitch, PitchEstimate,
)

SR = 16000


def _sine(freq=440.0, dur=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t)


class TestHPS:
    def test_pure_tone(self):
        sig = _sine(440, dur=1.0)
        est = harmonic_product_spectrum(sig, SR, n_harmonics=3)
        assert isinstance(est, PitchEstimate)
        assert est.frequency_hz > 50
        assert est.method == "hps"

    def test_returns_positive_freq(self):
        sig = _sine(200, dur=1.0)
        est = harmonic_product_spectrum(sig, SR, fmin=50)
        assert est.frequency_hz > 0

    def test_silence(self):
        est = harmonic_product_spectrum(np.zeros(4096), SR)
        assert est.frequency_hz == 0.0


class TestAutocorrelation:
    def test_pure_tone(self):
        sig = _sine(330, dur=0.5)
        est = autocorrelation_pitch(sig, SR)
        assert abs(est.frequency_hz - 330) < 15
        assert est.method == "autocorrelation"

    def test_very_short(self):
        est = autocorrelation_pitch(np.array([0.1]), SR)
        assert est.frequency_hz == 0.0

    def test_confidence_positive(self):
        sig = _sine(220, dur=0.5)
        est = autocorrelation_pitch(sig, SR)
        assert est.confidence > 0


class TestCepstral:
    def test_returns_positive_freq(self):
        sig = _sine(300, dur=1.0)
        est = cepstral_pitch(sig, SR)
        assert est.frequency_hz > 0
        assert est.method == "cepstral"

    def test_noise(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(4096) * 0.1
        est = cepstral_pitch(noise, SR)
        assert isinstance(est.confidence, float)


class TestMultiPitch:
    def test_single_tone(self):
        sig = _sine(440, dur=1.0)
        pitches = multi_pitch_estimate(sig, SR, max_pitches=3)
        assert len(pitches) >= 1
        assert pitches[0].frequency_hz > 50

    def test_two_tones(self):
        sig = _sine(300, dur=0.5) + _sine(600, dur=0.5)
        pitches = multi_pitch_estimate(sig, SR, max_pitches=3)
        assert len(pitches) >= 1

    def test_silence_empty(self):
        pitches = multi_pitch_estimate(np.zeros(4096), SR)
        assert len(pitches) == 0

    def test_method_label(self):
        sig = _sine(440)
        pitches = multi_pitch_estimate(sig, SR)
        assert all(p.method == "multi_hps" for p in pitches)


class TestConsensusPitch:
    def test_agreement(self):
        sig = _sine(440, dur=1.0)
        est = consensus_pitch(sig, SR)
        assert abs(est.frequency_hz - 440) < 30
        assert est.method == "consensus"

    def test_silence(self):
        est = consensus_pitch(np.zeros(4096), SR)
        assert est.frequency_hz == 0.0

    def test_confidence_bounded(self):
        sig = _sine(300, dur=1.0)
        est = consensus_pitch(sig, SR)
        assert 0.0 <= est.confidence <= 1.0
