"""Tests for onset detection module."""
import numpy as np
import pytest

from models.music_audio.onset_detection import OnsetDetector


class TestSpectralFlux:
    def test_silence_zero(self):
        spec = np.zeros((128, 10))
        sf = OnsetDetector.spectral_flux(spec)
        assert np.allclose(sf, 0.0)

    def test_positive_output(self):
        rng = np.random.default_rng(0)
        spec = rng.uniform(0, 1, (128, 20))
        sf = OnsetDetector.spectral_flux(spec)
        assert len(sf) == 19
        assert np.all(sf >= 0)

    def test_impulse_peak(self):
        spec = np.zeros((64, 20))
        spec[:, 10] = 1.0
        sf = OnsetDetector.spectral_flux(spec)
        assert sf[9] > sf[0]

    def test_single_frame(self):
        spec = np.ones((64, 1))
        sf = OnsetDetector.spectral_flux(spec)
        assert len(sf) == 1


class TestHFC:
    def test_shape(self):
        spec = np.ones((128, 10))
        hfc = OnsetDetector.high_frequency_content(spec)
        assert len(hfc) == 10

    def test_high_freq_emphasis(self):
        spec_low = np.zeros((128, 5))
        spec_low[:10, :] = 1.0
        spec_high = np.zeros((128, 5))
        spec_high[118:, :] = 1.0
        hfc_low = OnsetDetector.high_frequency_content(spec_low)
        hfc_high = OnsetDetector.high_frequency_content(spec_high)
        assert np.mean(hfc_high) > np.mean(hfc_low)


class TestComplexDomain:
    def test_shape(self):
        rng = np.random.default_rng(0)
        stft = rng.standard_normal((64, 20)) + 1j * rng.standard_normal((64, 20))
        odf = OnsetDetector.complex_domain(stft)
        assert len(odf) == 20

    def test_leading_zeros(self):
        rng = np.random.default_rng(0)
        stft = rng.standard_normal((64, 10)) + 1j * rng.standard_normal((64, 10))
        odf = OnsetDetector.complex_domain(stft)
        assert odf[0] == 0.0
        assert odf[1] == 0.0

    def test_short_input(self):
        stft = np.ones((32, 2)) + 0j
        odf = OnsetDetector.complex_domain(stft)
        assert len(odf) == 2


class TestSuperFlux:
    def test_shape(self):
        spec = np.random.default_rng(0).uniform(0, 1, (128, 30))
        sf = OnsetDetector.superflux(spec)
        assert len(sf) == 30

    def test_nonneg(self):
        spec = np.random.default_rng(0).uniform(0, 1, (128, 30))
        sf = OnsetDetector.superflux(spec)
        assert np.all(sf >= 0)


class TestPeakPicker:
    def test_no_peaks_below_threshold(self):
        fn = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
        peaks = OnsetDetector.pick_peaks(fn, threshold=0.5)
        assert len(peaks) == 0

    def test_finds_peak(self):
        fn = np.array([0, 0, 0, 1, 0, 0, 0])
        peaks = OnsetDetector.pick_peaks(fn, threshold=0.5)
        assert 3 in peaks

    def test_min_distance(self):
        fn = np.array([0, 1, 0, 1, 0, 1, 0])
        peaks = OnsetDetector.pick_peaks(fn, threshold=0.5, min_distance=3)
        assert len(peaks) <= 2

    def test_empty_input(self):
        peaks = OnsetDetector.pick_peaks(np.array([]))
        assert len(peaks) == 0


class TestOnsetStrength:
    def test_basic(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr)
        signal = np.sin(2 * np.pi * 440 * t)
        env = OnsetDetector.onset_strength_envelope(signal, sr)
        assert len(env) > 0

    def test_short_signal(self):
        env = OnsetDetector.onset_strength_envelope(np.zeros(100))
        assert len(env) == 1


class TestTempoFromOnsets:
    def test_regular_120bpm(self):
        onsets = np.arange(0, 5, 0.5)
        bpm = OnsetDetector.tempo_from_onsets(onsets)
        assert 100 < bpm < 140

    def test_single_onset(self):
        assert OnsetDetector.tempo_from_onsets(np.array([1.0])) == 0.0

    def test_empty(self):
        assert OnsetDetector.tempo_from_onsets(np.array([])) == 0.0
