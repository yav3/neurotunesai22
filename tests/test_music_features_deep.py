"""Tests for spectral, rhythm, harmonic, and temporal feature modules."""
import numpy as np
import pytest

from models.music_features.spectral_descriptors import (
    spectral_centroid, spectral_bandwidth, spectral_rolloff,
    spectral_flatness, spectral_slope, spectral_crest_factor,
    bark_band_energies, mfcc, spectral_contrast, spectral_feature_vector,
)
from models.music_features.rhythm_features import (
    onset_density, inter_onset_intervals, tempo_stability,
    beat_histogram, pulse_clarity, syncopation_index,
    swing_ratio, microtiming_deviation, npvi, rhythmic_complexity,
)
from models.music_features.harmonic_features import (
    key_strength, mode_strength, key_clarity,
    chroma_to_tonnetz, tonal_radius, hcdf,
    chord_change_rate, harmonic_complexity,
)
from models.music_features.temporal_features import (
    rms_energy, zero_crossing_rate, amplitude_envelope,
    attack_time, decay_time, temporal_centroid,
    effective_duration, crest_factor, dynamic_range,
    loudness_range, modulation_spectrum, temporal_feature_vector,
)


def _sine_spectrum(freq_hz: float = 440.0, sr: float = 22050.0, n_fft: int = 2048):
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    mag = np.zeros(n_bins)
    idx = np.argmin(np.abs(freqs - freq_hz))
    mag[idx] = 1.0
    return mag


def _white_noise_spectrum(n_fft: int = 2048, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return np.abs(rng.standard_normal(n_fft // 2 + 1))


class TestSpectralDescriptors:
    def test_centroid_pure_tone(self):
        mag = _sine_spectrum(1000.0)
        c = spectral_centroid(mag)
        assert abs(c - 1000.0) < 20.0

    def test_bandwidth_pure_tone_narrow(self):
        mag = _sine_spectrum(1000.0)
        bw = spectral_bandwidth(mag)
        assert bw < 50.0

    def test_rolloff_pure_tone(self):
        mag = _sine_spectrum(1000.0)
        ro = spectral_rolloff(mag)
        assert ro >= 990.0

    def test_flatness_noise_high(self):
        mag = _white_noise_spectrum()
        sf = spectral_flatness(mag)
        assert sf > 0.5

    def test_flatness_tone_low(self):
        mag = _sine_spectrum(440.0)
        sf = spectral_flatness(mag)
        assert sf < 0.1

    def test_crest_factor_tone_high(self):
        mag = _sine_spectrum(440.0)
        cf = spectral_crest_factor(mag)
        assert cf > 10.0

    def test_bark_bands_shape(self):
        mag = _white_noise_spectrum()
        bands = bark_band_energies(mag)
        assert len(bands) == 24
        assert np.all(bands >= 0)

    def test_mfcc_shape(self):
        mag = _white_noise_spectrum()
        coeffs = mfcc(mag, n_mfcc=13)
        assert len(coeffs) == 13

    def test_spectral_contrast_shape(self):
        mag = _white_noise_spectrum()
        sc = spectral_contrast(mag)
        assert len(sc) == 6

    def test_feature_vector_length(self):
        mag = _white_noise_spectrum()
        fv = spectral_feature_vector(mag, n_mfcc=13)
        assert len(fv) == 7 + 13


class TestRhythmFeatures:
    def test_onset_density(self):
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        d = onset_density(times, 2.0)
        assert d == pytest.approx(2.5)

    def test_ioi(self):
        times = np.array([0.0, 0.5, 1.0])
        ioi = inter_onset_intervals(times)
        np.testing.assert_allclose(ioi, [0.5, 0.5])

    def test_tempo_stability_regular(self):
        times = np.arange(0, 5, 0.5)
        ts = tempo_stability(times)
        assert ts < 0.01

    def test_beat_histogram_shape(self):
        times = np.arange(0, 10, 0.5)
        bpm, hist = beat_histogram(times)
        assert len(bpm) == 256
        assert len(hist) == 256

    def test_pulse_clarity_regular(self):
        times = np.arange(0, 10, 0.5)
        pc = pulse_clarity(times)
        assert pc > 0.5

    def test_syncopation_on_beat(self):
        onsets = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        si = syncopation_index(onsets)
        assert si < 0.1

    def test_syncopation_off_beat(self):
        onsets = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        si = syncopation_index(onsets)
        assert si > 0.0

    def test_npvi_regular(self):
        durs = np.array([0.5, 0.5, 0.5, 0.5])
        v = npvi(durs)
        assert v == pytest.approx(0.0)

    def test_npvi_irregular(self):
        durs = np.array([0.5, 1.0, 0.5, 1.0])
        v = npvi(durs)
        assert v > 30.0


class TestHarmonicFeatures:
    def test_key_c_major(self):
        chroma = np.array([1.0, 0, 0.5, 0, 0.8, 0.4, 0, 0.9, 0, 0.3, 0, 0.2])
        key, mode, corr = key_strength(chroma)
        assert key == "C"
        assert mode == "major"
        assert corr > 0.5

    def test_key_clarity_tonal(self):
        chroma = np.array([1.0, 0, 0.5, 0, 0.8, 0.4, 0, 0.9, 0, 0.3, 0, 0.2])
        kc = key_clarity(chroma)
        assert kc > 1.0

    def test_tonnetz_dimension(self):
        chroma = np.ones(12)
        t = chroma_to_tonnetz(chroma)
        assert len(t) == 6

    def test_tonal_radius_zero(self):
        chroma = np.zeros(12)
        r = tonal_radius(chroma)
        assert r == 0.0

    def test_harmonic_complexity_uniform(self):
        chroma = np.ones(12)
        hc = harmonic_complexity(chroma)
        assert abs(hc - np.log2(12)) < 0.01

    def test_hcdf_shape(self):
        rng = np.random.default_rng(42)
        seq = rng.random((12, 50))
        flux = hcdf(seq, sigma=2.0)
        assert len(flux) == 50


class TestTemporalFeatures:
    def test_rms_energy_silence(self):
        sig = np.zeros(4096)
        rms = rms_energy(sig)
        assert np.all(rms < 1e-10)

    def test_rms_energy_sine(self):
        t = np.linspace(0, 1, 22050)
        sig = np.sin(2 * np.pi * 440 * t)
        rms = rms_energy(sig)
        assert np.mean(rms) > 0.5

    def test_zcr_sine(self):
        t = np.linspace(0, 1, 22050)
        sig = np.sin(2 * np.pi * 440 * t)
        zcr = zero_crossing_rate(sig)
        mean_zcr = np.mean(zcr)
        assert 0.015 < mean_zcr < 0.06

    def test_amplitude_envelope_shape(self):
        sig = np.random.default_rng(0).standard_normal(4096)
        env = amplitude_envelope(sig)
        assert len(env) == len(sig)
        assert np.all(env >= 0)

    def test_temporal_centroid_symmetric(self):
        sig = np.ones(22050)
        tc = temporal_centroid(sig, sr=22050.0)
        assert abs(tc - 0.5) < 0.01

    def test_crest_factor_sine(self):
        t = np.linspace(0, 1, 22050)
        sig = np.sin(2 * np.pi * 440 * t)
        cf = crest_factor(sig)
        assert abs(cf - np.sqrt(2)) < 0.1

    def test_dynamic_range_constant(self):
        sig = 0.5 * np.ones(4096)
        dr = dynamic_range(sig)
        assert dr < 1.0

    def test_modulation_spectrum_shape(self):
        sig = np.random.default_rng(0).standard_normal(22050)
        freqs, spec = modulation_spectrum(sig)
        assert len(freqs) == 64
        assert len(spec) == 64

    def test_feature_vector_length(self):
        sig = np.random.default_rng(0).standard_normal(22050)
        fv = temporal_feature_vector(sig)
        assert len(fv) == 10
