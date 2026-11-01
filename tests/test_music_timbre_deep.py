"""Deep tests for music_timbre: spectral descriptors, attack transient
analysis, timbre space, harmonic analysis."""
import numpy as np
import pytest

from models.music_timbre.descriptors import (
    spectral_centroid_from_spectrum, spectral_spread, spectral_flatness,
    spectral_crest, spectral_skewness, spectral_kurtosis,
    brightness, warmth, odd_even_ratio, inharmonicity,
    roughness_sethares,
)
from models.music_timbre.attack_transient import (
    compute_attack_time, rise_slope, classify_attack,
    noise_ratio_at_onset, spectral_centroid_at_onset,
    analyze_transient, AttackType,
)
from models.music_timbre.harmonic_analysis import (
    find_harmonics, harmonic_energy_ratio, spectral_irregularity,
    tristimulus, harmonic_deviation,
)
from models.music_timbre.timbre_space import (
    compute_timbre_features, timbre_distance, build_timbre_space,
)

SR = 22050


def _sine(freq=440.0, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


def _spectrum(freq=440.0, dur=0.5, sr=SR):
    sig = _sine(freq, dur, sr)
    n = len(sig)
    mag = np.abs(np.fft.rfft(sig)) / n
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return mag, freqs


# ---- Spectral Descriptors ----

class TestSpectralDescriptors:
    def test_centroid_pure_tone(self):
        mag, freqs = _spectrum(440.0)
        c = spectral_centroid_from_spectrum(mag, freqs)
        assert abs(c - 440.0) < 20.0

    def test_centroid_higher_for_bright(self):
        mag_low, freqs = _spectrum(200.0)
        mag_high, _ = _spectrum(2000.0)
        c_low = spectral_centroid_from_spectrum(mag_low, freqs)
        c_high = spectral_centroid_from_spectrum(mag_high, freqs)
        assert c_high > c_low

    def test_spread_nonnegative(self):
        mag, freqs = _spectrum(440.0)
        assert spectral_spread(mag, freqs) >= 0.0

    def test_flatness_pure_tone_low(self):
        mag, freqs = _spectrum(440.0)
        f = spectral_flatness(mag)
        assert f < 0.1

    def test_flatness_noise_high(self):
        rng = np.random.default_rng(42)
        mag = rng.uniform(0.5, 1.5, 512)
        f = spectral_flatness(mag)
        assert f > 0.5

    def test_crest_pure_tone_high(self):
        mag, _ = _spectrum(440.0)
        c = spectral_crest(mag)
        assert c > 1.0

    def test_skewness_finite(self):
        mag, freqs = _spectrum(440.0)
        s = spectral_skewness(mag, freqs)
        assert np.isfinite(s)

    def test_kurtosis_finite(self):
        mag, freqs = _spectrum(440.0)
        k = spectral_kurtosis(mag, freqs)
        assert np.isfinite(k)

    def test_brightness_bounded(self):
        mag, freqs = _spectrum(440.0)
        b = brightness(mag, freqs, cutoff=1500.0)
        assert 0.0 <= b <= 1.0

    def test_warmth_bounded(self):
        mag, freqs = _spectrum(440.0)
        w = warmth(mag, freqs)
        assert 0.0 <= w <= 1.0

    def test_odd_even_sine_high(self):
        partials = np.array([1.0, 0.0, 0.5, 0.0, 0.3, 0.0])
        r = odd_even_ratio(partials)
        assert r > 1.0

    def test_inharmonicity_pure(self):
        f0 = 440.0
        partials = np.array([f0, 2 * f0, 3 * f0, 4 * f0])
        inh = inharmonicity(partials, f0)
        assert inh < 0.01

    def test_roughness_unison_low(self):
        f0 = 440.0
        freqs = np.array([f0, 2 * f0, 3 * f0])
        amps = np.array([1.0, 0.5, 0.3])
        r = roughness_sethares(freqs, amps)
        assert r >= 0.0


# ---- Attack Transient ----

class TestAttackTransient:
    def test_attack_time_positive(self):
        env = np.linspace(0, 1, 1000)
        at = compute_attack_time(env, sr=SR)
        assert at > 0.0

    def test_rise_slope_positive(self):
        env = np.linspace(0, 1, 500)
        rs = rise_slope(env, sr=SR)
        assert rs > 0.0

    def test_classify_attack_percussive(self):
        at = classify_attack(5.0, 0.8)
        assert at == AttackType.PERCUSSIVE

    def test_classify_attack_sustained(self):
        at = classify_attack(200.0, 0.05)
        assert at == AttackType.SUSTAINED

    def test_noise_ratio_bounded(self):
        sig = _sine(440, 0.1)
        nr = noise_ratio_at_onset(sig, sr=SR)
        assert 0.0 <= nr <= 1.0

    def test_spectral_centroid_at_onset_positive(self):
        sig = _sine(440, 0.1)
        sc = spectral_centroid_at_onset(sig, sr=SR)
        assert sc > 0.0

    def test_analyze_transient_returns_descriptor(self):
        sig = _sine(440, 0.1)
        desc = analyze_transient(sig, sr=SR)
        assert hasattr(desc, 'attack_time_ms')
        assert desc.attack_time_ms >= 0.0


# ---- Harmonic Analysis ----

class TestHarmonicAnalysis:
    def test_find_harmonics_pure(self):
        mag, freqs = _spectrum(440.0)
        h_freqs, h_amps = find_harmonics(mag, freqs, f0=440.0, n_harmonics=3)
        assert len(h_freqs) == 3
        assert abs(h_freqs[0] - 440.0) < 20.0

    def test_harmonic_energy_ratio(self):
        mag, freqs = _spectrum(440.0)
        her = harmonic_energy_ratio(mag, freqs, f0=440.0)
        assert 0.0 <= her <= 1.0

    def test_spectral_irregularity_nonneg(self):
        amps = np.array([1.0, 0.5, 0.3, 0.1, 0.05])
        irr = spectral_irregularity(amps)
        assert irr >= 0.0

    def test_tristimulus_sums_to_one(self):
        amps = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05])
        t1, t2, t3 = tristimulus(amps)
        assert abs(t1 + t2 + t3 - 1.0) < 0.01

    def test_harmonic_deviation_nonneg(self):
        amps = np.array([1.0, 0.4, 0.3, 0.2])
        hd = harmonic_deviation(amps)
        assert hd >= 0.0


# ---- Timbre Space ----

class TestTimbreSpace:
    def test_compute_features_shape(self):
        sig = _sine(440, 0.5)
        features = compute_timbre_features(sig, sr=SR)
        assert isinstance(features, np.ndarray)
        assert len(features) >= 5

    def test_timbre_distance_self_zero(self):
        sig = _sine(440, 0.5)
        f = compute_timbre_features(sig, sr=SR)
        d = timbre_distance(f, f)
        assert d < 0.01

    def test_timbre_distance_different_positive(self):
        f1 = compute_timbre_features(_sine(440, 0.5), sr=SR)
        f2 = compute_timbre_features(_sine(2000, 0.5), sr=SR)
        d = timbre_distance(f1, f2)
        assert d > 0.0

    def test_build_timbre_space_shape(self):
        signals = [_sine(f, 0.3) for f in [200, 440, 880, 1500]]
        space = build_timbre_space(signals, sr=SR)
        assert space.shape[0] == 4
