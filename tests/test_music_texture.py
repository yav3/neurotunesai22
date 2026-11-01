"""Tests for TextureAnalyzer: spectral contrast, roughness, polyphony, TPR."""
import numpy as np
import pytest

from models.music_features.texture import TextureAnalyzer

SR = 16000
FRAME = 1024


def _sine(freq, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


def _spectrum_and_freqs(signal, n=FRAME):
    chunk = signal[:n]
    w = np.hanning(len(chunk))
    spec = np.abs(np.fft.rfft(chunk * w))
    freqs = np.linspace(0, SR / 2, len(spec))
    return spec, freqs


@pytest.fixture
def ta():
    return TextureAnalyzer()


@pytest.fixture
def sine_spec():
    return _spectrum_and_freqs(_sine(440.0))


@pytest.fixture
def noise_spec():
    rng = np.random.default_rng(42)
    sig = rng.standard_normal(FRAME)
    return _spectrum_and_freqs(sig)


class TestSpectralContrast:
    def test_shape(self, ta, sine_spec):
        spec, freqs = sine_spec
        c = ta.spectral_contrast(spec, freqs, n_bands=6)
        assert c.shape == (6,)

    def test_nonneg(self, ta, sine_spec):
        spec, freqs = sine_spec
        c = ta.spectral_contrast(spec, freqs)
        assert np.all(c >= 0)

    def test_empty(self, ta):
        c = ta.spectral_contrast(np.array([]), np.array([]))
        assert np.all(c == 0)

    def test_single_bin(self, ta):
        c = ta.spectral_contrast(np.array([1.0]), np.array([100.0]))
        assert np.all(c == 0)

    def test_noise_has_contrast(self, ta, noise_spec):
        spec, freqs = noise_spec
        c = ta.spectral_contrast(spec, freqs)
        assert np.sum(c) > 0


class TestValleyEnergy:
    def test_shape(self, ta, sine_spec):
        spec, freqs = sine_spec
        v = ta.spectral_valley_energy(spec, freqs, n_bands=6)
        assert v.shape == (6,)

    def test_nonneg(self, ta, noise_spec):
        spec, freqs = noise_spec
        v = ta.spectral_valley_energy(spec, freqs)
        assert np.all(v >= 0)


class TestTonalPowerRatio:
    def test_sine_high_tpr(self, ta, sine_spec):
        spec, freqs = sine_spec
        tpr = ta.tonal_power_ratio(spec, freqs, f0=440.0)
        assert tpr > 0.3

    def test_zero_f0(self, ta, sine_spec):
        spec, freqs = sine_spec
        assert ta.tonal_power_ratio(spec, freqs, f0=0.0) == 0.0

    def test_range(self, ta, sine_spec):
        spec, freqs = sine_spec
        tpr = ta.tonal_power_ratio(spec, freqs, f0=440.0)
        assert 0 <= tpr <= 1.0

    def test_empty_spectrum(self, ta):
        assert ta.tonal_power_ratio(np.array([]), np.array([]), f0=440.0) == 0.0


class TestRoughness:
    def test_nonneg(self, ta, sine_spec):
        spec, freqs = sine_spec
        r = ta.roughness_sethares(spec, freqs)
        assert r >= 0

    def test_empty(self, ta):
        assert ta.roughness_sethares(np.array([]), np.array([])) == 0.0

    def test_unison_low_roughness(self, ta):
        sig = _sine(440.0)
        spec, freqs = _spectrum_and_freqs(sig)
        r_unison = ta.roughness_sethares(spec, freqs)
        sig2 = _sine(440.0) + 0.5 * _sine(466.16)
        spec2, freqs2 = _spectrum_and_freqs(sig2)
        r_minor = ta.roughness_sethares(spec2, freqs2)
        assert r_minor >= r_unison


class TestSpectralFlux:
    def test_shape(self, ta):
        frames = np.random.default_rng(0).standard_normal((10, 128))
        flux = ta.spectral_flux_texture(frames)
        assert len(flux) == 9

    def test_identical_frames_zero(self, ta):
        frame = np.ones((1, 64))
        frames = np.vstack([frame, frame, frame])
        flux = ta.spectral_flux_texture(frames)
        assert np.allclose(flux, 0.0)

    def test_1d_input(self, ta):
        flux = ta.spectral_flux_texture(np.array([1.0]))
        assert len(flux) == 1


class TestPolyphony:
    def test_sine_one_peak(self, ta, sine_spec):
        spec, freqs = sine_spec
        p = ta.polyphony_estimate(spec, freqs)
        assert p >= 1

    def test_empty(self, ta):
        assert ta.polyphony_estimate(np.array([]), np.array([])) == 0

    def test_noise_many_peaks(self, ta, noise_spec):
        spec, freqs = noise_spec
        p = ta.polyphony_estimate(spec, freqs, threshold=0.05)
        assert p > 1


class TestNHR:
    def test_sine_low_nhr(self, ta, sine_spec):
        spec, freqs = sine_spec
        nhr = ta.noise_to_harmonic_ratio(spec, freqs, f0=440.0)
        assert nhr < 3.0

    def test_zero_f0(self, ta, sine_spec):
        spec, freqs = sine_spec
        nhr = ta.noise_to_harmonic_ratio(spec, freqs, f0=0.0)
        assert nhr == 0.0


class TestTemporalFlatness:
    def test_constant_envelope(self, ta):
        env = np.ones(100)
        assert abs(ta.temporal_flatness(env) - 1.0) < 0.01

    def test_range(self, ta):
        env = np.abs(np.random.default_rng(0).standard_normal(200))
        tf = ta.temporal_flatness(env)
        assert 0 <= tf <= 1.0

    def test_empty(self, ta):
        assert ta.temporal_flatness(np.array([])) == 0.0


class TestOnsetDensity:
    def test_positive(self, ta):
        env = np.abs(_sine(5.0, dur=2.0, sr=100))
        od = ta.onset_density(env, sr=100, hop=1)
        assert od > 0

    def test_empty(self, ta):
        assert ta.onset_density(np.array([]), sr=SR, hop=512) == 0.0
