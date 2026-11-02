"""Tests for music_timbre: descriptors, envelope, timbre space, harmonic analysis."""
import numpy as np
import pytest

from models.music_timbre.descriptors import (
    spectral_centroid_from_spectrum, spectral_spread, spectral_skewness,
    spectral_kurtosis, spectral_flatness, spectral_crest,
    brightness, warmth, roughness_sethares, inharmonicity, odd_even_ratio,
)
from models.music_timbre.envelope import (
    ADSRParams, amplitude_envelope, rms_envelope,
    estimate_adsr, synthesise_adsr, extract_envelope,
)
from models.music_timbre.timbre_space import (
    TimbrePoint, TimbreSpace, grey_timbre_axes, mcadams_timbre_axes,
    descriptors_to_coordinates, timbre_similarity, mds_reduce, stress,
)
from models.music_timbre.timbre_descriptors import (
    brightness as td_brightness, warmth as td_warmth,
    spectral_irregularity, odd_even_ratio as td_odd_even,
    inharmonicity as td_inharmonicity, log_attack_time,
    temporal_centroid, spectral_tilt, roughness_sethares as td_roughness,
    sharpness_zwicker, estimate_spectral_envelope, compute_timbre_descriptors,
)
from models.music_timbre.harmonic_analysis import (
    extract_partials, harmonic_to_noise_ratio, spectral_envelope,
    estimate_formants, harmonic_deviation, analyse_harmonics,
)


# ── Spectral Descriptors ─────────────────────────────────────────────

class TestSpectralCentroid:
    def test_single_peak(self):
        freqs = np.arange(100, dtype=float)
        mag = np.zeros(100)
        mag[50] = 1.0
        assert spectral_centroid_from_spectrum(mag, freqs) == pytest.approx(50.0)

    def test_silent(self):
        assert spectral_centroid_from_spectrum(np.zeros(10), np.arange(10, dtype=float)) == 0.0


class TestSpectralSpread:
    def test_narrow_peak(self):
        freqs = np.arange(100, dtype=float)
        mag = np.zeros(100)
        mag[50] = 1.0
        assert spectral_spread(mag, freqs) == pytest.approx(0.0)

    def test_wide_distribution(self):
        freqs = np.arange(100, dtype=float)
        mag = np.ones(100)
        assert spectral_spread(mag, freqs) > 0


class TestSpectralFlatness:
    def test_white_noise(self):
        mag = np.ones(100)
        assert spectral_flatness(mag) == pytest.approx(1.0, abs=0.01)

    def test_tonal(self):
        mag = np.zeros(100)
        mag[30] = 10.0
        mag[31] = 0.1
        assert spectral_flatness(mag) < 0.5


class TestSpectralCrest:
    def test_flat(self):
        mag = np.ones(50)
        assert spectral_crest(mag) == pytest.approx(1.0)

    def test_peaky(self):
        mag = np.zeros(50)
        mag[25] = 10.0
        assert spectral_crest(mag) > 1.0


class TestBrightness:
    def test_all_above_cutoff(self):
        freqs = np.linspace(2000, 8000, 100)
        mag = np.ones(100)
        assert brightness(mag, freqs, cutoff=1500) == pytest.approx(1.0)

    def test_all_below_cutoff(self):
        freqs = np.linspace(100, 1000, 100)
        mag = np.ones(100)
        assert brightness(mag, freqs, cutoff=1500) == pytest.approx(0.0)


class TestWarmth:
    def test_energy_in_band(self):
        freqs = np.linspace(100, 500, 50)
        mag = np.ones(50)
        assert warmth(mag, freqs) == pytest.approx(1.0)


class TestRoughness:
    def test_unison_no_roughness(self):
        freqs = np.array([440.0, 440.0])
        amps = np.array([1.0, 1.0])
        assert roughness_sethares(freqs, amps) == pytest.approx(0.0)

    def test_semitone_beats(self):
        freqs = np.array([440.0, 466.0])
        amps = np.array([1.0, 1.0])
        assert roughness_sethares(freqs, amps) > 0

    def test_single_partial(self):
        assert roughness_sethares(np.array([440.0]), np.array([1.0])) == 0.0


class TestInharmonicity:
    def test_harmonic_series(self):
        f0 = 100.0
        partials = np.array([f0 * k for k in range(1, 6)])
        assert inharmonicity(partials, f0) == pytest.approx(0.0, abs=0.01)

    def test_inharmonic(self):
        f0 = 100.0
        partials = np.array([100, 205, 310, 420, 530], dtype=float)
        assert inharmonicity(partials, f0) > 0.01


class TestOddEvenRatio:
    def test_odd_dominant(self):
        amps = np.array([1.0, 0.0, 1.0, 0.0])
        assert odd_even_ratio(amps) == float('inf')

    def test_balanced(self):
        amps = np.ones(8)
        ratio = odd_even_ratio(amps)
        assert ratio == pytest.approx(1.0)


# ── ADSR Envelope ────────────────────────────────────────────────────

class TestAmplitudeEnvelope:
    def test_shape(self):
        signal = np.random.randn(4096)
        env = amplitude_envelope(signal, hop_size=256)
        assert len(env) == 4096 // 256


class TestEstimateADSR:
    def test_percussive(self):
        sr = 44100
        env = np.zeros(100)
        env[2] = 1.0  # fast attack
        env[3:10] = np.linspace(0.8, 0.3, 7)
        env[10:80] = 0.3
        env[80:] = np.linspace(0.3, 0.0, 20)
        adsr = estimate_adsr(env, sr, hop_size=256)
        assert adsr.attack_ms < adsr.decay_ms + adsr.release_ms

    def test_short_env(self):
        adsr = estimate_adsr(np.array([0.5, 1.0]), sr=44100, hop_size=256)
        assert adsr.attack_ms > 0


class TestSynthesiseADSR:
    def test_length(self):
        adsr = ADSRParams(10, 20, 0.5, 30)
        env = synthesise_adsr(adsr, total_ms=100, sr=44100)
        expected = int(44100 * 100 / 1000)
        assert len(env) == expected

    def test_starts_at_zero(self):
        adsr = ADSRParams(50, 50, 0.5, 50)
        env = synthesise_adsr(adsr, total_ms=200, sr=1000)
        assert env[0] == pytest.approx(0.0, abs=0.05)


class TestExtractEnvelope:
    def test_returns_result(self):
        sr = 16000
        t = np.arange(sr) / sr
        signal = np.sin(2 * np.pi * 440 * t) * np.exp(-3 * t)
        result = extract_envelope(signal, sr)
        assert result.adsr.attack_ms >= 0
        assert len(result.amplitude_envelope) > 0


# ── Timbre Space ─────────────────────────────────────────────────────

class TestTimbreSpace:
    def test_grey_axes(self):
        space = grey_timbre_axes()
        assert space.n_dims == 3
        assert space.n_points == 0

    def test_add_and_distance(self):
        space = grey_timbre_axes()
        space.add_point(TimbrePoint("a", np.array([0.0, 0.0, 0.0])))
        space.add_point(TimbrePoint("b", np.array([1.0, 1.0, 1.0])))
        d = space.distance("a", "b")
        assert d.euclidean == pytest.approx(np.sqrt(3.0))

    def test_nearest(self):
        space = grey_timbre_axes()
        space.add_point(TimbrePoint("a", np.array([0.0, 0.0, 0.0])))
        space.add_point(TimbrePoint("b", np.array([1.0, 0.0, 0.0])))
        nearest = space.nearest(np.array([0.1, 0.0, 0.0]), k=1)
        assert nearest[0][0] == "a"

    def test_interpolate(self):
        space = grey_timbre_axes()
        space.add_point(TimbrePoint("a", np.array([0.0, 0.0, 0.0])))
        space.add_point(TimbrePoint("b", np.array([2.0, 2.0, 2.0])))
        mid = space.interpolate("a", "b", 0.5)
        assert np.allclose(mid, [1.0, 1.0, 1.0])

    def test_centroid(self):
        space = grey_timbre_axes()
        space.add_point(TimbrePoint("a", np.array([0.0, 0.0, 0.0])))
        space.add_point(TimbrePoint("b", np.array([2.0, 2.0, 2.0])))
        c = space.centroid()
        assert np.allclose(c, [1.0, 1.0, 1.0])


class TestDescriptorsToCoordinates:
    def test_output_range(self):
        coords = descriptors_to_coordinates(50.0, 2000.0, 0.5, 0.5, 1.0)
        assert all(0 <= c <= 1 for c in coords)


class TestTimbreSimilarity:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert timbre_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert timbre_similarity(a, b) == pytest.approx(0.0)


class TestMDSReduce:
    def test_preserves_structure(self):
        dmat = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
        coords = mds_reduce(dmat, n_dims=2)
        assert coords.shape == (3, 2)

    def test_stress_low_for_perfect(self):
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        dmat = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                dmat[i, j] = np.linalg.norm(coords[i] - coords[j])
        recon = mds_reduce(dmat, n_dims=2)
        s = stress(dmat, recon)
        assert s < 0.1


# ── Timbre Descriptors (timbre_descriptors.py) ───────────────────────

class TestTimbreDescriptorsModule:
    def test_brightness(self):
        freqs = np.linspace(0, 8000, 200)
        spectrum = np.random.rand(200)
        b = td_brightness(spectrum, freqs)
        assert 0.0 <= b <= 1.0

    def test_warmth(self):
        freqs = np.linspace(0, 8000, 200)
        spectrum = np.random.rand(200)
        w = td_warmth(spectrum, freqs)
        assert 0.0 <= w <= 1.0

    def test_spectral_irregularity(self):
        mags = np.array([1.0, 0.5, 0.8, 0.3, 0.6])
        si = spectral_irregularity(mags)
        assert si >= 0.0

    def test_inharmonicity_perfect(self):
        f0 = 200.0
        partials = np.array([200, 400, 600, 800], dtype=float)
        assert td_inharmonicity(partials, f0) == pytest.approx(0.0, abs=0.01)

    def test_sharpness_zwicker(self):
        freqs = np.linspace(100, 8000, 200)
        spectrum = np.random.rand(200)
        s = sharpness_zwicker(spectrum, freqs)
        assert s >= 0.0

    def test_compute_timbre_descriptors(self):
        sr = 16000
        signal = np.random.randn(sr)
        spectrum = np.abs(np.fft.rfft(signal[:2048]))
        freqs = np.fft.rfftfreq(2048, 1.0 / sr)
        td = compute_timbre_descriptors(signal, float(sr), spectrum, freqs)
        assert td.brightness >= 0.0
        assert td.spectral_centroid_hz > 0


# ── Harmonic Analysis ────────────────────────────────────────────────

class TestExtractPartials:
    def test_finds_harmonics(self):
        n = 4096
        sr = 16000
        f0 = 200.0
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        magnitude = np.zeros(len(freqs))
        phase = np.zeros(len(freqs))
        for k in range(1, 6):
            idx = int(k * f0 / sr * n)
            if idx < len(magnitude):
                magnitude[idx] = 1.0 / k
        partials = extract_partials(magnitude, phase, freqs, f0, n_harmonics=5)
        assert len(partials.frequencies) >= 3

    def test_no_harmonics(self):
        freqs = np.linspace(0, 100, 50)
        mag = np.zeros(50)
        phase = np.zeros(50)
        partials = extract_partials(mag, phase, freqs, 440.0, n_harmonics=5)
        assert len(partials.frequencies) == 0


class TestHNR:
    def test_harmonic_signal_high_hnr(self):
        n = 2048
        sr = 16000
        f0 = 200.0
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        magnitude = np.ones(len(freqs)) * 0.001
        for k in range(1, 10):
            idx = int(k * f0 / sr * n)
            if idx < len(magnitude):
                magnitude[idx] = 1.0 / k
        phase = np.zeros(len(freqs))
        partials = extract_partials(magnitude, phase, freqs, f0)
        hnr = harmonic_to_noise_ratio(magnitude, partials, freqs)
        assert hnr > 5.0


class TestSpectralEnvelope:
    def test_output_length(self):
        mag = np.random.rand(100) + 0.01
        env = spectral_envelope(mag, order=10)
        assert len(env) == len(mag)


class TestEstimateFormants:
    def test_finds_peaks(self):
        freqs = np.linspace(0, 5000, 500)
        env = np.ones(500) * 0.1
        env[50] = 1.0
        env[150] = 0.8
        env[250] = 0.6
        formants = estimate_formants(env, freqs, n_formants=3)
        assert len(formants) >= 1


class TestHarmonicDeviation:
    def test_perfect_rolloff(self):
        from models.music_timbre.harmonic_analysis import PartialSet
        amps = np.array([1.0, 0.5, 0.25, 0.125])
        partials = PartialSet(
            frequencies=np.array([200, 400, 600, 800], dtype=float),
            amplitudes=amps,
            phases=np.zeros(4),
            f0=200.0,
        )
        dev = harmonic_deviation(partials)
        assert dev < 0.5


class TestAnalyseHarmonics:
    def test_full_pipeline(self):
        n = 4096
        sr = 16000
        f0 = 300.0
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        magnitude = np.random.rand(len(freqs)) * 0.01
        phase = np.random.rand(len(freqs)) * 2 * np.pi
        for k in range(1, 8):
            idx = int(k * f0 / sr * n)
            if idx < len(magnitude):
                magnitude[idx] = 1.0 / k
                phase[idx] = 0.0
        result = analyse_harmonics(magnitude, phase, freqs, f0)
        assert result.n_harmonics >= 3
        assert result.hnr_db > 0
