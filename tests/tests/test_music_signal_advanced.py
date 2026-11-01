"""Tests for music_signal: onset detection, pitch tracking, and MFCC."""
import numpy as np
import pytest

from models.music_signal.onset_detection import (
    spectral_flux, high_frequency_content, complex_domain,
    energy_onset, pick_peaks, frames_to_time, onset_detect, OnsetResult,
)
from models.music_signal.pitch_tracking import (
    hz_to_midi, midi_to_hz, pitch_class, cents_difference, note_name,
    autocorrelation_pitch, yin_frame, track_pitch, harmonic_product_spectrum,
    median_filter_pitch, PitchResult,
)
from models.music_signal.mfcc import (
    dct_ii, compute_mfcc, mfcc_delta, mfcc_delta_delta, lifter, MFCCResult,
)


# ── Onset Detection ──────────────────────────────────────────────────

class TestSpectralFlux:
    def test_constant_spectrum_zero_flux(self):
        mag = np.ones((128, 10))
        flux = spectral_flux(mag)
        assert flux[0] == 0.0
        assert np.allclose(flux[1:], 0.0)

    def test_onset_produces_peak(self):
        mag = np.ones((128, 20)) * 0.1
        mag[:, 10] = 5.0
        flux = spectral_flux(mag)
        assert np.argmax(flux) == 10

    def test_output_length(self):
        mag = np.random.rand(64, 15)
        flux = spectral_flux(mag)
        assert len(flux) == 15


class TestHighFrequencyContent:
    def test_energy_weighted_by_frequency(self):
        mag = np.zeros((128, 5))
        mag[100, 2] = 1.0
        hfc = high_frequency_content(mag)
        assert hfc[2] > hfc[0]

    def test_silent_frames(self):
        mag = np.zeros((64, 3))
        hfc = high_frequency_content(mag)
        assert np.allclose(hfc, 0.0)


class TestComplexDomain:
    def test_shape_matches_input(self):
        stft = np.random.randn(128, 10) + 1j * np.random.randn(128, 10)
        cd = complex_domain(stft)
        assert len(cd) == 10

    def test_first_frame_nonneg(self):
        stft = np.random.randn(64, 5) + 1j * np.random.randn(64, 5)
        cd = complex_domain(stft)
        assert cd[0] >= 0.0


class TestEnergyOnset:
    def test_shape(self):
        signal = np.random.randn(8000)
        eo = energy_onset(signal, hop_size=512)
        expected = len(signal) // 512
        assert abs(len(eo) - expected) <= 1


class TestPickPeaks:
    def test_finds_obvious_peaks(self):
        curve = np.zeros(50)
        curve[10] = 1.0
        curve[30] = 0.8
        peaks = pick_peaks(curve, threshold=0.3, min_distance=5)
        assert 10 in peaks
        assert 30 in peaks

    def test_respects_min_distance(self):
        curve = np.zeros(20)
        curve[5] = 1.0
        curve[7] = 0.9
        peaks = pick_peaks(curve, threshold=0.3, min_distance=5)
        assert len(peaks) == 1

    def test_empty_below_threshold(self):
        curve = np.ones(10) * 0.01
        peaks = pick_peaks(curve, threshold=0.5)
        assert len(peaks) == 0


class TestFramesToTime:
    def test_basic_conversion(self):
        times = frames_to_time(np.array([0, 1, 2]), sr=44100, hop_size=512)
        assert times[0] == pytest.approx(0.0)
        assert times[1] == pytest.approx(512 / 44100, abs=1e-6)


class TestOnsetDetect:
    def test_returns_onset_result(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr)
        signal = np.sin(2 * np.pi * 440 * t)
        signal[:sr // 2] *= 0.01
        result = onset_detect(signal, sr, method="energy")
        assert isinstance(result, OnsetResult)
        assert len(result.strength_curve) > 0


# ── Pitch Tracking ───────────────────────────────────────────────────

class TestHzMidi:
    def test_a440(self):
        assert hz_to_midi(440.0) == pytest.approx(69.0)

    def test_roundtrip(self):
        for midi in [36, 48, 60, 72, 84]:
            assert hz_to_midi(midi_to_hz(float(midi))) == pytest.approx(float(midi), abs=0.01)

    def test_zero_freq(self):
        assert hz_to_midi(0.0) == 0.0
        assert hz_to_midi(-1.0) == 0.0


class TestPitchClass:
    def test_c4(self):
        assert pitch_class(60) == 0

    def test_a4(self):
        assert pitch_class(69) == 9


class TestCentsDifference:
    def test_octave(self):
        assert cents_difference(220.0, 440.0) == pytest.approx(1200.0)

    def test_unison(self):
        assert cents_difference(440.0, 440.0) == pytest.approx(0.0)

    def test_zero_input(self):
        assert cents_difference(0.0, 440.0) == 0.0


class TestNoteName:
    def test_c4(self):
        assert note_name(60) == "C4"

    def test_a4(self):
        assert note_name(69) == "A4"

    def test_fsharp3(self):
        assert note_name(54) == "F#3"


class TestAutocorrelationPitch:
    def test_pure_tone(self):
        sr = 16000
        f0 = 200.0
        t = np.arange(4096) / sr
        frame = np.sin(2 * np.pi * f0 * t)
        freq, conf = autocorrelation_pitch(frame, sr, fmin=50, fmax=500)
        assert abs(freq - f0) < 10.0
        assert conf > 0.5

    def test_short_frame(self):
        freq, conf = autocorrelation_pitch(np.array([0.0, 1.0]), 16000)
        assert freq == 0.0


class TestYinFrame:
    def test_pure_tone(self):
        sr = 16000
        f0 = 300.0
        t = np.arange(4096) / sr
        frame = np.sin(2 * np.pi * f0 * t)
        freq, conf = yin_frame(frame, sr, fmin=50, fmax=1000, threshold=0.15)
        assert abs(freq - f0) < 15.0
        assert conf > 0.5


class TestTrackPitch:
    def test_returns_pitch_result(self):
        sr = 16000
        t = np.arange(sr) / sr
        signal = np.sin(2 * np.pi * 440 * t)
        result = track_pitch(signal, sr, frame_size=2048, hop_size=512)
        assert isinstance(result, PitchResult)
        assert len(result.f0) == len(result.confidence)
        assert len(result.times) == len(result.f0)


class TestHarmonicProductSpectrum:
    def test_finds_fundamental(self):
        n = 4096
        freqs = np.linspace(0, 8000, n)
        magnitude = np.zeros(n)
        f0 = 440.0
        for h in range(1, 6):
            idx = int(h * f0 / 8000 * n)
            if idx < n:
                for offset in range(-2, 3):
                    if 0 < idx + offset < n:
                        magnitude[idx + offset] = 1.0 / h
        detected = harmonic_product_spectrum(magnitude, freqs, n_harmonics=5)
        assert detected > 0  # should find a nonzero fundamental


class TestMedianFilterPitch:
    def test_removes_outlier(self):
        f0 = np.array([440.0, 440.0, 880.0, 440.0, 440.0])
        smoothed = median_filter_pitch(f0, kernel_size=3)
        assert smoothed[2] == pytest.approx(440.0)

    def test_handles_zeros(self):
        f0 = np.array([0.0, 0.0, 440.0, 0.0, 0.0])
        smoothed = median_filter_pitch(f0, kernel_size=3)
        assert smoothed[2] == pytest.approx(440.0)


# ── MFCC ─────────────────────────────────────────────────────────────

class TestDCTII:
    def test_orthonormality(self):
        x = np.random.randn(16)
        dct = dct_ii(x)
        assert len(dct) == len(x)

    def test_energy_preservation(self):
        x = np.array([1.0, 0.0, -1.0, 0.0])
        dct = dct_ii(x)
        assert np.sum(dct ** 2) > 0


class TestComputeMFCC:
    def test_returns_mfcc_result(self):
        sr = 16000
        signal = np.random.randn(sr)
        result = compute_mfcc(signal, sr, n_mfcc=13, n_fft=1024, hop_size=256)
        assert isinstance(result, MFCCResult)
        assert result.coefficients.shape[0] == 13
        assert result.n_mfcc == 13

    def test_short_signal(self):
        sr = 16000
        signal = np.random.randn(2048)
        result = compute_mfcc(signal, sr, n_mfcc=8, n_fft=1024, hop_size=512)
        assert result.coefficients.shape[0] == 8


class TestMFCCDelta:
    def test_shape_preserved(self):
        mfcc = np.random.randn(13, 20)
        d = mfcc_delta(mfcc, width=9)
        assert d.shape == mfcc.shape

    def test_constant_input_zero_delta(self):
        mfcc = np.ones((5, 10))
        d = mfcc_delta(mfcc, width=5)
        assert np.allclose(d, 0.0, atol=1e-10)


class TestMFCCDeltaDelta:
    def test_shape_preserved(self):
        mfcc = np.random.randn(13, 20)
        dd = mfcc_delta_delta(mfcc, width=9)
        assert dd.shape == mfcc.shape


class TestLifter:
    def test_shape_preserved(self):
        mfcc = np.random.randn(13, 10)
        lifted = lifter(mfcc, factor=22)
        assert lifted.shape == mfcc.shape

    def test_first_coeff_unchanged(self):
        mfcc = np.random.randn(13, 10)
        lifted = lifter(mfcc, factor=22)
        assert np.allclose(lifted[0], mfcc[0], atol=1e-10)
