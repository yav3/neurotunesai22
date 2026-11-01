"""Tests for audio restoration module."""
import numpy as np
import pytest

from models.music_audio.audio_restoration import (
    NoiseProfile, estimate_noise_profile, spectral_subtraction,
    wiener_filter, detect_clipping, declip_cubic,
    bandwidth_extension, estimate_reverb_tail, dereverberate,
    adaptive_noise_gate,
)


SR = 16000


def _sine(freq, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


class TestNoiseProfile:
    def test_estimate(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, SR)
        profile = estimate_noise_profile(noise, SR)
        assert isinstance(profile, NoiseProfile)
        assert profile.sr == SR
        assert len(profile.spectrum) == 1025

    def test_short_segment(self):
        noise = np.random.default_rng(42).normal(0, 0.01, 512)
        profile = estimate_noise_profile(noise, SR, n_fft=2048)
        assert len(profile.spectrum) == 1025


class TestSpectralSubtraction:
    def test_output_finite(self):
        rng = np.random.default_rng(42)
        clean = _sine(440, 1.0)
        noise = rng.normal(0, 0.1, len(clean))
        noisy = clean + noise
        profile = estimate_noise_profile(noise[:SR // 4], SR)
        restored = spectral_subtraction(noisy, profile)
        assert np.all(np.isfinite(restored))
        assert len(restored) == len(noisy)

    def test_preserves_length(self):
        rng = np.random.default_rng(42)
        sig = _sine(440, 0.5) + rng.normal(0, 0.05, SR // 2)
        profile = estimate_noise_profile(rng.normal(0, 0.05, SR // 4), SR)
        restored = spectral_subtraction(sig, profile)
        assert len(restored) == len(sig)


class TestWienerFilter:
    def test_output_finite(self):
        rng = np.random.default_rng(42)
        clean = _sine(440, 0.5)
        noise = rng.normal(0, 0.1, len(clean))
        noisy = clean + noise
        profile = estimate_noise_profile(noise[:2048], SR)
        restored = wiener_filter(noisy, profile)
        assert np.all(np.isfinite(restored))
        assert len(restored) == len(noisy)

    def test_output_length(self):
        rng = np.random.default_rng(42)
        sig = rng.normal(0, 0.1, SR)
        profile = estimate_noise_profile(sig[:2048], SR)
        out = wiener_filter(sig, profile)
        assert len(out) == len(sig)


class TestDetectClipping:
    def test_no_clipping(self):
        sig = _sine(440, 0.5) * 0.5
        regions = detect_clipping(sig)
        assert len(regions) == 0

    def test_detects_clipped(self):
        sig = _sine(440, 0.5) * 2.0
        sig = np.clip(sig, -1.0, 1.0)
        regions = detect_clipping(sig, threshold=0.99)
        assert len(regions) > 0

    def test_min_run(self):
        sig = np.zeros(100)
        sig[50] = 1.0
        regions = detect_clipping(sig, threshold=0.99, min_run=2)
        assert len(regions) == 0


class TestDeclipCubic:
    def test_restores_shape(self):
        sig = _sine(440, 0.1) * 2.0
        clipped = np.clip(sig, -1.0, 1.0)
        restored = declip_cubic(clipped)
        clipped_energy = np.sum(np.abs(clipped) >= 0.99)
        restored_energy = np.sum(np.abs(restored) >= 0.99)
        assert restored_energy <= clipped_energy

    def test_no_clipping_passthrough(self):
        sig = _sine(440, 0.1) * 0.5
        restored = declip_cubic(sig)
        np.testing.assert_allclose(restored, sig, atol=1e-10)


class TestBandwidthExtension:
    def test_extends(self):
        sig = _sine(440, 0.5)
        result = bandwidth_extension(sig, SR)
        assert len(result) == len(sig)

    def test_no_change_if_full_band(self):
        rng = np.random.default_rng(42)
        sig = rng.normal(0, 0.1, SR)
        result = bandwidth_extension(sig, SR)
        assert len(result) == len(sig)


class TestEstimateReverbTail:
    def test_short_ir(self):
        ir = np.zeros(SR)
        ir[0] = 1.0
        ir[1:100] = np.exp(-np.linspace(0, 10, 99))
        rt60 = estimate_reverb_tail(ir, SR)
        assert 0 < rt60 < 1.0

    def test_empty(self):
        assert estimate_reverb_tail(np.array([]), SR) == 0.0


class TestDereverberate:
    def test_reduces_reverb(self):
        rng = np.random.default_rng(42)
        sig = _sine(440, 1.0)
        reverb_tail = np.zeros(SR // 4)
        reverb_tail[:1000] = 0.3 * np.exp(-np.linspace(0, 5, 1000))
        reverbed = np.convolve(sig, np.concatenate([[1.0], reverb_tail]))[:len(sig)]
        result = dereverberate(reverbed, SR, rt60_estimate=0.2)
        assert len(result) == len(reverbed)

    def test_preserves_length(self):
        sig = _sine(440, 0.5)
        result = dereverberate(sig, SR)
        assert len(result) == len(sig)


class TestAdaptiveNoiseGate:
    def test_gates_silence(self):
        sig = np.zeros(SR)
        sig[SR // 4 : SR // 2] = _sine(440, 0.25)
        gated = adaptive_noise_gate(sig, SR, threshold_db=-30)
        silent_region = gated[:SR // 8]
        active_region = gated[SR // 4 + 1000 : SR // 2 - 1000]
        assert np.mean(silent_region ** 2) < np.mean(active_region ** 2)

    def test_empty(self):
        result = adaptive_noise_gate(np.array([]), SR)
        assert len(result) == 0
