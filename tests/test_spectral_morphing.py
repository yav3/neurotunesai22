"""Tests for spectral morphing."""
import numpy as np
import pytest
from models.music_timbre.spectral_morphing import (
    spectral_envelope, morph_spectral, formant_transplant,
    brightness_match, timbre_distance, harmonic_brightness,
    stft_frames, istft_frames,
)

SR = 44100

def _tone(freq=440, dur=0.5, amp=0.5):
    t = np.arange(int(SR * dur)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


class TestSpectralEnvelope:
    def test_shape(self):
        s = _tone(440, 0.1)
        env = spectral_envelope(s, SR)
        assert env.shape[0] == 1025

    def test_normalized(self):
        s = _tone(440, 0.1)
        env = spectral_envelope(s, SR)
        assert np.max(env) <= 1.0 + 1e-6


class TestSTFT:
    def test_roundtrip(self):
        s = _tone(440, 0.2)
        frames = stft_frames(s)
        rec = istft_frames(frames)
        min_len = min(len(s), len(rec))
        corr = np.corrcoef(s[:min_len], rec[:min_len])[0, 1]
        assert corr > 0.95


class TestMorph:
    def test_alpha_zero_is_source(self):
        src = _tone(440, 0.3)
        tgt = _tone(880, 0.3)
        out = morph_spectral(src, tgt, SR, alpha=0.0)
        assert out.shape[0] > 0

    def test_alpha_one_different(self):
        src = _tone(440, 0.3)
        tgt = _tone(880, 0.3)
        out0 = morph_spectral(src, tgt, SR, alpha=0.0)
        out1 = morph_spectral(src, tgt, SR, alpha=1.0)
        min_len = min(len(out0), len(out1))
        assert not np.allclose(out0[:min_len], out1[:min_len], atol=0.01)


class TestFormantTransplant:
    def test_output_shape(self):
        src = _tone(440, 0.2)
        tgt = _tone(880, 0.2)
        out = formant_transplant(src, tgt, SR)
        assert out.shape[0] > 0
        assert np.all(np.isfinite(out))


class TestTimbreDistance:
    def test_self_zero(self):
        env = np.ones(100)
        assert timbre_distance(env, env) < 1e-6

    def test_different_positive(self):
        a = np.ones(100)
        b = np.ones(100) * 2
        assert timbre_distance(a, b) > 0


class TestHarmonicBrightness:
    def test_fundamental_dominant(self):
        s = _tone(440, 0.5, amp=1.0)
        hb = harmonic_brightness(s, SR, 440, n_harmonics=5)
        assert hb[0] > hb[1]  # fundamental > first overtone
        assert abs(np.sum(hb) - 1.0) < 0.01
