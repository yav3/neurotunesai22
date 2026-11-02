"""Tests for subtractive and spectral synthesis modules."""
import numpy as np
import pytest

from models.music_synthesis.subtractive import (
    FilterType,
    biquad_coefficients,
    apply_biquad,
    apply_biquad_tv,
    FilterEnvelope,
    LFO,
    SubtractiveSynth,
)
from models.music_synthesis.spectral_synthesis import (
    stft_frames,
    detect_peaks,
    SinusoidalPeak,
    SinusoidalTrack,
    track_sinusoids,
    resynthesize_sinusoids,
    extract_residual,
    spectral_morph,
    sms_analyze,
)


# ── Subtractive synthesis ───────────────────────────────────────────

class TestBiquadCoefficients:
    def test_lowpass_shape(self):
        b, a = biquad_coefficients(FilterType.LOWPASS, 1000.0)
        assert b.shape == (3,)
        assert a.shape == (3,)
        assert a[0] == 1.0

    def test_highpass_shape(self):
        b, a = biquad_coefficients(FilterType.HIGHPASS, 5000.0)
        assert b.shape == (3,)
        assert a[0] == 1.0

    def test_bandpass_shape(self):
        b, a = biquad_coefficients(FilterType.BANDPASS, 2000.0, q=2.0)
        assert b.shape == (3,)
        assert a[0] == 1.0

    def test_cutoff_clamp(self):
        b_low, _ = biquad_coefficients(FilterType.LOWPASS, 5.0)
        b_high, _ = biquad_coefficients(FilterType.LOWPASS, 99999.0)
        assert np.all(np.isfinite(b_low))
        assert np.all(np.isfinite(b_high))


class TestApplyBiquad:
    def test_dc_passthrough_lowpass(self):
        b, a = biquad_coefficients(FilterType.LOWPASS, 5000.0)
        dc = np.ones(500)
        out = apply_biquad(dc, b, a)
        assert abs(out[-1] - 1.0) < 0.05

    def test_highpass_blocks_dc(self):
        b, a = biquad_coefficients(FilterType.HIGHPASS, 500.0)
        dc = np.ones(2000)
        out = apply_biquad(dc, b, a)
        assert abs(out[-1]) < 0.05

    def test_time_varying_runs(self):
        sig = np.random.randn(1000)
        cutoff = np.linspace(500, 3000, 1000)
        out = apply_biquad_tv(sig, FilterType.LOWPASS, cutoff)
        assert len(out) == 1000
        assert np.all(np.isfinite(out))


class TestFilterEnvelope:
    def test_shape(self):
        env = FilterEnvelope(attack=0.01, decay=0.05, sustain=0.5, release=0.1, amount=3000.0)
        curve = env.generate(base_cutoff=500.0, duration_s=0.5, sr=44100)
        assert len(curve) == int(44100 * 0.5)
        assert curve[0] == pytest.approx(500.0, abs=10)

    def test_peak_above_base(self):
        env = FilterEnvelope(amount=4000.0)
        curve = env.generate(base_cutoff=1000.0, duration_s=0.5)
        assert np.max(curve) > 1000.0

    def test_zero_duration(self):
        env = FilterEnvelope()
        curve = env.generate(500.0, 0.0)
        assert len(curve) == 0


class TestLFO:
    def test_shape_and_range(self):
        lfo = LFO(rate_hz=5.0, depth=1000.0)
        sig = lfo.generate(1.0)
        assert len(sig) == 44100
        assert np.max(sig) <= 1000.0 + 1.0
        assert np.min(sig) >= -1000.0 - 1.0

    def test_zero_depth(self):
        lfo = LFO(rate_hz=10.0, depth=0.0)
        sig = lfo.generate(0.5)
        assert np.allclose(sig, 0.0)


class TestSubtractiveSynth:
    def test_basic_render(self):
        synth = SubtractiveSynth(cutoff_hz=2000.0)
        out = synth.render(440.0, 0.2)
        assert len(out) == int(44100 * 0.2)
        assert out.dtype == np.float32
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_square_wave(self):
        synth = SubtractiveSynth(cutoff_hz=3000.0)
        out = synth.render(440.0, 0.1, waveform="square")
        assert len(out) > 0

    def test_with_envelope_and_lfo(self):
        synth = SubtractiveSynth(
            cutoff_hz=1000.0,
            filter_envelope=FilterEnvelope(amount=3000.0),
            lfo=LFO(rate_hz=6.0, depth=500.0),
        )
        out = synth.render(220.0, 0.3)
        assert len(out) == int(44100 * 0.3)
        assert np.all(np.isfinite(out))

    def test_high_resonance(self):
        synth = SubtractiveSynth(cutoff_hz=1500.0, resonance=8.0)
        out = synth.render(440.0, 0.1)
        assert np.all(np.isfinite(out))

    def test_zero_duration(self):
        synth = SubtractiveSynth()
        out = synth.render(440.0, 0.0)
        assert len(out) == 0


# ── Spectral synthesis ──────────────────────────────────────────────

class TestSTFT:
    def test_frame_shape(self):
        sig = np.random.randn(8000)
        mags, phases = stft_frames(sig, frame_size=1024, hop=256)
        assert mags.shape[1] == 513  # 1024//2 + 1
        assert mags.shape == phases.shape

    def test_magnitudes_nonneg(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4096) / 44100)
        mags, _ = stft_frames(sig)
        assert np.all(mags >= 0)


class TestPeakDetection:
    def test_sine_peak(self):
        sr = 44100
        t = np.arange(2048) / sr
        sig = np.sin(2 * np.pi * 1000 * t)
        mags, _ = stft_frames(sig, frame_size=2048, hop=2048)
        peaks = detect_peaks(mags[0], sr=sr, frame_size=2048)
        assert len(peaks) > 0
        freqs = [p.frequency_hz for p in peaks]
        assert any(abs(f - 1000) < 100 for f in freqs)

    def test_flat_spectrum_no_peaks(self):
        mags = np.ones(513) * 0.5
        peaks = detect_peaks(mags)
        assert len(peaks) == 0


class TestSinusoidalTracking:
    def test_tracking_on_sine(self):
        sr = 44100
        n = sr  # 1 second
        t = np.arange(n) / sr
        sig = np.sin(2 * np.pi * 440 * t)
        mags, phases = stft_frames(sig, frame_size=2048, hop=512)
        tracks = track_sinusoids(mags, phases, sr=sr, frame_size=2048)
        assert len(tracks) >= 1
        longest = max(tracks, key=lambda tr: tr.n_frames)
        assert abs(longest.mean_frequency - 440) < 50


class TestResynthesis:
    def test_resynth_length(self):
        tracks = [SinusoidalTrack(
            frequencies=[440.0] * 5,
            amplitudes=[0.5] * 5,
            phases=[0.0] * 5,
            start_frame=0,
        )]
        out = resynthesize_sinusoids(tracks, n_samples=4000, sr=44100, hop=512)
        assert len(out) == 4000

    def test_residual_shape(self):
        a = np.random.randn(1000)
        b = np.random.randn(1000)
        r = extract_residual(a, b)
        assert len(r) == 1000


class TestSpectralMorph:
    def test_extremes(self):
        a = np.ones((5, 10))
        b = np.zeros((5, 10))
        m0 = spectral_morph(a, b, alpha=0.0)
        assert np.allclose(m0, a)
        m1 = spectral_morph(a, b, alpha=1.0)
        assert np.allclose(m1, b)

    def test_midpoint(self):
        a = np.ones((5, 10)) * 2.0
        b = np.zeros((5, 10))
        m = spectral_morph(a, b, alpha=0.5)
        assert np.allclose(m, 1.0)


class TestSMSAnalyze:
    def test_full_pipeline(self):
        sr = 44100
        n = sr // 2
        t = np.arange(n) / sr
        sig = np.sin(2 * np.pi * 880 * t)
        tracks, residual = sms_analyze(sig, sr=sr, frame_size=1024, hop=256)
        assert len(tracks) >= 1
        assert len(residual) == n
