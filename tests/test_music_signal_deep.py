"""Deep tests for music_signal: windowing, filters, dynamics, onset detection,
pitch tracking, MFCC, envelope, signal quality, resampling, HPS, adaptive filtering."""
import numpy as np
import pytest

from models.music_signal.harmonic_product_spectrum import (
    harmonic_product_spectrum, autocorrelation_pitch as hps_ac,
    cepstral_pitch, multi_pitch_estimate, consensus_pitch, PitchEstimate,
)
from models.music_signal.windowing import (
    WindowType, window_function, equivalent_noise_bandwidth,
    coherent_gain, cola_check, optimal_hop,
)
from models.music_signal.filters import (
    FilterType, biquad_design, apply_biquad, a_weighting,
)
from models.music_signal.dynamics import (
    CompressorParams, compute_gain_reduction,
    compressor, limiter, rms_level, peak_level,
    crest_factor, dynamic_range,
)
from models.music_signal.onset_detection import (
    spectral_flux, pick_peaks, frames_to_time, onset_detect,
)
from models.music_signal.pitch_tracking import (
    hz_to_midi, midi_to_hz, pitch_class, cents_difference,
    autocorrelation_pitch, yin_frame, track_pitch,
)
from models.music_signal.mfcc import compute_mfcc, mfcc_delta, lifter
from models.music_signal.envelope import (
    analytic_signal, instantaneous_amplitude,
    rms_envelope, am_fm_decompose,
)
from models.music_signal.signal_quality import (
    snr, thd, dc_offset, clipping_detection, quality_report,
)
from models.music_signal.resampling import (
    resample_sinc, decimate, interpolate_linear, interpolate_cubic,
)
from models.music_signal.adaptive_filtering import (
    lms_update, nlms_update, AdaptiveFilterState,
    noise_cancellation, compute_misalignment,
)
from models.music_signal.source_separation import nmf, hpss, sdr

SR = 22050


def _sine(freq=440.0, dur=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


class TestWindowing:
    def test_hann_symmetry(self):
        w = window_function(WindowType.HANN, 256)
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)

    def test_hann_endpoints_near_zero(self):
        w = window_function(WindowType.HANN, 256)
        assert w[0] < 0.01
        assert w[127] > 0.99

    def test_rectangular_all_ones(self):
        w = window_function(WindowType.RECTANGULAR, 128)
        np.testing.assert_allclose(w, 1.0)

    def test_enbw_hann(self):
        w = window_function(WindowType.HANN, 1024)
        enbw = equivalent_noise_bandwidth(w)
        assert 1.4 < enbw < 1.7

    def test_coherent_gain_rect(self):
        w = window_function(WindowType.RECTANGULAR, 64)
        cg = coherent_gain(w)
        assert abs(cg - 1.0) < 0.01

    def test_empty_window(self):
        w = window_function(WindowType.HANN, 0)
        assert len(w) == 0

    def test_blackman_lower_sidelobes(self):
        hann = window_function(WindowType.HANN, 512)
        black = window_function(WindowType.BLACKMAN, 512)
        assert equivalent_noise_bandwidth(black) > equivalent_noise_bandwidth(hann)


class TestFilters:
    def test_lowpass_dc_unity(self):
        c = biquad_design(FilterType.LOWPASS, 1000, SR, q=0.707)
        dc_gain = abs((c.b0 + c.b1 + c.b2) / (c.a0 + c.a1 + c.a2))
        assert abs(dc_gain - 1.0) < 0.01

    def test_lowpass_attenuates_high(self):
        sig = _sine(8000, 0.5)
        c = biquad_design(FilterType.LOWPASS, 1000, SR, q=0.707)
        out = apply_biquad(sig, c)
        assert rms_level(out) < rms_level(sig) * 0.3

    def test_highpass_attenuates_low(self):
        sig = _sine(100, 0.5)
        c = biquad_design(FilterType.HIGHPASS, 1000, SR, q=0.707)
        out = apply_biquad(sig, c)
        assert rms_level(out) < rms_level(sig) * 0.3

    def test_a_weighting_1khz_ref(self):
        freqs = np.array([1000.0])
        aw = a_weighting(freqs)
        assert abs(aw[0]) < 0.5  # 0 dB reference at 1 kHz


class TestDynamics:
    def test_no_compression_below_threshold(self):
        p = CompressorParams(threshold_db=-10, ratio=4.0)
        gr = compute_gain_reduction(-20.0, p)
        assert gr == 0.0

    def test_compression_above_threshold(self):
        p = CompressorParams(threshold_db=-20, ratio=4.0, knee_db=0.0)
        gr = compute_gain_reduction(-10.0, p)
        assert gr < 0.0

    def test_limiter_reduces_peak(self):
        sig = _sine(440, 0.2) * 2.0
        out = limiter(sig, threshold_db=-3.0)
        assert peak_level(out) <= peak_level(sig)

    def test_crest_factor_sine(self):
        sig = _sine(440, 0.5)
        cf = crest_factor(sig)
        assert abs(cf - 3.01) < 0.5

    def test_dynamic_range_constant(self):
        sig = 0.5 * np.ones(4096)
        dr = dynamic_range(sig)
        assert dr < 1.0


class TestOnsetDetection:
    def test_spectral_flux_shape(self):
        rng = np.random.default_rng(42)
        mag = np.abs(rng.standard_normal((513, 100)))
        sf = spectral_flux(mag)
        assert sf.shape == (100,)

    def test_spectral_flux_nonneg(self):
        rng = np.random.default_rng(42)
        mag = np.abs(rng.standard_normal((513, 50)))
        sf = spectral_flux(mag)
        assert np.all(sf >= 0)

    def test_pick_peaks_basic(self):
        curve = np.array([0, 0.5, 1.0, 0.5, 0, 0, 0.8, 0.2])
        peaks = pick_peaks(curve, threshold=0.3)
        assert 2 in peaks
        assert 6 in peaks

    def test_frames_to_time(self):
        frames = np.array([0, 10, 20])
        times = frames_to_time(frames, sr=SR, hop_size=512)
        expected = frames * 512 / SR
        np.testing.assert_allclose(times, expected, atol=1e-6)


class TestPitchTracking:
    def test_hz_midi_roundtrip(self):
        for midi in [60, 69, 48, 84]:
            hz = midi_to_hz(midi)
            back = hz_to_midi(hz)
            assert abs(back - midi) < 0.01

    def test_a440_is_midi_69(self):
        assert abs(hz_to_midi(440.0) - 69.0) < 0.01

    def test_pitch_class_c(self):
        assert pitch_class(60) == 0

    def test_cents_unison(self):
        assert cents_difference(440.0, 440.0) == 0.0

    def test_cents_octave(self):
        assert abs(cents_difference(440.0, 880.0) - 1200.0) < 0.01

    def test_autocorrelation_440hz(self):
        sig = _sine(440, 0.1)
        f0 = autocorrelation_pitch(sig, sr=SR, fmin=80, fmax=2000)
        assert abs(f0 - 440.0) < 15.0


class TestHPS:
    def test_pure_tone(self):
        sig = _sine(440, dur=1.0, sr=16000)
        est = harmonic_product_spectrum(sig, 16000, n_harmonics=3)
        assert isinstance(est, PitchEstimate)
        assert est.frequency_hz > 50

    def test_silence(self):
        est = harmonic_product_spectrum(np.zeros(4096), 16000)
        assert est.frequency_hz == 0.0

    def test_consensus_agreement(self):
        sig = _sine(440, dur=1.0, sr=16000)
        est = consensus_pitch(sig, 16000)
        assert abs(est.frequency_hz - 440) < 30
        assert est.method == "consensus"


class TestMFCC:
    def test_mfcc_shape(self):
        sig = _sine(440, 0.5)
        result = compute_mfcc(sig, sr=SR, n_mfcc=13)
        assert result.coeffs.shape[0] == 13

    def test_mfcc_delta_shape(self):
        coeffs = np.random.default_rng(0).standard_normal((13, 50))
        d = mfcc_delta(coeffs)
        assert d.shape == coeffs.shape

    def test_lifter_identity_at_zero(self):
        coeffs = np.random.default_rng(0).standard_normal((13, 20))
        lifted = lifter(coeffs, L=0)
        np.testing.assert_allclose(lifted, coeffs)


class TestEnvelope:
    def test_analytic_real_part(self):
        sig = _sine(440, 0.1)
        ana = analytic_signal(sig)
        np.testing.assert_allclose(ana.real, sig, atol=1e-10)

    def test_instantaneous_amplitude_sine(self):
        sig = _sine(440, 0.1)
        amp = instantaneous_amplitude(sig)
        assert abs(np.mean(amp) - 1.0) < 0.05

    def test_rms_envelope_shape(self):
        sig = _sine(440, 0.5)
        env = rms_envelope(sig, frame_size=2048, hop_size=512)
        assert len(env) > 0
        assert np.all(env >= 0)


class TestSignalQuality:
    def test_snr_clean(self):
        sig = _sine(440, 0.5)
        noise = np.random.default_rng(42).standard_normal(len(sig)) * 0.001
        s = snr(sig, noise)
        assert s > 50.0

    def test_dc_offset_zero_mean(self):
        sig = _sine(440, 0.5)
        dc = dc_offset(sig)
        assert abs(dc) < 0.01

    def test_clipping_none(self):
        sig = _sine(440, 0.1) * 0.5
        result = clipping_detection(sig)
        assert result['clipped_samples'] == 0

    def test_clipping_detected(self):
        sig = _sine(440, 0.1) * 2.0
        sig = np.clip(sig, -1.0, 1.0)
        result = clipping_detection(sig, threshold=0.99)
        assert result['clipped_samples'] > 0


class TestResampling:
    def test_resample_double(self):
        sig = _sine(440, 0.1)
        out = resample_sinc(sig, SR, SR * 2)
        assert abs(len(out) - 2 * len(sig)) < 10

    def test_decimate_half(self):
        sig = _sine(100, 0.5)
        out = decimate(sig, 2)
        assert abs(len(out) - len(sig) // 2) < 5

    def test_interpolate_linear_preserves_endpoints(self):
        x = np.array([0.0, 1.0, 0.0])
        out = interpolate_linear(x, 5)
        assert abs(out[0]) < 1e-10
        assert abs(out[-1]) < 1e-10


class TestAdaptiveFiltering:
    def test_lms_update_reduces_error(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        h_true = np.array([0.5, -0.3, 0.1])
        d = np.convolve(x, h_true, mode='full')[:100]
        state = AdaptiveFilterState(weights=np.zeros(3))
        errors = []
        for n in range(3, 100):
            x_vec = x[n - 3:n][::-1]
            e = d[n] - float(x_vec @ state.weights)
            state = lms_update(state, x_vec, e, mu=0.01)
            errors.append(abs(e))
        assert np.mean(errors[-20:]) < np.mean(errors[:20])

    def test_nlms_converges(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        h_true = np.array([1.0, -0.5])
        d = np.convolve(x, h_true, mode='full')[:200]
        state = AdaptiveFilterState(weights=np.zeros(2))
        for n in range(2, 200):
            x_vec = x[n - 2:n][::-1]
            e = d[n] - float(x_vec @ state.weights)
            state = nlms_update(state, x_vec, e, mu=0.5)
        misalign = compute_misalignment(state.weights, h_true)
        assert misalign < -10.0  # < -10 dB misalignment


class TestSourceSeparation:
    def test_nmf_shapes(self):
        rng = np.random.default_rng(42)
        V = np.abs(rng.standard_normal((100, 50)))
        W, H = nmf(V, n_components=4, n_iter=20)
        assert W.shape == (100, 4)
        assert H.shape == (4, 50)

    def test_nmf_nonneg(self):
        rng = np.random.default_rng(42)
        V = np.abs(rng.standard_normal((50, 30)))
        W, H = nmf(V, n_components=3, n_iter=10)
        assert np.all(W >= 0)
        assert np.all(H >= 0)

    def test_sdr_perfect(self):
        sig = _sine(440, 0.1)
        s = sdr(sig, sig)
        assert s > 100.0  # near-infinite for identical signals
