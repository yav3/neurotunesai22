"""Deep tests for music_production: EQ design, dynamic range processing,
stereo imaging, gain staging, mix analysis."""
import numpy as np
import pytest

from models.music_production.eq_design import (
    EQBand, biquad_coefficients, apply_biquad, frequency_response,
    spectral_tilt, auto_eq_match,
)
from models.music_production.dynamic_range import (
    CompressorSettings, compression_curve, gain_reduction,
    apply_compression, limiter_ceiling, loudness_normalize,
    rms_level, peak_level, amplitude_to_db, db_to_amplitude,
    analyze_dynamic_range, multiband_split_points,
)
from models.music_production.stereo_imaging import (
    stereo_width_measure, haas_delay, mono_compatibility,
    mid_side_process,
)
from models.music_production.gain_staging import (
    headroom, optimal_gain, gain_stage_chain,
)
from models.music_production.mix_analysis import (
    frequency_balance, spectral_centroid_mix, crest_factor,
    loudness_range, dynamic_complexity,
)


SR = 44100


def _sine(freq=440.0, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


# ---- EQ Design ----

class TestEQDesign:
    def test_biquad_coefficients_shape(self):
        band = EQBand(frequency=1000.0, gain_db=6.0, q=1.0, filter_type="peaking")
        b, a = biquad_coefficients(band)
        assert b.shape == (3,)
        assert a.shape == (3,)

    def test_apply_biquad_preserves_length(self):
        sig = _sine(440, 0.1)
        band = EQBand(frequency=1000.0, gain_db=0.0, q=1.0, filter_type="peaking")
        b, a = biquad_coefficients(band)
        out = apply_biquad(sig, b, a)
        assert len(out) == len(sig)

    def test_frequency_response_shape(self):
        band = EQBand(frequency=1000.0, gain_db=6.0, q=1.0, filter_type="peaking")
        freqs = np.linspace(20, 20000, 256)
        resp = frequency_response(band, freqs)
        assert resp.shape == (256,)

    def test_spectral_tilt_flat(self):
        freqs = np.linspace(100, 10000, 100)
        flat = np.zeros(100)
        tilt = spectral_tilt(flat, freqs)
        assert abs(tilt) < 0.1

    def test_auto_eq_match_returns_bands(self):
        freqs = np.linspace(20, 20000, 512)
        source = np.random.default_rng(42).standard_normal(512)
        target = np.random.default_rng(0).standard_normal(512)
        bands = auto_eq_match(source, target, freqs, n_bands=3)
        assert len(bands) == 3
        assert all(isinstance(b, EQBand) for b in bands)


# ---- Dynamic Range ----

class TestDynamicRange:
    def test_db_amplitude_roundtrip(self):
        for db in [-20, -6, 0, 6]:
            amp = db_to_amplitude(float(db))
            back = amplitude_to_db(amp)
            assert abs(back - db) < 0.01

    def test_rms_level_sine(self):
        sig = _sine(440, 0.5)
        rms = rms_level(sig)
        assert abs(rms - 1.0 / np.sqrt(2)) < 0.01

    def test_peak_level_sine(self):
        sig = _sine(440, 0.5)
        pk = peak_level(sig)
        assert abs(pk - 1.0) < 0.01

    def test_compression_curve_below_threshold(self):
        settings = CompressorSettings(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        out = compression_curve(-30.0, settings)
        assert abs(out - (-30.0)) < 0.01

    def test_compression_curve_above_threshold(self):
        settings = CompressorSettings(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        out = compression_curve(-10.0, settings)
        assert out < -10.0
        assert out > -20.0

    def test_gain_reduction_below_zero(self):
        settings = CompressorSettings(threshold_db=-20.0, ratio=4.0, knee_db=0.0)
        gr = gain_reduction(-10.0, settings)
        assert gr <= 0.0

    def test_apply_compression_reduces_peak(self):
        sig = _sine(440, 0.2) * 2.0
        settings = CompressorSettings(threshold_db=-6.0, ratio=4.0)
        out = apply_compression(sig, settings, sample_rate=SR)
        assert peak_level(out) <= peak_level(sig)

    def test_limiter_ceiling_clips(self):
        sig = _sine(440, 0.1) * 2.0
        out = limiter_ceiling(sig, ceiling_db=-3.0)
        assert peak_level(out) <= db_to_amplitude(-3.0) + 0.01

    def test_loudness_normalize_target(self):
        sig = _sine(440, 0.5) * 0.1
        out = loudness_normalize(sig, target_rms_db=-14.0)
        out_rms_db = amplitude_to_db(rms_level(out))
        assert abs(out_rms_db - (-14.0)) < 1.0

    def test_analyze_dynamic_range_returns_stats(self):
        sig = _sine(440, 0.5)
        stats = analyze_dynamic_range(sig)
        assert hasattr(stats, 'peak_db')
        assert hasattr(stats, 'rms_db')

    def test_multiband_split_points(self):
        pts = multiband_split_points(n_bands=3)
        assert len(pts) == 2
        assert pts[0] < pts[1]


# ---- Stereo Imaging ----

class TestStereoImaging:
    def test_width_mono_zero(self):
        sig = _sine(440, 0.1)
        w = stereo_width_measure(sig, sig)
        assert w < 0.1

    def test_haas_delay_shape(self):
        sig = _sine(440, 0.1)
        left, right = haas_delay(sig, delay_ms=20.0, sr=SR)
        assert left.shape == right.shape

    def test_mono_compatibility_identical(self):
        sig = _sine(440, 0.1)
        mc = mono_compatibility(sig, sig)
        assert mc > 0.9

    def test_mid_side_process_shape(self):
        left = _sine(440, 0.1)
        right = _sine(660, 0.1)
        l2, r2 = mid_side_process(left, right, width=1.0)
        assert l2.shape == left.shape


# ---- Gain Staging ----

class TestGainStaging:
    def test_headroom_positive(self):
        sig = _sine(440, 0.1) * 0.5
        hr = headroom(sig)
        assert hr > 0.0

    def test_optimal_gain_boosts_quiet(self):
        sig = _sine(440, 0.1) * 0.01
        g = optimal_gain(sig, target_db=-14.0)
        assert g > 1.0

    def test_gain_stage_chain_returns_list(self):
        levels = [-30.0, -20.0, -10.0]
        chain = gain_stage_chain(levels, target_db=-14.0)
        assert isinstance(chain, list)
        assert len(chain) == len(levels)


# ---- Mix Analysis ----

class TestMixAnalysis:
    def test_frequency_balance_keys(self):
        sig = _sine(440, 0.5)
        balance = frequency_balance(sig, sr=SR)
        assert isinstance(balance, dict)

    def test_spectral_centroid_mix_positive(self):
        sig = _sine(440, 0.5)
        sc = spectral_centroid_mix(sig, sr=SR)
        assert sc > 0.0

    def test_crest_factor_sine(self):
        sig = _sine(440, 0.5)
        cf = crest_factor(sig)
        assert abs(cf - np.sqrt(2)) < 0.1

    def test_loudness_range_nonneg(self):
        sig = _sine(440, 0.5)
        lr = loudness_range(sig, sr=SR)
        assert lr >= 0.0

    def test_dynamic_complexity_nonneg(self):
        sig = _sine(440, 0.5)
        dc = dynamic_complexity(sig, sr=SR)
        assert dc >= 0.0
