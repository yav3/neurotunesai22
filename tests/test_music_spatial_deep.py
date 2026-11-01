"""Deep tests for music_spatial: HRTF, ambisonics, binaural rendering,
room acoustics, stereo field, surround panning."""
import numpy as np
import pytest

from models.music_spatial.hrtf import (
    HRTFParams, hrtf_filter_pair, interaural_time_difference,
    interaural_level_difference, distance_attenuation,
)
from models.music_spatial.ambisonics import (
    encode_source, n_channels, rotate_soundfield, energy_vector,
)
from models.music_spatial.binaural import (
    itd_from_azimuth, ild_from_azimuth, crossfeed,
    head_shadow_filter, BinauralSource, render_binaural_mono,
)
from models.music_spatial.room_acoustics import (
    sabine_rt60, eyring_rt60, critical_distance, schroeder_frequency,
    mean_free_path,
)
from models.music_spatial.stereo_field import (
    pan_constant_power, pan_linear, stereo_width, mid_side_encode,
    mid_side_decode, correlation_meter,
)
from models.music_spatial.surround_panning import (
    vbap_gains_2d, speaker_layout_5_1, speaker_layout_7_1,
)


SR = 44100

def _sine(freq=440.0, dur=0.1, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


# ---- HRTF ----

class TestHRTF:
    def test_itd_zero_at_front(self):
        assert interaural_time_difference(0.0) == 0.0

    def test_itd_max_at_side(self):
        itd = interaural_time_difference(np.pi / 2)
        assert itd > 0.0
        assert itd < 1e-3

    def test_itd_symmetric(self):
        left = interaural_time_difference(np.pi / 4)
        right = interaural_time_difference(-np.pi / 4)
        assert abs(left + right) < 1e-10

    def test_ild_zero_at_front(self):
        ild = interaural_level_difference(0.0, 1000.0)
        assert abs(ild) < 0.01

    def test_filter_pair_length(self):
        params = HRTFParams(azimuth_rad=0.5, elevation_rad=0.0)
        left, right = hrtf_filter_pair(params, n_taps=128)
        assert left.shape == (128,)
        assert right.shape == (128,)

    def test_distance_attenuation_inverse(self):
        a1 = distance_attenuation(1.0)
        a2 = distance_attenuation(2.0)
        assert abs(a2 - a1 / 2) < 0.01

    def test_distance_attenuation_reference(self):
        assert abs(distance_attenuation(1.0, reference=1.0) - 1.0) < 0.01


# ---- Ambisonics ----

class TestAmbisonics:
    def test_n_channels_first_order(self):
        assert n_channels(1) == 4

    def test_n_channels_second_order(self):
        assert n_channels(2) == 9

    def test_n_channels_third_order(self):
        assert n_channels(3) == 16

    def test_encode_first_order_shape(self):
        coeffs = encode_source(0.0, 0.0, order=1)
        assert coeffs.shape == (4,)

    def test_encode_front_w_channel(self):
        coeffs = encode_source(0.0, 0.0, order=1)
        assert abs(coeffs[0] - 1.0) < 0.01

    def test_rotate_soundfield_shape(self):
        b_format = np.random.default_rng(42).standard_normal((4, 1000))
        rotated = rotate_soundfield(b_format, yaw=0.5)
        assert rotated.shape == b_format.shape

    def test_rotate_zero_is_identity(self):
        b_format = np.random.default_rng(42).standard_normal((4, 100))
        rotated = rotate_soundfield(b_format, yaw=0.0)
        np.testing.assert_allclose(rotated, b_format, atol=1e-10)

    def test_energy_vector_unit(self):
        signals = np.eye(5)
        positions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1], [0.7, 0.7]
        ], dtype=float)
        direction, magnitude = energy_vector(signals[:, :1], positions)
        assert direction.shape == (2,)


# ---- Binaural ----

class TestBinaural:
    def test_itd_zero_front(self):
        assert abs(itd_from_azimuth(0.0)) < 1e-10

    def test_itd_positive_right(self):
        itd = itd_from_azimuth(90.0)
        assert itd > 0

    def test_ild_zero_front(self):
        ild = ild_from_azimuth(0.0, 1000.0)
        assert abs(ild) < 0.1

    def test_crossfeed_preserves_length(self):
        left = _sine(440, 0.05)
        right = _sine(660, 0.05)
        cl, cr = crossfeed(left, right, amount=0.3)
        assert cl.shape == left.shape
        assert cr.shape == right.shape

    def test_crossfeed_zero_is_passthrough(self):
        left = _sine(440, 0.05)
        right = _sine(660, 0.05)
        cl, cr = crossfeed(left, right, amount=0.0)
        np.testing.assert_allclose(cl, left, atol=1e-10)

    def test_head_shadow_attenuates(self):
        sig = _sine(4000, 0.05)
        filtered = head_shadow_filter(sig, azimuth_deg=90.0, sample_rate=SR)
        assert np.mean(filtered ** 2) < np.mean(sig ** 2)

    def test_render_binaural_shape(self):
        mono = _sine(440, 0.05)
        source = BinauralSource(azimuth_deg=45.0, distance_m=2.0)
        left, right = render_binaural_mono(mono, source, sample_rate=SR)
        assert left.shape == mono.shape
        assert right.shape == mono.shape


# ---- Room Acoustics ----

class TestRoomAcoustics:
    def test_sabine_rt60_positive(self):
        rt = sabine_rt60(volume=100.0, surface_area=200.0, avg_absorption=0.2)
        assert rt > 0.0

    def test_sabine_more_absorption_shorter(self):
        rt1 = sabine_rt60(100.0, 200.0, 0.1)
        rt2 = sabine_rt60(100.0, 200.0, 0.5)
        assert rt2 < rt1

    def test_eyring_rt60_positive(self):
        rt = eyring_rt60(volume=100.0, surface_area=200.0, avg_absorption=0.2)
        assert rt > 0.0

    def test_critical_distance_positive(self):
        d = critical_distance(volume=200.0, rt60=1.5)
        assert d > 0.0

    def test_schroeder_frequency(self):
        f = schroeder_frequency(volume=200.0, rt60=1.5)
        assert f > 20.0

    def test_mean_free_path_positive(self):
        mfp = mean_free_path(volume=200.0, surface_area=300.0)
        assert mfp > 0.0


# ---- Stereo Field ----

class TestStereoField:
    def test_constant_power_center(self):
        l, r = pan_constant_power(0.5)
        assert abs(l - r) < 0.01
        assert abs(l ** 2 + r ** 2 - 1.0) < 0.01

    def test_constant_power_hard_left(self):
        l, r = pan_constant_power(0.0)
        assert l > 0.99
        assert r < 0.01

    def test_linear_pan_center(self):
        l, r = pan_linear(0.5)
        assert abs(l - r) < 0.01

    def test_stereo_width_mono(self):
        sig = _sine(440, 0.05)
        w = stereo_width(sig, sig)
        assert w < 0.1

    def test_mid_side_roundtrip(self):
        left = _sine(440, 0.05)
        right = _sine(660, 0.05)
        mid, side = mid_side_encode(left, right)
        l2, r2 = mid_side_decode(mid, side)
        np.testing.assert_allclose(l2, left, atol=1e-10)
        np.testing.assert_allclose(r2, right, atol=1e-10)

    def test_correlation_meter_identical(self):
        sig = _sine(440, 0.05)
        corr = correlation_meter(sig, sig)
        assert abs(corr - 1.0) < 0.01


# ---- Surround Panning ----

class TestSurroundPanning:
    def test_vbap_gains_sum(self):
        layout = speaker_layout_5_1()
        gains = vbap_gains_2d(30.0, layout)
        assert len(gains) == len(layout)
        assert abs(sum(g ** 2 for g in gains) - 1.0) < 0.1

    def test_speaker_layout_5_1_count(self):
        layout = speaker_layout_5_1()
        assert len(layout) >= 5

    def test_speaker_layout_7_1_count(self):
        layout = speaker_layout_7_1()
        assert len(layout) >= 7
