"""Tests for the music_spatial module."""
import math
import pytest
import numpy as np

from models.music_spatial.hrtf import (
    HRTFParams, SPEED_OF_SOUND,
    interaural_time_difference, interaural_level_difference,
    hrtf_filter_pair, distance_attenuation,
)
from models.music_spatial.ambisonics import (
    AmbisonicsOrder, n_channels, encode_source,
    AmbisonicsDecoder, rotate_soundfield, energy_vector,
)
from models.music_spatial.stereo_field import (
    pan_law, stereo_width, mid_side_encode, mid_side_decode,
    stereo_balance, phantom_center_level, StereoField,
)
from models.music_spatial.room_model import (
    RoomGeometry, rt60_sabine, rt60_eyring,
    image_source_model, early_decay_time, clarity_index,
    direct_to_reverberant_ratio,
)


class TestHRTF:
    def test_itd_center(self):
        assert interaural_time_difference(0.0) == 0.0

    def test_itd_side(self):
        itd = interaural_time_difference(math.pi / 2)
        assert itd > 0
        assert itd < 0.001  # ~0.6 ms max

    def test_ild_low_frequency(self):
        assert interaural_level_difference(0.5, 100) == 0.0

    def test_ild_high_frequency(self):
        ild = interaural_level_difference(math.pi / 4, 4000)
        assert ild > 0

    def test_hrtf_filter_pair_shape(self):
        params = HRTFParams(azimuth=30.0, elevation=0.0)
        left, right = hrtf_filter_pair(params)
        assert left.shape == (128,)
        assert right.shape == (128,)

    def test_distance_attenuation(self):
        assert distance_attenuation(1.0) == 1.0
        assert distance_attenuation(2.0) == pytest.approx(0.5)

    def test_distance_attenuation_near(self):
        assert distance_attenuation(0.0) == 1.0

    def test_speed_of_sound(self):
        assert SPEED_OF_SOUND == pytest.approx(343.0)


class TestAmbisonics:
    def test_n_channels_first(self):
        assert n_channels(1) == 4

    def test_n_channels_second(self):
        assert n_channels(2) == 9

    def test_n_channels_third(self):
        assert n_channels(3) == 16

    def test_encode_source_w(self):
        coeffs = encode_source(0.0, 0.0, order=1)
        assert len(coeffs) == 4
        assert coeffs[0] == pytest.approx(1.0)

    def test_encode_source_front(self):
        coeffs = encode_source(0.0, 0.0, order=1)
        assert coeffs[3] == pytest.approx(1.0, abs=0.01)

    def test_decoder_matrix_shape(self):
        positions = np.array([[0, 0], [math.pi/2, 0], [math.pi, 0], [-math.pi/2, 0]])
        dec = AmbisonicsDecoder(speaker_positions=positions)
        D = dec.decode_matrix(order=1)
        assert D.shape == (4, 4)

    def test_rotate_yaw(self):
        b = np.zeros((4, 10))
        b[0] = 1.0
        b[3] = 1.0  # X (front)
        rotated = rotate_soundfield(b, yaw=math.pi / 2)
        assert rotated[0, 0] == pytest.approx(1.0)

    def test_energy_vector(self):
        signals = np.random.randn(4, 100)
        positions = np.array([[0, 0], [math.pi/2, 0], [math.pi, 0], [-math.pi/2, 0]])
        direction, spread = energy_vector(signals, positions)
        assert len(direction) == 3
        assert 0.0 <= spread <= 1.0


class TestStereoField:
    def test_pan_center(self):
        l, r = pan_law(0.0)
        assert l == pytest.approx(r, abs=0.01)

    def test_pan_left(self):
        l, r = pan_law(-1.0)
        assert l > r

    def test_pan_right(self):
        l, r = pan_law(1.0)
        assert r > l

    def test_pan_linear(self):
        l, r = pan_law(0.0, "linear")
        assert l == pytest.approx(0.5)
        assert r == pytest.approx(0.5)

    def test_stereo_width_mono(self):
        mono = np.random.randn(1000)
        assert stereo_width(mono, mono) == pytest.approx(0.0, abs=0.01)

    def test_stereo_width_uncorrelated(self):
        np.random.seed(42)
        left = np.random.randn(10000)
        right = np.random.randn(10000)
        w = stereo_width(left, right)
        assert w > 0.8

    def test_mid_side_roundtrip(self):
        left = np.array([1.0, 2.0, 3.0])
        right = np.array([0.5, 1.5, 2.5])
        mid, side = mid_side_encode(left, right)
        l2, r2 = mid_side_decode(mid, side)
        np.testing.assert_array_almost_equal(l2, left)
        np.testing.assert_array_almost_equal(r2, right)

    def test_stereo_balance_centered(self):
        signal = np.random.randn(1000)
        b = stereo_balance(signal, signal)
        assert abs(b) < 0.01

    def test_stereo_field_analyze(self):
        left = np.random.randn(1000)
        right = np.random.randn(1000)
        sf = StereoField.analyze(left, right)
        assert isinstance(sf.width, float)
        assert isinstance(sf.balance, float)
        assert isinstance(sf.correlation, float)


class TestRoomModel:
    def test_rt60_sabine_small_room(self):
        room = RoomGeometry(5.0, 4.0, 3.0)
        rt = rt60_sabine(room)
        assert 0.1 < rt < 5.0

    def test_rt60_eyring_small_room(self):
        room = RoomGeometry(5.0, 4.0, 3.0)
        rt = rt60_eyring(room)
        assert 0.1 < rt < 5.0

    def test_sabine_vs_eyring(self):
        room = RoomGeometry(5.0, 4.0, 3.0)
        assert rt60_eyring(room) < rt60_sabine(room)

    def test_image_source_model(self):
        room = RoomGeometry(6.0, 5.0, 3.0)
        src = np.array([1.0, 1.0, 1.5])
        lis = np.array([4.0, 3.0, 1.5])
        sources = image_source_model(room, src, lis, max_order=2)
        assert len(sources) > 0
        assert all(s.delay_s > 0 for s in sources)

    def test_clarity_index(self):
        ir = np.zeros(44100)
        ir[0] = 1.0
        ir[4410:] = np.random.randn(44100 - 4410) * 0.0001
        c80 = clarity_index(ir, 44100)
        assert c80 > 0

    def test_drr(self):
        ir = np.zeros(44100)
        ir[0] = 1.0
        ir[200:] = np.random.randn(44100 - 200) * 0.001
        drr = direct_to_reverberant_ratio(ir, 44100)
        assert drr > 0


class TestPackageExports:
    def test_all_exports(self):
        import models.music_spatial as ms
        assert hasattr(ms, "HRTFParams")
        assert hasattr(ms, "AmbisonicsOrder")
        assert hasattr(ms, "pan_law")
        assert hasattr(ms, "RoomGeometry")
