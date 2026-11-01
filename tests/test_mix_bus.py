"""Tests for mix bus processing chain."""
import numpy as np
import pytest
from models.music_production.mix_bus import (
    sum_stems, rms_db, peak_db, lufs_integrated,
    bus_compressor, master_eq, loudness_normalize,
    true_peak, process_mix_bus, MixBusConfig,
)

SR = 44100


def _tone(freq, dur=0.5, amp=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


class TestSumStems:
    def test_empty(self):
        assert sum_stems([]).shape == (0, 2)

    def test_mono_center(self):
        s = np.ones(100)
        mix = sum_stems([s], pans=[0.0])
        assert mix.shape == (100, 2)
        assert np.allclose(mix[:, 0], mix[:, 1], atol=1e-6)

    def test_mono_hard_left(self):
        s = np.ones(100)
        mix = sum_stems([s], pans=[-1.0])
        assert mix[0, 0] > mix[0, 1]

    def test_stereo_passthrough(self):
        s = np.column_stack([np.ones(50), np.zeros(50)])
        mix = sum_stems([s])
        assert np.allclose(mix[:50, 0], 1.0)
        assert np.allclose(mix[:50, 1], 0.0)

    def test_gain_applied(self):
        s = np.ones(100)
        mix_0 = sum_stems([s], gains_db=[0.0])
        mix_6 = sum_stems([s], gains_db=[6.0])
        assert np.max(np.abs(mix_6)) > np.max(np.abs(mix_0))

    def test_different_lengths(self):
        s1 = np.ones(100)
        s2 = np.ones(200)
        mix = sum_stems([s1, s2])
        assert mix.shape[0] == 200


class TestLevels:
    def test_rms_db_silence(self):
        assert rms_db(np.zeros(100)) < -200

    def test_rms_db_fullscale(self):
        s = np.ones(1000)
        assert abs(rms_db(s) - 0.0) < 0.1

    def test_peak_db_sine(self):
        s = 0.5 * np.ones(100)
        assert abs(peak_db(s) - (-6.02)) < 0.2

    def test_lufs_silence(self):
        assert lufs_integrated(np.zeros(SR), SR) == -70.0

    def test_lufs_loud_signal(self):
        s = _tone(1000, 2.0, amp=0.8)
        val = lufs_integrated(s, SR)
        assert val > -20

    def test_lufs_stereo(self):
        s = np.column_stack([_tone(440, 1.0), _tone(440, 1.0)])
        val = lufs_integrated(s, SR)
        assert -30 < val < 0


class TestBusCompressor:
    def test_shape_preserved(self):
        s = _tone(440, 0.2)
        out = bus_compressor(s, SR)
        assert out.shape[0] == len(s)

    def test_loud_signal_reduced(self):
        s = np.ones(SR)
        out = bus_compressor(s, SR, threshold_db=-20.0, ratio=4.0)
        tail = out[SR // 2:]
        assert np.max(np.abs(tail)) < np.max(np.abs(s))

    def test_quiet_signal_untouched(self):
        s = 0.001 * np.ones(SR)
        out = bus_compressor(s, SR, threshold_db=-6.0, ratio=4.0)
        assert np.allclose(out.flatten(), s, atol=1e-6)


class TestMasterEQ:
    def test_flat_eq_passthrough(self):
        s = _tone(1000, 0.5)
        out = master_eq(s, SR)
        assert np.allclose(out, s, atol=1e-10)

    def test_low_shelf_boost(self):
        low = _tone(50, 0.5)
        out = master_eq(low, SR, low_shelf_db=6.0)
        assert np.max(np.abs(out)) > np.max(np.abs(low))

    def test_high_shelf_cut(self):
        hi = _tone(12000, 0.5)
        out = master_eq(hi, SR, high_shelf_db=-6.0)
        assert np.max(np.abs(out)) < np.max(np.abs(hi))


class TestLoudnessNormalize:
    def test_normalize_to_target(self):
        s = 0.5 * _tone(440, 2.0)
        out = loudness_normalize(s, SR, target_lufs=-14.0)
        measured = lufs_integrated(out, SR)
        assert abs(measured - (-14.0)) < 1.5

    def test_silent_signal_unchanged(self):
        s = np.zeros(SR)
        out = loudness_normalize(s, SR)
        assert np.allclose(out, s)


class TestTruePeak:
    def test_sine_peak(self):
        s = _tone(1000, 0.1, amp=1.0)
        tp = true_peak(s, SR)
        assert abs(tp - 0.0) < 1.0

    def test_oversampled_catches_intersample(self):
        s = _tone(1000, 0.1, amp=0.9)
        tp1 = true_peak(s, SR, oversample=1)
        tp4 = true_peak(s, SR, oversample=4)
        assert tp4 >= tp1 - 0.5


class TestProcessMixBus:
    def test_full_chain(self):
        stems = [_tone(440, 1.0, amp=0.3), _tone(880, 1.0, amp=0.2)]
        out = process_mix_bus(stems, SR)
        assert out.shape == (SR, 2)
        assert np.all(np.isfinite(out))

    def test_empty_stems(self):
        out = process_mix_bus([], SR)
        assert out.size == 0

    def test_custom_config(self):
        cfg = MixBusConfig(target_lufs=-16.0, headroom_db=-2.0)
        stems = [_tone(440, 1.0, amp=0.5)]
        out = process_mix_bus(stems, SR, config=cfg)
        tp = true_peak(out, SR)
        assert tp <= cfg.headroom_db + 1.0

    def test_headroom_respected(self):
        cfg = MixBusConfig(headroom_db=-1.0)
        stems = [np.ones(SR)]
        out = process_mix_bus(stems, SR, config=cfg)
        tp = true_peak(out, SR)
        assert tp <= 0.0
