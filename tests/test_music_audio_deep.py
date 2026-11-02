"""Deep tests for audio segmentation and loudness measurement."""
import numpy as np
import pytest

from models.music_audio.segmentation import (
    rms_energy, detect_silence, detect_activity,
    energy_novelty, find_boundaries, Segment,
)
from models.music_audio.loudness import (
    integrated_loudness, true_peak_dbtp,
    measure_loudness, LoudnessResult,
    _k_weight_filter,
)

SR = 16000


def _sine(freq=440.0, dur=1.0, amp=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _silence(dur=1.0, sr=SR):
    return np.zeros(int(sr * dur))


# ── Segmentation ─────────────────────────────────────────────────────

class TestRMSEnergy:
    def test_silence_low(self):
        e = rms_energy(_silence(), frame_size=512, hop=256)
        assert np.all(e < -60)

    def test_loud_signal_high(self):
        e = rms_energy(_sine(amp=0.9), frame_size=512, hop=256)
        assert np.all(e > -10)

    def test_shape(self):
        sig = _sine(dur=0.5)
        e = rms_energy(sig, frame_size=512, hop=256)
        expected = max(1, (len(sig) - 512) // 256 + 1)
        assert len(e) == expected


class TestDetectSilence:
    def test_finds_silence(self):
        loud = _sine(dur=0.5, amp=0.5)
        quiet = _silence(dur=1.0)
        signal = np.concatenate([loud, quiet, loud])
        segs = detect_silence(signal, SR, threshold_db=-40, min_duration_sec=0.3)
        assert len(segs) >= 1
        assert all(s.label == "silence" for s in segs)

    def test_no_silence_in_loud(self):
        segs = detect_silence(_sine(dur=2.0, amp=0.8), SR)
        assert len(segs) == 0

    def test_all_silence(self):
        segs = detect_silence(_silence(dur=2.0), SR, min_duration_sec=0.5)
        assert len(segs) >= 1


class TestDetectActivity:
    def test_finds_activity(self):
        signal = np.concatenate([_silence(0.5), _sine(dur=1.0, amp=0.5), _silence(0.5)])
        segs = detect_activity(signal, SR, threshold_db=-20)
        assert len(segs) >= 1
        assert all(s.label == "active" for s in segs)

    def test_all_silent(self):
        segs = detect_activity(_silence(1.0), SR, threshold_db=-20)
        assert len(segs) == 0


class TestEnergyNovelty:
    def test_shape(self):
        sig = _sine(dur=1.0)
        n = energy_novelty(sig, frame_size=1024, hop=512)
        assert len(n) > 0

    def test_first_zero(self):
        n = energy_novelty(_sine(dur=0.5))
        assert n[0] == 0.0

    def test_nonneg(self):
        n = energy_novelty(_sine(dur=1.0))
        assert np.all(n >= 0)


class TestFindBoundaries:
    def test_onset_detected(self):
        signal = np.concatenate([_silence(1.0), _sine(dur=1.0, amp=0.9)])
        bounds = find_boundaries(signal, SR, threshold=1.5)
        assert len(bounds) >= 1

    def test_constant_no_boundaries(self):
        bounds = find_boundaries(_sine(dur=2.0), SR, threshold=5.0)
        assert len(bounds) == 0


# ── Loudness ─────────────────────────────────────────────────────────

class TestKWeightFilter:
    def test_output_length(self):
        sig = _sine(dur=0.5)
        out = _k_weight_filter(sig, SR)
        assert len(out) == len(sig)

    def test_preserves_energy(self):
        sig = _sine(freq=1000.0, dur=1.0, amp=0.5)
        out = _k_weight_filter(sig, SR)
        assert np.sum(out ** 2) > 0


class TestIntegratedLoudness:
    def test_silence_very_low(self):
        lufs = integrated_loudness(_silence(2.0), SR)
        assert lufs <= -60

    def test_loud_signal_higher(self):
        loud = integrated_loudness(_sine(dur=2.0, amp=0.8), SR)
        quiet = integrated_loudness(_sine(dur=2.0, amp=0.05), SR)
        assert loud > quiet

    def test_returns_float(self):
        assert isinstance(integrated_loudness(_sine(), SR), float)


class TestTruePeak:
    def test_clipped_signal(self):
        sig = np.clip(_sine(amp=1.5), -1.0, 1.0)
        tp = true_peak_dbtp(sig)
        assert tp <= 0.1

    def test_empty(self):
        tp = true_peak_dbtp(np.array([]))
        assert tp == -np.inf

    def test_quiet_low(self):
        tp = true_peak_dbtp(_sine(amp=0.001))
        assert tp < -40


class TestMeasureLoudness:
    def test_returns_result(self):
        r = measure_loudness(_sine(dur=2.0), SR)
        assert isinstance(r, LoudnessResult)
        assert isinstance(r.integrated_lufs, float)
        assert len(r.momentary_lufs) > 0

    def test_range_positive(self):
        sig = np.concatenate([_sine(amp=0.1, dur=2.0), _sine(amp=0.8, dur=2.0)])
        r = measure_loudness(sig, SR)
        assert r.loudness_range_lu >= 0

    def test_peak_consistency(self):
        r = measure_loudness(_sine(amp=0.5), SR)
        assert r.true_peak_dbtp >= r.sample_peak_dbfs - 1.0
