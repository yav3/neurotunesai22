"""Tests for deep accessibility models: sonification, speech-to-music."""
import numpy as np
import pytest

from models.music_accessibility.sonification import (
    ParameterMapping, PitchSonifier, RhythmSonifier,
    MultidimensionalSonifier, SonificationEvaluator,
)
from models.music_accessibility.speech_to_music import (
    PitchTracker, VoiceToMIDI, VocalGestureDetector, VoiceCommandClassifier,
)


SR = 16000


def _sine(freq, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


class TestParameterMapping:
    def test_linear_min(self):
        pm = ParameterMapping(0, 10, 100, 1000)
        assert abs(pm.map_value(0) - 100) < 1e-6

    def test_linear_max(self):
        pm = ParameterMapping(0, 10, 100, 1000)
        assert abs(pm.map_value(10) - 1000) < 1e-6

    def test_log_mapping(self):
        pm = ParameterMapping(0, 1, 0, 1, mapping_type="log")
        mid = pm.map_value(0.5)
        assert 0 < mid < 1

    def test_exponential_mapping(self):
        pm = ParameterMapping(0, 1, 0, 1, mapping_type="exponential")
        mid = pm.map_value(0.5)
        assert 0 < mid < 1

    def test_map_array(self):
        pm = ParameterMapping(0, 10, 0, 100)
        data = np.array([0, 5, 10])
        mapped = pm.map_array(data)
        np.testing.assert_allclose(mapped, [0, 50, 100], atol=1e-6)

    def test_constant_data(self):
        pm = ParameterMapping(5, 5, 100, 200)
        assert abs(pm.map_value(5) - 150) < 1e-6


class TestPitchSonifier:
    def test_output_length(self):
        ps = PitchSonifier()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        audio = ps.sonify(data, sr=SR, note_duration=0.1)
        expected = int(SR * 0.1) * 5
        assert len(audio) == expected

    def test_different_data_different_audio(self):
        ps = PitchSonifier()
        a1 = ps.sonify(np.array([1.0, 1.0, 1.0]), sr=SR, note_duration=0.1)
        a2 = ps.sonify(np.array([1.0, 5.0, 10.0]), sr=SR, note_duration=0.1)
        assert not np.allclose(a1, a2)

    def test_pentatonic_scale(self):
        ps = PitchSonifier(scale="pentatonic")
        data = np.linspace(0, 1, 10)
        audio = ps.sonify(data, sr=SR, note_duration=0.05)
        assert len(audio) > 0


class TestRhythmSonifier:
    def test_output_length(self):
        rs = RhythmSonifier(tempo=120)
        events = np.array([1, 0, 1, 0])
        audio = rs.sonify(events, sr=SR)
        expected = int(SR * 4 * 0.5)
        assert len(audio) == expected

    def test_clicks_at_events(self):
        rs = RhythmSonifier(tempo=60)
        events = np.array([1, 0, 0, 0])
        audio = rs.sonify(events, sr=SR)
        first_beat_energy = np.sum(audio[:int(SR * 0.01)] ** 2)
        later_energy = np.sum(audio[SR:SR + int(SR * 0.01)] ** 2)
        assert first_beat_energy > later_energy


class TestMultidimensionalSonifier:
    def test_output_shape(self):
        ms = MultidimensionalSonifier()
        data = np.random.default_rng(42).standard_normal((20, 3))
        audio = ms.sonify(data, sr=SR, duration=2.0)
        assert len(audio) == int(SR * 2.0)

    def test_1d_data(self):
        ms = MultidimensionalSonifier()
        data = np.array([1.0, 2.0, 3.0])
        audio = ms.sonify(data, sr=SR, duration=1.0)
        assert len(audio) == SR

    def test_normalized(self):
        ms = MultidimensionalSonifier()
        data = np.random.default_rng(42).standard_normal((50, 2))
        audio = ms.sonify(data, sr=SR, duration=3.0)
        assert np.max(np.abs(audio)) <= 1.0 + 1e-6


class TestSonificationEvaluator:
    def test_discriminability_perfect(self):
        se = SonificationEvaluator()
        data = np.array([1.0, 2.0, 3.0, 4.0])
        assert se.discriminability(data, data) > 0.99

    def test_discriminability_range(self):
        se = SonificationEvaluator()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        d = se.discriminability(a, b)
        assert 0 <= d <= 1.0

    def test_information_density_positive(self):
        se = SonificationEvaluator()
        data = np.arange(100, dtype=float)
        density = se.information_density(data, 5.0)
        assert density > 0


class TestPitchTracker:
    def test_detect_440(self):
        pt = PitchTracker(min_freq=100, max_freq=600)
        sig = _sine(440.0, dur=0.5)
        f0 = pt.track(sig, SR)
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            assert abs(np.median(voiced) - 440) < 50

    def test_silence_no_pitch(self):
        pt = PitchTracker()
        sig = np.zeros(SR)
        f0 = pt.track(sig, SR)
        assert np.all(f0 == 0)

    def test_voiced_segments(self):
        pt = PitchTracker()
        f0 = np.array([0, 0, 440, 442, 438, 0, 0, 330, 332, 0])
        segs = pt.voiced_segments(f0)
        assert len(segs) >= 1


class TestVoiceToMIDI:
    def test_single_note(self):
        vm = VoiceToMIDI(quantize=True)
        f0 = np.array([440.0] * 20)
        events = vm.convert(f0, SR, 512)
        assert len(events) >= 1
        assert events[0]["pitch"] == 69

    def test_two_notes(self):
        vm = VoiceToMIDI(quantize=True)
        f0 = np.concatenate([np.full(20, 440.0), np.full(20, 880.0)])
        events = vm.convert(f0, SR, 512)
        assert len(events) >= 2

    def test_scale_quantize(self):
        vm = VoiceToMIDI(quantize=True, scale=[60, 62, 64, 65, 67, 69, 71, 72])
        f0 = np.array([440.0] * 10)
        events = vm.convert(f0, SR, 512)
        assert events[0]["pitch"] in [60, 62, 64, 65, 67, 69, 71, 72]

    def test_empty_track(self):
        vm = VoiceToMIDI()
        events = vm.convert(np.zeros(10), SR, 512)
        assert len(events) == 0


class TestVocalGestureDetector:
    def test_detect_onset(self):
        vgd = VocalGestureDetector()
        sig = np.zeros(SR)
        sig[SR // 2:SR // 2 + 1000] = _sine(440.0, dur=1000 / SR)[:1000]
        onsets = vgd.detect_onset(sig, SR)
        assert isinstance(onsets, list)

    def test_detect_vibrato(self):
        vgd = VocalGestureDetector()
        t = np.arange(100)
        f0 = 440.0 + 10.0 * np.sin(2 * np.pi * 5.5 * t / 100)
        result = vgd.detect_vibrato(f0, SR, 512)
        assert "rate_hz" in result
        assert "depth_cents" in result
        assert "regularity" in result

    def test_detect_glissando(self):
        vgd = VocalGestureDetector()
        f0 = np.linspace(200, 400, 50)
        gliss = vgd.detect_glissando(f0, SR, 512, threshold_cents=100)
        assert isinstance(gliss, list)


class TestVoiceCommandClassifier:
    def test_extract_features_shape(self):
        vcc = VoiceCommandClassifier()
        sig = _sine(440.0, dur=0.3)
        feats = vcc.extract_features(sig, SR)
        assert len(feats) == 6

    def test_classify_returns_valid(self):
        vcc = VoiceCommandClassifier()
        sig = _sine(440.0, dur=0.3)
        cmd, conf = vcc.classify(sig, SR)
        assert cmd in vcc.commands
        assert 0 <= conf <= 1.0

    def test_empty_signal(self):
        vcc = VoiceCommandClassifier()
        feats = vcc.extract_features(np.array([]), SR)
        np.testing.assert_array_equal(feats, np.zeros(6))
