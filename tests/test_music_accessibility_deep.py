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


# ---- Audio description (enhanced) ----

from models.music_accessibility.audio_description import (
    describe_tempo, describe_rhythm_features, describe_harmony_features,
    describe_dynamics_features, describe_timbre_features,
    generate_full_description, DescriptionLevel, AudioDescription,
    describe_pitch, describe_interval, describe_rhythm,
    describe_dynamics, describe_chord, describe_progression,
    generate_audio_guide,
)


class TestDescribeTempo:
    def test_grave(self):
        assert "grave" in describe_tempo(50)

    def test_allegro(self):
        assert "allegro" in describe_tempo(130)

    def test_bpm_in_output(self):
        assert "120" in describe_tempo(120)

    def test_presto(self):
        assert "presto" in describe_tempo(200)


class TestDescribeRhythmFeatures:
    def test_basic(self):
        r = describe_rhythm_features(120)
        assert "120" in r and "4/4" in r

    def test_swing(self):
        r = describe_rhythm_features(100, swing_ratio=1.5)
        assert "swung" in r.lower()

    def test_reverse_swing(self):
        r = describe_rhythm_features(100, swing_ratio=0.6)
        assert "reverse" in r.lower()

    def test_waltz_meter(self):
        r = describe_rhythm_features(100, time_sig=(3, 4))
        assert "3/4" in r


class TestDescribeHarmonyFeatures:
    def test_key_only(self):
        r = describe_harmony_features("C", "minor")
        assert "C minor" in r

    def test_with_progression(self):
        r = describe_harmony_features("G", "major", ["I", "IV", "V", "I"])
        assert "I" in r and "V" in r


class TestDescribeDynamicsFeatures:
    def test_quiet(self):
        env = np.full(100, 0.08)
        r = describe_dynamics_features(env)
        assert "quiet" in r.lower()

    def test_loud(self):
        env = np.full(100, 0.6)
        r = describe_dynamics_features(env)
        assert "loud" in r.lower()

    def test_empty(self):
        r = describe_dynamics_features(np.array([]))
        assert "no dynamic" in r.lower()

    def test_wide_range(self):
        env = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
        r = describe_dynamics_features(env)
        assert "wide" in r.lower()

    def test_peak_db(self):
        env = np.full(50, 0.3)
        r = describe_dynamics_features(env, peak_db=-3.2)
        assert "-3.2" in r


class TestDescribeTimbreFeatures:
    def test_dark(self):
        r = describe_timbre_features(300)
        assert "dark" in r.lower() or "warm" in r.lower()

    def test_bright(self):
        r = describe_timbre_features(3000)
        assert "bright" in r.lower()

    def test_roughness(self):
        r = describe_timbre_features(1000, roughness=0.8)
        assert "rough" in r.lower() or "distort" in r.lower()


class TestGenerateFullDescription:
    def test_standard(self):
        feats = {"tempo": 120, "key": "D", "mode": "minor",
                 "rms_envelope": np.full(50, 0.3), "spectral_centroid": 2500}
        ad = generate_full_description(feats)
        assert isinstance(ad, AudioDescription)
        assert ad.rhythm != ""
        assert ad.harmony != ""
        assert ad.timbre != ""

    def test_brief(self):
        feats = {"tempo": 90, "key": "A"}
        ad = generate_full_description(feats, DescriptionLevel.BRIEF)
        assert ad.level == DescriptionLevel.BRIEF
        assert "90" in ad.rhythm

    def test_full_text(self):
        feats = {"tempo": 100, "key": "C"}
        ad = generate_full_description(feats)
        text = ad.full_text()
        assert text.endswith(".")


# ---- Haptic mapping ----

from models.music_accessibility.haptic_mapping import (
    HapticPattern, HapticEvent,
    beat_to_haptic, melody_to_haptic, dynamics_to_intensity,
    rhythm_pattern_to_vibration, continuous_haptic_envelope,
)


class TestBeatToHaptic:
    def test_basic(self):
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        events = beat_to_haptic(beats, 120.0)
        assert len(events) == 4

    def test_downbeat_accent(self):
        beats = np.arange(8) * 0.5
        events = beat_to_haptic(beats, 120.0, beats_per_bar=4)
        assert events[0].intensity > events[1].intensity
        assert events[0].pattern == HapticPattern.DOUBLE_TAP
        assert events[1].pattern == HapticPattern.PULSE

    def test_empty(self):
        assert beat_to_haptic(np.array([]), 120) == []


class TestMelodyToHaptic:
    def test_basic(self):
        pitches = np.array([60, 64, 67])
        durs = np.array([0.5, 0.5, 1.0])
        events = melody_to_haptic(pitches, durs)
        assert len(events) == 3
        assert all(e.pattern == HapticPattern.SUSTAINED for e in events)

    def test_pitch_to_freq(self):
        pitches = np.array([36, 96])
        durs = np.array([1.0, 1.0])
        events = melody_to_haptic(pitches, durs)
        assert events[0].frequency_hz < events[1].frequency_hz

    def test_empty(self):
        assert melody_to_haptic(np.array([]), np.array([])) == []


class TestDynamicsToIntensity:
    def test_range(self):
        rms = np.array([0.0, 0.5, 1.0])
        out = dynamics_to_intensity(rms, floor=0.1, ceiling=0.9)
        assert out[0] == pytest.approx(0.1)
        assert out[-1] == pytest.approx(0.9)

    def test_constant(self):
        rms = np.full(10, 0.5)
        out = dynamics_to_intensity(rms)
        assert np.std(out) < 1e-6

    def test_empty(self):
        out = dynamics_to_intensity(np.array([]))
        assert len(out) == 0


class TestRhythmPatternToVibration:
    def test_basic(self):
        pat = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        events = rhythm_pattern_to_vibration(pat)
        assert len(events) == 4

    def test_accented(self):
        pat = np.array([2, 0, 1, 0])
        events = rhythm_pattern_to_vibration(pat, grid_resolution=4)
        assert events[0].frequency_hz > events[1].frequency_hz

    def test_empty(self):
        assert rhythm_pattern_to_vibration(np.array([])) == []


class TestContinuousHapticEnvelope:
    def test_shape(self):
        events = [HapticEvent(0.1, 0.8, 200, 50)]
        env = continuous_haptic_envelope(events, 1.0, sample_rate=250)
        assert len(env) == 250

    def test_zero_duration(self):
        env = continuous_haptic_envelope([], 0.0)
        assert len(env) == 0

    def test_ramp_up(self):
        events = [HapticEvent(0.0, 0.9, 200, 200, HapticPattern.RAMP_UP)]
        env = continuous_haptic_envelope(events, 0.5, sample_rate=250)
        assert env[0] < env[20]

    def test_double_tap(self):
        events = [HapticEvent(0.0, 0.8, 200, 400, HapticPattern.DOUBLE_TAP)]
        env = continuous_haptic_envelope(events, 1.0, sample_rate=250)
        assert np.max(env) > 0


# ---- Cognitive load ----

from models.music_accessibility.cognitive_load import (
    CognitiveLoadEstimate,
    harmonic_complexity_load, rhythmic_complexity_load,
    textural_density_load, combined_cognitive_load,
    passage_cognitive_profile,
)


class TestHarmonicComplexityLoad:
    def test_empty(self):
        assert harmonic_complexity_load([]) == 0.0

    def test_single_chord(self):
        load = harmonic_complexity_load([(0, 4, 7)])
        assert 0 <= load <= 1

    def test_high_diversity(self):
        chords = [(0, 4, 7), (2, 5, 9), (4, 7, 11), (5, 9, 0),
                  (7, 11, 2), (9, 0, 4), (11, 2, 6)]
        load = harmonic_complexity_load(chords)
        assert load > 0.3

    def test_low_diversity(self):
        chords = [(0, 4, 7)] * 20
        load = harmonic_complexity_load(chords)
        assert load < 0.3


class TestRhythmicComplexityLoad:
    def test_empty(self):
        assert rhythmic_complexity_load(np.array([])) == 0.0

    def test_simple_pattern(self):
        pat = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        load = rhythmic_complexity_load(pat)
        assert 0 <= load <= 1

    def test_dense_pattern(self):
        pat = np.ones(16)
        load_dense = rhythmic_complexity_load(pat)
        pat_sparse = np.zeros(16)
        pat_sparse[0] = 1
        load_sparse = rhythmic_complexity_load(pat_sparse)
        assert load_dense > load_sparse

    def test_syncopation(self):
        pat = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        meter = np.array([4, 1, 2, 1, 3, 1, 2, 1], dtype=float)
        load = rhythmic_complexity_load(pat, meter)
        assert load > 0


class TestTexturalDensityLoad:
    def test_single_voice(self):
        load = textural_density_load(1, 0)
        assert load < 0.2

    def test_many_voices(self):
        load = textural_density_load(8, 48)
        assert load > 0.8

    def test_bounds(self):
        load = textural_density_load(4, 24)
        assert 0 <= load <= 1


class TestCombinedCognitiveLoad:
    def test_low(self):
        est = combined_cognitive_load(0.1, 0.1, 0.1)
        assert est.category == "low"
        assert est.combined < 0.25

    def test_high(self):
        est = combined_cognitive_load(0.9, 0.9, 0.9)
        assert est.category == "very_high"

    def test_weights(self):
        est = combined_cognitive_load(1.0, 0.0, 0.0, weights=(1.0, 0.0, 0.0))
        assert est.combined == pytest.approx(1.0)


class TestPassageCognitiveProfile:
    def test_end_to_end(self):
        chords = [(0, 4, 7), (5, 9, 0), (7, 11, 2), (0, 4, 7)]
        onset = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        est = passage_cognitive_profile(chords, onset, n_voices=3, register_spread=24)
        assert isinstance(est, CognitiveLoadEstimate)
        assert 0 <= est.combined <= 1
        assert est.category in ("low", "moderate", "high", "very_high")


# ---- Exports ----

class TestAccessibilityDeepExports:
    def test_haptic_exports(self):
        from models.music_accessibility import (
            HapticPattern, HapticEvent,
            beat_to_haptic, melody_to_haptic, dynamics_to_intensity,
            rhythm_pattern_to_vibration, continuous_haptic_envelope,
        )
        assert callable(beat_to_haptic)

    def test_cognitive_exports(self):
        from models.music_accessibility import (
            CognitiveLoadEstimate,
            harmonic_complexity_load, rhythmic_complexity_load,
            textural_density_load, combined_cognitive_load,
            passage_cognitive_profile,
        )
        assert callable(passage_cognitive_profile)

    def test_description_exports(self):
        from models.music_accessibility import (
            DescriptionLevel, AudioDescription,
            describe_tempo, describe_rhythm_features,
            describe_harmony_features, describe_dynamics_features,
            describe_timbre_features, generate_full_description,
        )
        assert callable(generate_full_description)
