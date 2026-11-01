"""Tests for round 18 deepening: assistive playback, expressive dynamics, ear training."""
import numpy as np
import pytest

from models.music_accessibility.assistive_playback import (
    PlaybackRegion, CuePoint, PlaybackProfile,
    time_stretch_factor, pitch_shift_ratio, resample_signal,
    mono_mixdown, dynamic_compression_simple, extract_loop_region,
    apply_crossfade, find_nearest_cue, generate_practice_regions,
    playback_duration, apply_profile, accessibility_score,
)
from models.music_dynamics.expressive_dynamics import (
    DynamicMarking, DynamicContourType, DynamicEvent, DynamicProfile,
    rms_envelope, db_envelope, classify_dynamic_level,
    detect_crescendo, detect_decrescendo, detect_sforzando,
    terraced_dynamics, dynamic_range, dynamic_variance,
    accent_pattern, loudness_contour_similarity, analyze_dynamics,
)
from models.music_education.ear_training import (
    IntervalQuestion, ChordQuestion, DictationExercise, EarTrainingScore,
    interval_name, interval_difficulty, generate_interval_question,
    score_interval_answer, chord_pitches, generate_chord_question,
    score_chord_answer, generate_dictation, score_dictation,
    contour_score, adaptive_difficulty, session_summary,
    CHORD_TEMPLATES, INTERVAL_DIFFICULTY,
)


# ── Assistive Playback ──────────────────────────────────────────────

class TestTimeStretch:
    def test_double_speed(self):
        assert time_stretch_factor(120, 240) == pytest.approx(2.0)

    def test_half_speed(self):
        assert time_stretch_factor(120, 60) == pytest.approx(0.5)

    def test_same(self):
        assert time_stretch_factor(100, 100) == pytest.approx(1.0)

    def test_zero_bpm(self):
        assert time_stretch_factor(0, 120) == 1.0


class TestPitchShift:
    def test_octave_up(self):
        assert pitch_shift_ratio(12) == pytest.approx(2.0)

    def test_octave_down(self):
        assert pitch_shift_ratio(-12) == pytest.approx(0.5)

    def test_no_shift(self):
        assert pitch_shift_ratio(0) == pytest.approx(1.0)


class TestResample:
    def test_identity(self):
        sig = np.random.randn(100)
        out = resample_signal(sig, 1.0)
        np.testing.assert_allclose(out, sig)

    def test_downsample(self):
        sig = np.random.randn(100)
        out = resample_signal(sig, 2.0)
        assert len(out) == 50


class TestMonoMixdown:
    def test_identical_channels(self):
        sig = np.random.randn(100)
        out = mono_mixdown(sig, sig)
        np.testing.assert_allclose(out, sig)


class TestLoopRegion:
    def test_single_loop(self):
        sig = np.arange(100, dtype=float)
        region = PlaybackRegion(0.0, 0.5, "A", 1)
        out = extract_loop_region(sig, 100, region)
        assert len(out) == 50

    def test_double_loop(self):
        sig = np.arange(100, dtype=float)
        region = PlaybackRegion(0.0, 0.5, "A", 2)
        out = extract_loop_region(sig, 100, region)
        assert len(out) == 100


class TestCueNavigation:
    def test_forward(self):
        cues = [CuePoint(1.0, "A"), CuePoint(3.0, "B"), CuePoint(5.0, "C")]
        c = find_nearest_cue(cues, 2.0, "forward")
        assert c.label == "B"

    def test_backward(self):
        cues = [CuePoint(1.0, "A"), CuePoint(3.0, "B"), CuePoint(5.0, "C")]
        c = find_nearest_cue(cues, 4.0, "backward")
        assert c.label == "B"

    def test_no_forward(self):
        cues = [CuePoint(1.0, "A")]
        assert find_nearest_cue(cues, 2.0, "forward") is None


class TestAccessibilityScore:
    def test_default_profile(self):
        p = PlaybackProfile()
        assert 0.0 <= accessibility_score(p) <= 1.0

    def test_full_profile(self):
        p = PlaybackProfile(
            tempo_factor=0.7, mono_mixdown=True, reduce_dynamics=True,
            cue_points=[CuePoint(0, "A"), CuePoint(1, "B"), CuePoint(2, "C")],
            loop_regions=[PlaybackRegion(0, 1)])
        assert accessibility_score(p) >= 0.8


# ── Expressive Dynamics ─────────────────────────────────────────────

class TestRMSEnvelope:
    def test_shape(self):
        sig = np.random.randn(1024)
        env = rms_envelope(sig, hop=256)
        assert len(env) == 4

    def test_nonnegative(self):
        sig = np.random.randn(512)
        env = rms_envelope(sig)
        assert np.all(env >= 0)


class TestDynamicLevel:
    def test_forte(self):
        assert classify_dynamic_level(-10) == DynamicMarking.FF

    def test_piano(self):
        assert classify_dynamic_level(-35) == DynamicMarking.P

    def test_ppp(self):
        assert classify_dynamic_level(-60) == DynamicMarking.PPP


class TestCrescendo:
    def test_rising_signal(self):
        env = np.linspace(-40, -10, 20)
        cresc = detect_crescendo(env, min_frames=3, threshold_db=5.0)
        assert len(cresc) >= 1
        assert cresc[0][2] > 5.0

    def test_flat_no_crescendo(self):
        env = np.full(20, -20.0)
        cresc = detect_crescendo(env, min_frames=3, threshold_db=3.0)
        assert len(cresc) == 0


class TestDecrescendo:
    def test_falling_signal(self):
        env = np.linspace(-10, -40, 20)
        decresc = detect_decrescendo(env, min_frames=3, threshold_db=5.0)
        assert len(decresc) >= 1


class TestSforzando:
    def test_spike(self):
        env = np.full(20, -30.0)
        env[10] = -5.0
        sfz = detect_sforzando(env, threshold_db=10.0)
        assert 10 in sfz


class TestDynamicRange:
    def test_range(self):
        env = np.array([-40, -30, -20, -10, 0], dtype=float)
        dr = dynamic_range(env)
        assert dr > 0

    def test_constant(self):
        env = np.full(10, -20.0)
        assert dynamic_range(env) == pytest.approx(0.0)


class TestAnalyzeDynamics:
    def test_returns_profile(self):
        sig = np.random.randn(4096) * np.linspace(0.01, 1.0, 4096)
        profile = analyze_dynamics(sig, sr=44100, hop=512)
        assert isinstance(profile, DynamicProfile)
        assert profile.dynamic_range_db >= 0


# ── Ear Training ─────────────────────────────────────────────────

class TestIntervalName:
    def test_perfect_fifth(self):
        assert interval_name(7) == "P5"

    def test_minor_third(self):
        assert interval_name(3) == "m3"

    def test_unison(self):
        assert interval_name(0) == "P1"


class TestIntervalDifficulty:
    def test_unison_easy(self):
        assert interval_difficulty(0) == 1

    def test_tritone_hard(self):
        assert interval_difficulty(6) == 5


class TestChordPitches:
    def test_major_root(self):
        p = chord_pitches(60, "major")
        assert p == (60, 64, 67)

    def test_minor_first_inversion(self):
        p = chord_pitches(60, "minor", inversion=1)
        assert p[0] == 63  # E♭ moved up

    def test_dom7(self):
        p = chord_pitches(60, "dom7")
        assert len(p) == 4


class TestGenerateQuestion:
    def test_interval_question(self):
        rng = np.random.RandomState(42)
        q = generate_interval_question(difficulty=3, rng=rng)
        assert isinstance(q, IntervalQuestion)
        assert 48 <= q.root_midi < 72

    def test_chord_question(self):
        rng = np.random.RandomState(42)
        q = generate_chord_question(difficulty=2, rng=rng)
        assert isinstance(q, ChordQuestion)
        assert q.chord_type in CHORD_TEMPLATES


class TestScoring:
    def test_correct_interval(self):
        q = IntervalQuestion(60, 7)
        assert score_interval_answer(q, 7)

    def test_wrong_interval(self):
        q = IntervalQuestion(60, 7)
        assert not score_interval_answer(q, 6)

    def test_correct_chord(self):
        q = ChordQuestion(60, "major")
        assert score_chord_answer(q, "major")


class TestDictation:
    def test_generate(self):
        rng = np.random.RandomState(42)
        d = generate_dictation(8, key=60, rng=rng)
        assert len(d.notes_midi) == 8
        assert len(d.rhythm_beats) == 8

    def test_perfect_score(self):
        d = DictationExercise([60, 62, 64, 65], [1, 1, 1, 1])
        assert score_dictation(d, [60, 62, 64, 65]) == pytest.approx(1.0)

    def test_contour_perfect(self):
        d = DictationExercise([60, 62, 64, 62], [1, 1, 1, 1])
        assert contour_score(d, [48, 50, 52, 50]) == pytest.approx(1.0)


class TestAdaptiveDifficulty:
    def test_increase(self):
        scores = [1.0, 1.0, 0.9, 0.95, 0.85]
        assert adaptive_difficulty(scores, 3) == 4

    def test_decrease(self):
        scores = [0.2, 0.3, 0.1, 0.2, 0.3]
        assert adaptive_difficulty(scores, 3) == 2

    def test_stable(self):
        scores = [0.6, 0.65, 0.55, 0.6, 0.7]
        assert adaptive_difficulty(scores, 3) == 3
