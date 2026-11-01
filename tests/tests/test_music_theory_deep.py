"""Tests for counterpoint rules and modulation analysis."""
import pytest

from models.music_theory.counterpoint import (
    IntervalClass,
    MotionType,
    classify_interval,
    is_consonant,
    is_step,
    motion_type,
    validate_cantus_firmus,
    check_first_species,
    check_second_species,
    voice_leading_score,
)
from models.music_theory.modulation import (
    circle_of_fifths_distance,
    relative_key,
    parallel_key,
    key_distance,
    pivot_chords,
    common_tone_count,
    common_tone_modulation_score,
    chromatic_mediant_relationship,
    detect_enharmonic_reinterpretation,
    analyze_modulation,
)


# ── Counterpoint ─────────────────────────────────────────────────────

class TestIntervalClassification:
    def test_unison(self):
        assert classify_interval(0) == IntervalClass.PERFECT_CONSONANCE

    def test_perfect_fifth(self):
        assert classify_interval(7) == IntervalClass.PERFECT_CONSONANCE

    def test_minor_third(self):
        assert classify_interval(3) == IntervalClass.IMPERFECT_CONSONANCE

    def test_major_sixth(self):
        assert classify_interval(9) == IntervalClass.IMPERFECT_CONSONANCE

    def test_tritone(self):
        assert classify_interval(6) == IntervalClass.DISSONANCE

    def test_major_second(self):
        assert classify_interval(2) == IntervalClass.DISSONANCE

    def test_consonance(self):
        assert is_consonant(7) is True
        assert is_consonant(6) is False

    def test_step(self):
        assert is_step(1) is True
        assert is_step(2) is True
        assert is_step(3) is False


class TestMotion:
    def test_parallel(self):
        assert motion_type(2, 2) == MotionType.PARALLEL

    def test_contrary(self):
        assert motion_type(2, -3) == MotionType.CONTRARY

    def test_similar(self):
        assert motion_type(2, 4) == MotionType.SIMILAR

    def test_oblique(self):
        assert motion_type(0, 3) == MotionType.OBLIQUE


class TestCantusFirmus:
    def test_valid_cf(self):
        # C major stepwise, single climax, ends on tonic
        cf = [60, 62, 64, 65, 67, 65, 64, 62, 60]
        errors = validate_cantus_firmus(cf)
        assert len(errors) == 0

    def test_repeated_note(self):
        cf = [60, 62, 62, 64, 60]
        errors = validate_cantus_firmus(cf)
        kinds = [e.message for e in errors]
        assert any("Repeated" in m for m in kinds)

    def test_multiple_climaxes(self):
        cf = [60, 67, 64, 67, 60]
        errors = validate_cantus_firmus(cf)
        kinds = [e.message for e in errors]
        assert any("climax" in m.lower() for m in kinds)

    def test_too_short(self):
        errors = validate_cantus_firmus([60, 62, 60])
        assert any("short" in e.message.lower() for e in errors)

    def test_wrong_ending(self):
        cf = [60, 62, 64, 65, 67]  # ends on G, not C
        errors = validate_cantus_firmus(cf)
        assert any("tonic" in e.message.lower() for e in errors)


class TestFirstSpecies:
    def test_parallel_fifths(self):
        cantus = [60, 62]
        cp = [67, 69]  # both intervals are P5 (7 semitones), parallel motion
        errors = check_first_species(cantus, cp)
        kinds = [e.kind for e in errors]
        assert "parallel_perfects" in kinds

    def test_no_errors_contrary(self):
        # C→D in cantus, G→F in counterpoint = contrary motion, both consonant
        cantus = [60, 62]
        cp = [67, 65]  # P5 then m3
        errors = check_first_species(cantus, cp)
        assert len(errors) == 0

    def test_dissonance_flagged(self):
        cantus = [60, 62]
        cp = [61, 65]  # m2 then m3
        errors = check_first_species(cantus, cp)
        assert any(e.kind == "dissonance" for e in errors)


class TestSecondSpecies:
    def test_strong_beat_dissonance(self):
        cantus = [60, 62]
        cp = [61, 62, 66, 65]  # strong beat 0: m2 (dissonant)
        errors = check_second_species(cantus, cp)
        assert any("strong" in e.kind for e in errors)

    def test_passing_tone_ok(self):
        # Cantus: C, D.  CP: E, D, F#, (next would need to be step)
        # strong=E (M3 consonant), weak=D (M2 dissonant but step from E, step to F#)
        cantus = [60, 62]
        cp = [64, 62, 66, 64]  # E→D (step), D→F# (too large) but also need to check
        # Actually: D to F# = 4 semitones, not a step. Let's fix:
        # CP: E, D, F, E — E(M3), D(step down, M2 dissonant, step up to F, F is consonant)
        cantus = [60, 62]
        cp = [64, 62, 65, 64]  # E→D(step), D→F(step): passing tone
        errors = check_second_species(cantus, cp)
        strong_errors = [e for e in errors if "strong" in e.kind]
        assert len(strong_errors) == 0


class TestVoiceLeadingScore:
    def test_perfect_score(self):
        cantus = [60, 62, 64]
        cp = [67, 65, 67]  # all consonant, contrary motion
        score = voice_leading_score(cantus, cp)
        assert score == 1.0

    def test_penalized(self):
        cantus = [60, 62]
        cp = [67, 69]  # parallel fifths
        score = voice_leading_score(cantus, cp)
        assert 0.0 < score < 1.0


# ── Modulation ───────────────────────────────────────────────────────

class TestKeyDistance:
    def test_same_key(self):
        assert circle_of_fifths_distance((0, "major"), (0, "major")) == 0

    def test_fifth_apart(self):
        # C major to G major = 1 step on CoF
        assert circle_of_fifths_distance((0, "major"), (7, "major")) == 1

    def test_tritone_apart(self):
        # C to F# = 6 steps
        assert circle_of_fifths_distance((0, "major"), (6, "major")) == 6

    def test_mode_penalty(self):
        # C major to C minor = 0 CoF steps + 1 mode penalty
        assert circle_of_fifths_distance((0, "major"), (0, "minor")) == 1


class TestRelativeParallel:
    def test_relative_minor_of_c(self):
        assert relative_key(0, "major") == (9, "minor")  # A minor

    def test_relative_major_of_a_minor(self):
        assert relative_key(9, "minor") == (0, "major")  # C major

    def test_parallel_key(self):
        assert parallel_key(0, "major") == (0, "minor")


class TestPivotChords:
    def test_close_keys_have_pivots(self):
        # C major and G major share many triads
        pivots = pivot_chords((0, "major"), (7, "major"))
        assert len(pivots) >= 2

    def test_remote_keys_fewer_pivots(self):
        # C major and F# major — remote
        close = pivot_chords((0, "major"), (7, "major"))
        remote = pivot_chords((0, "major"), (6, "major"))
        assert len(remote) <= len(close)


class TestCommonTone:
    def test_same_key(self):
        assert common_tone_count((0, "major"), (0, "major")) == 7

    def test_relative_keys(self):
        # C major and A minor share all pitch classes (same set)
        ct = common_tone_count((0, "major"), (9, "minor"))
        assert ct == 7

    def test_score_range(self):
        score = common_tone_modulation_score((0, "major"), (6, "major"))
        assert 0.0 <= score <= 1.0


class TestChromaticMediant:
    def test_c_to_e_major(self):
        cm = chromatic_mediant_relationship((0, "major"), (4, "major"))
        assert cm is not None
        assert cm.root_interval == 4

    def test_not_mediant(self):
        # C to G = P5, not a mediant
        cm = chromatic_mediant_relationship((0, "major"), (7, "major"))
        assert cm is None

    def test_minor_not_applicable(self):
        cm = chromatic_mediant_relationship((0, "minor"), (4, "major"))
        assert cm is None


class TestModulationPlan:
    def test_close_modulation(self):
        plan = analyze_modulation((0, "major"), (7, "major"))
        assert plan.distance <= 2
        assert plan.pivot_count >= 1
        assert plan.smoothness > 0.5

    def test_remote_modulation(self):
        plan = analyze_modulation((0, "major"), (6, "major"))
        assert plan.distance >= 5
        assert plan.smoothness < plan.common_tones / 7.0 + 0.01
