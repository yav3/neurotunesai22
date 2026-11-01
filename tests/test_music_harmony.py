"""Tests for music_harmony: chords, progressions, voice leading, functional harmony."""
from __future__ import annotations
import pytest

from models.music_harmony.chords import (
    Chord, ChordQuality, CHORD_TYPES, chord_from_name, chord_from_pitches,
    chord_similarity, NOTE_NAMES,
)
from models.music_harmony.progressions import (
    ChordProgression, RomanNumeral, COMMON_PROGRESSIONS,
    analyze_roman_numerals, progression_complexity, progression_tension_curve,
)
from models.music_harmony.voice_leading import (
    VoiceLeadingRules, voice_lead, voice_leading_cost,
    parallel_fifths_check, parallel_octaves_check,
)
from models.music_harmony.functional import (
    HarmonicFunction, classify_function, cadence_type,
    CADENCE_TYPES, tonicization_target,
)


# ── Chords ──────────────────────────────────────────────────────────────

class TestChords:
    def test_major_triad_pcs(self):
        c = Chord(root=0, quality=ChordQuality.MAJOR)
        assert c.pitch_classes == frozenset({0, 4, 7})

    def test_minor_triad_pcs(self):
        c = Chord(root=9, quality=ChordQuality.MINOR)
        assert c.pitch_classes == frozenset({9, 0, 4})

    def test_dominant7_pcs(self):
        g7 = Chord(root=7, quality=ChordQuality.DOMINANT7)
        assert g7.pitch_classes == frozenset({7, 11, 2, 5})

    def test_diminished_pcs(self):
        bdim = Chord(root=11, quality=ChordQuality.DIMINISHED)
        assert bdim.pitch_classes == frozenset({11, 2, 5})

    def test_augmented_pcs(self):
        caug = Chord(root=0, quality=ChordQuality.AUGMENTED)
        assert caug.pitch_classes == frozenset({0, 4, 8})

    def test_chord_name_major(self):
        assert Chord(root=0, quality=ChordQuality.MAJOR).name == "C"

    def test_chord_name_minor(self):
        assert Chord(root=9, quality=ChordQuality.MINOR).name == "Am"

    def test_chord_name_dom7(self):
        assert Chord(root=7, quality=ChordQuality.DOMINANT7).name == "G7"

    def test_chord_name_slash(self):
        c = Chord(root=0, quality=ChordQuality.MAJOR, bass=7)
        assert c.name == "C/G"

    def test_root_name(self):
        assert Chord(root=5, quality=ChordQuality.MAJOR).root_name == "F"

    def test_chord_types_all_have_root(self):
        for q, intervals in CHORD_TYPES.items():
            assert intervals[0] == 0

    def test_from_name_major(self):
        c = chord_from_name("C")
        assert c.root == 0 and c.quality == ChordQuality.MAJOR

    def test_from_name_minor(self):
        c = chord_from_name("Am")
        assert c.root == 9 and c.quality == ChordQuality.MINOR

    def test_from_name_dom7(self):
        c = chord_from_name("G7")
        assert c.root == 7 and c.quality == ChordQuality.DOMINANT7

    def test_from_name_maj7(self):
        c = chord_from_name("Cmaj7")
        assert c.quality == ChordQuality.MAJOR7

    def test_from_name_dim(self):
        c = chord_from_name("Bdim")
        assert c.root == 11 and c.quality == ChordQuality.DIMINISHED

    def test_from_name_slash(self):
        c = chord_from_name("C/G")
        assert c.root == 0 and c.bass == 7

    def test_from_name_flat(self):
        c = chord_from_name("Bbm")
        assert c.root == 10 and c.quality == ChordQuality.MINOR

    def test_from_pitches_major(self):
        c = chord_from_pitches([60, 64, 67])
        assert c.quality == ChordQuality.MAJOR and c.root == 0

    def test_from_pitches_minor(self):
        c = chord_from_pitches([57, 60, 64])
        assert c.quality == ChordQuality.MINOR

    def test_similarity_identical(self):
        c = Chord(root=0, quality=ChordQuality.MAJOR)
        assert chord_similarity(c, c) == pytest.approx(1.0)

    def test_similarity_different(self):
        c = Chord(root=0, quality=ChordQuality.MAJOR)
        f = Chord(root=5, quality=ChordQuality.MAJOR)
        sim = chord_similarity(c, f)
        assert 0.0 <= sim < 1.0

    def test_similarity_relative_minor(self):
        c = Chord(root=0, quality=ChordQuality.MAJOR)
        am = Chord(root=9, quality=ChordQuality.MINOR)
        sim = chord_similarity(c, am)
        assert sim > 0.3


# ── Progressions ────────────────────────────────────────────────────────

class TestProgressions:
    def test_analyze_I_IV_V_I(self):
        chords = [
            Chord(0, ChordQuality.MAJOR), Chord(5, ChordQuality.MAJOR),
            Chord(7, ChordQuality.MAJOR), Chord(0, ChordQuality.MAJOR),
        ]
        numerals = analyze_roman_numerals(chords, key_root=0)
        assert [rn.numeral for rn in numerals] == ["I", "IV", "V", "I"]

    def test_analyze_minor_key(self):
        chords = [
            Chord(9, ChordQuality.MINOR), Chord(0, ChordQuality.MAJOR),
        ]
        numerals = analyze_roman_numerals(chords, key_root=9, mode="minor")
        assert numerals[0].numeral == "i"
        assert numerals[1].numeral == "III"

    def test_analyze_dom7(self):
        chords = [
            Chord(7, ChordQuality.DOMINANT7), Chord(0, ChordQuality.MAJOR),
        ]
        numerals = analyze_roman_numerals(chords, key_root=0)
        assert numerals[0].numeral == "V7"

    def test_complexity_simple(self):
        numerals = [
            RomanNumeral(0, "I", ChordQuality.MAJOR),
            RomanNumeral(5, "IV", ChordQuality.MAJOR),
            RomanNumeral(7, "V", ChordQuality.MAJOR),
            RomanNumeral(0, "I", ChordQuality.MAJOR),
        ]
        c = progression_complexity(numerals)
        assert 0.0 < c < 0.5

    def test_complexity_chromatic(self):
        numerals = [
            RomanNumeral(0, "I", ChordQuality.MAJOR),
            RomanNumeral(1, "?1", ChordQuality.MAJOR),
            RomanNumeral(6, "?6", ChordQuality.MAJOR),
            RomanNumeral(0, "I", ChordQuality.MAJOR),
        ]
        c = progression_complexity(numerals)
        assert c > 0.2

    def test_complexity_single(self):
        assert progression_complexity([RomanNumeral(0, "I", ChordQuality.MAJOR)]) == 0.0

    def test_tension_curve_length(self):
        numerals = [
            RomanNumeral(0, "I", ChordQuality.MAJOR),
            RomanNumeral(7, "V", ChordQuality.MAJOR),
        ]
        curve = progression_tension_curve(numerals)
        assert len(curve) == 2

    def test_tension_tonic_low(self):
        numerals = [RomanNumeral(0, "I", ChordQuality.MAJOR)]
        assert progression_tension_curve(numerals) == [0.0]

    def test_tension_dominant_high(self):
        numerals = [RomanNumeral(7, "V7", ChordQuality.DOMINANT7)]
        assert progression_tension_curve(numerals)[0] > 0.7

    def test_chord_progression_length(self):
        cp = ChordProgression([Chord(0, ChordQuality.MAJOR)] * 4)
        assert cp.length == 4

    def test_transposed(self):
        cp = ChordProgression(
            [Chord(0, ChordQuality.MAJOR), Chord(7, ChordQuality.MAJOR)],
            key_root=0,
        )
        t = cp.transposed(5)
        assert t.chords[0].root == 5
        assert t.chords[1].root == 0
        assert t.key_root == 5

    def test_common_progressions_exist(self):
        assert len(COMMON_PROGRESSIONS) >= 8


# ── Voice Leading ───────────────────────────────────────────────────────

class TestVoiceLeading:
    def test_cost_zero_for_same(self):
        source = [60, 64, 67]
        cost = voice_leading_cost(source, source)
        assert cost == 0.0

    def test_cost_increases_with_leaps(self):
        source = [60, 64, 67]
        close = [60, 65, 67]
        far = [60, 72, 67]
        assert voice_leading_cost(source, close) < voice_leading_cost(source, far)

    def test_cost_mismatched_voices(self):
        assert voice_leading_cost([60, 64], [60]) == float("inf")

    def test_cost_empty(self):
        assert voice_leading_cost([], []) == 0.0

    def test_parallel_fifths_detected(self):
        source = [60, 67]
        target = [62, 69]
        violations = parallel_fifths_check(source, target)
        assert len(violations) == 1

    def test_no_parallel_fifths(self):
        source = [60, 67]
        target = [60, 65]
        violations = parallel_fifths_check(source, target)
        assert len(violations) == 0

    def test_parallel_octaves_detected(self):
        source = [48, 60]
        target = [50, 62]
        violations = parallel_octaves_check(source, target)
        assert len(violations) == 1

    def test_no_parallel_octaves_stationary(self):
        source = [48, 60]
        target = [48, 60]
        violations = parallel_octaves_check(source, target)
        assert len(violations) == 0

    def test_voice_lead_returns_correct_count(self):
        source = [60, 64, 67]
        target_pcs = frozenset({0, 5, 9})
        result = voice_lead(source, target_pcs)
        assert len(result) == 3

    def test_voice_lead_minimal_motion(self):
        source = [60, 64, 67]
        target_pcs = frozenset({0, 4, 7})
        result = voice_lead(source, target_pcs)
        total_motion = sum(abs(r - s) for r, s in zip(result, source))
        assert total_motion == 0

    def test_voice_lead_empty_target(self):
        source = [60, 64, 67]
        result = voice_lead(source, frozenset())
        assert result == [60, 64, 67]


# ── Functional Harmony ──────────────────────────────────────────────────

class TestFunctionalHarmony:
    def test_tonic_major(self):
        c = Chord(0, ChordQuality.MAJOR)
        assert classify_function(c, key_root=0) == HarmonicFunction.TONIC

    def test_dominant_major(self):
        g = Chord(7, ChordQuality.MAJOR)
        assert classify_function(g, key_root=0) == HarmonicFunction.DOMINANT

    def test_predominant_major(self):
        f = Chord(5, ChordQuality.MAJOR)
        assert classify_function(f, key_root=0) == HarmonicFunction.PREDOMINANT

    def test_chromatic(self):
        db = Chord(1, ChordQuality.MAJOR)
        assert classify_function(db, key_root=0) == HarmonicFunction.CHROMATIC

    def test_minor_tonic(self):
        am = Chord(9, ChordQuality.MINOR)
        assert classify_function(am, key_root=9, mode="minor") == HarmonicFunction.TONIC

    def test_cadence_authentic(self):
        v = Chord(7, ChordQuality.MAJOR)
        i = Chord(0, ChordQuality.MAJOR)
        assert cadence_type(v, i) == "authentic"

    def test_cadence_plagal(self):
        iv = Chord(5, ChordQuality.MAJOR)
        i = Chord(0, ChordQuality.MAJOR)
        assert cadence_type(iv, i) == "plagal"

    def test_cadence_deceptive(self):
        v = Chord(7, ChordQuality.MAJOR)
        vi = Chord(9, ChordQuality.MINOR)
        assert cadence_type(v, vi) == "deceptive"

    def test_cadence_half(self):
        i = Chord(0, ChordQuality.MAJOR)
        v = Chord(7, ChordQuality.MAJOR)
        assert cadence_type(i, v) == "half"

    def test_cadence_none(self):
        iii = Chord(4, ChordQuality.MINOR)
        ii = Chord(2, ChordQuality.MINOR)
        assert cadence_type(iii, ii) is None

    def test_cadence_dom7_authentic(self):
        v7 = Chord(7, ChordQuality.DOMINANT7)
        i = Chord(0, ChordQuality.MAJOR)
        assert cadence_type(v7, i) == "authentic"

    def test_tonicization_V_of_V(self):
        d = Chord(2, ChordQuality.MAJOR)
        g = Chord(7, ChordQuality.MAJOR)
        assert tonicization_target(d, g) == 7

    def test_tonicization_no_resolution(self):
        d = Chord(2, ChordQuality.MAJOR)
        c = Chord(0, ChordQuality.MAJOR)
        assert tonicization_target(d, c) is None

    def test_tonicization_minor_chord(self):
        dm = Chord(2, ChordQuality.MINOR)
        g = Chord(7, ChordQuality.MAJOR)
        assert tonicization_target(dm, g) is None

    def test_cadence_types_count(self):
        assert len(CADENCE_TYPES) >= 10
