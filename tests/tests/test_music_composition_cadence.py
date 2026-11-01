"""Tests for music_composition.cadence — cadence detection and classification."""
import pytest
from models.music_composition.cadence import (
    CadenceType, Cadence, detect_cadence, generate_cadence,
    cadential_strength, find_cadences, suggest_cadence,
    is_cadential_64, cadence_rate, ROMAN_TO_DEGREE,
)


class TestDetectCadence:
    def test_pac(self):
        assert detect_cadence("V", "I") == CadenceType.PERFECT_AUTHENTIC

    def test_iac(self):
        assert detect_cadence("V", "i") == CadenceType.IMPERFECT_AUTHENTIC

    def test_half(self):
        assert detect_cadence("IV", "V") == CadenceType.HALF

    def test_deceptive(self):
        assert detect_cadence("V", "vi") == CadenceType.DECEPTIVE

    def test_plagal(self):
        assert detect_cadence("IV", "I") == CadenceType.PLAGAL

    def test_phrygian(self):
        assert detect_cadence("bII", "V") == CadenceType.PHRYGIAN

    def test_no_cadence(self):
        assert detect_cadence("ii", "IV") is None

    def test_unknown_chord(self):
        assert detect_cadence("X", "Y") is None

    def test_half_from_ii(self):
        assert detect_cadence("ii", "V") == CadenceType.HALF

    def test_half_from_I(self):
        assert detect_cadence("I", "V") == CadenceType.HALF


class TestGenerateCadence:
    def test_pac(self):
        assert generate_cadence(CadenceType.PERFECT_AUTHENTIC) == ("V", "I")

    def test_deceptive(self):
        assert generate_cadence(CadenceType.DECEPTIVE) == ("V", "vi")

    def test_plagal(self):
        assert generate_cadence(CadenceType.PLAGAL) == ("IV", "I")

    def test_phrygian(self):
        assert generate_cadence(CadenceType.PHRYGIAN) == ("bII", "V")

    def test_roundtrip(self):
        for ct in CadenceType:
            a, b = generate_cadence(ct)
            detected = detect_cadence(a, b)
            assert detected is not None


class TestCadentialStrength:
    def test_pac_strongest(self):
        c = Cadence(type=CadenceType.PERFECT_AUTHENTIC, chords=("V", "I"), position_beats=4.0, key="C")
        assert cadential_strength(c) == 1.0

    def test_deceptive_weak(self):
        c = Cadence(type=CadenceType.DECEPTIVE, chords=("V", "vi"), position_beats=4.0, key="C")
        assert cadential_strength(c) == 0.3

    def test_ordering(self):
        types = [CadenceType.PERFECT_AUTHENTIC, CadenceType.IMPERFECT_AUTHENTIC,
                 CadenceType.PLAGAL, CadenceType.PHRYGIAN,
                 CadenceType.HALF, CadenceType.DECEPTIVE]
        strengths = []
        for ct in types:
            c = Cadence(type=ct, chords=("V", "I"), position_beats=0, key="C")
            strengths.append(cadential_strength(c))
        assert strengths[0] == max(strengths)


class TestFindCadences:
    def test_simple_progression(self):
        chords = ["I", "IV", "V", "I"]
        cads = find_cadences(chords, "C")
        types = [c.type for c in cads]
        assert CadenceType.HALF in types
        assert CadenceType.PERFECT_AUTHENTIC in types

    def test_empty(self):
        assert find_cadences([], "C") == []

    def test_single_chord(self):
        assert find_cadences(["I"], "C") == []

    def test_position_beats(self):
        chords = ["I", "V", "I"]
        cads = find_cadences(chords)
        positions = [c.position_beats for c in cads]
        assert 1.0 in positions or 2.0 in positions

    def test_all_dominants(self):
        chords = ["V", "vi", "V", "I"]
        cads = find_cadences(chords)
        assert len(cads) >= 2


class TestSuggestCadence:
    def test_high_strength_pac(self):
        a, b = suggest_cadence(["I", "IV"], strength=0.95)
        assert a == "V" and b == "I"

    def test_medium_strength_plagal(self):
        a, b = suggest_cadence(["I"], strength=0.7)
        assert a == "IV" and b == "I"

    def test_low_strength_half(self):
        a, b = suggest_cadence(["I"], strength=0.45)
        assert b == "V"

    def test_very_low_deceptive(self):
        a, b = suggest_cadence(["I"], strength=0.2)
        assert a == "V" and b == "vi"


class TestIsCadential64:
    def test_positive(self):
        assert is_cadential_64("I64", "V") is True

    def test_negative_no_64(self):
        assert is_cadential_64("I", "V") is False

    def test_negative_wrong_degree(self):
        assert is_cadential_64("IV64", "V") is False


class TestCadenceRate:
    def test_all_cadences(self):
        chords = ["V", "I", "V", "I"]
        rate = cadence_rate(chords)
        assert 0.0 < rate <= 1.0

    def test_no_cadences(self):
        chords = ["ii", "iii", "ii", "iii"]
        rate = cadence_rate(chords)
        assert rate == 0.0

    def test_empty(self):
        assert cadence_rate([]) == 0.0

    def test_single(self):
        assert cadence_rate(["I"]) == 0.0


class TestRomanToDegree:
    def test_tonic(self):
        assert ROMAN_TO_DEGREE["I"] == 0
        assert ROMAN_TO_DEGREE["i"] == 0

    def test_dominant(self):
        assert ROMAN_TO_DEGREE["V"] == 7

    def test_flat_vi(self):
        assert ROMAN_TO_DEGREE["bVI"] == 8
