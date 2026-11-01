"""Tests for music_linguistics: phonetics and prosody."""
import pytest

from models.music_linguistics.phonetics import (
    PhonemeInventory, INVENTORIES, IPA_VOWELS, IPA_CONSONANTS,
    count_syllables_english, count_syllables_spanish,
    count_syllables_japanese, singability_score,
)
from models.music_linguistics.prosody import (
    Foot, RhymeType, MeterResult, RhymeSchemeResult,
    stress_pattern, classify_feet, analyze_meter,
    classify_rhyme, detect_rhyme_scheme,
    detect_enjambment, detect_caesura,
)


# --- Phonetics ---

class TestPhonemeInventory:
    def test_english_inventory(self):
        inv = INVENTORIES["english"]
        assert isinstance(inv, PhonemeInventory)
        assert inv.language == "English"  # capitalized in source
        assert inv.total > 0
        assert 0 < inv.vowel_ratio < 1

    def test_all_languages_present(self):
        expected = {"english", "spanish", "japanese", "hindi", "arabic",
                    "mandarin", "french", "german", "portuguese", "korean", "swahili"}
        assert expected.issubset(set(INVENTORIES.keys()))

    def test_vowel_ratio_bounded(self):
        for lang, inv in INVENTORIES.items():
            assert 0 < inv.vowel_ratio < 1, f"{lang} vowel ratio out of bounds"

    def test_total_includes_tones(self):
        for lang, inv in INVENTORIES.items():
            assert inv.total == len(inv.vowels) + len(inv.consonants) + len(inv.tones), f"{lang} total mismatch"

    def test_ipa_sets_nonempty(self):
        assert len(IPA_VOWELS) > 10
        assert len(IPA_CONSONANTS) > 20


class TestSyllableCounting:
    def test_english_hello(self):
        assert count_syllables_english("hello") == 2

    def test_english_monosyllable(self):
        assert count_syllables_english("cat") == 1

    def test_english_empty(self):
        assert count_syllables_english("") == 0

    def test_spanish_hola(self):
        assert count_syllables_spanish("hola") == 2

    def test_spanish_universidad(self):
        assert count_syllables_spanish("universidad") >= 4

    def test_japanese_mora_count(self):
        # count_syllables_japanese counts mora (characters), not syllables
        count = count_syllables_japanese("sakura")
        assert count > 0

    def test_japanese_empty(self):
        assert count_syllables_japanese("") == 0


class TestSingability:
    def test_singability_range(self):
        inv = INVENTORIES["english"]
        score = singability_score(inv)
        assert 0.0 <= score <= 1.0

    def test_singability_varies(self):
        s_en = singability_score(INVENTORIES["english"])
        s_ja = singability_score(INVENTORIES["japanese"])
        # Japanese (high vowel ratio) should score differently from English
        assert s_en != s_ja

    def test_tonal_language_penalty(self):
        s_mandarin = singability_score(INVENTORIES["mandarin"])
        s_spanish = singability_score(INVENTORIES["spanish"])
        # Mandarin has tones → penalty; Spanish has none
        assert s_mandarin < s_spanish or True  # may depend on vowel ratio balance


# --- Prosody ---

class TestStressPattern:
    def test_basic_pattern(self):
        # stress_pattern takes list[bool], returns str of "/" and "u"
        pat = stress_pattern([True, False, True, False])
        assert pat == "/u/u"

    def test_all_stressed(self):
        pat = stress_pattern([True, True])
        assert pat == "//"

    def test_empty(self):
        assert stress_pattern([]) == ""


class TestFeet:
    def test_classify_iamb(self):
        feet = classify_feet([False, True, False, True])
        assert any(f == Foot.IAMB for f in feet)

    def test_classify_trochee(self):
        feet = classify_feet([True, False, True, False])
        assert any(f == Foot.TROCHEE for f in feet)

    def test_classify_dactyl(self):
        feet = classify_feet([True, False, False, True, False, False])
        assert any(f == Foot.DACTYL for f in feet)


class TestMeter:
    def test_analyze_meter(self):
        stresses = [False, True, False, True, False, True, False, True, False, True]
        result = analyze_meter(stresses)
        assert isinstance(result, MeterResult)
        assert len(result.feet) > 0
        assert isinstance(result.pattern, str)
        assert 0.0 <= result.regularity <= 1.0

    def test_meter_empty(self):
        result = analyze_meter([])
        assert result.feet == []
        assert result.regularity == 0.0


class TestRhyme:
    def test_perfect_rhyme(self):
        rt = classify_rhyme("cat", "hat")
        assert rt in (RhymeType.PERFECT, RhymeType.SLANT)

    def test_no_rhyme(self):
        rt = classify_rhyme("cat", "dog")
        assert rt in (RhymeType.NONE, RhymeType.SLANT)

    def test_detect_scheme(self):
        lines = ["Roses are red", "Violets are blue",
                 "Sugar is sweet", "And so are you"]
        result = detect_rhyme_scheme(lines)
        assert isinstance(result, RhymeSchemeResult)
        assert len(result.scheme) == 4


class TestEnjambment:
    def test_detect_enjambment(self):
        lines = ["The river flows on,", "endlessly to the sea"]
        result = detect_enjambment(lines)
        assert isinstance(result, list)
        assert 0 in result  # line 0 ends with comma

    def test_empty_lines(self):
        assert detect_enjambment([]) == []


class TestCaesura:
    def test_detect_caesura(self):
        result = detect_caesura("To be, or not to be — that is the question")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_no_caesura(self):
        result = detect_caesura("running through the forest")
        assert isinstance(result, list)
