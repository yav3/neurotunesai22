"""Tests for lyric generation module."""
import numpy as np
import pytest
from models.music_lyrics.lyric_generation import (
    count_syllables_approx, line_syllable_count, stress_from_syllable_count,
    enforce_syllable_count, end_phoneme, words_rhyme, MarkovLyricModel,
    generate_verse, COMMON_TEMPLATES, COMMON_STRUCTURES,
    song_section_count, total_lines, lyric_density, repetition_ratio,
    rhyme_distance, PhonemeCluster, SyllableTemplate, RHYME_SCHEMES,
)


class TestSyllableCount:
    def test_monosyllable(self):
        assert count_syllables_approx("cat") == 1

    def test_polysyllable(self):
        assert count_syllables_approx("beautiful") >= 3

    def test_silent_e(self):
        assert count_syllables_approx("lake") == 1

    def test_empty(self):
        assert count_syllables_approx("") == 0

    def test_line(self):
        assert line_syllable_count("the cat sat on the mat") >= 5


class TestStress:
    def test_iambic(self):
        pat = stress_from_syllable_count(10, "iambic")
        assert len(pat) == 10
        assert pat[0] == 0 and pat[1] == 1

    def test_trochaic(self):
        pat = stress_from_syllable_count(8, "trochaic")
        assert pat[0] == 1 and pat[1] == 0

    def test_dactylic(self):
        pat = stress_from_syllable_count(9, "dactylic")
        assert pat[0] == 1 and pat[3] == 1


class TestRhyme:
    def test_end_phoneme(self):
        assert end_phoneme("cat") == end_phoneme("hat")

    def test_words_rhyme(self):
        assert words_rhyme("cat", "hat")

    def test_not_rhyme(self):
        assert not words_rhyme("cat", "dog")


class TestPhonemeCluster:
    def test_same_cluster(self):
        a = PhonemeCluster("iː", "t")
        b = PhonemeCluster("iː", "t")
        assert rhyme_distance(a, b) == 0.0

    def test_different(self):
        a = PhonemeCluster("iː", "t")
        b = PhonemeCluster("ɒ", "k")
        assert rhyme_distance(a, b) > 0.5


class TestMarkovModel:
    def test_train_generate(self):
        corpus = [
            "the sun is shining bright today",
            "the moon is rising high tonight",
            "the stars are shining in the sky",
        ]
        model = MarkovLyricModel(order=1)
        model.train(corpus)
        assert len(model.vocabulary) > 5
        words = model.generate_words(10)
        assert len(words) == 10

    def test_generate_line(self):
        corpus = ["love is all you need"] * 10
        model = MarkovLyricModel(order=1)
        model.train(corpus)
        line = model.generate_line(8)
        assert line.syllables <= 10


class TestVerseGeneration:
    def test_haiku(self):
        corpus = ["cherry blossoms fall soft on the ground below"] * 20
        model = MarkovLyricModel(order=1)
        model.train(corpus)
        verse = generate_verse(model, COMMON_TEMPLATES["haiku"])
        assert len(verse.lines) == 3

    def test_with_rhyme(self):
        corpus = [f"word{i} is good" for i in range(50)]
        model = MarkovLyricModel(order=1)
        model.train(corpus)
        verse = generate_verse(model, COMMON_TEMPLATES["ballad"], "ABAB")
        assert len(verse.lines) == 4


class TestSongStructure:
    def test_section_count(self):
        counts = song_section_count("verse_chorus")
        assert counts["verse"] == 2
        assert counts["chorus"] == 2

    def test_total_lines(self):
        n = total_lines("verse_chorus")
        assert n == 16

    def test_pop(self):
        counts = song_section_count("pop")
        assert "bridge" in counts

    def test_unknown(self):
        assert total_lines("nonexistent") == 0


class TestLyricMetrics:
    def test_density(self):
        d = lyric_density("the the the cat cat")
        assert 0 < d < 1

    def test_density_unique(self):
        d = lyric_density("one two three four five")
        assert d == 1.0

    def test_repetition_none(self):
        assert repetition_ratio(["a", "b", "c"]) == 0.0

    def test_repetition_some(self):
        r = repetition_ratio(["hello", "hello", "world"])
        assert r > 0.0

    def test_empty(self):
        assert lyric_density("") == 0.0
        assert repetition_ratio([]) == 0.0
