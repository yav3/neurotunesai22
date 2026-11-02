"""Tests for phonetic alignment."""
import numpy as np
import pytest
from models.music_nlp.phonetic_alignment import (
    count_syllables, syllabify, Syllable, NoteSlot,
    align_syllables_to_notes, stress_alignment_score,
    vowel_on_high_note, lyric_density, phrasing_breath_points,
)


class TestSyllableCount:
    def test_monosyllable(self):
        assert count_syllables("cat") == 1

    def test_bisyllable(self):
        assert count_syllables("hello") == 2

    def test_trisyllable(self):
        assert count_syllables("banana") == 3

    def test_silent_e(self):
        assert count_syllables("make") == 1

    def test_empty(self):
        assert count_syllables("") == 0


class TestSyllabify:
    def test_simple(self):
        syls = syllabify("cat")
        assert len(syls) == 1
        assert syls[0].text == "cat"

    def test_multi(self):
        syls = syllabify("hello")
        assert len(syls) == 2

    def test_singability(self):
        syl = Syllable(text="ah", onset="", nucleus="a", coda="")
        assert syl.singability > 0.5


class TestAlignment:
    def test_one_to_one(self):
        syls = [Syllable("hel"), Syllable("lo")]
        notes = [NoteSlot(60, 0.0, 0.5), NoteSlot(62, 0.5, 0.5)]
        result = align_syllables_to_notes(syls, notes)
        assert len(result.pairs) == 2
        assert result.alignment_score == 1.0

    def test_unmatched_syllables(self):
        syls = [Syllable("a"), Syllable("b"), Syllable("c")]
        notes = [NoteSlot(60, 0.0, 0.5)]
        result = align_syllables_to_notes(syls, notes)
        assert result.unmatched_syllables == 2

    def test_rests_skipped(self):
        syls = [Syllable("yes")]
        notes = [NoteSlot(0, 0.0, 0.5, is_rest=True), NoteSlot(60, 0.5, 0.5)]
        result = align_syllables_to_notes(syls, notes)
        assert len(result.pairs) == 1


class TestMetrics:
    def test_lyric_density(self):
        syls = [Syllable("a")] * 10
        assert lyric_density(syls, 5.0) == 2.0

    def test_breath_points(self):
        notes = [NoteSlot(60, 0.0, 0.4), NoteSlot(62, 1.0, 0.4)]
        pts = phrasing_breath_points(notes, min_gap=0.3)
        assert len(pts) == 1


class TestEqualLoudness:
    def test_contour_shape(self):
        from models.music_psychoacoustics.equal_loudness import equal_loudness_contour
        freqs, spl = equal_loudness_contour(40)
        assert len(freqs) == 29
        assert np.all(np.isfinite(spl))

    def test_phon_sone_roundtrip(self):
        from models.music_psychoacoustics.equal_loudness import phon_to_sone, sone_to_phon
        for phon in [20, 40, 60, 80]:
            sone = phon_to_sone(phon)
            back = sone_to_phon(sone)
            assert abs(back - phon) < 2.0

    def test_a_weighting(self):
        from models.music_psychoacoustics.equal_loudness import a_weighting
        freqs = np.array([1000.0, 100.0, 10000.0])
        w = a_weighting(freqs)
        assert w[0] > w[1]  # 1kHz penalized less than 100Hz
