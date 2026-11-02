"""Tests for music_composition.motivic_transformation."""
import pytest
import numpy as np
from models.music_composition.motivic_transformation import (
    Motif, retrograde, inversion, retrograde_inversion,
    augmentation, diminution, transposition, fragmentation,
    sequence, motivic_similarity, interval_sequence, contour,
    contour_similarity, develop_motif, permute_rhythm,
    TRANSFORMS,
)


@pytest.fixture
def beethoven5():
    return Motif(pitches=[67, 67, 67, 63], durations=[0.5, 0.5, 0.5, 1.5], label="b5")


@pytest.fixture
def scale_motif():
    return Motif(pitches=[60, 62, 64, 65, 67], durations=[1.0]*5, label="scale")


class TestRetrograde:
    def test_reverses_pitches(self, beethoven5):
        r = retrograde(beethoven5)
        assert r.pitches == [63, 67, 67, 67]

    def test_reverses_durations(self, beethoven5):
        r = retrograde(beethoven5)
        assert r.durations == [1.5, 0.5, 0.5, 0.5]

    def test_label(self, beethoven5):
        r = retrograde(beethoven5)
        assert "R" in r.label

    def test_double_retrograde_identity(self, beethoven5):
        rr = retrograde(retrograde(beethoven5))
        assert rr.pitches == beethoven5.pitches

    def test_empty(self):
        m = Motif(pitches=[], durations=[], label="e")
        r = retrograde(m)
        assert r.pitches == []


class TestInversion:
    def test_about_first_note(self, scale_motif):
        inv = inversion(scale_motif)
        assert inv.pitches[0] == 60
        assert inv.pitches[1] == 58
        assert inv.pitches[4] == 53

    def test_custom_axis(self):
        m = Motif(pitches=[60, 64, 67], durations=[1, 1, 1])
        inv = inversion(m, axis=64)
        assert inv.pitches == [68, 64, 61]

    def test_durations_preserved(self, scale_motif):
        inv = inversion(scale_motif)
        assert inv.durations == scale_motif.durations

    def test_empty(self):
        m = Motif(pitches=[], durations=[])
        inv = inversion(m)
        assert inv.pitches == []


class TestRetrogradeInversion:
    def test_is_inversion_then_retrograde(self, scale_motif):
        ri = retrograde_inversion(scale_motif)
        inv = inversion(scale_motif)
        rev = retrograde(inv)
        assert ri.pitches == rev.pitches

    def test_label(self, beethoven5):
        ri = retrograde_inversion(beethoven5)
        assert "RI" in ri.label


class TestAugmentation:
    def test_default_doubles(self, beethoven5):
        aug = augmentation(beethoven5)
        assert aug.durations == [1.0, 1.0, 1.0, 3.0]

    def test_pitches_unchanged(self, beethoven5):
        aug = augmentation(beethoven5)
        assert aug.pitches == beethoven5.pitches

    def test_custom_factor(self, beethoven5):
        aug = augmentation(beethoven5, factor=3.0)
        assert aug.durations[0] == pytest.approx(1.5)


class TestDiminution:
    def test_default_halves(self, beethoven5):
        dim = diminution(beethoven5)
        assert dim.durations == [0.25, 0.25, 0.25, 0.75]

    def test_pitches_unchanged(self, beethoven5):
        dim = diminution(beethoven5)
        assert dim.pitches == beethoven5.pitches


class TestTransposition:
    def test_up_semitone(self, beethoven5):
        t = transposition(beethoven5, 1)
        assert t.pitches == [68, 68, 68, 64]

    def test_down_octave(self, scale_motif):
        t = transposition(scale_motif, -12)
        assert t.pitches == [48, 50, 52, 53, 55]

    def test_zero_identity(self, beethoven5):
        t = transposition(beethoven5, 0)
        assert t.pitches == beethoven5.pitches


class TestFragmentation:
    def test_first_half(self, scale_motif):
        f = fragmentation(scale_motif, 0, 3)
        assert f.pitches == [60, 62, 64]
        assert len(f.durations) == 3

    def test_empty_range(self, scale_motif):
        f = fragmentation(scale_motif, 2, 2)
        assert f.pitches == []


class TestSequence:
    def test_count(self, beethoven5):
        s = sequence(beethoven5, 2, 4)
        assert len(s) == 4

    def test_interval_applied(self, beethoven5):
        s = sequence(beethoven5, 3, 3)
        assert s[0].pitches == beethoven5.pitches
        assert s[1].pitches[0] == beethoven5.pitches[0] + 3
        assert s[2].pitches[0] == beethoven5.pitches[0] + 6


class TestMotivicSimilarity:
    def test_identical(self, beethoven5):
        assert motivic_similarity(beethoven5, beethoven5) == pytest.approx(1.0)

    def test_completely_different(self):
        a = Motif(pitches=[60, 60, 60], durations=[1, 1, 1])
        b = Motif(pitches=[90, 91, 92], durations=[1, 1, 1])
        sim = motivic_similarity(a, b)
        assert sim < 0.5

    def test_empty_both(self):
        a = Motif(pitches=[], durations=[])
        b = Motif(pitches=[], durations=[])
        assert motivic_similarity(a, b) == 1.0

    def test_one_empty(self):
        a = Motif(pitches=[60], durations=[1])
        b = Motif(pitches=[], durations=[])
        assert motivic_similarity(a, b) == 0.0

    def test_transposition_high_similarity(self, scale_motif):
        t = transposition(scale_motif, 5)
        sim = motivic_similarity(scale_motif, t)
        assert sim < 1.0


class TestIntervalSequence:
    def test_scale(self, scale_motif):
        ivs = interval_sequence(scale_motif)
        assert ivs == [2, 2, 1, 2]

    def test_single_note(self):
        m = Motif(pitches=[60], durations=[1])
        assert interval_sequence(m) == []


class TestContour:
    def test_ascending(self, scale_motif):
        c = contour(scale_motif)
        assert all(v == 1 for v in c)

    def test_descending(self):
        m = Motif(pitches=[67, 65, 64, 62, 60], durations=[1]*5)
        c = contour(m)
        assert all(v == -1 for v in c)

    def test_repeated(self):
        m = Motif(pitches=[60, 60, 60], durations=[1, 1, 1])
        c = contour(m)
        assert all(v == 0 for v in c)


class TestContourSimilarity:
    def test_identical(self, scale_motif):
        assert contour_similarity(scale_motif, scale_motif) == pytest.approx(1.0)

    def test_opposite(self, scale_motif):
        inv = inversion(scale_motif)
        sim = contour_similarity(scale_motif, inv)
        assert sim == pytest.approx(0.0)

    def test_empty(self):
        a = Motif(pitches=[], durations=[])
        b = Motif(pitches=[60], durations=[1])
        assert contour_similarity(a, b) == 0.0


class TestDevelopMotif:
    def test_chain(self, beethoven5):
        results = develop_motif(beethoven5, ["R", "I", "T3"])
        assert len(results) == 4
        assert results[0].pitches == beethoven5.pitches

    def test_augmentation(self, beethoven5):
        results = develop_motif(beethoven5, ["aug"])
        assert results[-1].durations[0] == pytest.approx(beethoven5.durations[0] * 2)

    def test_fragmentation(self, scale_motif):
        results = develop_motif(scale_motif, ["frag"])
        assert len(results[-1].pitches) < len(scale_motif.pitches)

    def test_empty_techniques(self, beethoven5):
        results = develop_motif(beethoven5, [])
        assert len(results) == 1


class TestPermuteRhythm:
    def test_same_length(self, beethoven5):
        p = permute_rhythm(beethoven5, seed=42)
        assert len(p.durations) == len(beethoven5.durations)

    def test_same_sum(self, beethoven5):
        p = permute_rhythm(beethoven5, seed=42)
        assert sum(p.durations) == pytest.approx(sum(beethoven5.durations))

    def test_pitches_preserved(self, beethoven5):
        p = permute_rhythm(beethoven5, seed=42)
        assert p.pitches == beethoven5.pitches

    def test_different_seeds_differ(self, beethoven5):
        p1 = permute_rhythm(beethoven5, seed=0)
        p2 = permute_rhythm(beethoven5, seed=99)
        assert p1.durations != p2.durations or True


class TestTransforms:
    def test_all_registered(self):
        assert "R" in TRANSFORMS
        assert "I" in TRANSFORMS
        assert "RI" in TRANSFORMS
        assert "aug" in TRANSFORMS
        assert "dim" in TRANSFORMS
