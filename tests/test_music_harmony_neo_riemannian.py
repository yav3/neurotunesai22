"""Tests for neo-Riemannian transformations."""
import pytest

from models.music_harmony.neo_riemannian import (
    Triad, TriadQuality,
    parallel, leading_tone, relative,
    apply_sequence, hexatonic_cycle, octatonic_cycle,
    tonnetz_coordinates, triad_tonnetz_centroid, tonnetz_distance,
    voice_leading_distance, plr_distance,
    analyze_harmonic_path,
)


def _major(root: int) -> Triad:
    return Triad(root, TriadQuality.MAJOR)

def _minor(root: int) -> Triad:
    return Triad(root, TriadQuality.MINOR)


class TestPLR:
    def test_parallel_toggles(self):
        c = _major(0)
        cm = parallel(c)
        assert cm.quality == TriadQuality.MINOR
        assert cm.root == 0
        assert parallel(cm).quality == TriadQuality.MAJOR

    def test_parallel_involution(self):
        c = _major(0)
        assert parallel(parallel(c)).root == c.root
        assert parallel(parallel(c)).quality == c.quality

    def test_leading_tone_c_major(self):
        c = _major(0)
        result = leading_tone(c)
        assert result.quality == TriadQuality.MINOR
        assert result.root == 4

    def test_leading_tone_minor(self):
        am = _minor(9)
        result = leading_tone(am)
        assert result.quality == TriadQuality.MAJOR

    def test_relative_c_major(self):
        c = _major(0)
        am = relative(c)
        assert am.quality == TriadQuality.MINOR
        assert am.root == 9

    def test_relative_involution(self):
        c = _major(0)
        assert relative(relative(c)).root == c.root
        assert relative(relative(c)).quality == c.quality


class TestApplySequence:
    def test_single_op(self):
        path = apply_sequence(_major(0), "P")
        assert len(path) == 2
        assert path[1].quality == TriadQuality.MINOR

    def test_pr(self):
        path = apply_sequence(_major(0), "PR")
        assert len(path) == 3

    def test_invalid_op(self):
        with pytest.raises(ValueError):
            apply_sequence(_major(0), "X")


class TestCycles:
    def test_hexatonic_cycle_length(self):
        cycle = hexatonic_cycle(_major(0))
        assert len(cycle) == 9
        assert cycle[0] == cycle[-1]

    def test_octatonic_returns_to_start(self):
        cycle = octatonic_cycle(_major(0))
        assert len(cycle) == 9
        assert cycle[0].root == cycle[-1].root
        assert cycle[0].quality == cycle[-1].quality

    def test_hexatonic_eight_distinct(self):
        cycle = hexatonic_cycle(_major(0))
        pcs = {(t.root % 12, t.quality) for t in cycle[:-1]}
        assert len(pcs) == 8


class TestTonnetz:
    def test_c_coordinates(self):
        x, y = tonnetz_coordinates(0)
        assert x == 0.0 and y == 0.0

    def test_g_one_fifth(self):
        x, y = tonnetz_coordinates(7)
        assert x == 1.0

    def test_distance_same(self):
        c = _major(0)
        assert tonnetz_distance(c, c) == pytest.approx(0.0)

    def test_distance_positive(self):
        d = tonnetz_distance(_major(0), _major(7))
        assert d > 0.0


class TestVoiceLeading:
    def test_same_chord_zero(self):
        assert voice_leading_distance(_major(0), _major(0)) == 0

    def test_parallel_one_semitone(self):
        d = voice_leading_distance(_major(0), _minor(0))
        assert d == 1

    def test_relative(self):
        d = voice_leading_distance(_major(0), _minor(9))
        assert d <= 4


class TestPLRDistance:
    def test_identity(self):
        assert plr_distance(_major(0), _major(0)) == 0

    def test_parallel_one(self):
        assert plr_distance(_major(0), _minor(0)) == 1

    def test_relative_one(self):
        assert plr_distance(_major(0), _minor(9)) == 1

    def test_unreachable(self):
        d = plr_distance(_major(0), _major(1), max_depth=1)
        assert d == -1 or d > 1


class TestHarmonicPath:
    def test_single_chord(self):
        s = analyze_harmonic_path([_major(0)])
        assert s.n_chords == 1
        assert s.total_vl_distance == 0

    def test_two_chords(self):
        s = analyze_harmonic_path([_major(0), _minor(0)])
        assert s.n_chords == 2
        assert s.total_vl_distance == 1
        assert s.quality_changes == 1

    def test_pl_path(self):
        path = apply_sequence(_major(0), "PLPL")
        s = analyze_harmonic_path(path)
        assert s.n_chords == 5
        assert s.quality_changes == 4
        assert s.mean_vl_distance > 0
