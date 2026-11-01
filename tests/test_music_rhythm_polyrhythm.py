"""Tests for music_rhythm.polyrhythm — polyrhythmic patterns and Bjorklund."""
import pytest
import numpy as np
from models.music_rhythm.polyrhythm import (
    PolyrhythmPattern, compute_lcm_cycle, generate_polyrhythm,
    euclidean_rhythm_bjorklund, cross_rhythm_density,
    polymetric_offset, rhythmic_oddity, maximal_evenness,
    polyrhythm_complexity, isochronous_deviation,
    resultant_rhythm, aksak_pattern, tresillo,
    son_clave_3_2, rumba_clave_3_2,
)


class TestComputeLcmCycle:
    def test_coprime(self):
        assert compute_lcm_cycle(3, 4) == 12

    def test_same(self):
        assert compute_lcm_cycle(5, 5) == 5

    def test_one_divides_other(self):
        assert compute_lcm_cycle(3, 6) == 6

    def test_large_coprime(self):
        assert compute_lcm_cycle(7, 11) == 77


class TestGeneratePolyrhythm:
    def test_3_over_4(self):
        pr = generate_polyrhythm(3, 4, 12)
        assert len(pr.voices) == 2
        assert pr.cycle_length == 12
        assert sum(pr.voices[0]) == 3
        assert sum(pr.voices[1]) == 4

    def test_onset_positions_in_range(self):
        pr = generate_polyrhythm(5, 7, 35)
        assert len(pr.voices[0]) == 35
        assert len(pr.voices[1]) == 35

    def test_2_over_3(self):
        pr = generate_polyrhythm(2, 3, 6)
        assert sum(pr.voices[0]) == 2
        assert sum(pr.voices[1]) == 3
        assert pr.voices[0][0] == 1
        assert pr.voices[1][0] == 1

    def test_equal_streams(self):
        pr = generate_polyrhythm(4, 4, 16)
        assert pr.voices[0] == pr.voices[1]


class TestEuclideanRhythmBjorklund:
    def test_tresillo_3_8(self):
        pattern = euclidean_rhythm_bjorklund(3, 8)
        assert len(pattern) == 8
        assert sum(pattern) == 3

    def test_4_of_12(self):
        pattern = euclidean_rhythm_bjorklund(4, 12)
        assert sum(pattern) == 4
        assert len(pattern) == 12

    def test_all_pulses(self):
        pattern = euclidean_rhythm_bjorklund(5, 5)
        assert pattern == [1, 1, 1, 1, 1]

    def test_single_pulse(self):
        pattern = euclidean_rhythm_bjorklund(1, 8)
        assert sum(pattern) == 1
        assert pattern[0] == 1

    def test_zero_pulses(self):
        pattern = euclidean_rhythm_bjorklund(0, 8)
        assert sum(pattern) == 0

    def test_5_of_8(self):
        pattern = euclidean_rhythm_bjorklund(5, 8)
        assert sum(pattern) == 5
        assert len(pattern) == 8

    def test_maximal_evenness_property(self):
        pattern = euclidean_rhythm_bjorklund(3, 8)
        onsets = [i for i, v in enumerate(pattern) if v == 1]
        gaps = [onsets[(i+1) % len(onsets)] - onsets[i] for i in range(len(onsets))]
        gaps = [g if g > 0 else g + 8 for g in gaps]
        assert max(gaps) - min(gaps) <= 1


class TestCrossRhythmDensity:
    def test_3_4(self):
        pr = generate_polyrhythm(3, 4, 12)
        d = cross_rhythm_density(pr)
        assert 0.0 < d <= 1.0

    def test_unison(self):
        pr = generate_polyrhythm(4, 4, 16)
        d = cross_rhythm_density(pr)
        assert d == pytest.approx(4 / 16)


class TestPolymetricOffset:
    def test_basic(self):
        offset = polymetric_offset(3, 4, 12)
        assert isinstance(offset, float)
        assert 0.0 <= offset <= 1.0

    def test_zero_at_downbeat(self):
        offset = polymetric_offset(3, 4, 0)
        assert offset == pytest.approx(0.0)

    def test_nonzero_off_downbeat(self):
        offset = polymetric_offset(3, 4, 1)
        assert offset > 0


class TestRhythmicOddity:
    def test_tresillo(self):
        pattern = euclidean_rhythm_bjorklund(3, 8)
        odd = rhythmic_oddity(pattern)
        assert isinstance(odd, bool)

    def test_uniform_pattern(self):
        odd = rhythmic_oddity([1, 0, 1, 0, 1, 0])
        assert isinstance(odd, bool)

    def test_single_onset(self):
        assert rhythmic_oddity([1, 0, 0, 0]) is False


class TestMaximalEvenness:
    def test_4_of_12(self):
        pattern = maximal_evenness(4, 12)
        assert len(pattern) == 12
        assert sum(pattern) == 4

    def test_matches_bjorklund(self):
        me = maximal_evenness(3, 8)
        bj = euclidean_rhythm_bjorklund(3, 8)
        assert me == bj

    def test_all_pulses(self):
        assert maximal_evenness(5, 5) == [1, 1, 1, 1, 1]


class TestPolyrhythmComplexity:
    def test_simple(self):
        c = polyrhythm_complexity(2, 4)
        assert c >= 0

    def test_complex_higher(self):
        c1 = polyrhythm_complexity(3, 4)
        c2 = polyrhythm_complexity(7, 11)
        assert c2 > c1


class TestIsochronousDeviation:
    def test_perfect_isochronous(self):
        onsets = np.array([0.0, 1.0, 2.0, 3.0])
        dev = isochronous_deviation(onsets, period=1.0)
        np.testing.assert_array_almost_equal(dev, 0.0)

    def test_non_isochronous(self):
        onsets = np.array([0.0, 0.5, 2.0, 3.0])
        dev = isochronous_deviation(onsets, period=1.0)
        assert np.any(np.abs(dev) > 0)

    def test_empty(self):
        dev = isochronous_deviation(np.array([]), period=1.0)
        assert len(dev) == 0


class TestResultantRhythm:
    def test_3_4(self):
        r = resultant_rhythm(3, 4)
        assert isinstance(r, list)
        assert all(v in (0, 1) for v in r)
        assert len(r) == 12
        assert r[0] == 1

    def test_2_3(self):
        r = resultant_rhythm(2, 3)
        assert len(r) == 6


class TestAksak:
    def test_basic(self):
        pat = aksak_pattern([2, 2, 3])
        assert len(pat) == 7
        assert sum(pat) == 3

    def test_single_group(self):
        pat = aksak_pattern([5])
        assert len(pat) == 5
        assert sum(pat) == 1


class TestClave:
    def test_tresillo(self):
        t = tresillo()
        assert len(t) == 8
        assert sum(t) == 3

    def test_son_clave(self):
        c = son_clave_3_2()
        assert len(c) == 16
        assert sum(c) == 5

    def test_rumba_clave(self):
        c = rumba_clave_3_2()
        assert len(c) == 16
        assert sum(c) == 5

    def test_son_vs_rumba_differ(self):
        assert son_clave_3_2() != rumba_clave_3_2()
