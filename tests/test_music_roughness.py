"""Tests for psychoacoustic roughness and sensory dissonance models."""
import pytest
import numpy as np

from models.music_psychoacoustics.roughness import (
    plomp_levelt_curve, critical_bandwidth, sethares_dissonance,
    hutchinson_knopoff, DissonanceCurve, interval_dissonance_12tet,
    consonance_ranking_12tet, RoughnessAnalyzer,
)


class TestPlompLevelt:
    def test_zero_diff(self):
        assert plomp_levelt_curve(0.0, 440.0) == 0.0

    def test_peak_near_quarter_cbw(self):
        cbw = critical_bandwidth(440.0)
        peak_diff = 0.25 * cbw
        d = plomp_levelt_curve(peak_diff, 440.0)
        assert d > 0.3

    def test_wide_diff_low(self):
        d = plomp_levelt_curve(500.0, 440.0)
        assert d < 0.05

    def test_low_freq_zero(self):
        assert plomp_levelt_curve(10.0, 0.0) == 0.0

    def test_symmetric(self):
        d1 = plomp_levelt_curve(30.0, 440.0)
        d2 = plomp_levelt_curve(-30.0, 440.0)
        assert abs(d1 - d2) < 1e-9


class TestCriticalBandwidth:
    def test_positive(self):
        assert critical_bandwidth(1000.0) > 0

    def test_increases_with_freq(self):
        assert critical_bandwidth(4000.0) > critical_bandwidth(500.0)

    def test_low_freq_floor(self):
        assert critical_bandwidth(0.0) >= 25.0


class TestSetharesDissonance:
    def test_single_partial(self):
        assert sethares_dissonance(np.array([440.0]), np.array([1.0])) == 0.0

    def test_unison_low_dissonance(self):
        d = sethares_dissonance(np.array([440.0, 440.0]), np.array([1.0, 1.0]))
        assert d < 0.01

    def test_semitone_high_dissonance(self):
        d_semi = sethares_dissonance(np.array([440.0, 466.16]), np.array([1.0, 1.0]))
        d_oct = sethares_dissonance(np.array([440.0, 880.0]), np.array([1.0, 1.0]))
        assert d_semi > d_oct

    def test_amplitude_scaling(self):
        d1 = sethares_dissonance(np.array([440.0, 466.0]), np.array([1.0, 1.0]))
        d2 = sethares_dissonance(np.array([440.0, 466.0]), np.array([0.5, 0.5]))
        assert d1 > d2


class TestHutchinsonKnopoff:
    def test_single_partial(self):
        assert hutchinson_knopoff(np.array([440.0]), np.array([1.0])) == 0.0

    def test_positive_for_close_partials(self):
        r = hutchinson_knopoff(np.array([440.0, 466.0]), np.array([1.0, 1.0]))
        assert r > 0

    def test_octave_less_rough(self):
        r_semi = hutchinson_knopoff(np.array([440.0, 466.0]), np.array([1.0, 1.0]))
        r_oct = hutchinson_knopoff(np.array([440.0, 880.0]), np.array([1.0, 1.0]))
        assert r_semi > r_oct


class TestDissonanceCurve:
    @pytest.fixture
    def harmonic_dc(self):
        return DissonanceCurve([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_unison_low(self, harmonic_dc):
        d = harmonic_dc.evaluate(1.0)
        assert d < 0.01

    def test_curve_shape(self, harmonic_dc):
        ratios = np.linspace(1.0, 2.0, 100)
        vals = harmonic_dc.curve(ratios)
        assert len(vals) == 100
        assert vals[0] < vals[10]

    def test_local_minima_include_octave_region(self, harmonic_dc):
        minima = harmonic_dc.local_minima(n_points=2000)
        assert len(minima) > 0
        ratios = [r for r, _ in minima]
        near_fifth = any(abs(r - 1.5) < 0.05 for r in ratios)
        assert near_fifth

    def test_default_amps_decay(self):
        dc = DissonanceCurve([1.0, 2.0, 3.0])
        assert dc.amps[0] > dc.amps[2]


class TestIntervalDissonance:
    def test_unison(self):
        d = interval_dissonance_12tet(0)
        assert d < 0.01

    def test_octave_consonant(self):
        d_oct = interval_dissonance_12tet(12)
        d_semi = interval_dissonance_12tet(1)
        assert d_oct < d_semi

    def test_fifth_relatively_consonant(self):
        d_fifth = interval_dissonance_12tet(7)
        d_tritone = interval_dissonance_12tet(6)
        assert d_fifth < d_tritone

    def test_ranking_top_is_consonant(self):
        ranking = consonance_ranking_12tet()
        assert ranking[0][0] in (0, 12)


class TestRoughnessAnalyzer:
    def test_single_frame(self):
        ra = RoughnessAnalyzer(method="sethares")
        r = ra.frame_roughness(np.array([440.0, 466.0]), np.array([1.0, 1.0]))
        assert r > 0

    def test_trajectory_length(self):
        ra = RoughnessAnalyzer()
        spectra = [
            (np.array([440.0, 466.0]), np.array([1.0, 1.0])),
            (np.array([440.0, 660.0]), np.array([1.0, 0.8])),
        ]
        traj = ra.trajectory(spectra)
        assert len(traj) == 2

    def test_mean_and_range(self):
        ra = RoughnessAnalyzer()
        spectra = [
            (np.array([440.0, 466.0]), np.array([1.0, 1.0])),
            (np.array([440.0, 880.0]), np.array([1.0, 1.0])),
        ]
        m = ra.mean_roughness(spectra)
        assert m > 0
        r = ra.roughness_range(spectra)
        assert r > 0

    def test_hutchinson_method(self):
        ra = RoughnessAnalyzer(method="hutchinson")
        r = ra.frame_roughness(np.array([440.0, 466.0]), np.array([1.0, 1.0]))
        assert r > 0

    def test_empty_trajectory(self):
        ra = RoughnessAnalyzer()
        assert ra.mean_roughness([]) == 0.0
        assert ra.roughness_range([]) == 0.0
