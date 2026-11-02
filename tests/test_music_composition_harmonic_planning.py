"""Tests for tension-guided harmonic planning."""
import numpy as np
import pytest

from models.music_composition.harmonic_planning import (
    ChordSymbol, TensionScore, HarmonicPlan,
    surface_dissonance, harmonic_distance, tonal_distance,
    compute_tension, arch_tension, ramp_tension, wave_tension,
    plan_progression, valid_next_chords, degree_to_chord,
    FUNCTIONAL_GRAMMAR,
)


class TestChordSymbol:
    def test_from_name_major(self):
        c = ChordSymbol.from_name("C", "maj")
        assert c.root == 0
        assert c.pcs == (0, 4, 7)

    def test_from_name_minor(self):
        am = ChordSymbol.from_name("A", "min")
        assert am.root == 9
        assert am.pcs == (9, 0, 4)

    def test_dom7(self):
        g7 = ChordSymbol.from_name("G", "dom7")
        assert len(g7.pcs) == 4
        assert 5 in g7.pcs  # F (m7 above G)


class TestSurfaceDissonance:
    def test_major_triad_low(self):
        c = ChordSymbol.from_name("C", "maj")
        d = surface_dissonance(c)
        assert d < 0.5

    def test_dim_higher(self):
        dim = ChordSymbol.from_name("B", "dim")
        maj = ChordSymbol.from_name("C", "maj")
        assert surface_dissonance(dim) > surface_dissonance(maj)


class TestHarmonicDistance:
    def test_same_chord_zero(self):
        c = ChordSymbol.from_name("C", "maj")
        assert harmonic_distance(c, c) == pytest.approx(0.0)

    def test_fifth_motion_small(self):
        c = ChordSymbol.from_name("C", "maj")
        g = ChordSymbol.from_name("G", "maj")
        d = harmonic_distance(c, g)
        assert 0 < d < 2.0

    def test_tritone_large(self):
        c = ChordSymbol.from_name("C", "maj")
        fs = ChordSymbol.from_name("F#", "maj")
        d = harmonic_distance(c, fs)
        assert d > 2.0


class TestTonalDistance:
    def test_tonic_zero(self):
        c = ChordSymbol.from_name("C", "maj")
        assert tonal_distance(c, 0) == pytest.approx(0.0)

    def test_tritone_max(self):
        c = ChordSymbol.from_name("C", "maj")
        fs = ChordSymbol.from_name("F#", "maj")
        assert tonal_distance(fs, 0) == 6.0


class TestComputeTension:
    def test_returns_tension_score(self):
        c = ChordSymbol.from_name("C", "maj")
        g = ChordSymbol.from_name("G", "maj")
        ts = compute_tension(g, c, key_root=0)
        assert isinstance(ts, TensionScore)
        assert ts.total >= 0

    def test_no_prev(self):
        c = ChordSymbol.from_name("C", "maj")
        ts = compute_tension(c, None, key_root=0)
        assert ts.harmonic_distance == 0.0


class TestTensionCurves:
    def test_arch_shape(self):
        curve = arch_tension(8)
        assert len(curve) == 8
        assert curve[0] < curve[4]
        assert curve[-1] < curve[4]

    def test_ramp(self):
        curve = ramp_tension(5, 0.0, 1.0)
        assert curve[0] == pytest.approx(0.0)
        assert curve[-1] == pytest.approx(1.0)

    def test_wave(self):
        curve = wave_tension(10)
        assert len(curve) == 10
        assert curve.min() >= 0.0


class TestPlanProgression:
    def test_returns_plan(self):
        target = arch_tension(4)
        plan = plan_progression(target, key_root=0)
        assert isinstance(plan, HarmonicPlan)
        assert len(plan.chords) == 4
        assert len(plan.tensions) == 4

    def test_fit_error_finite(self):
        target = ramp_tension(6)
        plan = plan_progression(target)
        assert np.isfinite(plan.fit_error)

    def test_achieved_matches_tensions(self):
        target = arch_tension(5)
        plan = plan_progression(target)
        for i, ts in enumerate(plan.tensions):
            assert plan.achieved_curve[i] == pytest.approx(ts.total)


class TestFunctionalGrammar:
    def test_tonic_has_options(self):
        nexts = valid_next_chords("I")
        assert len(nexts) > 0
        assert "V" in nexts

    def test_dominant_resolves(self):
        nexts = valid_next_chords("V")
        assert "I" in nexts

    def test_unknown_degree(self):
        nexts = valid_next_chords("bVII")
        assert "I" in nexts  # fallback

    def test_degree_to_chord(self):
        c = degree_to_chord("I", key_root=0)
        assert c.root == 0
        assert c.quality == "maj"

    def test_degree_to_chord_transposed(self):
        c = degree_to_chord("V", key_root=2)  # D major: V = A
        assert c.root == 9  # A
