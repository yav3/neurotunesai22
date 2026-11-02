"""Tests for expansion round 14: microtonal systems, sidechain compression, stereo imaging, ornaments."""
from __future__ import annotations

import math
import pytest
import numpy as np

from models.music_ethnomusicology.microtonal import (
    MicroPitch, edo_scale, just_interval, JUST_INTERVALS,
    syntonic_comma, pythagorean_comma, quarter_tone_scale,
    LatticeNode, ji_lattice, lattice_distance, cents_distance,
    closest_edo_pitch, edo_approximation_error, best_edo_for_intervals,
)
from models.music_production.sidechain import (
    SidechainParams, db_to_linear, linear_to_db,
    envelope_follower, compute_gain_reduction,
    apply_sidechain, ducking_curve, pumping_rate,
)
from models.music_production.stereo_imaging import (
    StereoAnalysis as StereoAnalysisResult,
    stereo_correlation, stereo_width, stereo_balance,
    mono_compatibility, mid_side_ratio_db, analyze_stereo,
    to_mid_side, from_mid_side, widen_stereo,
)
from models.music_improvisation.ornaments import (
    OrnamentType, Ornament, trill, mordent, turn, grace_note,
    appoggiatura, acciaccatura, tremolo, glissando_pitches,
    ornament_duration, apply_ornament_timing,
)


class TestMicroPitch:
    def test_unison_cents(self):
        p = MicroPitch(0.0)
        assert p.frequency_ratio == pytest.approx(1.0)

    def test_octave_cents(self):
        p = MicroPitch(1200.0)
        assert p.frequency_ratio == pytest.approx(2.0)

    def test_to_hz_a440(self):
        p = MicroPitch(0.0)
        assert p.to_hz(440.0) == pytest.approx(440.0)

    def test_to_hz_octave_above(self):
        p = MicroPitch(1200.0)
        assert p.to_hz(440.0) == pytest.approx(880.0)

    def test_from_ratio_perfect_fifth(self):
        p = MicroPitch.from_ratio(3 / 2)
        assert p.cents == pytest.approx(701.955, abs=0.01)

    def test_from_hz_roundtrip(self):
        p = MicroPitch.from_hz(880.0, 440.0)
        assert p.to_hz(440.0) == pytest.approx(880.0)

    def test_nearest_12tet(self):
        p = MicroPitch(701.955)
        assert p.nearest_12tet() == 7  # perfect fifth = 7 semitones

    def test_deviation_cents(self):
        p = MicroPitch(701.955)
        assert abs(p.deviation_cents()) < 2.0  # JI fifth is ~2 cents sharp of 12TET


class TestEDOScales:
    def test_12edo_size(self):
        scale = edo_scale(12)
        assert len(scale) == 12

    def test_12edo_first_last(self):
        scale = edo_scale(12)
        assert scale[0].cents == pytest.approx(0.0)
        assert scale[11].cents == pytest.approx(1100.0)

    def test_24edo_quarter_tones(self):
        scale = quarter_tone_scale()
        assert len(scale) == 24
        assert scale[1].cents == pytest.approx(50.0)

    def test_19edo_step(self):
        scale = edo_scale(19)
        step = scale[1].cents
        assert step == pytest.approx(1200.0 / 19, abs=0.01)

    def test_53edo_approximates_ji_fifth(self):
        scale = edo_scale(53)
        ji_fifth = MicroPitch.from_ratio(3 / 2)
        closest = closest_edo_pitch(ji_fifth, 53)
        assert abs(closest.cents - ji_fifth.cents % 1200) < 1.0


class TestJustIntonation:
    def test_unison(self):
        assert JUST_INTERVALS["unison"].cents == pytest.approx(0.0)

    def test_octave(self):
        assert JUST_INTERVALS["octave"].cents == pytest.approx(1200.0)

    def test_perfect_fifth(self):
        fifth = JUST_INTERVALS["perfect_fifth"]
        assert fifth.cents == pytest.approx(701.955, abs=0.01)

    def test_major_third(self):
        third = JUST_INTERVALS["major_third"]
        assert third.cents == pytest.approx(386.314, abs=0.01)

    def test_syntonic_comma_value(self):
        sc = syntonic_comma()
        assert sc.cents == pytest.approx(21.506, abs=0.01)

    def test_pythagorean_comma_value(self):
        pc = pythagorean_comma()
        assert pc.cents == pytest.approx(23.460, abs=0.01)


class TestJILattice:
    def test_lattice_size(self):
        nodes = ji_lattice(2, 1)
        assert len(nodes) == 5 * 3  # (-2..2) x (-1..1)

    def test_origin_ratio(self):
        n = LatticeNode(0, 0)
        assert n.ratio == pytest.approx(1.0)

    def test_fifth_node(self):
        n = LatticeNode(1, 0)
        assert n.ratio == pytest.approx(1.5)

    def test_lattice_distance_self(self):
        a = LatticeNode(0, 0)
        assert lattice_distance(a, a) == pytest.approx(0.0)

    def test_lattice_distance_unit(self):
        a = LatticeNode(0, 0)
        b = LatticeNode(1, 0)
        assert lattice_distance(a, b) == pytest.approx(1.0)

    def test_cents_distance_symmetry(self):
        a = MicroPitch(300.0)
        b = MicroPitch(500.0)
        assert cents_distance(a, b) == pytest.approx(cents_distance(b, a))


class TestEDOApproximation:
    def test_closest_12edo_for_ji_fifth(self):
        ji_fifth = JUST_INTERVALS["perfect_fifth"]
        closest = closest_edo_pitch(ji_fifth, 12)
        assert closest.cents == pytest.approx(700.0)

    def test_approximation_error_12edo(self):
        ji_list = [JUST_INTERVALS["perfect_fifth"], JUST_INTERVALS["major_third"]]
        err = edo_approximation_error(ji_list, 12)
        assert err > 0  # 12-EDO is not just intonation

    def test_approximation_error_empty(self):
        assert edo_approximation_error([], 12) == 0.0

    def test_best_edo_includes_53(self):
        ji_list = list(JUST_INTERVALS.values())
        best = best_edo_for_intervals(ji_list)
        assert best in [53, 72]  # 53-EDO is famously close to JI

    def test_53edo_better_than_12(self):
        ji_list = list(JUST_INTERVALS.values())
        err_12 = edo_approximation_error(ji_list, 12)
        err_53 = edo_approximation_error(ji_list, 53)
        assert err_53 < err_12


class TestSidechain:
    def test_db_linear_roundtrip(self):
        db_val = -6.0
        assert linear_to_db(db_to_linear(db_val)) == pytest.approx(db_val)

    def test_zero_linear_to_db(self):
        assert linear_to_db(0.0) == -120.0

    def test_envelope_follower_shape(self):
        sig = np.sin(2 * np.pi * 100 * np.arange(44100) / 44100)
        env = envelope_follower(sig, 44100, attack_ms=1.0, release_ms=50.0)
        assert len(env) == len(sig)
        assert np.all(env >= 0)

    def test_gain_reduction_below_threshold(self):
        env_db = np.array([-30.0, -25.0, -35.0])
        gr = compute_gain_reduction(env_db, threshold_db=-20.0, ratio=4.0)
        assert np.all(gr == 0.0)

    def test_gain_reduction_above_threshold(self):
        env_db = np.array([-10.0])
        gr = compute_gain_reduction(env_db, threshold_db=-20.0, ratio=4.0)
        assert gr[0] > 0

    def test_sidechain_reduces_target(self):
        sr = 44100
        t = np.arange(sr) / sr
        target = np.sin(2 * np.pi * 440 * t) * 0.5
        kick = np.zeros(sr)
        kick[:2000] = 0.9
        params = SidechainParams(threshold_db=-20.0, ratio=8.0, attack_ms=1.0, release_ms=100.0)
        out = apply_sidechain(target, kick, sr, params)
        assert np.mean(np.abs(out[:2000])) < np.mean(np.abs(target[:2000]))

    def test_ducking_curve_shape(self):
        kick = np.zeros(44100)
        kick[:1000] = 0.8
        curve = ducking_curve(kick, 44100.0, depth_db=-12.0, threshold_db=-20.0)
        assert len(curve) == len(kick)
        assert curve[500] < 1.0  # ducked during kick

    def test_pumping_rate_silent(self):
        sig = np.zeros(44100)
        rate = pumping_rate(sig, 44100.0, threshold_db=-20.0)
        assert rate == 0.0


class TestStereoImaging:
    def test_mono_signal_correlation(self):
        sig = np.random.randn(1000)
        assert stereo_correlation(sig, sig) == pytest.approx(1.0, abs=0.01)

    def test_opposite_phase_correlation(self):
        sig = np.random.randn(1000)
        assert stereo_correlation(sig, -sig) == pytest.approx(-1.0, abs=0.01)

    def test_mono_width_zero(self):
        sig = np.random.randn(1000)
        assert stereo_width(sig, sig) == pytest.approx(0.0, abs=0.01)

    def test_stereo_width_uncorrelated(self):
        np.random.seed(42)
        l = np.random.randn(10000)
        r = np.random.randn(10000)
        w = stereo_width(l, r)
        assert 0.4 < w < 0.6  # should be ~0.5 for uncorrelated

    def test_balance_centered(self):
        sig = np.random.randn(1000)
        assert stereo_balance(sig, sig) == pytest.approx(0.0, abs=0.01)

    def test_balance_hard_left(self):
        left = np.ones(100) * 0.5
        right = np.zeros(100)
        assert stereo_balance(left, right) < -0.9

    def test_mono_compatibility_mono(self):
        sig = np.random.randn(1000)
        assert mono_compatibility(sig, sig) == pytest.approx(1.0, abs=0.05)

    def test_mid_side_roundtrip(self):
        left = np.random.randn(100)
        right = np.random.randn(100)
        mid, side = to_mid_side(left, right)
        l2, r2 = from_mid_side(mid, side)
        np.testing.assert_allclose(l2, left, atol=1e-10)
        np.testing.assert_allclose(r2, right, atol=1e-10)

    def test_widen_increases_width(self):
        np.random.seed(99)
        left = np.random.randn(5000) + 0.3 * np.random.randn(5000)
        right = np.random.randn(5000) + 0.3 * np.random.randn(5000)
        w_orig = stereo_width(left, right)
        l2, r2 = widen_stereo(left, right, factor=2.0)
        w_wide = stereo_width(l2, r2)
        assert w_wide > w_orig

    def test_analyze_stereo_struct(self):
        left = np.random.randn(500)
        right = np.random.randn(500)
        result = analyze_stereo(left, right)
        assert hasattr(result, "width")
        assert hasattr(result, "correlation")
        assert hasattr(result, "balance")
        assert hasattr(result, "mono_compatibility")


class TestOrnaments:
    def test_trill_alternation(self):
        pitches = trill(60, interval=2, n_notes=6)
        assert pitches == [60, 62, 60, 62, 60, 62]

    def test_mordent_upper(self):
        pitches = mordent(60, upper=True, interval=2)
        assert pitches == [60, 62, 60]

    def test_mordent_lower(self):
        pitches = mordent(60, upper=False, interval=1)
        assert pitches == [60, 59, 60]

    def test_turn_four_notes(self):
        pitches = turn(60, upper_interval=2, lower_interval=1)
        assert len(pitches) == 4
        assert pitches[0] > 60  # upper neighbor
        assert pitches[2] < 60  # lower neighbor
        assert pitches[3] == 60  # resolution

    def test_grace_note(self):
        pitches = grace_note(60, approach_from=-2)
        assert pitches == [58, 60]

    def test_appoggiatura_above(self):
        pitches = appoggiatura(60, above=True, interval=2)
        assert pitches == [62, 60]

    def test_acciaccatura_below(self):
        pitches = acciaccatura(60, above=False, interval=1)
        assert pitches == [59, 60]

    def test_tremolo_repeated(self):
        pitches = tremolo(60, repetitions=4)
        assert pitches == [60] * 4

    def test_glissando_ascending(self):
        pitches = glissando_pitches(60, 65)
        assert pitches == [60, 61, 62, 63, 64, 65]

    def test_glissando_descending(self):
        pitches = glissando_pitches(65, 60)
        assert pitches == [65, 64, 63, 62, 61, 60]

    def test_ornament_duration_trill(self):
        orn = Ornament(OrnamentType.TRILL, [60, 62, 60, 62], beat_duration=0.5)
        dur = ornament_duration(orn)
        assert dur == pytest.approx(0.5)

    def test_apply_ornament_timing(self):
        orn = Ornament(OrnamentType.TRILL, [60, 62, 60, 62], beat_duration=1.0)
        times = apply_ornament_timing(orn)
        assert len(times) == 4
        assert times[0] == pytest.approx(0.0)
        assert times[-1] < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
