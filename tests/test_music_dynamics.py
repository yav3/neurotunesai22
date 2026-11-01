"""Tests for music_dynamics: velocity curves, dynamic markings, articulation, expression, analysis."""
import numpy as np
import pytest

from models.music_dynamics.velocity_curves import (
    VelocityCurve, velocity_to_amplitude, amplitude_to_velocity,
    apply_velocity_curve, humanize_velocities, accent_pattern,
    velocity_gradient,
)
from models.music_dynamics.dynamic_markings import (
    DynamicLevel, DynamicMarking, dynamic_to_velocity, velocity_to_dynamic,
    dynamic_to_velocity_range, interpret_hairpin, sforzando, subito_change,
    parse_dynamic_sequence,
)
from models.music_dynamics.articulation import (
    ArticulationType, apply_articulation, legato_overlap,
    detect_articulation_from_ratio, batch_articulate,
)
from models.music_dynamics.expression_contour import (
    phrase_arc, crescendo_contour, diminuendo_contour, swell,
    terraced_dynamics, smooth_transition, rubato_warp, agogic_accent,
    combine_contours, build_expression_contour,
)
from models.music_dynamics.dynamic_analysis import (
    rms_envelope, rms_to_db, velocity_histogram, dynamic_range_midi,
    loudness_variation, detect_dynamic_changes, classify_dynamic_level,
    analyse_dynamics, compute_compression_ratio, normalise_velocities,
)


# ── Velocity Curves ──────────────────────────────────────────────────

class TestVelocityToAmplitude:
    def test_linear_endpoints(self):
        assert velocity_to_amplitude(0) == pytest.approx(0.0)
        assert velocity_to_amplitude(127) == pytest.approx(1.0)

    def test_fixed_curve(self):
        assert velocity_to_amplitude(0, VelocityCurve.FIXED) == 0.0
        assert velocity_to_amplitude(1, VelocityCurve.FIXED) == 1.0
        assert velocity_to_amplitude(127, VelocityCurve.FIXED) == 1.0

    def test_monotonic_all_curves(self):
        for curve in VelocityCurve:
            prev = -1.0
            for v in range(0, 128, 10):
                a = velocity_to_amplitude(v, curve)
                assert a >= prev - 1e-6
                prev = a


class TestAmplitudeToVelocity:
    def test_linear_roundtrip(self):
        for v in [0, 32, 64, 96, 127]:
            a = velocity_to_amplitude(v, VelocityCurve.LINEAR)
            v2 = amplitude_to_velocity(a, VelocityCurve.LINEAR)
            assert abs(v2 - v) <= 1

    def test_clamping(self):
        assert amplitude_to_velocity(-0.5) == 0
        assert amplitude_to_velocity(1.5) == 127


class TestHumanizeVelocities:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        result = humanize_velocities([64] * 50, amount=0.5, rng=rng)
        assert all(1 <= v <= 127 for v in result)

    def test_zero_amount_no_change(self):
        result = humanize_velocities([80, 80, 80], amount=0.0)
        assert result == [80, 80, 80]


class TestAccentPattern:
    def test_accents_boosted(self):
        vels = [64, 64, 64, 64]
        pattern = [True, False, False, False]
        result = accent_pattern(vels, pattern, accent_boost=20)
        assert result[0] == 84
        assert result[1] == 64

    def test_pattern_cycles(self):
        vels = [64] * 6
        pattern = [True, False]
        result = accent_pattern(vels, pattern, accent_boost=10)
        assert result == [74, 64, 74, 64, 74, 64]


class TestVelocityGradient:
    def test_endpoints(self):
        grad = velocity_gradient(30, 100, 5)
        assert grad[0] == 30
        assert grad[-1] == 100

    def test_single_note(self):
        assert velocity_gradient(60, 100, 1) == [60]

    def test_monotonic_crescendo(self):
        grad = velocity_gradient(40, 100, 10)
        for i in range(len(grad) - 1):
            assert grad[i] <= grad[i + 1]


# ── Dynamic Markings ─────────────────────────────────────────────────

class TestDynamicToVelocity:
    def test_ordering(self):
        levels = list(DynamicLevel)
        vels = [dynamic_to_velocity(l) for l in levels]
        for i in range(len(vels) - 1):
            assert vels[i] < vels[i + 1]

    def test_range(self):
        for level in DynamicLevel:
            v = dynamic_to_velocity(level)
            lo, hi = dynamic_to_velocity_range(level)
            assert lo <= v <= hi


class TestVelocityToDynamic:
    def test_midpoints_roundtrip(self):
        for level in DynamicLevel:
            v = dynamic_to_velocity(level)
            assert velocity_to_dynamic(v) == level

    def test_clamping(self):
        assert velocity_to_dynamic(0) is not None
        assert velocity_to_dynamic(200) is not None


class TestInterpretHairpin:
    def test_crescendo(self):
        vels = interpret_hairpin(DynamicLevel.P, DynamicLevel.F, 5)
        assert len(vels) == 5
        assert vels[0] < vels[-1]

    def test_single_note(self):
        vels = interpret_hairpin(DynamicLevel.MF, DynamicLevel.FF, 1)
        assert len(vels) == 1

    def test_empty(self):
        assert interpret_hairpin(DynamicLevel.P, DynamicLevel.F, 0) == []


class TestSforzando:
    def test_boost(self):
        assert sforzando(80, 40) == 120
        assert sforzando(120, 40) == 127  # clamped


class TestParseDynamicSequence:
    def test_empty_markings_returns_mf(self):
        vels = parse_dynamic_sequence([], 4.0, 0.25)
        expected_n = max(1, int(4.0 / 0.25))
        assert len(vels) == expected_n
        assert all(v == dynamic_to_velocity(DynamicLevel.MF) for v in vels)

    def test_single_marking(self):
        marks = [DynamicMarking(DynamicLevel.F, 0.0)]
        vels = parse_dynamic_sequence(marks, 2.0, 0.5)
        assert all(v == dynamic_to_velocity(DynamicLevel.F) for v in vels)


# ── Articulation ─────────────────────────────────────────────────────

class TestApplyArticulation:
    def test_staccato_shortens(self):
        note = apply_articulation(60, 0.0, 1.0, 80, ArticulationType.STACCATO)
        assert note.sounded_duration_beats == pytest.approx(0.5)

    def test_legato_full(self):
        note = apply_articulation(60, 0.0, 1.0, 80, ArticulationType.LEGATO)
        assert note.sounded_duration_beats == pytest.approx(1.0)

    def test_marcato_louder(self):
        note = apply_articulation(60, 0.0, 1.0, 80, ArticulationType.MARCATO)
        assert note.velocity > 80

    def test_velocity_clamped(self):
        note = apply_articulation(60, 0.0, 1.0, 127, ArticulationType.MARCATO)
        assert note.velocity <= 127


class TestDetectArticulation:
    def test_staccato(self):
        assert detect_articulation_from_ratio(0.5, 1.0) == ArticulationType.STACCATO

    def test_staccatissimo(self):
        assert detect_articulation_from_ratio(0.2, 1.0) == ArticulationType.STACCATISSIMO

    def test_legato(self):
        assert detect_articulation_from_ratio(1.0, 1.0) == ArticulationType.LEGATO

    def test_zero_notated(self):
        assert detect_articulation_from_ratio(0.5, 0.0) == ArticulationType.NORMAL


class TestLegato:
    def test_extends_duration(self):
        notes = [
            apply_articulation(60, 0.0, 1.0, 80, ArticulationType.LEGATO),
            apply_articulation(62, 1.0, 1.0, 80, ArticulationType.LEGATO),
            apply_articulation(64, 2.0, 1.0, 80, ArticulationType.LEGATO),
        ]
        extended = legato_overlap(notes, overlap_beats=0.1)
        assert extended[0].sounded_duration_beats > 1.0
        assert extended[-1].sounded_duration_beats == notes[-1].sounded_duration_beats


class TestBatchArticulate:
    def test_length(self):
        result = batch_articulate([60, 62, 64], [0.0, 0.5, 1.0], [0.5, 0.5, 0.5],
                                  [80, 80, 80], ArticulationType.STACCATO)
        assert len(result) == 3


# ── Expression Contour ───────────────────────────────────────────────

class TestPhraseArc:
    def test_starts_low_ends_low(self):
        arc = phrase_arc(100)
        assert arc[0] < arc[50]
        assert arc[-1] < arc[50]

    def test_peak_position(self):
        arc = phrase_arc(100, peak_position=0.3)
        peak_idx = np.argmax(arc)
        assert abs(peak_idx - 30) <= 5


class TestCrescendoDiminuendo:
    def test_crescendo_monotonic(self):
        c = crescendo_contour(50)
        for i in range(len(c) - 1):
            assert c[i] <= c[i + 1] + 1e-10

    def test_diminuendo_monotonic(self):
        d = diminuendo_contour(50)
        for i in range(len(d) - 1):
            assert d[i] >= d[i + 1] - 1e-10


class TestSwell:
    def test_starts_and_ends_at_base(self):
        s = swell(100, base=0.3, peak=1.0)
        assert s[0] == pytest.approx(0.3, abs=0.05)
        assert s[-1] == pytest.approx(0.3, abs=0.05)

    def test_peak_in_middle(self):
        s = swell(100)
        peak_idx = np.argmax(s)
        assert 40 <= peak_idx <= 60


class TestTerracedDynamics:
    def test_step_values(self):
        td = terraced_dynamics([0.3, 0.8, 0.5], [10, 10, 10])
        assert len(td) == 30
        assert np.allclose(td[:10], 0.3)
        assert np.allclose(td[10:20], 0.8)


class TestRubatoWarp:
    def test_preserves_total_duration(self):
        warp = rubato_warp(100)
        assert np.sum(warp) == pytest.approx(100.0, abs=0.5)


class TestAgogicAccent:
    def test_stretches_correct_note(self):
        durs = [1.0, 1.0, 1.0, 1.0]
        result = agogic_accent(durs, [1], stretch=1.5)
        assert result[0] == 1.0
        assert result[1] == 1.5


class TestCombineContours:
    def test_weighted_sum(self):
        c1 = np.ones(10) * 0.5
        c2 = np.ones(10) * 1.0
        combined = combine_contours([c1, c2], weights=[0.5, 0.5])
        assert np.allclose(combined, 0.75)

    def test_empty(self):
        result = combine_contours([])
        assert len(result) == 0


# ── Dynamic Analysis ─────────────────────────────────────────────────

class TestRMSEnvelope:
    def test_constant_signal(self):
        signal = np.ones(4096)
        rms = rms_envelope(signal, frame_size=1024, hop_size=512)
        assert np.allclose(rms, 1.0, atol=0.01)

    def test_silent_signal(self):
        signal = np.zeros(2048)
        rms = rms_envelope(signal)
        assert np.allclose(rms, 0.0)


class TestRMSToDb:
    def test_unity(self):
        rms = np.array([1.0])
        db = rms_to_db(rms)
        assert db[0] == pytest.approx(0.0)

    def test_small_value(self):
        rms = np.array([0.01])
        db = rms_to_db(rms)
        assert db[0] < 0


class TestVelocityHistogram:
    def test_shape(self):
        counts, edges = velocity_histogram([64, 64, 80, 100])
        assert len(counts) == 16
        assert len(edges) == 17


class TestDynamicRangeMidi:
    def test_basic(self):
        assert dynamic_range_midi([30, 50, 100]) == 70

    def test_empty(self):
        assert dynamic_range_midi([]) == 0


class TestDetectDynamicChanges:
    def test_finds_subito(self):
        vels = [40, 40, 100, 100]
        changes = detect_dynamic_changes(vels, threshold=30)
        assert len(changes) == 1
        assert changes[0][0] == 2

    def test_no_changes(self):
        vels = [64, 65, 63, 64]
        changes = detect_dynamic_changes(vels, threshold=15)
        assert len(changes) == 0


class TestClassifyDynamic:
    def test_boundaries(self):
        assert classify_dynamic_level(10) == "ppp"
        assert classify_dynamic_level(25) == "pp"
        assert classify_dynamic_level(40) == "p"
        assert classify_dynamic_level(55) == "mp"
        assert classify_dynamic_level(75) == "mf"
        assert classify_dynamic_level(90) == "f"
        assert classify_dynamic_level(105) == "ff"
        assert classify_dynamic_level(120) == "fff"


class TestAnalyseDynamics:
    def test_midi_only(self):
        profile = analyse_dynamics([40, 60, 80, 100])
        assert profile.min_velocity == 40
        assert profile.max_velocity == 100
        assert profile.dynamic_range == 60

    def test_with_audio(self):
        signal = np.random.randn(8000)
        profile = analyse_dynamics([64, 80], signal=signal, sr=16000)
        assert len(profile.rms_db) > 0


class TestNormaliseVelocities:
    def test_target_range(self):
        vels = [20, 30, 40, 50, 60]
        normed = normalise_velocities(vels, target_mean=80, target_range=40)
        assert all(1 <= v <= 127 for v in normed)
        assert max(normed) - min(normed) == pytest.approx(40, abs=2)

    def test_empty(self):
        assert normalise_velocities([]) == []


class TestCompressionRatio:
    def test_no_compression_needed(self):
        assert compute_compression_ratio([60, 70, 80], target_range=40) == 1.0

    def test_needs_compression(self):
        ratio = compute_compression_ratio([10, 110], target_range=40)
        assert ratio > 1.0
