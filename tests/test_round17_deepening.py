"""Tests for round 17 deepening: dose-response, notation, prosody, harmony-emotion."""
import numpy as np
import pytest

from models.music_therapy.dose_response import (
    TherapyDose, DoseResponseCurve, fit_dose_response,
    therapeutic_threshold, optimal_dose, dose_efficiency,
    diminishing_returns_point, session_schedule, cumulative_response,
)
from models.music_historical.notation_systems import (
    NotationEra, MensuralDuration, MensuralNote, Mensuration, NeumeGroup,
    ShapeNote, NEUME_CATALOG, SHAPE_NOTE_FOUR, SHAPE_NOTE_SEVEN,
    mensuration_ratio, mensural_to_modern_duration, neume_to_intervals,
    intervals_to_neume, TabNote, parse_tab_line, tab_to_pitches,
    notation_complexity, era_pitch_precision,
)
from models.music_lyrics.prosody import (
    Syllable, MetricPosition, ProsodyAlignment, syllabify, estimate_stress,
    metric_strength_profile, align_syllables_to_beats, prosody_score,
    mismatch_count, mismatch_ratio, syllable_density, melisma_ratio,
    classify_text_setting, stress_contour, contour_correlation,
)
from models.music_emotion.harmony_emotion import (
    AffectPoint, CHORD_AFFECT, chord_affect, progression_valence,
    progression_arousal, harmonic_tension, mode_brightness,
    emotional_trajectory, emotional_range, emotional_centroid,
    emotional_variance, valence_arousal_quadrant, major_minor_ratio,
)


# ── Dose-Response ───────────────────────────────────────────────────

class TestDoseResponseCurve:
    def test_zero_dose(self):
        c = DoseResponseCurve(ec50=10, hill_coefficient=1.5, e_max=100)
        assert c.response(0) == 0.0

    def test_ec50_gives_half(self):
        c = DoseResponseCurve(ec50=10, hill_coefficient=2.0, e_max=100)
        assert c.response(10) == pytest.approx(50.0, abs=0.1)

    def test_large_dose_saturates(self):
        c = DoseResponseCurve(ec50=10, hill_coefficient=2.0, e_max=100)
        assert c.response(1000) > 99.0

    def test_inverse_round_trip(self):
        c = DoseResponseCurve(ec50=10, hill_coefficient=1.5, e_max=100)
        for target in [20, 50, 80]:
            dose = c.inverse(target)
            assert c.response(dose) == pytest.approx(target, abs=0.5)

    def test_baseline(self):
        c = DoseResponseCurve(ec50=10, hill_coefficient=1.0, e_max=50, baseline=20)
        assert c.response(0) == 20.0
        assert c.response(10) == pytest.approx(45.0, abs=0.5)


class TestFitDoseResponse:
    def test_recovers_parameters(self):
        true = DoseResponseCurve(ec50=10, hill_coefficient=2.0, e_max=100)
        doses = np.linspace(0.1, 50, 20)
        responses = np.array([true.response(d) for d in doses])
        fitted = fit_dose_response(doses, responses)
        assert fitted.response(10) == pytest.approx(50.0, abs=10)


class TestTherapySchedule:
    def test_session_count(self):
        c = DoseResponseCurve(ec50=20, hill_coefficient=1.5, e_max=100)
        sched = session_schedule(c, total_sessions=8)
        assert len(sched) == 8

    def test_tapering(self):
        c = DoseResponseCurve(ec50=20, hill_coefficient=1.5, e_max=100)
        sched = session_schedule(c, total_sessions=8)
        assert sched[0].duration_min >= sched[-1].duration_min

    def test_cumulative_monotonic(self):
        c = DoseResponseCurve(ec50=10, hill_coefficient=1.5, e_max=100)
        resp = cumulative_response(c, [10, 10, 10, 10])
        assert resp[-1] >= resp[0]

    def test_threshold_below_optimal(self):
        c = DoseResponseCurve(ec50=10, hill_coefficient=2.0, e_max=100)
        assert therapeutic_threshold(c) < optimal_dose(c)


# ── Notation Systems ────────────────────────────────────────────────

class TestNeumes:
    def test_catalog_populated(self):
        assert len(NEUME_CATALOG) >= 8

    def test_podatus_ascending(self):
        assert NEUME_CATALOG["podatus"].ascending is True

    def test_clivis_descending(self):
        assert NEUME_CATALOG["clivis"].ascending is False

    def test_interval_lookup(self):
        ivs = neume_to_intervals("scandicus")
        assert ivs == (2, 2)

    def test_reverse_lookup(self):
        name = intervals_to_neume((2,))
        assert name == "podatus"


class TestMensural:
    def test_semibrevis_beats(self):
        n = MensuralNote(MensuralDuration.SEMIBREVIS, "C", 4)
        assert n.beats == 1.0

    def test_dotted_augments(self):
        n = MensuralNote(MensuralDuration.SEMIBREVIS, "C", 4, dotted=True)
        assert n.beats == 1.5

    def test_colored_reduces(self):
        n = MensuralNote(MensuralDuration.BREVIS, "D", 4, colored=True)
        assert n.beats < MensuralDuration.BREVIS.value

    def test_tempus_perfectum(self):
        t, p = mensuration_ratio(Mensuration.TEMPUS_PERFECTUM_PROLATIO_MAJOR)
        assert t == 3 and p == 3


class TestShapeNote:
    def test_four_shape_has_four(self):
        assert len(SHAPE_NOTE_FOUR) == 4

    def test_seven_shape_has_seven(self):
        assert len(SHAPE_NOTE_SEVEN) == 7

    def test_do_pitch_class(self):
        assert SHAPE_NOTE_SEVEN["do"].pitch_class == 0


class TestTab:
    def test_tab_note_midi(self):
        n = TabNote(string=0, fret=5)
        assert n.to_midi() == 45

    def test_parse_simple(self):
        notes = parse_tab_line("e|--0--2--3--")
        assert len(notes) == 3

    def test_notation_complexity_ordering(self):
        assert notation_complexity(NotationEra.NEUMATIC) < notation_complexity(NotationEra.MODERN)


# ── Prosody ─────────────────────────────────────────────────────────

class TestSyllabify:
    def test_monosyllable(self):
        assert syllabify("cat") == ["cat"]

    def test_multisyllable(self):
        syls = syllabify("music")
        assert len(syls) >= 2

    def test_empty(self):
        assert syllabify("") == [""]


class TestStress:
    def test_single_stressed(self):
        assert estimate_stress(["word"]) == [True]

    def test_trochaic_default(self):
        s = estimate_stress(["mu", "sic"])
        assert s[0] is True
        assert s[1] is False


class TestAlignment:
    def test_stressed_on_strong(self):
        syls = [Syllable("mu", True), Syllable("sic", False)]
        positions = [MetricPosition(1, 0, 1.0), MetricPosition(1, 1, 0.3)]
        aligned = align_syllables_to_beats(syls, positions)
        assert aligned[0].fit_score == 1.0

    def test_mismatch_detection(self):
        syls = [Syllable("the", False)]
        positions = [MetricPosition(1, 0, 0.9)]
        aligned = align_syllables_to_beats(syls, positions)
        assert aligned[0].is_mismatch


class TestProsodyMetrics:
    def test_syllable_density(self):
        d = syllable_density("hello world", 2.0)
        assert d > 0

    def test_syllabic_classification(self):
        assert classify_text_setting(10, 10) == "syllabic"

    def test_melismatic_classification(self):
        assert classify_text_setting(5, 20) == "melismatic"

    def test_contour_correlation(self):
        stress = np.array([1, 0, 1, 0, 1])
        melody = np.array([5, 2, 6, 1, 7])
        r = contour_correlation(stress.astype(float), melody.astype(float))
        assert -1.0 <= r <= 1.0


# ── Harmony-Emotion ─────────────────────────────────────────────────

class TestChordAffect:
    def test_major_high_valence(self):
        assert chord_affect("major").valence > 0.6

    def test_minor_low_valence(self):
        assert chord_affect("minor").valence < 0.4

    def test_dim7_high_arousal(self):
        assert chord_affect("dim7").arousal > 0.7

    def test_unknown_returns_neutral(self):
        p = chord_affect("nonexistent")
        assert p.valence == 0.5


class TestProgression:
    def test_valence_trajectory_length(self):
        v = progression_valence(["major", "minor", "dom7"])
        assert len(v) == 3

    def test_tension_tritone_highest(self):
        t = harmonic_tension([5, 6, 7])
        assert t[1] > t[0]  # tritone > P4
        assert t[1] > t[2]  # tritone > P5

    def test_mode_brightness_lydian(self):
        lydian = (2, 2, 2, 1, 2, 2, 1)
        locrian = (1, 2, 2, 1, 2, 2, 2)
        assert mode_brightness(lydian) > mode_brightness(locrian)


class TestEmotionalTrajectory:
    def test_trajectory_length(self):
        traj = emotional_trajectory(["major", "minor", "dim7"], [5, 3])
        assert len(traj) == 3

    def test_range_nonneg(self):
        traj = emotional_trajectory(["major", "minor"], [7])
        assert emotional_range(traj) >= 0.0

    def test_centroid_bounded(self):
        traj = emotional_trajectory(["major", "dom7", "minor"], [5, 7])
        c = emotional_centroid(traj)
        assert 0 <= c.valence <= 1
        assert 0 <= c.arousal <= 1

    def test_quadrant_happy(self):
        assert valence_arousal_quadrant(AffectPoint(0.8, 0.8)) == "happy/excited"

    def test_quadrant_sad(self):
        assert valence_arousal_quadrant(AffectPoint(0.2, 0.2)) == "sad/depressed"

    def test_major_minor_ratio(self):
        r = major_minor_ratio(["major", "minor", "major", "dom7"])
        assert r == pytest.approx(0.75)
