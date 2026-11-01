"""Tests for tension curves, lick library, and solo analysis."""
import numpy as np
import pytest

from models.music_improvisation.tension_curve import (
    sensory_dissonance,
    voice_leading_distance,
    tonal_distance,
    TensionPoint,
    tension_curve,
    tension_release_ratio,
    peak_tension_position,
    mean_tension,
    tension_variance,
)
from models.music_improvisation.lick_library import (
    LickTemplate,
    BEBOP_LICKS,
    BLUES_LICKS,
    ALL_TEMPLATES,
    match_score,
    find_licks,
    vocabulary_richness,
    suggest_next_lick,
)
from models.music_improvisation.solo_analysis import (
    note_choice_distribution,
    detect_approaches,
    approach_type_distribution,
    detect_enclosures,
    register_profile,
    rhythmic_density_curve,
    interval_histogram,
    summarise_solo,
)


# ---- Tension curve ----

class TestSensoryDissonance:
    def test_unison_zero(self):
        assert sensory_dissonance([0, 0]) == 0.0

    def test_tritone_high(self):
        d = sensory_dissonance([0, 6])
        assert d > 0.5

    def test_perfect_fifth_low(self):
        d = sensory_dissonance([0, 7])
        assert d < sensory_dissonance([0, 1])

    def test_single_note(self):
        assert sensory_dissonance([5]) == 0.0

    def test_empty(self):
        assert sensory_dissonance([]) == 0.0


class TestVoiceLeadingDistance:
    def test_same_chord_zero(self):
        assert voice_leading_distance([0, 4, 7], [0, 4, 7]) == 0.0

    def test_half_step_motion(self):
        d = voice_leading_distance([0, 4, 7], [0, 4, 8])
        assert d == pytest.approx(1.0)

    def test_empty(self):
        assert voice_leading_distance([], [0, 4, 7]) == 0.0


class TestTonalDistance:
    def test_same_key(self):
        assert tonal_distance(0, 0) == 0.0

    def test_fifth_apart(self):
        d = tonal_distance(0, 7)
        assert d > 0

    def test_tritone_far(self):
        d_tri = tonal_distance(0, 6)
        d_fifth = tonal_distance(0, 7)
        assert d_tri >= d_fifth


class TestTensionCurve:
    def test_single_chord(self):
        curve = tension_curve([[0, 4, 7]])
        assert len(curve) == 1
        assert curve[0].dissonance == sensory_dissonance([0, 4, 7])

    def test_two_chords(self):
        curve = tension_curve([[0, 4, 7], [0, 3, 7]])
        assert len(curve) == 2
        assert curve[1].voice_leading > 0

    def test_empty(self):
        assert tension_curve([]) == []

    def test_custom_keys(self):
        curve = tension_curve([[0, 4, 7], [5, 9, 0]], key_roots=[0, 5])
        assert len(curve) == 2
        assert curve[1].tonal > 0


class TestTensionMetrics:
    def test_release_ratio_balanced(self):
        pts = [TensionPoint(i, 0, 0, 0, float(i % 2)) for i in range(6)]
        r = tension_release_ratio(pts)
        assert r > 0

    def test_peak_position(self):
        pts = [TensionPoint(i, 0, 0, 0, float(i)) for i in range(5)]
        assert peak_tension_position(pts) == pytest.approx(1.0)

    def test_mean(self):
        pts = [TensionPoint(0, 0, 0, 0, 2.0), TensionPoint(1, 0, 0, 0, 4.0)]
        assert mean_tension(pts) == pytest.approx(3.0)

    def test_variance(self):
        pts = [TensionPoint(0, 0, 0, 0, 1.0), TensionPoint(1, 0, 0, 0, 1.0)]
        assert tension_variance(pts) == pytest.approx(0.0)


# ---- Lick library ----

class TestMatchScore:
    def test_identical(self):
        assert match_score([2, 2, 1], [2, 2, 1]) == pytest.approx(1.0)

    def test_different(self):
        s = match_score([2, 2, 1], [-5, -5, -5])
        assert s < 0.5

    def test_empty(self):
        assert match_score([], [1, 2]) == 0.0


class TestFindLicks:
    def test_bebop_enclosure(self):
        pitches = [60, 61, 59, 60]
        matches = find_licks(pitches, BEBOP_LICKS, threshold=0.5)
        names = {m.template.name for m in matches}
        assert "bebop_enclosure_above" in names or len(matches) > 0

    def test_chromatic_run(self):
        pitches = [60, 61, 62, 63, 64]
        matches = find_licks(pitches, threshold=0.5)
        assert any(m.template.name == "chromatic_run_up" for m in matches)

    def test_short_sequence(self):
        assert find_licks([60]) == []

    def test_threshold_filtering(self):
        pitches = [60, 62, 64, 65, 67]
        strict = find_licks(pitches, threshold=0.95)
        loose = find_licks(pitches, threshold=0.3)
        assert len(strict) <= len(loose)


class TestVocabularyRichness:
    def test_chromatic_passage(self):
        pitches = [60, 61, 62, 63, 64, 65, 66, 67]
        vr = vocabulary_richness(pitches, threshold=0.5)
        assert vr["coverage"] > 0
        assert vr["unique_patterns"] > 0

    def test_empty(self):
        vr = vocabulary_richness([60])
        assert vr["coverage"] == 0.0


class TestSuggestNextLick:
    def test_ascending_context(self):
        suggestions = suggest_next_lick([60, 62, 64])
        assert len(suggestions) > 0
        assert suggestions[0].intervals[0] > 0

    def test_short_context(self):
        suggestions = suggest_next_lick([60])
        assert len(suggestions) > 0


# ---- Solo analysis ----

class TestNoteChoiceDistribution:
    def test_all_chord_tones(self):
        chord_pcs = {0, 4, 7, 10}
        dist = note_choice_distribution([60, 64, 67], 0, chord_pcs, {0, 2, 4, 5, 7, 9, 10})
        assert dist["chord_tone"] == pytest.approx(1.0)

    def test_empty(self):
        dist = note_choice_distribution([], 0, {0, 4, 7}, {0, 2, 4, 5, 7})
        assert dist["chord_tone"] == 0.0


class TestDetectApproaches:
    def test_chromatic_approach_below(self):
        pitches = [59, 60]  # B -> C, C is chord tone of C major
        events = detect_approaches(pitches, {0, 4, 7})
        assert len(events) > 0
        assert events[0].approach_type == "chromatic_below"

    def test_no_approach(self):
        pitches = [60, 64]  # both chord tones
        events = detect_approaches(pitches, {0, 4, 7})
        assert len(events) == 0


class TestDetectEnclosures:
    def test_enclosure_pattern(self):
        pitches = [61, 59, 60]  # above, below, target (C)
        enc = detect_enclosures(pitches, {0, 4, 7})
        assert len(enc) == 1

    def test_no_enclosure(self):
        pitches = [60, 64, 67]  # all chord tones, no enclosure geometry
        enc = detect_enclosures(pitches, {0, 4, 7})
        assert len(enc) == 0


class TestRegisterProfile:
    def test_basic(self):
        rp = register_profile([48, 60, 72, 84])
        assert rp.lowest == 48
        assert rp.highest == 84
        assert rp.range_semitones == 36
        assert rp.std_pitch > 0

    def test_empty(self):
        rp = register_profile([])
        assert rp.range_semitones == 0


class TestRhythmicDensityCurve:
    def test_uniform(self):
        onsets = [float(i) * 0.5 for i in range(10)]
        dc = rhythmic_density_curve(onsets, window_sec=2.0, hop_sec=0.5)
        assert len(dc) > 0
        assert np.all(dc >= 0)

    def test_empty(self):
        dc = rhythmic_density_curve([])
        assert dc[0] == 0.0


class TestIntervalHistogram:
    def test_chromatic_scale(self):
        pitches = list(range(60, 73))
        ih = interval_histogram(pitches)
        assert ih[1] == pytest.approx(1.0)

    def test_empty(self):
        ih = interval_histogram([60])
        assert ih.sum() == 0.0


class TestSummariseSolo:
    def test_basic(self):
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        onsets = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        s = summarise_solo(pitches, onsets, chord_root=0)
        assert s.n_notes == 8
        assert s.duration_sec == pytest.approx(3.5)
        assert s.avg_density > 0
        assert sum(s.note_choice.values()) == pytest.approx(1.0)
        assert s.register.range_semitones == 12
