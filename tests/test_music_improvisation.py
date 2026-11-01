"""Tests for the music_improvisation module."""

import pytest
from models.music_improvisation.scale_navigation import (
    ScaleMode, scale_degrees, chord_tones_in_scale,
    available_tensions, approach_note_options, target_note_gravity,
)
from models.music_improvisation.motivic_development import (
    Motif, transpose_motif, invert_motif, retrograde_motif,
    augment_motif, diminish_motif, sequence_motif, motivic_similarity,
)
from models.music_improvisation.call_response import (
    call_response_score, rhythmic_complement, melodic_answer, phrase_balance,
)
from models.music_improvisation.jazz_voicings import (
    VoicingType, voice_chord, rootless_voicing,
    drop2_voicing, shell_voicing, quartal_voicing, voice_leading_cost,
)


class TestScaleNavigation:
    def test_c_ionian(self):
        degrees = scale_degrees(0, ScaleMode.IONIAN)
        assert degrees == [0, 2, 4, 5, 7, 9, 11]

    def test_d_dorian(self):
        degrees = scale_degrees(2, ScaleMode.DORIAN)
        assert 2 in degrees
        assert len(degrees) == 7

    def test_chord_tones_in_scale(self):
        result = chord_tones_in_scale([0, 4, 7], 0, ScaleMode.IONIAN)
        assert all(result)

    def test_chord_tone_outside_scale(self):
        result = chord_tones_in_scale([0, 4, 7, 10], 0, ScaleMode.IONIAN)
        assert result[3] is False

    def test_available_tensions_dom7_mixolydian(self):
        tensions = available_tensions("dom7", ScaleMode.MIXOLYDIAN)
        assert len(tensions) > 0

    def test_approach_notes(self):
        opts = approach_note_options(0)
        assert 11 in opts["chromatic_below"]
        assert 1 in opts["chromatic_above"]

    def test_target_gravity_root(self):
        g = target_note_gravity(60, 60, [60, 64, 67])
        assert g == 1.0

    def test_target_gravity_chord_tone(self):
        g = target_note_gravity(64, 60, [60, 64, 67])
        assert g == pytest.approx(0.7)


class TestMotivicDevelopment:
    def test_transpose(self):
        m = Motif([60, 62, 64], [0.5, 0.5, 1.0])
        t = transpose_motif(m, 5)
        assert t.pitches == [65, 67, 69]
        assert t.durations == m.durations

    def test_invert(self):
        m = Motif([60, 64, 67], [0.5, 0.5, 1.0])
        inv = invert_motif(m)
        assert inv.pitches[0] == 60
        assert inv.pitches[1] == 56
        assert inv.pitches[2] == 53

    def test_retrograde(self):
        m = Motif([60, 62, 64], [0.25, 0.5, 1.0])
        r = retrograde_motif(m)
        assert r.pitches == [64, 62, 60]
        assert r.durations == [1.0, 0.5, 0.25]

    def test_augment(self):
        m = Motif([60], [0.5])
        a = augment_motif(m, 2.0)
        assert a.durations == [1.0]

    def test_diminish(self):
        m = Motif([60], [1.0])
        d = diminish_motif(m, 0.5)
        assert d.durations == [0.5]

    def test_sequence(self):
        m = Motif([60, 62], [0.5, 0.5])
        seq = sequence_motif(m, [2, 2])
        assert len(seq) == 3
        assert seq[1].pitches == [62, 64]
        assert seq[2].pitches == [64, 66]

    def test_similarity_identical(self):
        m = Motif([60, 62, 64, 65], [0.5] * 4)
        assert motivic_similarity(m, m) == pytest.approx(1.0)

    def test_similarity_transposed(self):
        m = Motif([60, 62, 64, 65], [0.5] * 4)
        t = transpose_motif(m, 7)
        sim = motivic_similarity(m, t)
        assert sim > 0.8


class TestCallResponse:
    def test_score_balanced(self):
        s = call_response_score([60, 62, 64], [67, 65, 63], [0.5] * 3, [0.5] * 3)
        assert 0.0 < s <= 1.0

    def test_rhythmic_complement(self):
        comp = rhythmic_complement([0.5, 0.5, 0.5], 4.0)
        assert sum(comp) == pytest.approx(2.5)

    def test_melodic_answer(self):
        call = [60, 64, 67]
        answer = melodic_answer(call, 72, [0, 2, 4, 5, 7, 9, 11])
        assert len(answer) == 3
        assert answer[0] == 72

    def test_phrase_balance_equal(self):
        assert phrase_balance([4, 4, 4, 4]) == pytest.approx(1.0)

    def test_phrase_balance_unequal(self):
        b = phrase_balance([1, 10, 1, 10])
        assert b < 0.7


class TestJazzVoicings:
    def test_close_voicing(self):
        v = voice_chord(60, "maj7", VoicingType.CLOSE)
        assert v == [60, 64, 67, 71]

    def test_drop2(self):
        close = [60, 64, 67, 71]
        d2 = drop2_voicing(close)
        assert d2[0] == 55
        assert len(d2) == 4

    def test_rootless_a(self):
        close = [60, 64, 67, 71]
        rl = rootless_voicing(close, "A")
        assert 60 not in rl
        assert len(rl) == 3

    def test_shell(self):
        s = shell_voicing(60, "dom7")
        assert len(s) == 3
        assert s[0] == 60

    def test_quartal(self):
        q = quartal_voicing(60, 4)
        assert len(q) == 4
        assert all(q[i + 1] - q[i] == 5 for i in range(3))

    def test_voice_leading_cost_zero(self):
        v = [60, 64, 67, 71]
        assert voice_leading_cost(v, v) == 0.0

    def test_voice_leading_cost_positive(self):
        a = [60, 64, 67, 71]
        b = [62, 65, 69, 72]
        cost = voice_leading_cost(a, b)
        assert cost > 0


class TestModalInterchange:
    def test_mode_pitch_classes_ionian(self):
        from models.music_improvisation.modal_interchange_improv import (
            Mode, mode_pitch_classes, common_tones, mode_distance,
            find_interchange_options, pivot_note, interchange_tension,
            build_interchange_sequence,
        )
        pcs = mode_pitch_classes(0, Mode.IONIAN)
        assert pcs == {0, 2, 4, 5, 7, 9, 11}

    def test_common_tones_self(self):
        from models.music_improvisation.modal_interchange_improv import Mode, common_tones
        ct = common_tones(0, Mode.IONIAN, 0, Mode.IONIAN)
        assert len(ct) == 7

    def test_mode_distance_self_zero(self):
        from models.music_improvisation.modal_interchange_improv import Mode, mode_distance
        assert mode_distance(0, Mode.IONIAN, 0, Mode.IONIAN) == 0

    def test_mode_distance_positive(self):
        from models.music_improvisation.modal_interchange_improv import Mode, mode_distance
        d = mode_distance(0, Mode.IONIAN, 0, Mode.ALTERED)
        assert d > 0

    def test_find_options_ranked_by_smoothness(self):
        from models.music_improvisation.modal_interchange_improv import Mode, find_interchange_options
        opts = find_interchange_options(0, Mode.IONIAN)
        assert len(opts) > 0
        smoothnesses = [o.smoothness for o in opts]
        assert smoothnesses == sorted(smoothnesses, reverse=True)

    def test_pivot_note_exists(self):
        from models.music_improvisation.modal_interchange_improv import Mode, pivot_note
        p = pivot_note(0, Mode.IONIAN, Mode.MIXOLYDIAN)
        assert p is not None

    def test_interchange_tension_range(self):
        from models.music_improvisation.modal_interchange_improv import Mode, interchange_tension
        t = interchange_tension(0, Mode.IONIAN, Mode.ALTERED)
        assert 0.0 <= t <= 1.0

    def test_build_sequence_length(self):
        from models.music_improvisation.modal_interchange_improv import Mode, build_interchange_sequence
        import numpy as np
        seq = build_interchange_sequence(0, Mode.DORIAN, 4, 8.0, rng=np.random.default_rng(42))
        assert len(seq.modes) == 4
        assert seq.total_beats == 32.0

    def test_sequence_tension_profile(self):
        from models.music_improvisation.modal_interchange_improv import Mode, build_interchange_sequence
        import numpy as np
        seq = build_interchange_sequence(0, Mode.IONIAN, 3, 4.0, rng=np.random.default_rng(0))
        profile = seq.tension_profile()
        assert len(profile) == 2
        assert all(0.0 <= t <= 1.0 for t in profile)


class TestRhythmPatterns:
    def test_swing_straight_eighth(self):
        from models.music_improvisation.rhythm_patterns import swing_offset
        assert swing_offset(0.0, 0.67) == pytest.approx(0.0)

    def test_swing_upbeat_delayed(self):
        from models.music_improvisation.rhythm_patterns import swing_offset
        swung = swing_offset(0.5, 0.67)
        assert swung > 0.5

    def test_apply_swing_preserves_count(self):
        from models.music_improvisation.rhythm_patterns import apply_swing, RhythmicEvent
        events = [RhythmicEvent(i * 0.5, 0.5) for i in range(4)]
        swung = apply_swing(events, 0.67)
        assert len(swung) == 4

    def test_syncopation_downbeats_low(self):
        from models.music_improvisation.rhythm_patterns import syncopation_score
        assert syncopation_score([0.0, 2.0, 4.0], 4) < 0.5

    def test_syncopation_upbeats_high(self):
        from models.music_improvisation.rhythm_patterns import syncopation_score
        assert syncopation_score([0.5, 1.5, 2.5], 4) == 1.0

    def test_metric_displacement(self):
        from models.music_improvisation.rhythm_patterns import metric_displacement, RhythmicEvent
        events = [RhythmicEvent(0.0, 1.0), RhythmicEvent(1.0, 1.0)]
        shifted = metric_displacement(events, 0.5)
        assert shifted[0].onset_beat == pytest.approx(0.5)

    def test_clave_alignment_perfect(self):
        from models.music_improvisation.rhythm_patterns import clave_alignment, CLAVE_PATTERNS
        onsets = CLAVE_PATTERNS["son_32"]
        assert clave_alignment(onsets, "son_32") == pytest.approx(1.0)

    def test_generate_clave_pattern_count(self):
        from models.music_improvisation.rhythm_patterns import generate_clave_pattern
        events = generate_clave_pattern("son_32", 2)
        assert len(events) == 10

    def test_tresillo_pattern(self):
        from models.music_improvisation.rhythm_patterns import tresillo_pattern
        events = tresillo_pattern(1)
        assert len(events) == 3
        onsets = [e.onset_beat for e in events]
        assert onsets == pytest.approx([0.0, 1.5, 3.0])

    def test_anticipation_before_target(self):
        from models.music_improvisation.rhythm_patterns import anticipation_pattern
        events = anticipation_pattern([1.0, 2.0], 0.5)
        assert events[0].onset_beat == pytest.approx(0.5)

    def test_density_curve_nonempty(self):
        from models.music_improvisation.rhythm_patterns import density_curve, RhythmicEvent
        events = [RhythmicEvent(i * 0.5, 0.25) for i in range(8)]
        curve = density_curve(events, 2.0)
        assert len(curve) > 0


class TestPackageExports:
    def test_all_exports(self):
        import models.music_improvisation as mi
        assert hasattr(mi, "ScaleMode")
        assert hasattr(mi, "Motif")
        assert hasattr(mi, "call_response_score")
        assert hasattr(mi, "VoicingType")
        assert hasattr(mi, "Mode")
        assert hasattr(mi, "RhythmFeel")
