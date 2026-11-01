"""Tests for the music_arrangement module."""

import pytest
from models.music_arrangement.orchestration import (
    InstrumentRange, ORCHESTRA_RANGES,
    fits_range, suggest_transposition, doubling_score, orchestral_balance,
)
from models.music_arrangement.scoring import (
    ScoreFormat, NotatedPart, reduce_to_piano,
    expand_lead_sheet, part_extraction, transpose_part,
)
from models.music_arrangement.texture_mapping import (
    TextureType, classify_texture, texture_density, recommended_texture,
)
from models.music_arrangement.reharmonization import (
    tritone_substitution, backdoor_progression,
    modal_interchange, chromatic_mediant, reharmonize_melody_note,
)
from models.music_arrangement.voice_leading import (
    VoiceLeadingRule, check_parallel_motion, voice_crossing_check,
    smooth_voice_leading, voice_leading_cost_matrix,
    optimal_voicing, total_voice_motion,
)
from models.music_arrangement.arranging_styles import (
    ArrangingStyle, StyleTemplate, STYLE_TEMPLATES,
    assign_voices, double_at_octave, unison_passage, tutti_score,
)


class TestOrchestration:
    def test_fits_range_violin(self):
        assert fits_range(60, "violin") is True
        assert fits_range(30, "violin") is False

    def test_suggest_transposition_in_range(self):
        assert suggest_transposition(60, "violin") == 0

    def test_suggest_transposition_too_low(self):
        trans = suggest_transposition(30, "violin")
        assert trans is not None and trans > 0

    def test_doubling_unison(self):
        assert doubling_score([60, 64, 67], [60, 64, 67]) == 1.0

    def test_doubling_disjoint(self):
        assert doubling_score([60, 64, 67], [61, 63, 66]) == 0.0

    def test_orchestral_balance(self):
        assignments = {
            "violin": [60, 62, 64],
            "flute": [72],
            "trombone": [48, 50],
        }
        bal = orchestral_balance(assignments)
        assert "strings" in bal
        assert "woodwind" in bal
        assert "brass" in bal
        assert abs(sum(bal.values()) - 1.0) < 1e-6

    def test_orchestra_ranges_count(self):
        assert len(ORCHESTRA_RANGES) >= 15


class TestScoring:
    def test_reduce_to_piano(self):
        parts = [
            NotatedPart("violin", [[72, 76]], [1.0]),
            NotatedPart("cello", [[48, 52]], [1.0]),
        ]
        treble, bass = reduce_to_piano(parts)
        assert len(treble) == 1
        assert 72 in treble[0]
        assert 48 in bass[0]

    def test_expand_lead_sheet(self):
        melody = [60, 62, 64]
        chords = ["Cmaj7", "Dm7", "G7"]
        voicings = {"Cmaj7": [60, 64, 67, 71], "Dm7": [62, 65, 69, 72], "G7": [67, 71, 74, 77]}
        mel, acc = expand_lead_sheet(melody, chords, voicings)
        assert mel == melody
        assert len(acc) == 3

    def test_part_extraction(self):
        score = {"violin": [[60], [62]], "cello": [[48], [50]]}
        assert part_extraction(score, "violin") == [[60], [62]]
        assert part_extraction(score, "flute") is None

    def test_transpose_part(self):
        pitches = [[60, 64], [67]]
        result = transpose_part(pitches, 2)
        assert result == [[62, 66], [69]]


class TestTextureMapping:
    def test_monophonic(self):
        assert classify_texture(1, 0.5, 0.5) == TextureType.MONOPHONIC

    def test_homorhythmic(self):
        assert classify_texture(4, 0.95, 0.2) == TextureType.HOMORHYTHMIC

    def test_polyphonic(self):
        assert classify_texture(4, 0.3, 0.8) == TextureType.POLYPHONIC

    def test_density(self):
        d = texture_density(4, 24)
        assert d == pytest.approx(4 / 24)

    def test_density_zero_range(self):
        assert texture_density(4, 0) == 0.0

    def test_recommended_fast_tempo(self):
        t = recommended_texture(4, 180, 0.7)
        assert t == TextureType.HOMORHYTHMIC


class TestReharmonization:
    def test_tritone_sub(self):
        assert tritone_substitution(0) == 6
        assert tritone_substitution(7) == 1

    def test_backdoor(self):
        prog = backdoor_progression(0)
        assert len(prog) == 3
        assert prog[-1] == 0

    def test_modal_interchange_maj(self):
        options = modal_interchange(0, "maj7")
        assert len(options) > 0
        qualities = [q for _, q in options]
        assert "min7" in qualities

    def test_chromatic_mediant(self):
        options = chromatic_mediant(0, "maj7")
        assert len(options) == 4
        roots = [r for r, _ in options]
        assert 3 in roots and 4 in roots

    def test_reharmonize_melody_note(self):
        options = reharmonize_melody_note(60, 0, "maj7")
        assert len(options) > 0
        assert all(r != 0 or q != "maj7" for r, q in options)


class TestVoiceLeading:
    def test_parallel_fifths_detected(self):
        v1 = [60, 67, 74]
        v2 = [53, 60, 67]
        violations = check_parallel_motion(v1, v2)
        assert len(violations) == 2
        assert all(rule == VoiceLeadingRule.PARALLEL_FIFTHS for _, rule in violations)

    def test_parallel_octaves_detected(self):
        v1 = [60, 62, 64]
        v2 = [48, 50, 52]
        violations = check_parallel_motion(v1, v2)
        assert any(rule == VoiceLeadingRule.PARALLEL_OCTAVES for _, rule in violations)

    def test_no_violations_contrary(self):
        v1 = [60, 62, 64]
        v2 = [67, 65, 63]
        violations = check_parallel_motion(v1, v2)
        assert len(violations) == 0

    def test_voice_crossing(self):
        s = [60, 60, 60]
        a = [64, 64, 64]
        t = [55, 55, 55]
        b = [48, 48, 48]
        crossings = voice_crossing_check(s, a, t, b)
        assert len(crossings) == 3
        assert all(upper == "soprano" and lower == "alto" for _, upper, lower in crossings)

    def test_no_voice_crossing(self):
        s = [72, 72]
        a = [67, 67]
        t = [60, 60]
        b = [48, 48]
        assert voice_crossing_check(s, a, t, b) == []

    def test_smooth_voice_leading(self):
        from_chord = [60, 64, 67]
        to_pcs = [5, 9, 0]
        result = smooth_voice_leading(from_chord, to_pcs)
        assert len(result) == 3
        total_motion = sum(abs(a - b) for a, b in zip(from_chord, result))
        assert total_motion < 12

    def test_smooth_voice_leading_empty(self):
        assert smooth_voice_leading([], [0, 4, 7]) == [0, 4, 7]

    def test_cost_matrix(self):
        m = voice_leading_cost_matrix([60, 64], [65, 67])
        assert len(m) == 2
        assert len(m[0]) == 2
        assert m[0][0] == 5
        assert m[0][1] == 7

    def test_optimal_voicing(self):
        voicing = optimal_voicing([0, 4, 7], [60, 64, 67])
        assert len(voicing) == 3
        assert all(36 <= p <= 84 for p in voicing)
        for p, pc in zip(voicing, [0, 4, 7]):
            assert p % 12 == pc

    def test_total_motion_static(self):
        assert total_voice_motion([[60, 64, 67], [60, 64, 67]]) == 0.0

    def test_total_motion_step(self):
        motion = total_voice_motion([[60, 64, 67], [62, 65, 69]])
        assert motion == pytest.approx(2 + 1 + 2)


class TestArrangingStyles:
    def test_all_styles_have_templates(self):
        for style in ArrangingStyle:
            assert style in STYLE_TEMPLATES

    def test_big_band_voice_count(self):
        assert STYLE_TEMPLATES[ArrangingStyle.BIG_BAND].voice_count == 16

    def test_string_quartet_instruments(self):
        t = STYLE_TEMPLATES[ArrangingStyle.STRING_QUARTET]
        assert t.instruments == ["violin1", "violin2", "viola", "cello"]

    def test_assign_voices_melody_on_lead(self):
        melody = [60, 62, 64]
        chords = [0, 4, 7]
        result = assign_voices(melody, chords, ArrangingStyle.JAZZ_COMBO)
        assert result["lead"] == [60, 62, 64]
        assert len(result) > 1

    def test_assign_voices_no_harmony(self):
        result = assign_voices([60], [], ArrangingStyle.ROCK_BAND)
        assert result["vocal"] == [60]
        assert len(result) == 1

    def test_double_at_octave(self):
        assert double_at_octave([60, 64]) == [60, 64, 72, 76]

    def test_double_at_fifth(self):
        assert double_at_octave([60], interval=7) == [60, 67]

    def test_unison_passage(self):
        result = unison_passage([60, 62], ["violin1", "violin2", "viola"])
        assert len(result) == 3
        assert result["viola"] == [60, 62]

    def test_tutti_score_shape(self):
        melody = [60, 62]
        harmony = [[64, 67], [65, 69]]
        result = tutti_score(melody, harmony, ArrangingStyle.STRING_QUARTET)
        assert "violin1" in result
        assert result["violin1"] == [[60], [62]]
        assert len(result) == 4

    def test_tutti_score_harmony_distribution(self):
        melody = [72]
        harmony = [[0, 4, 7]]
        result = tutti_score(melody, harmony, ArrangingStyle.CHORAL_SATB)
        assert result["soprano"] == [[72]]
        assert len(result["alto"]) == 1


class TestPackageExports:
    def test_all_exports(self):
        import models.music_arrangement as ma
        assert hasattr(ma, "InstrumentRange")
        assert hasattr(ma, "ScoreFormat")
        assert hasattr(ma, "TextureType")
        assert hasattr(ma, "tritone_substitution")
        assert hasattr(ma, "VoiceLeadingRule")
        assert hasattr(ma, "ArrangingStyle")
