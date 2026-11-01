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


class TestPackageExports:
    def test_all_exports(self):
        import models.music_arrangement as ma
        assert hasattr(ma, "InstrumentRange")
        assert hasattr(ma, "ScoreFormat")
        assert hasattr(ma, "TextureType")
        assert hasattr(ma, "tritone_substitution")
