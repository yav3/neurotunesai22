"""Tests for Gjerdingen galant schemata recognition."""
import numpy as np
import pytest

from models.music_cognition.schema_theory import (
    SchemaTemplate, SCHEMA_CATALOG, ROMANESCA, PRINNER, FONTE, MONTE,
    DO_RE_MI, SOL_FA_MI, QUIESCENZA, FENAROLI,
    pitch_to_degree, pitches_to_degrees,
    match_schema, find_schemata,
    generate_schema_melody, generate_schema_bass,
    schema_complexity, schema_catalog_summary,
    MAJOR_SCALE,
)


class TestPitchToDegree:
    def test_c_is_1(self):
        assert pitch_to_degree(60, key_root=0) == 1  # C4

    def test_g_is_5(self):
        assert pitch_to_degree(67, key_root=0) == 5  # G4

    def test_transposed(self):
        assert pitch_to_degree(62, key_root=2) == 1  # D in D major

    def test_chromatic_nearest(self):
        deg = pitch_to_degree(61, key_root=0)  # C# → nearest is 1 (C) or 2 (D)
        assert deg in (1, 2)

    def test_batch(self):
        degs = pitches_to_degrees([60, 62, 64, 65, 67], key_root=0)
        assert degs == [1, 2, 3, 4, 5]


class TestSchemaTemplates:
    def test_romanesca(self):
        assert ROMANESCA.melody_degrees == (8, 7, 6, 5)
        assert len(ROMANESCA.harmonic_labels) == 4

    def test_prinner(self):
        assert PRINNER.melody_degrees == (6, 5, 4, 3)

    def test_catalog_count(self):
        assert len(SCHEMA_CATALOG) == 10

    def test_all_have_bass(self):
        for name, s in SCHEMA_CATALOG.items():
            assert len(s.bass_degrees) == len(s.melody_degrees), f"{name}: length mismatch"


class TestMatchSchema:
    def test_exact_romanesca(self):
        mel = [8, 7, 6, 5]
        bass = [1, 7, 6, 5]
        m = match_schema(mel, bass, ROMANESCA, start=0)
        assert m is not None
        assert m.combined_score == pytest.approx(1.0)

    def test_no_match_short(self):
        m = match_schema([1, 2], [3, 4], ROMANESCA, start=0)
        assert m is None

    def test_partial_match(self):
        mel = [8, 7, 6, 4]  # last note wrong
        bass = [1, 7, 6, 5]
        m = match_schema(mel, bass, ROMANESCA, start=0)
        assert m is not None
        assert m.combined_score < 1.0

    def test_threshold_filters(self):
        mel = [1, 1, 1, 1]
        bass = [1, 1, 1, 1]
        m = match_schema(mel, bass, FONTE, start=0)
        # Combined score too low
        if m is not None:
            assert m.combined_score >= 0.5


class TestFindSchemata:
    def test_finds_romanesca(self):
        melody = [60, 71, 69, 67]  # C5, B4, A4, G4 → 1,7,6,5 (octave equiv: 8,7,6,5)
        bass = [48, 59, 57, 55]    # C3, B3, A3, G3 → 1,7,6,5
        matches = find_schemata(melody, bass, key_root=0, threshold=0.5)
        names = [m.schema_name for m in matches]
        assert "Romanesca" in names

    def test_empty_input(self):
        assert find_schemata([], [], key_root=0) == []

    def test_sorted_by_score(self):
        melody = list(range(60, 72))
        bass = list(range(36, 48))
        matches = find_schemata(melody, bass, key_root=0, threshold=0.5)
        scores = [m.combined_score for m in matches]
        assert scores == sorted(scores, reverse=True)


class TestGeneration:
    def test_melody_length(self):
        mel = generate_schema_melody(ROMANESCA, key_root=0, octave=5)
        assert len(mel) == len(ROMANESCA.melody_degrees)

    def test_bass_length(self):
        bass = generate_schema_bass(PRINNER, key_root=0, octave=3)
        assert len(bass) == len(PRINNER.bass_degrees)

    def test_melody_in_range(self):
        mel = generate_schema_melody(DO_RE_MI, key_root=0, octave=5)
        assert all(48 <= p <= 84 for p in mel)

    def test_transposition(self):
        mel_c = generate_schema_melody(ROMANESCA, key_root=0, octave=5)
        mel_g = generate_schema_melody(ROMANESCA, key_root=7, octave=5)
        diffs = [g - c for g, c in zip(mel_g, mel_c)]
        assert all(d == diffs[0] for d in diffs)  # uniform transposition


class TestComplexity:
    def test_positive(self):
        for name, s in SCHEMA_CATALOG.items():
            c = schema_complexity(s)
            assert c > 0, f"{name} has zero complexity"

    def test_catalog_summary(self):
        summary = schema_catalog_summary()
        assert len(summary) == len(SCHEMA_CATALOG)
        assert all(v > 0 for v in summary.values())

    def test_monte_vs_do_re_mi(self):
        cm = schema_complexity(MONTE)
        cd = schema_complexity(DO_RE_MI)
        # Monte has wider range + more harmonic variety
        assert cm > cd
