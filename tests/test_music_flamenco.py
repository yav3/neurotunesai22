"""Tests for flamenco: compás, palos, Phrygian modes, cadence detection."""
import pytest
import numpy as np

from models.music_ethnomusicology.flamenco import (
    Palo, PALO_CATALOG, COMPAS_PATTERNS,
    PHRYGIAN_INTERVALS, PHRYGIAN_DOMINANT_INTERVALS,
    is_phrygian, is_phrygian_dominant, andalusian_cadence_score,
    classify_palo, compas_correlation, rasgueado_density,
)


class TestPaloCatalog:
    def test_count(self):
        assert len(PALO_CATALOG) == 12

    def test_bulerias_12_beats(self):
        assert PALO_CATALOG["bulerias"].compas_beats == 12

    def test_tangos_4_beats(self):
        assert PALO_CATALOG["tangos"].compas_beats == 4

    def test_all_have_tempo_range(self):
        for name, palo in PALO_CATALOG.items():
            lo, hi = palo.typical_tempo_range
            assert lo < hi, f"{name} invalid tempo range"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            PALO_CATALOG["solea"].name = "x"


class TestCompasPatterns:
    def test_bulerias_12_positions(self):
        assert len(COMPAS_PATTERNS["bulerias"]) == 12

    def test_binary_values(self):
        for name, pattern in COMPAS_PATTERNS.items():
            assert all(v in (0, 1) for v in pattern), f"{name} has non-binary"

    def test_all_start_on_beat(self):
        for name, pattern in COMPAS_PATTERNS.items():
            assert pattern[0] == 1, f"{name} doesn't start on downbeat"


class TestPhrygianDetection:
    def test_phrygian_scale(self):
        assert is_phrygian([0, 1, 3, 5, 7, 8, 10])

    def test_non_phrygian(self):
        assert not is_phrygian([0, 2, 4, 5, 7, 9, 11])

    def test_phrygian_dominant(self):
        assert is_phrygian_dominant([0, 1, 4, 5, 7, 8, 10])

    def test_phrygian_subset(self):
        assert is_phrygian([0, 1, 3, 7])

    def test_transposed_phrygian(self):
        assert is_phrygian([4, 5, 7, 9, 11, 0, 2], root=4)


class TestAndalusianCadence:
    def test_perfect_cadence(self):
        roots = [5, 3, 1, 0]
        score = andalusian_cadence_score(roots)
        assert score == 1.0

    def test_no_match(self):
        roots = [0, 0, 0, 0]
        score = andalusian_cadence_score(roots)
        assert score < 0.5

    def test_too_short(self):
        assert andalusian_cadence_score([0, 5]) == 0.0


class TestClassifyPalo:
    def test_bulerias_fast_12(self):
        results = classify_palo(200, "12/8", "phrygian_dominant")
        assert len(results) > 0
        names = [n for n, _ in results]
        assert "bulerias" in names

    def test_tangos_4_4(self):
        results = classify_palo(120, "4/4", "phrygian")
        names = [n for n, _ in results]
        assert "tangos" in names

    def test_sorted_descending(self):
        results = classify_palo(100, "12/8", "phrygian")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestCompasCorrelation:
    def test_perfect_alignment(self):
        beat_dur = 0.5
        pattern = COMPAS_PATTERNS["bulerias"]
        onsets = np.array([i * beat_dur for i, v in enumerate(pattern) if v == 1])
        corr = compas_correlation(onsets, "bulerias", beat_dur)
        assert corr > 0.8

    def test_unknown_palo(self):
        assert compas_correlation(np.array([0.0, 0.5]), "nonexistent", 0.5) == 0.0

    def test_empty_onsets(self):
        assert compas_correlation(np.array([]), "bulerias", 0.5) == 0.0


class TestRasgueadoDensity:
    def test_basic(self):
        onsets = np.array([0.1, 0.12, 0.14, 0.16, 0.18, 1.0, 2.0])
        densities = rasgueado_density(onsets, window=0.5)
        assert densities[0] > densities[-1]

    def test_short_input(self):
        result = rasgueado_density(np.array([1.0]))
        assert len(result) == 0


class TestPackageExports:
    def test_import_from_package(self):
        from models.music_ethnomusicology import (
            Palo, PALO_CATALOG, COMPAS_PATTERNS,
            is_phrygian, classify_palo,
        )
        assert callable(classify_palo)
        assert len(PALO_CATALOG) > 0
