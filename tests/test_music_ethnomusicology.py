"""Tests for music_ethnomusicology: raga, maqam, gamelan, african rhythm."""
from __future__ import annotations
import pytest

from models.music_ethnomusicology.raga import (
    SWARAS, SWARA_SEMITONES, THAAT_SCALES, RAGA_CATALOG,
    Raga, thaat_to_ragas, raga_similarity, classify_raga,
)
from models.music_ethnomusicology.maqam import (
    MAQAM_CATALOG, Maqam, maqam_modulation_distance, classify_maqam,
)
from models.music_ethnomusicology.gamelan import (
    SLENDRO_INTERVALS, PELOG_INTERVALS, GAMELAN_SCALES, PATHET_MAP,
    GamelanScale, interlocking_pattern, colotomic_structure,
)
from models.music_ethnomusicology.african_rhythm import (
    STANDARD_PATTERNS, TimelinePattern, euclidean_rhythm,
    cross_rhythm, rotate_pattern, pattern_density, syncopation_index,
)


# ── Raga ──────────────────────────────────────────────────────────────────

class TestRaga:
    def test_swaras_count(self):
        assert len(SWARAS) == 12

    def test_swara_semitones_covers_all(self):
        assert set(SWARA_SEMITONES.values()) == set(range(12))

    def test_thaat_count(self):
        assert len(THAAT_SCALES) == 10

    def test_thaat_scales_are_7_note(self):
        for thaat, swaras in THAAT_SCALES.items():
            assert len(swaras) == 7, f"{thaat} has {len(swaras)} swaras"

    def test_catalog_size(self):
        assert len(RAGA_CATALOG) >= 20

    def test_raga_notes_auto_computed(self):
        yaman = RAGA_CATALOG["Yaman"]
        assert isinstance(yaman.notes, frozenset)
        assert len(yaman.notes) > 0

    def test_thaat_to_ragas_bilawal(self):
        ragas = thaat_to_ragas("Bilawal")
        names = [r.name for r in ragas]
        assert "Bilawal" in names

    def test_raga_similarity_self(self):
        yaman = RAGA_CATALOG["Yaman"]
        assert raga_similarity(yaman, yaman) == 1.0

    def test_raga_similarity_different(self):
        yaman = RAGA_CATALOG["Yaman"]
        bhairav = RAGA_CATALOG["Bhairav"]
        sim = raga_similarity(yaman, bhairav)
        assert 0.0 < sim < 1.0

    def test_classify_raga_exact(self):
        yaman = RAGA_CATALOG["Yaman"]
        matches = classify_raga(yaman.notes)
        best_name = matches[0][0].name
        assert best_name == "Yaman"

    def test_classify_raga_threshold(self):
        matches = classify_raga(frozenset({"Sa", "Re", "Ga"}))
        for raga, sim in matches:
            assert sim >= 0.3

    def test_raga_vadi_samvadi(self):
        yaman = RAGA_CATALOG["Yaman"]
        assert yaman.vadi in yaman.notes
        assert yaman.samvadi in yaman.notes


# ── Maqam ─────────────────────────────────────────────────────────────────

class TestMaqam:
    def test_catalog_size(self):
        assert len(MAQAM_CATALOG) >= 15

    def test_maqam_intervals(self):
        rast = MAQAM_CATALOG["Rast"]
        assert len(rast.intervals) == len(rast.jins_lower) + len(rast.jins_upper)

    def test_maqam_total_cents(self):
        rast = MAQAM_CATALOG["Rast"]
        assert rast.total_cents > 0

    def test_modulation_distance_self(self):
        rast = MAQAM_CATALOG["Rast"]
        assert maqam_modulation_distance(rast, rast) == 0.0

    def test_modulation_distance_different(self):
        rast = MAQAM_CATALOG["Rast"]
        hijaz = MAQAM_CATALOG["Hijaz"]
        d = maqam_modulation_distance(rast, hijaz)
        assert d > 0.0

    def test_modulation_distance_symmetry(self):
        rast = MAQAM_CATALOG["Rast"]
        bayati = MAQAM_CATALOG["Bayati"]
        assert abs(maqam_modulation_distance(rast, bayati) -
                   maqam_modulation_distance(bayati, rast)) < 1e-10

    def test_classify_maqam(self):
        rast = MAQAM_CATALOG["Rast"]
        matches = classify_maqam(list(rast.intervals))
        best = matches[0]
        assert best[0].name == "Rast"
        assert best[1] > 0.9

    def test_rast_octave(self):
        rast = MAQAM_CATALOG["Rast"]
        assert rast.total_cents == sum(rast.jins_lower) + sum(rast.jins_upper)


# ── Gamelan ───────────────────────────────────────────────────────────────

class TestGamelan:
    def test_slendro_intervals(self):
        assert len(SLENDRO_INTERVALS) == 5
        assert all(c == 240 for c in SLENDRO_INTERVALS)

    def test_pelog_intervals(self):
        assert len(PELOG_INTERVALS) == 7

    def test_gamelan_scales_count(self):
        assert len(GAMELAN_SCALES) >= 4

    def test_frequency_ratios(self):
        slendro = GAMELAN_SCALES["slendro"]
        ratios = slendro.frequency_ratios()
        assert ratios[0] == pytest.approx(220.0)
        assert all(r > 0 for r in ratios)
        assert ratios[-1] > ratios[0]

    def test_interlocking_pattern(self):
        melody = [1, 2, 3, 4, 5, 0]
        polos, sangsih = interlocking_pattern(melody, 6)
        assert len(polos) == len(melody)
        assert len(sangsih) == len(melody)

    def test_colotomic_lancaran(self):
        structure = colotomic_structure(16, "lancaran")
        assert "gong" in structure
        assert "kenong" in structure
        assert all(0 <= pos < 16 for pos in structure["gong"])

    def test_colotomic_ketawang(self):
        structure = colotomic_structure(32, "ketawang")
        assert "gong" in structure

    def test_pathet_map_non_empty(self):
        assert len(PATHET_MAP) > 0

    def test_n_tones(self):
        slendro = GAMELAN_SCALES["slendro"]
        assert slendro.n_tones == 5


# ── African Rhythm ────────────────────────────────────────────────────────

class TestAfricanRhythm:
    def test_standard_patterns_count(self):
        assert len(STANDARD_PATTERNS) >= 12

    def test_pattern_properties(self):
        clave = STANDARD_PATTERNS["son_clave_32"]
        assert clave.cycle_length == len(clave.pulses)
        assert clave.onset_count == sum(clave.pulses)

    def test_euclidean_rhythm_basic(self):
        pattern = euclidean_rhythm(3, 8)
        assert len(pattern) == 8
        assert sum(pattern) == 3

    def test_euclidean_rhythm_full(self):
        pattern = euclidean_rhythm(8, 8)
        assert all(p == 1 for p in pattern)

    def test_euclidean_rhythm_empty(self):
        pattern = euclidean_rhythm(0, 8)
        assert all(p == 0 for p in pattern)

    def test_cross_rhythm(self):
        a, b = cross_rhythm(3, 4, 12)
        assert len(a) == 12
        assert len(b) == 12
        assert sum(a) == 3
        assert sum(b) == 4

    def test_rotate_pattern(self):
        clave = STANDARD_PATTERNS["son_clave_32"]
        rotated = rotate_pattern(clave.pulses, 1)
        assert len(rotated) == clave.cycle_length
        assert sum(rotated) == clave.onset_count
        assert rotated != clave.pulses

    def test_rotate_full_cycle(self):
        clave = STANDARD_PATTERNS["son_clave_32"]
        rotated = rotate_pattern(clave.pulses, clave.cycle_length)
        assert rotated == clave.pulses

    def test_pattern_density(self):
        clave = STANDARD_PATTERNS["son_clave_32"]
        d = pattern_density(clave.pulses)
        assert 0.0 < d < 1.0

    def test_syncopation_index_range(self):
        for name, pat in STANDARD_PATTERNS.items():
            idx = syncopation_index(pat.pulses)
            assert idx >= 0.0, f"{name} has negative syncopation"

    def test_tresillo_density(self):
        tresillo = STANDARD_PATTERNS["tresillo"]
        d = pattern_density(tresillo.pulses)
        assert d == pytest.approx(3.0 / 8.0)
