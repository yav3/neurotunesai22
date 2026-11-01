"""Tests for harmonic similarity, rhythm fingerprint, and public domain modules."""
import numpy as np
import pytest

from models.music_copyright.harmonic_similarity import (
    chord_to_pc,
    chords_to_intervals,
    chords_to_roman,
    harmonic_sequence_similarity,
    transposition_invariant_harmonic_similarity,
    functional_reduction,
    functional_similarity,
    chord_ngram_overlap,
    ngram_significance,
    sliding_harmonic_compare,
)
from models.music_copyright.rhythm_fingerprint import (
    onsets_to_grid,
    grid_hamming_similarity,
    metric_weight_vector,
    weighted_onset_similarity,
    extract_motifs,
    motif_jaccard,
    ioi_vector,
    ioi_ratio_vector,
    ioi_ratio_similarity,
    cross_rhythm_correlation,
    compare_rhythms,
)
from models.music_copyright.public_domain import (
    Jurisdiction,
    WorkType,
    CopyrightQuery,
    determine_copyright_status,
    is_public_domain,
)


# ── Harmonic similarity ───────────────────────────────────────────

class TestChordParsing:
    def test_chord_to_pc_c_major(self):
        pc, q = chord_to_pc("C", "maj")
        assert pc == 0
        assert q == "M"

    def test_chord_to_pc_flat(self):
        pc, q = chord_to_pc("Bb", "min")
        assert pc == 10
        assert q == "m"

    def test_chords_to_intervals(self):
        roots = [0, 7, 5, 0]  # C G F C
        ivs = chords_to_intervals(roots)
        assert ivs == [7, 10, 7]

    def test_chords_to_roman(self):
        roots = [0, 7, 5, 0]
        roman = chords_to_roman(roots, tonic=0)
        assert roman == ["I", "V", "IV", "I"]


class TestHarmonicSimilarity:
    def test_identical_sequences(self):
        seq = [(0, "M"), (7, "M"), (5, "M"), (0, "M")]
        assert harmonic_sequence_similarity(seq, seq) == 1.0

    def test_empty(self):
        assert harmonic_sequence_similarity([], []) == 1.0

    def test_completely_different(self):
        a = [(0, "M"), (2, "m"), (4, "M")]
        b = [(1, "d"), (3, "A"), (5, "7")]
        sim = harmonic_sequence_similarity(a, b)
        assert sim < 0.5

    def test_quality_weight(self):
        a = [(0, "M"), (7, "M")]
        b = [(0, "m"), (7, "m")]
        sim_low = harmonic_sequence_similarity(a, b, quality_weight=0.1)
        sim_high = harmonic_sequence_similarity(a, b, quality_weight=0.9)
        assert sim_low > sim_high

    def test_transposition_invariant(self):
        a = [(0, "M"), (7, "M"), (5, "M")]       # C G F in C
        b = [(2, "M"), (9, "M"), (7, "M")]       # D A G in D — same intervals
        sim, shift = transposition_invariant_harmonic_similarity(a, b)
        assert sim > 0.9
        assert shift == 10  # shift b up 10 semitones ≡ down 2 to align


class TestFunctionalSimilarity:
    def test_identical_functions(self):
        a = ["I", "IV", "V", "I"]
        b = ["I", "IV", "V", "I"]
        assert functional_similarity(a, b) == 1.0

    def test_substitution_within_class(self):
        a = ["I", "IV", "V", "I"]
        b = ["III", "II", "VII", "VI"]  # tonic, subdominant, dominant, tonic
        assert functional_similarity(a, b) == 1.0

    def test_reduction(self):
        reduced = functional_reduction(["I", "IV", "V", "VI"])
        assert reduced == ["tonic", "subdominant", "dominant", "tonic"]


class TestNgramOverlap:
    def test_identical_overlap(self):
        seq = [(0, "M"), (7, "M"), (5, "M"), (0, "M")]
        assert chord_ngram_overlap(seq, seq, n=2) == 1.0

    def test_no_overlap(self):
        a = [(0, "M"), (2, "M"), (4, "M")]
        b = [(1, "m"), (3, "m"), (5, "m")]
        assert chord_ngram_overlap(a, b, n=2) == 0.0

    def test_significance_random_is_high(self):
        p = ngram_significance(0.0, 20, 20, n=3, vocab_size=48)
        assert p > 0.5

    def test_significance_identical_is_low(self):
        p = ngram_significance(1.0, 20, 20, n=3, vocab_size=48)
        assert p < 0.01


class TestSlidingHarmonicCompare:
    def test_finds_match(self):
        prog = [(0, "M"), (7, "M"), (5, "M"), (0, "M"), (2, "m"), (7, "M")]
        sub = [(0, "M"), (7, "M"), (5, "M"), (0, "M")]
        matches = sliding_harmonic_compare(prog, sub, window=4, threshold=0.9)
        assert len(matches) >= 1
        assert matches[0].offset_a == 0

    def test_empty_short(self):
        assert sliding_harmonic_compare([(0, "M")], [(0, "M")], window=4) == []


# ── Rhythm fingerprint ────────────────────────────────────────────

class TestOnsetGrid:
    def test_basic_grid(self):
        onsets = np.array([0.0, 0.5, 1.0, 1.5])
        grid = onsets_to_grid(onsets, grid_resolution=4, beats=4, beat_duration=0.5)
        assert grid[0] == 1
        assert grid[4] == 1
        assert len(grid) == 16

    def test_empty_onsets(self):
        grid = onsets_to_grid(np.array([]), grid_resolution=4, beats=4)
        assert np.sum(grid) == 0


class TestGridSimilarity:
    def test_identical(self):
        g = np.array([1, 0, 0, 1, 0, 0, 1, 0])
        assert grid_hamming_similarity(g, g) == 1.0

    def test_opposite(self):
        a = np.array([1, 0, 1, 0])
        b = np.array([0, 1, 0, 1])
        assert grid_hamming_similarity(a, b) == 0.0

    def test_weighted_similarity(self):
        a = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        b = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        sim = weighted_onset_similarity(a, b)
        assert sim == pytest.approx(1.0)


class TestMetricWeight:
    def test_downbeat_heaviest(self):
        mw = metric_weight_vector(grid_resolution=4, beats=4)
        assert mw[0] == 1.0  # downbeat of bar


class TestMotifs:
    def test_extract_and_jaccard(self):
        g = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        motifs = extract_motifs(g, min_len=4, max_len=4)
        assert len(motifs) == 5
        j = motif_jaccard(motifs, motifs)
        assert j == 1.0


class TestIOI:
    def test_ioi_vector(self):
        onsets = np.array([0.0, 0.5, 1.0, 1.5])
        ioi = ioi_vector(onsets)
        np.testing.assert_array_almost_equal(ioi, [0.5, 0.5, 0.5])

    def test_ratio_straight(self):
        onsets = np.array([0.0, 0.5, 1.0, 1.5])
        ratios = ioi_ratio_vector(onsets)
        np.testing.assert_array_almost_equal(ratios, [1.0, 1.0])

    def test_ratio_swing(self):
        onsets = np.array([0.0, 0.667, 1.0, 1.667, 2.0])
        ratios = ioi_ratio_vector(onsets)
        assert ratios[0] > 1.5  # long-short

    def test_ratio_similarity_identical(self):
        r = np.array([1.0, 1.0, 1.0])
        assert ioi_ratio_similarity(r, r) == pytest.approx(1.0)


class TestCrossRhythm:
    def test_identical_grids(self):
        g = np.array([1, 0, 0, 1, 0, 0, 1, 0])
        xc = cross_rhythm_correlation(g, g)
        assert xc > 0.9

    def test_shifted_grids(self):
        a = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        b = np.array([0, 0, 1, 0, 0, 0, 1, 0])  # shifted by 2
        xc = cross_rhythm_correlation(a, b)
        assert xc > 0.9  # circular correlation finds the alignment


class TestCompareRhythms:
    def test_identical_rhythms(self):
        onsets = np.array([0.0, 0.5, 1.0, 1.5])
        result = compare_rhythms(onsets, onsets)
        assert result.composite > 0.9
        assert result.grid_similarity == 1.0


# ── Public domain ─────────────────────────────────────────────────

class TestUSPublicDomain:
    def test_pre_1929(self):
        assert is_public_domain(publication_year=1920, jurisdiction="us") is True

    def test_post_1978_dead_author(self):
        assert is_public_domain(
            publication_year=1980, author_death_year=1950, jurisdiction="us"
        ) is True  # 1950 + 70 = 2020 < 2026

    def test_post_1978_recent_author(self):
        assert is_public_domain(
            publication_year=1990, author_death_year=2000, jurisdiction="us"
        ) is False  # 2000 + 70 = 2070

    def test_1929_1977_renewed(self):
        q = CopyrightQuery(
            publication_year=1960, jurisdiction=Jurisdiction.US,
            registered_us=True, renewed_us=True,
        )
        status = determine_copyright_status(q)
        assert status.is_public_domain is False  # 1960+95=2055

    def test_1929_1977_not_renewed_explicit(self):
        q = CopyrightQuery(
            publication_year=1960, jurisdiction=Jurisdiction.US,
            registered_us=True, renewed_us=False,
        )
        status = determine_copyright_status(q)
        assert status.is_public_domain is True  # 1960+28=1988 < 2026

    def test_work_for_hire(self):
        q = CopyrightQuery(
            publication_year=1920, jurisdiction=Jurisdiction.US,
            work_type=WorkType.WORK_FOR_HIRE,
        )
        status = determine_copyright_status(q)
        assert status.is_public_domain is True  # 1920+95 = 2015 < 2026


class TestEUPublicDomain:
    def test_life_plus_70(self):
        assert is_public_domain(
            author_death_year=1950, jurisdiction="eu"
        ) is True  # 1950+70=2020

    def test_recent_death(self):
        assert is_public_domain(
            author_death_year=1990, jurisdiction="eu"
        ) is False  # 1990+70=2060

    def test_sound_recording(self):
        q = CopyrightQuery(
            publication_year=1950, jurisdiction=Jurisdiction.EU,
            work_type=WorkType.SOUND_RECORDING,
        )
        status = determine_copyright_status(q)
        assert status.is_public_domain is True  # 1950+70=2020


class TestBernePublicDomain:
    def test_life_plus_50(self):
        assert is_public_domain(
            author_death_year=1970, jurisdiction="berne_default"
        ) is True  # 1970+50=2020

    def test_recent(self):
        assert is_public_domain(
            author_death_year=1990, jurisdiction="berne_default"
        ) is False  # 1990+50=2040


class TestJapanCanada:
    def test_japan_life_plus_70(self):
        assert is_public_domain(
            author_death_year=1950, jurisdiction="japan"
        ) is True

    def test_canada_life_plus_70(self):
        assert is_public_domain(
            author_death_year=1950, jurisdiction="canada"
        ) is True

    def test_canada_recent(self):
        assert is_public_domain(
            author_death_year=1990, jurisdiction="canada"
        ) is False


class TestCopyrightStatusDetails:
    def test_rule_and_confidence(self):
        q = CopyrightQuery(publication_year=1900, jurisdiction=Jurisdiction.US)
        status = determine_copyright_status(q)
        assert status.rule_applied == "us_pre_1929"
        assert status.confidence == 1.0
        assert status.expiration_year == 1900

    def test_insufficient_data(self):
        q = CopyrightQuery(jurisdiction=Jurisdiction.US)
        status = determine_copyright_status(q)
        assert status.confidence < 0.5
