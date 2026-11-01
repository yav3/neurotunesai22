"""Tests for music_copyright: melody comparison, originality scoring, plagiarism detection."""
from __future__ import annotations
import pytest

from models.music_copyright.melody_comparator import (
    pitch_sequence_similarity, rhythm_sequence_similarity,
    interval_contour_similarity, transposition_invariant_similarity,
)
from models.music_copyright.originality import (
    n_gram_novelty, OriginalityScorer,
)
from models.music_copyright.plagiarism_detector import (
    PlagiarismReport, PlagiarismDetector,
)


# ── Melody Comparator ────────────────────────────────────────────────────

class TestMelodyComparator:
    def test_pitch_similarity_identical(self):
        seq = [60, 62, 64, 65, 67]
        assert pitch_sequence_similarity(seq, seq) == pytest.approx(1.0)

    def test_pitch_similarity_octave_equivalent(self):
        a = [60, 62, 64, 65, 67]
        b = [72, 74, 76, 77, 79]
        sim = pitch_sequence_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_pitch_similarity_all_different_pc(self):
        a = [60, 62, 64, 65, 67]
        b = [61, 63, 66, 68, 70]
        sim = pitch_sequence_similarity(a, b)
        assert sim < 1.0

    def test_pitch_similarity_empty(self):
        assert pitch_sequence_similarity([], []) == pytest.approx(1.0)

    def test_pitch_similarity_one_empty(self):
        assert pitch_sequence_similarity([60], []) == pytest.approx(0.0)

    def test_rhythm_similarity_identical(self):
        r = [1.0, 0.5, 0.5, 1.0]
        assert rhythm_sequence_similarity(r, r) == pytest.approx(1.0)

    def test_rhythm_similarity_different(self):
        a = [1.0, 1.0, 1.0, 1.0]
        b = [0.5, 0.5, 0.5, 0.5]
        sim = rhythm_sequence_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_contour_similarity_identical(self):
        seq = [60, 62, 64, 62, 60]
        assert interval_contour_similarity(seq, seq) == pytest.approx(1.0)

    def test_contour_similarity_inverted(self):
        a = [60, 62, 64, 66, 68]
        b = [68, 66, 64, 62, 60]
        sim = interval_contour_similarity(a, b)
        assert sim < 0.5

    def test_transposition_invariant_identical(self):
        a = [60, 62, 64, 65, 67]
        b = [72, 74, 76, 77, 79]
        sim = transposition_invariant_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_transposition_invariant_different(self):
        a = [60, 62, 64, 65, 67]
        b = [60, 63, 65, 68, 70]
        sim = transposition_invariant_similarity(a, b)
        assert sim < 1.0

    def test_contour_short(self):
        assert interval_contour_similarity([60], [60]) == pytest.approx(1.0)


# ── Originality ──────────────────────────────────────────────────────────

class TestOriginality:
    def test_ngram_novelty_all_novel(self):
        seq = [1, 2, 3, 4, 5]
        corpus = [[10, 11, 12, 13, 14]]
        novelty = n_gram_novelty(seq, corpus, n=3)
        assert novelty == pytest.approx(1.0)

    def test_ngram_novelty_none_novel(self):
        seq = [1, 2, 3, 4, 5]
        corpus = [[1, 2, 3, 4, 5]]
        novelty = n_gram_novelty(seq, corpus, n=3)
        assert novelty == pytest.approx(0.0)

    def test_ngram_novelty_partial(self):
        seq = [1, 2, 3, 4, 5, 6, 7]
        corpus = [[1, 2, 3, 10, 11, 12, 13]]
        novelty = n_gram_novelty(seq, corpus, n=3)
        assert 0.0 < novelty < 1.0

    def test_originality_scorer_melodic(self):
        scorer = OriginalityScorer()
        melody = [60, 62, 64, 65, 67, 69, 71, 72]
        corpus = [[60, 64, 67, 72, 76, 79, 84]]
        score = scorer.melodic_originality(melody, corpus)
        assert 0.0 <= score <= 1.0

    def test_originality_scorer_harmonic(self):
        scorer = OriginalityScorer()
        chords = ["Cmaj", "Fmaj", "Gmaj", "Cmaj"]
        corpus = [["Cmaj", "Gmaj", "Amin", "Fmaj"]]
        score = scorer.harmonic_originality(chords, corpus)
        assert 0.0 <= score <= 1.0

    def test_originality_scorer_rhythmic(self):
        scorer = OriginalityScorer()
        rhythms = [1.0, 0.5, 0.5, 1.0, 0.25, 0.25, 0.5]
        corpus = [[1.0, 1.0, 1.0, 1.0]]
        score = scorer.rhythmic_originality(rhythms, corpus)
        assert 0.0 <= score <= 1.0

    def test_composite_originality(self):
        scorer = OriginalityScorer()
        melody = [60, 62, 64, 65, 67, 69, 71, 72]
        chords = ["Cmaj", "Fmaj", "Gmaj", "Cmaj"]
        rhythms = [1.0, 0.5, 0.5, 1.0]
        m_corpus = [[48, 50, 52, 53, 55]]
        c_corpus = [["Dmin", "Gmaj", "Cmaj"]]
        r_corpus = [[0.5, 0.5, 0.5, 0.5]]
        score = scorer.composite_originality(
            melody, chords, rhythms, m_corpus, c_corpus, r_corpus
        )
        assert 0.0 <= score <= 1.0


# ── Plagiarism Detector ──────────────────────────────────────────────────

class TestPlagiarismDetector:
    def test_compare_identical(self):
        det = PlagiarismDetector(window_size=4, threshold=0.8)
        melody = [60, 62, 64, 65, 67, 69, 71, 72]
        report = det.compare_melodies(melody, melody)
        assert report.similarity_score >= 0.8
        assert report.confidence > 0.5

    def test_compare_transposed(self):
        det = PlagiarismDetector(window_size=4, threshold=0.8)
        a = [60, 62, 64, 65, 67, 69, 71, 72]
        b = [x + 5 for x in a]
        report = det.compare_melodies(a, b)
        assert report.similarity_score >= 0.8

    def test_compare_different(self):
        det = PlagiarismDetector(window_size=4, threshold=0.8)
        a = [60, 62, 64, 65, 67, 69, 71, 72]
        b = [60, 63, 67, 70, 74, 77, 81, 84]
        report = det.compare_melodies(a, b)
        assert report.similarity_score < 0.9

    def test_scan_corpus(self):
        det = PlagiarismDetector(window_size=4, threshold=0.3)
        query = [60, 62, 64, 65, 67]
        corpus = [
            [60, 62, 64, 65, 67, 69, 71],
            [48, 50, 52, 53, 55, 57, 59],
            [72, 74, 76, 77, 79, 81, 83],
        ]
        results = det.scan_corpus(query, corpus)
        assert len(results) >= 1
        assert results[0][1].similarity_score >= results[-1][1].similarity_score

    def test_best_transposition(self):
        det = PlagiarismDetector()
        a = [60, 62, 64, 65, 67]
        b = [65, 67, 69, 70, 72]
        report = det.compare_melodies(a, b)
        assert report.transposition_offset == 5

    def test_short_melody(self):
        det = PlagiarismDetector(window_size=4, threshold=0.8)
        a = [60, 62]
        b = [60, 62]
        report = det.compare_melodies(a, b)
        assert isinstance(report, PlagiarismReport)
