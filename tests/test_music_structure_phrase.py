"""Tests for music_structure.phrase_structure — phrase hierarchy and grouping."""
import pytest
import numpy as np
from models.music_structure.phrase_structure import (
    Phrase, PhraseTree, detect_phrase_boundaries,
    group_into_periods, phrase_length_regularity,
    melodic_arch_score, build_phrase_tree,
    sentence_structure, hypermeter_level,
)


class TestPhrase:
    def test_length(self):
        p = Phrase(start_beat=0.0, end_beat=4.0)
        assert p.length == 4.0

    def test_level_default(self):
        p = Phrase(start_beat=0, end_beat=8)
        assert p.level == 0

    def test_children_default(self):
        p = Phrase(start_beat=0, end_beat=4)
        assert p.children == []


class TestPhraseTree:
    def test_depth_leaf(self):
        root = Phrase(start_beat=0, end_beat=8)
        tree = PhraseTree(root)
        assert tree.depth() == 0

    def test_depth_one_level(self):
        c1 = Phrase(start_beat=0, end_beat=4)
        c2 = Phrase(start_beat=4, end_beat=8)
        root = Phrase(start_beat=0, end_beat=8, level=1, children=[c1, c2])
        tree = PhraseTree(root)
        assert tree.depth() == 1

    def test_depth_two_levels(self):
        gc = Phrase(start_beat=0, end_beat=2)
        c = Phrase(start_beat=0, end_beat=4, children=[gc])
        root = Phrase(start_beat=0, end_beat=8, children=[c])
        tree = PhraseTree(root)
        assert tree.depth() == 2

    def test_leaf_phrases(self):
        c1 = Phrase(start_beat=0, end_beat=4)
        c2 = Phrase(start_beat=4, end_beat=8)
        root = Phrase(start_beat=0, end_beat=8, children=[c1, c2])
        tree = PhraseTree(root)
        leaves = tree.leaf_phrases()
        assert len(leaves) == 2
        assert c1 in leaves
        assert c2 in leaves

    def test_all_phrases(self):
        c1 = Phrase(start_beat=0, end_beat=4)
        c2 = Phrase(start_beat=4, end_beat=8)
        root = Phrase(start_beat=0, end_beat=8, children=[c1, c2])
        tree = PhraseTree(root)
        assert len(tree.all_phrases()) == 3


class TestDetectPhraseBoundaries:
    def test_simple_arch(self):
        contour = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0])
        rhythm = np.ones(13)
        boundaries = detect_phrase_boundaries(contour, rhythm)
        assert boundaries[0] == 0
        assert boundaries[-1] == 12

    def test_gap_detection(self):
        contour = np.ones(20)
        rhythm = np.ones(20)
        rhythm[10] = 5.0
        boundaries = detect_phrase_boundaries(contour, rhythm, gap_threshold=1.5)
        assert 10 in boundaries

    def test_minimum_length(self):
        contour = np.array([5, 4, 5, 4, 5, 4, 5, 4, 5, 4])
        rhythm = np.ones(10)
        boundaries = detect_phrase_boundaries(contour, rhythm)
        for i in range(len(boundaries) - 1):
            assert boundaries[i + 1] - boundaries[i] >= 1

    def test_flat_contour(self):
        contour = np.ones(15)
        rhythm = np.ones(15)
        boundaries = detect_phrase_boundaries(contour, rhythm)
        assert boundaries[0] == 0
        assert boundaries[-1] == 14

    def test_short_input(self):
        contour = np.array([1, 2, 3])
        rhythm = np.array([1, 1, 1])
        boundaries = detect_phrase_boundaries(contour, rhythm)
        assert 0 in boundaries


class TestGroupIntoPeriods:
    def test_even_count(self):
        phrases = [Phrase(0, 4), Phrase(4, 8), Phrase(8, 12), Phrase(12, 16)]
        pairs = group_into_periods(phrases)
        assert len(pairs) == 2
        assert pairs[0] == (phrases[0], phrases[1])
        assert pairs[1] == (phrases[2], phrases[3])

    def test_odd_count(self):
        phrases = [Phrase(0, 4), Phrase(4, 8), Phrase(8, 12)]
        pairs = group_into_periods(phrases)
        assert len(pairs) == 1

    def test_empty(self):
        assert group_into_periods([]) == []

    def test_single(self):
        assert group_into_periods([Phrase(0, 4)]) == []


class TestPhraseLengthRegularity:
    def test_perfectly_regular(self):
        phrases = [Phrase(0, 4), Phrase(4, 8), Phrase(8, 12)]
        reg = phrase_length_regularity(phrases)
        assert reg == pytest.approx(1.0)

    def test_irregular(self):
        phrases = [Phrase(0, 2), Phrase(2, 10), Phrase(10, 12)]
        reg = phrase_length_regularity(phrases)
        assert reg < 0.7

    def test_single(self):
        assert phrase_length_regularity([Phrase(0, 4)]) == 1.0

    def test_empty(self):
        assert phrase_length_regularity([]) == 1.0

    def test_range(self):
        phrases = [Phrase(0, 3), Phrase(3, 8), Phrase(8, 10)]
        reg = phrase_length_regularity(phrases)
        assert 0.0 <= reg <= 1.0


class TestMelodicArchScore:
    def test_perfect_arch(self):
        contour = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2])
        score = melodic_arch_score(contour)
        assert score > 0.5

    def test_flat(self):
        contour = np.ones(10)
        score = melodic_arch_score(contour)
        assert isinstance(score, float)

    def test_short(self):
        assert melodic_arch_score(np.array([1, 2])) == 0.0

    def test_range(self):
        contour = np.random.randn(20)
        score = melodic_arch_score(contour)
        assert 0.0 <= score <= 1.0

    def test_descending_low(self):
        contour = np.array([10, 8, 6, 4, 2, 0])
        score = melodic_arch_score(contour)
        assert score < 0.7


class TestBuildPhraseTree:
    def test_basic(self):
        boundaries = [0, 4, 8, 12]
        tree = build_phrase_tree(boundaries)
        assert isinstance(tree, PhraseTree)
        assert len(tree.root.children) == 3

    def test_with_levels(self):
        boundaries = [0, 4, 8, 16]
        levels = [0, 1, 0]
        tree = build_phrase_tree(boundaries, levels)
        assert tree.root.children[1].level == 1

    def test_empty_boundaries(self):
        tree = build_phrase_tree([])
        assert tree.root.start_beat == 0

    def test_single_boundary(self):
        tree = build_phrase_tree([0])
        assert tree.root.start_beat == 0

    def test_root_spans_all(self):
        boundaries = [0, 8, 16, 24]
        tree = build_phrase_tree(boundaries)
        assert tree.root.start_beat == 0
        assert tree.root.end_beat == 24


class TestSentenceStructure:
    def test_period(self):
        phrases = [Phrase(0, 4), Phrase(4, 8)]
        assert sentence_structure(phrases) == "period"

    def test_sentence(self):
        phrases = [Phrase(0, 4), Phrase(4, 8), Phrase(8, 16)]
        assert sentence_structure(phrases) == "sentence"

    def test_fragment(self):
        assert sentence_structure([Phrase(0, 4)]) == "fragment"
        assert sentence_structure([]) == "fragment"

    def test_through_composed(self):
        phrases = [Phrase(0, 3), Phrase(3, 7), Phrase(7, 15), Phrase(15, 20)]
        assert sentence_structure(phrases) == "through-composed"


class TestHypermeterLevel:
    def test_phrase_start(self):
        assert hypermeter_level(0.0) == 2

    def test_downbeat(self):
        assert hypermeter_level(4.0, beats_per_bar=4, bars_per_phrase=4) == 1

    def test_regular_beat(self):
        assert hypermeter_level(1.0) == 0

    def test_phrase_boundary(self):
        assert hypermeter_level(16.0, beats_per_bar=4, bars_per_phrase=4) == 2

    def test_custom_meter(self):
        assert hypermeter_level(0.0, beats_per_bar=3, bars_per_phrase=4) == 2
        assert hypermeter_level(3.0, beats_per_bar=3, bars_per_phrase=4) == 1
        assert hypermeter_level(1.0, beats_per_bar=3) == 0
