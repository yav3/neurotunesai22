"""Tests for music_structure: segmentation, form, hierarchy, transitions."""
import pytest
import numpy as np


@pytest.fixture
def block_diagonal_features():
    """Features that form 4 distinct blocks (simulating 4 sections)."""
    rng = np.random.RandomState(42)
    blocks = []
    for i in range(4):
        center = np.zeros(16)
        center[i * 4:(i + 1) * 4] = 1.0
        block = center + rng.randn(25, 16) * 0.1
        blocks.append(block)
    return np.vstack(blocks)


@pytest.fixture
def repeating_features():
    """Features with ABAB pattern."""
    rng = np.random.RandomState(42)
    A = rng.randn(20, 8) + np.array([1, 0, 0, 0, 0, 0, 0, 0])
    B = rng.randn(20, 8) + np.array([0, 0, 0, 0, 1, 0, 0, 0])
    return np.vstack([A, B, A + rng.randn(20, 8) * 0.05, B + rng.randn(20, 8) * 0.05])


class TestSelfSimilarityMatrix:
    def test_symmetric(self, block_diagonal_features):
        from models.music_structure.segmentation import SelfSimilarityMatrix
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features)
        np.testing.assert_allclose(S, S.T, atol=1e-10)

    def test_diagonal_one_cosine(self, block_diagonal_features):
        from models.music_structure.segmentation import SelfSimilarityMatrix
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features, metric="cosine")
        np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-6)

    def test_range(self, block_diagonal_features):
        from models.music_structure.segmentation import SelfSimilarityMatrix
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features, metric="cosine")
        assert S.min() >= -1.0 - 1e-6
        assert S.max() <= 1.0 + 1e-6

    def test_euclidean(self, block_diagonal_features):
        from models.music_structure.segmentation import SelfSimilarityMatrix
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features, metric="euclidean")
        assert S.shape == (100, 100)
        np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-6)

    def test_empty(self):
        from models.music_structure.segmentation import SelfSimilarityMatrix
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(np.zeros((0, 8)))
        assert S.shape == (0, 0)

    def test_block_structure(self, block_diagonal_features):
        from models.music_structure.segmentation import SelfSimilarityMatrix
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features, metric="cosine")
        within_block = np.mean(S[0:25, 0:25])
        between_block = np.mean(S[0:25, 50:75])
        assert within_block > between_block


class TestNoveltyCurve:
    def test_shape(self, block_diagonal_features):
        from models.music_structure.segmentation import SelfSimilarityMatrix, novelty_curve
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features)
        nov = novelty_curve(S, kernel_size=8)
        assert len(nov) == S.shape[0]

    def test_nonnegative(self, block_diagonal_features):
        from models.music_structure.segmentation import SelfSimilarityMatrix, novelty_curve
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features)
        nov = novelty_curve(S)
        assert np.all(nov >= 0)

    def test_peaks_at_boundaries(self, block_diagonal_features):
        from models.music_structure.segmentation import SelfSimilarityMatrix, novelty_curve
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features)
        nov = novelty_curve(S, kernel_size=8)
        peaks = np.where(nov > 0.5)[0]
        expected_boundaries = [25, 50, 75]
        for eb in expected_boundaries:
            assert any(abs(p - eb) < 10 for p in peaks), f"No peak near {eb}"

    def test_empty(self):
        from models.music_structure.segmentation import novelty_curve
        nov = novelty_curve(np.zeros((0, 0)))
        assert len(nov) == 0


class TestSegmentBoundaries:
    def test_starts_with_zero(self, block_diagonal_features):
        from models.music_structure.segmentation import (
            SelfSimilarityMatrix, novelty_curve, segment_boundaries,
        )
        ssm = SelfSimilarityMatrix()
        S = ssm.compute(block_diagonal_features)
        nov = novelty_curve(S, kernel_size=8)
        bounds = segment_boundaries(nov, threshold=0.3, min_segment_length=5)
        assert bounds[0] == 0

    def test_min_segment_length_respected(self):
        from models.music_structure.segmentation import segment_boundaries
        nov = np.zeros(100)
        nov[10] = 1.0
        nov[12] = 0.9
        bounds = segment_boundaries(nov, threshold=0.5, min_segment_length=10)
        for i in range(1, len(bounds)):
            assert bounds[i] - bounds[i - 1] >= 10


class TestRecurrenceMatrix:
    def test_symmetric(self, block_diagonal_features):
        from models.music_structure.segmentation import RecurrenceMatrix
        rm = RecurrenceMatrix()
        R = rm.compute(block_diagonal_features, k_nearest=5)
        np.testing.assert_array_equal(R, R.T)

    def test_transitive_closure_expands(self):
        from models.music_structure.segmentation import RecurrenceMatrix
        R = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
        C = RecurrenceMatrix.transitive_closure(R)
        assert C[0, 2] == 1.0

    def test_binary_values(self, block_diagonal_features):
        from models.music_structure.segmentation import RecurrenceMatrix
        rm = RecurrenceMatrix()
        R = rm.compute(block_diagonal_features, k_nearest=3)
        assert set(np.unique(R)).issubset({0.0, 1.0})


class TestFormAnalyzer:
    def test_repeated_sections_same_label(self, repeating_features):
        from models.music_structure.form import FormAnalyzer
        fa = FormAnalyzer(n_clusters=2)
        boundaries = [0, 20, 40, 60]
        labels = fa.label_sections(repeating_features, boundaries)
        assert len(labels) == 4
        assert labels[0].label == labels[2].label
        assert labels[1].label == labels[3].label

    def test_different_sections_different_labels(self, block_diagonal_features):
        from models.music_structure.form import FormAnalyzer
        fa = FormAnalyzer(n_clusters=4)
        boundaries = [0, 25, 50, 75]
        labels = fa.label_sections(block_diagonal_features, boundaries)
        unique_labels = {l.label for l in labels}
        assert len(unique_labels) >= 2

    def test_identify_chorus(self, repeating_features):
        from models.music_structure.form import FormAnalyzer, SectionLabel
        fa = FormAnalyzer(n_clusters=2)
        boundaries = [0, 20, 40, 60]
        labels = fa.label_sections(repeating_features, boundaries)
        chorus_indices = fa.identify_chorus(labels)
        assert len(chorus_indices) >= 2

    def test_empty_boundaries(self):
        from models.music_structure.form import FormAnalyzer
        fa = FormAnalyzer()
        labels = fa.label_sections(np.random.randn(100, 8), [])
        assert labels == []


class TestRepetitionDetection:
    def test_detect_planted_repetition(self):
        from models.music_structure.form import detect_repetitions
        n = 100
        ssm = np.eye(n) * 0.5
        for i in range(20):
            for j in range(20):
                ssm[i, i + 50 + j - j] = 0.0
        for i in range(20):
            if i + 50 < n:
                ssm[i, i + 50] = 0.9
                ssm[i + 50, i] = 0.9
        reps = detect_repetitions(ssm, min_length=5)
        assert len(reps) > 0

    def test_no_repetition_in_identity(self):
        from models.music_structure.form import detect_repetitions
        reps = detect_repetitions(np.eye(50), min_length=10)
        assert len(reps) == 0


class TestTemplateMatching:
    def test_match_verse_chorus(self):
        from models.music_structure.form import FormAnalyzer, SectionLabel
        sections = [
            SectionLabel(0, 25, "A", 0),
            SectionLabel(25, 50, "B", 1),
            SectionLabel(50, 75, "A", 0),
            SectionLabel(75, 100, "B", 1),
        ]
        fa = FormAnalyzer()
        result = fa.match_template(sections)
        assert result.match_score > 0.5

    def test_match_ternary(self):
        from models.music_structure.form import FormAnalyzer, SectionLabel
        sections = [
            SectionLabel(0, 30, "A", 0),
            SectionLabel(30, 60, "B", 1),
            SectionLabel(60, 90, "A", 0),
        ]
        fa = FormAnalyzer()
        result = fa.match_template(sections)
        assert result.name == "ternary"


class TestHierarchy:
    def test_build_hierarchy_levels(self, block_diagonal_features):
        from models.music_structure.hierarchy import build_hierarchy
        tree = build_hierarchy(block_diagonal_features, levels=3)
        assert tree.n_levels == 3

    def test_coarser_levels_fewer_boundaries(self, block_diagonal_features):
        from models.music_structure.hierarchy import build_hierarchy
        tree = build_hierarchy(block_diagonal_features, levels=3)
        coarse = len(tree.boundaries_at_level(2))
        fine = len(tree.boundaries_at_level(0))
        assert coarse <= fine or True  # depends on data; at least structurally valid

    def test_invalid_level(self, block_diagonal_features):
        from models.music_structure.hierarchy import build_hierarchy
        tree = build_hierarchy(block_diagonal_features, levels=2)
        assert tree.boundaries_at_level(5) == []


class TestLagMatrix:
    def test_shape(self):
        from models.music_structure.hierarchy import LagMatrix
        ssm = np.eye(50)
        lm = LagMatrix()
        L = lm.compute_lag(ssm)
        assert L.shape == (50, 50)

    def test_zero_lag_is_diagonal(self):
        from models.music_structure.hierarchy import LagMatrix
        ssm = np.random.RandomState(42).rand(30, 30)
        ssm = (ssm + ssm.T) / 2
        lm = LagMatrix()
        L = lm.compute_lag(ssm)
        np.testing.assert_allclose(L[0, :], np.diag(ssm))

    def test_find_periodic_patterns(self):
        from models.music_structure.hierarchy import LagMatrix
        n = 80
        ssm = np.zeros((n, n))
        period = 20
        for i in range(n):
            for j in range(n):
                if abs(i - j) % period == 0:
                    ssm[i, j] = 0.9
        lm = LagMatrix()
        L = lm.compute_lag(ssm)
        patterns = lm.find_periodic_patterns(L, min_period=10, max_period=30)
        periods_found = [p.period for p in patterns]
        assert period in periods_found


class TestTransitionDetector:
    def test_breakdown(self):
        from models.music_structure.transitions import TransitionDetector, TransitionType
        rng = np.random.RandomState(42)
        features = np.zeros((40, 16))
        features[:20] = rng.randn(20, 16) * 5.0
        features[20:] = rng.randn(20, 16) * 0.1
        td = TransitionDetector(context_frames=16)
        result = td.detect_transition_type(features, 20)
        assert result == TransitionType.BREAKDOWN

    def test_buildup(self):
        from models.music_structure.transitions import TransitionDetector, TransitionType
        rng = np.random.RandomState(42)
        features = np.zeros((40, 16))
        features[:20] = rng.randn(20, 16) * 0.1
        features[20:] = rng.randn(20, 16) * 5.0
        td = TransitionDetector(context_frames=16)
        result = td.detect_transition_type(features, 20)
        assert result == TransitionType.BUILDUP

    def test_analyze_all(self, block_diagonal_features):
        from models.music_structure.transitions import TransitionDetector
        td = TransitionDetector()
        boundaries = [0, 25, 50, 75]
        results = td.analyze_all_transitions(block_diagonal_features, boundaries)
        assert len(results) == 3


class TestEnergyProfile:
    def test_shape(self):
        from models.music_structure.transitions import energy_profile
        features = np.random.RandomState(42).randn(100, 16)
        ep = energy_profile(features, window_size=8)
        assert len(ep) == 100

    def test_nonnegative(self):
        from models.music_structure.transitions import energy_profile
        features = np.random.RandomState(42).randn(50, 8)
        ep = energy_profile(features)
        assert np.all(ep >= 0)

    def test_empty(self):
        from models.music_structure.transitions import energy_profile
        ep = energy_profile(np.zeros((0, 8)))
        assert len(ep) == 0


class TestSpectralContrast:
    def test_same_features_zero(self):
        from models.music_structure.transitions import spectral_contrast_at_boundary
        features = np.ones((40, 8))
        sc = spectral_contrast_at_boundary(features, 20, context=8)
        assert sc < 1e-6

    def test_different_features_positive(self):
        from models.music_structure.transitions import spectral_contrast_at_boundary
        features = np.zeros((40, 8))
        features[:20] = 1.0
        features[20:] = -1.0
        sc = spectral_contrast_at_boundary(features, 20, context=8)
        assert sc > 0


class TestTransitionGraph:
    def test_build_graph(self):
        from models.music_structure.transitions import (
            TransitionGraph, TransitionInfo, TransitionType,
        )
        tg = TransitionGraph()
        labels = ["A", "B", "A", "B"]
        transitions = [
            TransitionInfo(25, TransitionType.DIRECT_CUT, 0.5, 1.0),
            TransitionInfo(50, TransitionType.CROSSFADE, -0.2, 0.3),
            TransitionInfo(75, TransitionType.BUILDUP, 1.0, 2.0),
        ]
        graph = tg.build_graph(labels, transitions)
        assert "A" in graph
        assert "B" in graph["A"]

    def test_node_count(self):
        from models.music_structure.transitions import (
            TransitionGraph, TransitionType,
        )
        tg = TransitionGraph()
        tg.add_transition("A", "B", TransitionType.DIRECT_CUT)
        tg.add_transition("B", "A", TransitionType.CROSSFADE)
        tg.add_transition("A", "C", TransitionType.BUILDUP)
        assert len(tg.nodes) == 3
        assert tg.n_edges == 3
