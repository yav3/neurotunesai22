"""Tests for embedding distance metrics and structural similarity."""
import numpy as np
import pytest

from models.music_similarity.embedding_distance import (
    EmbeddingDistanceMetrics,
    StructuralSimilarity,
)


class TestEmbeddingDistanceMetrics:
    @pytest.fixture
    def edm(self):
        return EmbeddingDistanceMetrics()

    def test_cosine_identical(self, edm):
        a = np.array([1.0, 2.0, 3.0])
        assert abs(edm.cosine_distance(a, a)) < 1e-10

    def test_cosine_orthogonal(self, edm):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(edm.cosine_distance(a, b) - 1.0) < 1e-10

    def test_cosine_zero_vector(self, edm):
        assert edm.cosine_distance(np.zeros(3), np.array([1, 2, 3])) == 1.0

    def test_euclidean_known(self, edm):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(edm.euclidean_distance(a, b) - 5.0) < 1e-10

    def test_manhattan_known(self, edm):
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert abs(edm.manhattan_distance(a, b) - 7.0) < 1e-10

    def test_mahalanobis_identity_cov(self, edm):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        I = np.eye(2)
        d = edm.mahalanobis_distance(a, b, I)
        assert abs(d - 5.0) < 1e-10

    def test_emd_identical(self, edm):
        h = np.array([0.2, 0.3, 0.5])
        assert abs(edm.earth_movers_distance(h, h)) < 1e-10

    def test_emd_nonneg(self, edm):
        a = np.array([1, 0, 0])
        b = np.array([0, 0, 1])
        assert edm.earth_movers_distance(a, b) >= 0

    def test_kl_identical(self, edm):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert edm.kl_divergence(p, p) < 0.01

    def test_kl_nonneg(self, edm):
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        assert edm.kl_divergence(p, q) >= 0

    def test_symmetric_kl(self, edm):
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        skl = edm.symmetric_kl(p, q)
        assert skl >= 0

    def test_bhattacharyya_identical(self, edm):
        p = np.array([0.5, 0.5])
        assert edm.bhattacharyya_distance(p, p) < 0.01

    def test_pairwise_symmetric(self, edm):
        emb = np.random.default_rng(0).standard_normal((5, 10))
        mat = edm.pairwise_distance_matrix(emb, "cosine")
        np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_pairwise_diagonal_zero(self, edm):
        emb = np.random.default_rng(0).standard_normal((4, 8))
        mat = edm.pairwise_distance_matrix(emb, "euclidean")
        np.testing.assert_allclose(np.diag(mat), 0, atol=1e-10)


class TestStructuralSimilarity:
    @pytest.fixture
    def ss(self):
        return StructuralSimilarity()

    def test_ssm_diagonal_one(self, ss):
        seq = np.random.default_rng(0).standard_normal((10, 5))
        ssm = ss.self_similarity_matrix(seq)
        np.testing.assert_allclose(np.diag(ssm), 1.0, atol=1e-6)

    def test_ssm_shape(self, ss):
        seq = np.random.default_rng(0).standard_normal((8, 3))
        ssm = ss.self_similarity_matrix(seq)
        assert ssm.shape == (8, 8)

    def test_novelty_curve_shape(self, ss):
        ssm = np.eye(20)
        nc = ss.novelty_curve(ssm, kernel_size=4)
        assert len(nc) == 20

    def test_recurrence_diagonal_true(self, ss):
        seq = np.random.default_rng(0).standard_normal((10, 4))
        rec = ss.recurrence_matrix(seq, k_nearest=3)
        assert np.all(np.diag(rec))

    def test_dtw_identical_zero(self, ss):
        seq = np.array([1.0, 2.0, 3.0, 4.0])
        assert ss.dtw_distance(seq, seq) < 1e-10

    def test_dtw_nonneg(self, ss):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert ss.dtw_distance(a, b) >= 0

    def test_dtw_empty(self, ss):
        assert ss.dtw_distance(np.array([]), np.array([1.0])) == 0.0

    def test_cover_song_transposition(self, ss):
        chroma = np.zeros(12)
        chroma[0] = 1.0
        chroma[4] = 0.8
        chroma[7] = 0.6
        shifted = np.roll(chroma, 5)
        d = ss.cover_song_similarity(chroma, shifted)
        assert d < 0.1

    def test_harmonic_similarity_identical(self, ss):
        c = np.ones(12) / 12
        assert ss.harmonic_similarity(c, c) < 1e-10

    def test_harmonic_similarity_different(self, ss):
        a = np.zeros(12)
        a[0] = 1.0
        b = np.zeros(12)
        b[6] = 1.0
        assert ss.harmonic_similarity(a, b) > 0


from models.music_similarity.sequence_alignment import (
    SequenceAlignment, ChromaAlignment, MelodyAlignment, RhythmAlignment,
)
from models.music_similarity.embedding_similarity import (
    FeatureAggregator, EmbeddingSimilarity, TripletEvaluator, CrossModalSimilarity,
)


class TestSequenceAlignmentClass:
    def test_dtw_identical(self):
        sa = SequenceAlignment()
        x = np.array([1.0, 2.0, 3.0])
        d, path = sa.dtw(x, x)
        assert d == 0.0

    def test_dtw_band_constraint(self):
        sa = SequenceAlignment()
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float) + 0.1
        d_full, _ = sa.dtw(x, y)
        d_band, _ = sa.dtw(x, y, band=2)
        assert d_band >= d_full - 1e-6

    def test_dtw_empty(self):
        sa = SequenceAlignment()
        d, path = sa.dtw(np.array([]), np.array([1.0]))
        assert d == 0.0
        assert len(path) == 0

    def test_edit_distance_identical(self):
        sa = SequenceAlignment()
        x = np.array([60, 62, 64])
        assert sa.edit_distance(x, x) == 0

    def test_edit_distance_single_sub(self):
        sa = SequenceAlignment()
        a = np.array([60, 62, 64])
        b = np.array([60, 63, 64])
        assert sa.edit_distance(a, b) == 1

    def test_smith_waterman_local(self):
        sa = SequenceAlignment()
        a = np.array([10.0, 1.0, 2.0, 3.0, 10.0])
        b = np.array([1.0, 2.0, 3.0])
        score, path = sa.smith_waterman(a, b)
        assert score > 0

    def test_normalized_dtw_positive(self):
        sa = SequenceAlignment()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 3.5])
        nd = sa.normalized_dtw(a, b)
        assert nd > 0


class TestChromaAlignmentClass:
    def test_oti_identical(self):
        ca = ChromaAlignment()
        chroma = np.random.default_rng(42).random((10, 12))
        assert ca.oti(chroma, chroma) == 0

    def test_oti_shifted(self):
        ca = ChromaAlignment()
        chroma = np.zeros(12)
        chroma[0] = 1.0
        shifted = np.roll(chroma, 3)
        assert ca.oti(chroma, shifted) == 9

    def test_cover_similarity_range(self):
        ca = ChromaAlignment()
        rng = np.random.default_rng(42)
        a = rng.random((8, 12))
        b = rng.random((8, 12))
        sim = ca.cover_song_similarity(a, b)
        assert 0 < sim <= 1.0

    def test_cover_similarity_identical(self):
        ca = ChromaAlignment()
        chroma = np.random.default_rng(42).random((10, 12))
        sim = ca.cover_song_similarity(chroma, chroma)
        assert sim > 0.9


class TestMelodyAlignmentClass:
    def test_interval_sequence(self):
        ma = MelodyAlignment()
        pitches = np.array([60, 62, 65, 64])
        intervals = ma.interval_sequence(pitches)
        np.testing.assert_array_equal(intervals, [2, 3, -1])

    def test_interval_edit_distance_transposed(self):
        ma = MelodyAlignment()
        a = np.array([60, 62, 64])
        b = np.array([72, 74, 76])
        assert ma.interval_edit_distance(a, b) == 0

    def test_contour_similarity_identical(self):
        ma = MelodyAlignment()
        p = np.array([60.0, 62, 64, 63, 65])
        sim = ma.contour_similarity(p, p)
        assert sim == 1.0

    def test_pitch_class_similarity_octave(self):
        ma = MelodyAlignment()
        a = np.array([60, 64, 67])
        b = np.array([72, 76, 79])
        assert ma.pitch_class_similarity(a, b) == 1.0


class TestRhythmAlignmentClass:
    def test_onset_identical(self):
        ra = RhythmAlignment()
        onsets = np.array([0.0, 0.5, 1.0, 1.5])
        assert ra.onset_pattern_distance(onsets, onsets) == 0.0

    def test_ioi_proportional(self):
        ra = RhythmAlignment()
        a = np.array([0.0, 0.5, 1.0, 1.5])
        b = np.array([0.0, 1.0, 2.0, 3.0])
        d = ra.ioi_sequence_distance(a, b)
        assert d < 0.01

    def test_groove_identical(self):
        ra = RhythmAlignment()
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        sim = ra.groove_similarity(onsets, onsets)
        assert sim == 1.0


class TestFeatureAggregatorClass:
    def test_mean_std_1d(self):
        fa = FeatureAggregator()
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = fa.mean_std(x)
        assert len(result) == 2
        assert abs(result[0] - 2.5) < 1e-6

    def test_mean_std_2d(self):
        fa = FeatureAggregator()
        x = np.random.default_rng(42).standard_normal((20, 5))
        result = fa.mean_std(x)
        assert len(result) == 10

    def test_gmm_output_length(self):
        fa = FeatureAggregator()
        x = np.random.default_rng(42).standard_normal((50, 3))
        result = fa.gaussian_mixture(x, n_components=2)
        assert len(result) == 2 * 3 + 2 * 3

    def test_temporal_pooling_segments(self):
        fa = FeatureAggregator()
        x = np.random.default_rng(42).standard_normal((40, 5))
        result = fa.temporal_pooling(x, n_segments=4)
        assert len(result) == 4 * 5


class TestEmbeddingSimilarityClass:
    def test_cosine_identical(self):
        es = EmbeddingSimilarity()
        x = np.array([1.0, 2.0, 3.0])
        assert abs(es.cosine(x, x) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        es = EmbeddingSimilarity()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(es.cosine(a, b)) < 1e-6

    def test_pairwise_symmetric(self):
        es = EmbeddingSimilarity()
        emb = np.random.default_rng(42).standard_normal((5, 3))
        sim = es.pairwise_matrix(emb)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)


class TestTripletEvaluatorClass:
    def test_perfect_separation(self):
        emb = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=float)
        labels = np.array([0, 0, 1, 1])
        te = TripletEvaluator(emb, labels)
        acc = te.triplet_accuracy(margin=0.0, n_triplets=500)
        assert acc > 0.9

    def test_map_perfect(self):
        emb = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=float)
        labels = np.array([0, 0, 1, 1])
        te = TripletEvaluator(emb, labels)
        m = te.mean_average_precision()
        assert m > 0.8

    def test_recall_at_k(self):
        emb = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=float)
        labels = np.array([0, 0, 1, 1])
        te = TripletEvaluator(emb, labels)
        r = te.recall_at_k(k=1)
        assert r > 0.8


class TestCrossModalSimilarityClass:
    def test_cca_correlations(self):
        rng = np.random.default_rng(42)
        z = rng.standard_normal((50, 3))
        X = z @ rng.standard_normal((3, 5)) + 0.1 * rng.standard_normal((50, 5))
        Y = z @ rng.standard_normal((3, 4)) + 0.1 * rng.standard_normal((50, 4))
        cms = CrossModalSimilarity()
        corrs, Xp, Yp = cms.cca(X, Y, n_components=3)
        assert corrs[0] > 0.5

    def test_projection_similarity_range(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        Y = rng.standard_normal((40, 4))
        cms = CrossModalSimilarity()
        sim = cms.projection_similarity(X, Y, projection_dim=3)
        assert -1.0 <= sim <= 1.0
