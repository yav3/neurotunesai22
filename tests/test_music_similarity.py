"""Tests for the music_similarity module: embedding, similarity, recommender, clustering."""
from __future__ import annotations

import pytest
import numpy as np

from models.music_similarity.embedding import TrackEmbedding, EmbeddingSpace
from models.music_similarity.similarity import SimilarityEngine, SimilarityResult
from models.music_similarity.recommender import (
    ContentBasedRecommender,
    PlaylistRecommender,
    Recommendation,
)
from models.music_similarity.clustering import MusicClusterer, ClusterProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_space(n: int = 20, dim: int = 8, seed: int = 42) -> EmbeddingSpace:
    rng = np.random.default_rng(seed)
    space = EmbeddingSpace(dim=dim)
    for i in range(n):
        space.add(TrackEmbedding(
            track_id=f"t{i}",
            vector=rng.standard_normal(dim).astype(np.float32),
            metadata={"index": i},
        ))
    return space


# ---------------------------------------------------------------------------
# EmbeddingSpace
# ---------------------------------------------------------------------------

class TestEmbeddingSpace:
    def test_add_and_len(self):
        space = EmbeddingSpace()
        space.add(TrackEmbedding("t1", np.array([1.0, 2.0])))
        assert len(space) == 1
        assert space.dim == 2

    def test_dim_mismatch_raises(self):
        space = EmbeddingSpace(dim=3)
        with pytest.raises(ValueError):
            space.add(TrackEmbedding("t1", np.array([1.0, 2.0])))

    def test_get(self):
        space = EmbeddingSpace()
        space.add(TrackEmbedding("t1", np.array([1.0, 0.0])))
        assert space.get("t1") is not None
        assert space.get("missing") is None

    def test_nearest_cosine(self):
        space = EmbeddingSpace()
        space.add(TrackEmbedding("t1", np.array([1.0, 0.0])))
        space.add(TrackEmbedding("t2", np.array([0.9, 0.1])))
        space.add(TrackEmbedding("t3", np.array([0.0, 1.0])))
        results = space.nearest(np.array([1.0, 0.0]), n=2, metric="cosine")
        assert len(results) == 2
        assert results[0][0] == "t1"

    def test_nearest_euclidean(self):
        space = EmbeddingSpace()
        space.add(TrackEmbedding("t1", np.array([0.0, 0.0])))
        space.add(TrackEmbedding("t2", np.array([1.0, 0.0])))
        space.add(TrackEmbedding("t3", np.array([10.0, 10.0])))
        results = space.nearest(np.array([0.5, 0.0]), n=2, metric="euclidean")
        assert results[0][0] in ("t1", "t2")

    def test_nearest_exclude(self):
        space = EmbeddingSpace()
        space.add(TrackEmbedding("t1", np.array([1.0, 0.0])))
        space.add(TrackEmbedding("t2", np.array([0.9, 0.1])))
        results = space.nearest(np.array([1.0, 0.0]), n=2, exclude={"t1"})
        assert all(tid != "t1" for tid, _ in results)

    def test_nearest_empty(self):
        space = EmbeddingSpace(dim=3)
        results = space.nearest(np.array([1.0, 0.0, 0.0]), n=5)
        assert results == []

    def test_reduce_dim(self):
        space = _make_space(n=10, dim=8)
        reduced = space.reduce_dim(target_dim=2)
        assert reduced.shape == (10, 2)

    def test_add_many(self):
        space = EmbeddingSpace()
        embs = [TrackEmbedding(f"t{i}", np.array([float(i), 0.0])) for i in range(5)]
        n = space.add_many(embs)
        assert n == 5
        assert len(space) == 5

    def test_to_matrix(self):
        space = _make_space(n=5, dim=4)
        mat, ids = space.to_matrix()
        assert mat.shape == (5, 4)
        assert len(ids) == 5

    def test_all_ids(self):
        space = _make_space(n=3)
        assert len(space.all_ids()) == 3


# ---------------------------------------------------------------------------
# SimilarityEngine
# ---------------------------------------------------------------------------

class TestSimilarityEngine:
    def test_cosine_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert SimilarityEngine.cosine(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert SimilarityEngine.cosine(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_cosine_zero_vector(self):
        assert SimilarityEngine.cosine(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_euclidean(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert SimilarityEngine.euclidean(a, b) == pytest.approx(5.0)

    def test_manhattan(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert SimilarityEngine.manhattan(a, b) == pytest.approx(7.0)

    def test_weighted_feature_similarity(self):
        engine = SimilarityEngine()
        a = {"bpm": 0.5, "energy": 0.8}
        b = {"bpm": 0.5, "energy": 0.6}
        result = engine.weighted_feature_similarity(a, b)
        assert 0.0 <= result.score <= 1.0
        assert "bpm" in result.components
        assert result.components["bpm"] == 1.0

    def test_weighted_feature_custom_weights(self):
        engine = SimilarityEngine()
        a = {"x": 0.0, "y": 1.0}
        b = {"x": 1.0, "y": 1.0}
        w1 = engine.weighted_feature_similarity(a, b, weights={"x": 1.0, "y": 0.0})
        w2 = engine.weighted_feature_similarity(a, b, weights={"x": 0.0, "y": 1.0})
        assert w1.score < w2.score

    def test_harmonic_compatibility_same(self):
        engine = SimilarityEngine()
        assert engine.harmonic_compatibility("A minor", "A minor") == 1.0

    def test_harmonic_compatibility_parallel(self):
        engine = SimilarityEngine()
        score = engine.harmonic_compatibility("A minor", "C major")
        assert score >= 0.7

    def test_harmonic_compatibility_distant(self):
        engine = SimilarityEngine()
        score = engine.harmonic_compatibility("C major", "F# major")
        assert score < 0.7

    def test_temporal_similarity_identical(self):
        engine = SimilarityEngine()
        arc = [0.1, 0.3, 0.5, 0.7, 0.9]
        assert engine.temporal_similarity(arc, arc) == pytest.approx(1.0, abs=0.01)

    def test_temporal_similarity_inverse(self):
        engine = SimilarityEngine()
        a = [0.1, 0.3, 0.5, 0.7, 0.9]
        b = [0.9, 0.7, 0.5, 0.3, 0.1]
        score = engine.temporal_similarity(a, b)
        assert score < 0.2

    def test_temporal_similarity_empty(self):
        engine = SimilarityEngine()
        assert engine.temporal_similarity([], [0.5]) == 0.0

    def test_composite_similarity(self):
        engine = SimilarityEngine()
        a = {"bpm": 120, "energy": 0.8, "danceability": 0.7,
             "acousticness": 0.1, "valence": 0.6, "key": "A minor",
             "arousal": 0.7, "dominance": 0.5}
        b = {"bpm": 125, "energy": 0.75, "danceability": 0.65,
             "acousticness": 0.15, "valence": 0.55, "key": "E minor",
             "arousal": 0.65, "dominance": 0.5}
        result = engine.composite_similarity(a, b)
        assert 0.0 <= result.score <= 1.0
        assert "feature" in result.components


# ---------------------------------------------------------------------------
# ContentBasedRecommender
# ---------------------------------------------------------------------------

class TestContentBasedRecommender:
    def test_recommend_similar(self):
        space = _make_space(n=20)
        rec = ContentBasedRecommender()
        results = rec.recommend_similar("t0", space, n=5)
        assert len(results) == 5
        assert all(isinstance(r, Recommendation) for r in results)
        assert all(r.track_id != "t0" for r in results)

    def test_recommend_missing_seed(self):
        space = _make_space(n=5)
        rec = ContentBasedRecommender()
        assert rec.recommend_similar("missing", space) == []

    def test_recommend_from_seeds(self):
        space = _make_space(n=20)
        rec = ContentBasedRecommender()
        results = rec.recommend_from_seeds(["t0", "t1", "t2"], space, n=5)
        assert len(results) == 5
        assert all(r.track_id not in ("t0", "t1", "t2") for r in results)

    def test_recommend_mmr_diversity(self):
        space = _make_space(n=30)
        rec = ContentBasedRecommender()
        low_div = rec.recommend_mmr("t0", space, n=5, lambda_=1.0)
        high_div = rec.recommend_mmr("t0", space, n=5, lambda_=0.3)
        assert len(low_div) == 5
        assert len(high_div) == 5

    def test_recommend_with_exclude(self):
        space = _make_space(n=10)
        rec = ContentBasedRecommender()
        results = rec.recommend_similar("t0", space, n=5, exclude={"t1", "t2"})
        assert all(r.track_id not in ("t0", "t1", "t2") for r in results)


# ---------------------------------------------------------------------------
# PlaylistRecommender
# ---------------------------------------------------------------------------

class TestPlaylistRecommender:
    def test_generate_playlist_basic(self):
        space = _make_space(n=30)
        rec = PlaylistRecommender()
        track_data = {f"t{i}": {"energy": 0.3 + (i / 30) * 0.6} for i in range(30)}
        playlist = rec.generate_playlist("t0", space, track_data, n_tracks=8, seed=42)
        assert len(playlist) <= 8

    def test_generate_playlist_with_energy_curve(self):
        space = _make_space(n=30)
        rec = PlaylistRecommender()
        track_data = {f"t{i}": {"energy": 0.3 + (i / 30) * 0.6} for i in range(30)}
        energy_target = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8]
        playlist = rec.generate_playlist(
            "t0", space, track_data, n_tracks=8,
            energy_target=energy_target, seed=42
        )
        assert len(playlist) <= 8

    def test_extend_playlist(self):
        space = _make_space(n=20)
        rec = PlaylistRecommender()
        ext = rec.extend_playlist(["t0", "t1", "t2"], space, n_extend=3)
        assert len(ext) <= 3
        assert all(r.track_id not in ("t0", "t1", "t2") for r in ext)

    def test_extend_empty(self):
        space = _make_space(n=10)
        rec = PlaylistRecommender()
        assert rec.extend_playlist([], space) == []


# ---------------------------------------------------------------------------
# MusicClusterer
# ---------------------------------------------------------------------------

class TestMusicClusterer:
    def test_kmeans_basic(self):
        space = _make_space(n=20, dim=4)
        cl = MusicClusterer()
        result = cl.kmeans(space, k=3, seed=42)
        assert result.n_clusters == 3
        assert len(result.labels) == 20
        assert len(result.profiles) == 3
        assert result.inertia >= 0

    def test_kmeans_k_larger_than_n(self):
        space = _make_space(n=3, dim=4)
        cl = MusicClusterer()
        result = cl.kmeans(space, k=10, seed=42)
        assert result.n_clusters == 3

    def test_kmeans_empty(self):
        space = EmbeddingSpace(dim=4)
        cl = MusicClusterer()
        result = cl.kmeans(space, k=3)
        assert result.n_clusters == 0
        assert len(result.labels) == 0

    def test_hierarchical_basic(self):
        space = _make_space(n=10, dim=4)
        cl = MusicClusterer()
        result = cl.hierarchical(space, k=3)
        assert result.n_clusters == 3
        assert len(result.labels) == 10
        assert len(result.profiles) == 3

    def test_cluster_profiles_have_members(self):
        space = _make_space(n=15, dim=4)
        cl = MusicClusterer()
        result = cl.kmeans(space, k=3, seed=42)
        total_members = sum(p.size for p in result.profiles)
        assert total_members == 15
        for p in result.profiles:
            assert len(p.member_ids) == p.size

    def test_cluster_profile_summary(self):
        space = _make_space(n=10, dim=4)
        cl = MusicClusterer()
        result = cl.kmeans(space, k=2, seed=42)
        for p in result.profiles:
            assert "dim_0_mean" in p.feature_summary
            assert "dim_0_std" in p.feature_summary

    def test_silhouette_score(self):
        space = _make_space(n=20, dim=4)
        cl = MusicClusterer()
        result = cl.kmeans(space, k=3, seed=42)
        score = cl.silhouette_score(space, result.labels)
        assert -1.0 <= score <= 1.0

    def test_silhouette_single_cluster(self):
        space = _make_space(n=5, dim=4)
        cl = MusicClusterer()
        labels = np.zeros(5, dtype=int)
        score = cl.silhouette_score(space, labels)
        assert score == 0.0

    def test_deterministic_with_seed(self):
        space = _make_space(n=20, dim=4)
        cl = MusicClusterer()
        r1 = cl.kmeans(space, k=3, seed=123)
        r2 = cl.kmeans(space, k=3, seed=123)
        np.testing.assert_array_equal(r1.labels, r2.labels)
