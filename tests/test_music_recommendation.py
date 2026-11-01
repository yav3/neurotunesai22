"""Tests for the music_recommendation module."""
from __future__ import annotations
import pytest
import numpy as np
from models.music_data.schema import TrackRecord
from models.music_recommendation.similarity import (
    cosine_similarity, euclidean_distance, weighted_feature_similarity,
    mood_similarity, harmonic_compatibility, rhythm_similarity,
    FeatureSimilarity, HarmonicSimilarity, RhythmicSimilarity,
    EmotionalSimilarity, CompositeSimilarity,
)
from models.music_recommendation.clustering import (
    MusicClusterer, MoodClusterer, GenreProfiler, ClusterLabel,
)
from models.music_recommendation.recommender import (
    ContentBasedRecommender, MoodBasedRecommender,
    PlaylistContinuationRecommender, ContextualRecommender, DiversityMixer,
)
from models.music_recommendation.taste_profile import (
    ListeningHistory, ListenEvent, TasteProfile, TasteEvolution, TasteDistance,
)
from models.music_recommendation.discovery import (
    NoveltyScorer, ExplorationExploitation, GenreBridging, MoodJourney,
)
from models.music_recommendation.embeddings import (
    build_embedding, pca_reduce, EmbeddingSpace, TrackEmbedding,
)
from models.music_recommendation.filters import (
    BPMFilter, GenreFilter, KeyFilter, DurationFilter, EnergyFilter,
    ExclusionFilter, CompositeFilter,
)


def _make_tracks(n=10):
    return [
        TrackRecord(
            id=f"t{i}", title=f"Track {i}",
            artist=f"Artist {'ABCDE'[i % 5]}",
            genre=["Pop", "Rock", "Jazz", "Electronic", "Classical"][i % 5],
            bpm=100 + i * 10, valence=0.1 + i * 0.08,
            arousal=0.2 + i * 0.06, dominance=0.3 + i * 0.05,
            energy=0.2 + i * 0.07, key_str="C", mode="major",
        )
        for i in range(n)
    ]


# --- Similarity ---

class TestFeatureSimilarity:
    def test_identical(self):
        t = TrackRecord(bpm=120, valence=0.8, energy=0.7)
        assert FeatureSimilarity().score(t, t) > 0.99

    def test_different(self):
        a = TrackRecord(bpm=60, valence=0.1, energy=0.1, arousal=0.1,
                        spectral_centroid=100, bass_energy=0.9, treble_energy=0.1)
        b = TrackRecord(bpm=180, valence=0.9, energy=0.9, arousal=0.9,
                        spectral_centroid=5000, bass_energy=0.1, treble_energy=0.9)
        assert FeatureSimilarity().score(a, b) < 0.95

    def test_range(self):
        a, b = _make_tracks(2)
        s = FeatureSimilarity().score(a, b)
        assert 0 <= s <= 1


class TestHarmonicSimilarity:
    def test_same_key(self):
        a = TrackRecord(key_str="A", mode="minor")
        b = TrackRecord(key_str="A", mode="minor")
        assert HarmonicSimilarity().score(a, b) == 1.0

    def test_unknown_key(self):
        a = TrackRecord(key_str="X", mode="")
        b = TrackRecord(key_str="Y", mode="")
        assert HarmonicSimilarity().score(a, b) == 0.0


class TestRhythmicSimilarity:
    def test_same_bpm(self):
        a = TrackRecord(bpm=128)
        b = TrackRecord(bpm=128)
        assert RhythmicSimilarity().score(a, b) == 1.0

    def test_double_time(self):
        a = TrackRecord(bpm=64)
        b = TrackRecord(bpm=128)
        s = RhythmicSimilarity().score(a, b)
        assert s >= 0.85

    def test_zero_bpm(self):
        a = TrackRecord(bpm=0)
        b = TrackRecord(bpm=128)
        assert RhythmicSimilarity().score(a, b) == 0.5


class TestEmotionalSimilarity:
    def test_identical(self):
        a = TrackRecord(valence=0.5, arousal=0.5, dominance=0.5)
        assert EmotionalSimilarity().score(a, a) == 1.0

    def test_opposite(self):
        a = TrackRecord(valence=0, arousal=0, dominance=0)
        b = TrackRecord(valence=1, arousal=1, dominance=1)
        assert EmotionalSimilarity().score(a, b) < 0.1


class TestCompositeSimilarity:
    def test_range(self):
        tracks = _make_tracks(2)
        s = CompositeSimilarity().score(tracks[0], tracks[1])
        assert 0 <= s <= 1

    def test_detailed(self):
        tracks = _make_tracks(2)
        d = CompositeSimilarity().score_detailed(tracks[0], tracks[1])
        assert "composite" in d
        assert "feature" in d


# --- Clustering ---

class TestMusicClusterer:
    def test_fit_predict(self):
        tracks = _make_tracks(20)
        mc = MusicClusterer(n_clusters=3)
        mc.fit(tracks, seed=0)
        assert mc.labels is not None
        assert len(mc.labels) == 20
        assert mc.predict(tracks[0]) in range(3)

    def test_empty(self):
        mc = MusicClusterer()
        mc.fit([])
        assert mc.labels is None


class TestMoodClusterer:
    def test_cluster(self):
        tracks = _make_tracks(20)
        groups = MoodClusterer().cluster(tracks, n_clusters=4, seed=0)
        assert len(groups) <= 4
        total = sum(len(v) for v in groups.values())
        assert total == 20


class TestGenreProfiler:
    def test_profile(self):
        tracks = _make_tracks(10)
        profiles = GenreProfiler().profile(tracks)
        assert "Pop" in profiles
        assert profiles["Pop"].count >= 1


class TestClusterLabel:
    def test_label_mood(self):
        tracks = [TrackRecord(valence=0.8, arousal=0.8)]
        assert ClusterLabel.label_mood_cluster(tracks) == "Excited/Happy"

    def test_label_energy(self):
        tracks = [TrackRecord(energy=0.9)]
        assert ClusterLabel.label_energy_cluster(tracks) == "High Energy"

    def test_empty(self):
        assert ClusterLabel.label_mood_cluster([]) == "Unknown"


# --- Recommender ---

class TestContentBasedRecommender:
    def test_recommend(self):
        tracks = _make_tracks(20)
        recs = ContentBasedRecommender().recommend(tracks[0], tracks[1:], n=5)
        assert len(recs) == 5
        assert all(isinstance(r[1], float) for r in recs)

    def test_excludes_seed(self):
        tracks = _make_tracks(5)
        recs = ContentBasedRecommender().recommend(tracks[0], tracks, n=10)
        ids = [r[0].id for r in recs]
        assert tracks[0].id not in ids


class TestMoodBasedRecommender:
    def test_recommend(self):
        tracks = _make_tracks(20)
        recs = MoodBasedRecommender().recommend(0.5, 0.5, 0.5, tracks, n=5)
        assert len(recs) == 5
        dists = [d for _, d in recs]
        assert dists == sorted(dists)


class TestPlaylistContinuationRecommender:
    def test_recommend(self):
        tracks = _make_tracks(20)
        recs = PlaylistContinuationRecommender().recommend(tracks[:3], tracks, n=5)
        assert len(recs) == 5

    def test_empty_playlist(self):
        tracks = _make_tracks(5)
        recs = PlaylistContinuationRecommender().recommend([], tracks, n=3)
        assert len(recs) == 3


class TestContextualRecommender:
    def test_genre_filter(self):
        tracks = _make_tracks(20)
        cr = ContextualRecommender(preferred_genres=["Pop"])
        recs = cr.recommend(tracks, n=10)
        assert all(r.genre == "Pop" for r in recs)

    def test_bpm_filter(self):
        tracks = _make_tracks(20)
        cr = ContextualRecommender(bpm_range=(100, 130))
        recs = cr.recommend(tracks)
        assert all(100 <= r.bpm <= 130 for r in recs)


class TestDiversityMixer:
    def test_diversify(self):
        tracks = _make_tracks(10)
        ranked = [(t, 1.0 - i * 0.1) for i, t in enumerate(tracks)]
        mixed = DiversityMixer.diversify(ranked, window=2)
        assert len(mixed) == len(ranked)

    def test_empty(self):
        assert DiversityMixer.diversify([]) == []


# --- Taste Profile ---

class TestListeningHistory:
    def test_add_and_len(self):
        h = ListeningHistory()
        h.add(ListenEvent(track_id="t1", timestamp=1.0))
        assert len(h) == 1

    def test_play_counts(self):
        h = ListeningHistory()
        h.add(ListenEvent(track_id="t1", timestamp=1.0))
        h.add(ListenEvent(track_id="t1", timestamp=2.0))
        h.add(ListenEvent(track_id="t2", timestamp=3.0))
        assert h.play_counts() == {"t1": 2, "t2": 1}


class TestTasteProfile:
    def test_from_tracks(self):
        tracks = _make_tracks(10)
        tp = TasteProfile.from_tracks(tracks)
        assert tp.bpm_mean > 0
        assert len(tp.preferred_genres) > 0

    def test_empty(self):
        tp = TasteProfile.from_tracks([])
        assert tp.bpm_mean == 0

    def test_to_vector(self):
        tp = TasteProfile(bpm_mean=120, energy_mean=0.7, valence_mean=0.5, arousal_mean=0.6, dominance_mean=0.4)
        v = tp.to_vector()
        assert len(v) == 5


class TestTasteEvolution:
    def test_drift(self):
        te = TasteEvolution()
        te.add_snapshot(1.0, TasteProfile(energy_mean=0.3, valence_mean=0.3))
        te.add_snapshot(2.0, TasteProfile(energy_mean=0.8, valence_mean=0.8))
        assert te.drift() > 0

    def test_single_snapshot(self):
        te = TasteEvolution()
        te.add_snapshot(1.0, TasteProfile())
        assert te.drift() == 0.0


class TestTasteDistance:
    def test_same(self):
        tp = TasteProfile(bpm_mean=120, energy_mean=0.5)
        assert TasteDistance.distance(tp, tp) == 0.0

    def test_genre_overlap(self):
        a = TasteProfile(preferred_genres={"Pop": 0.5, "Rock": 0.5})
        b = TasteProfile(preferred_genres={"Pop": 0.3, "Jazz": 0.7})
        assert 0 < TasteDistance.genre_overlap(a, b) < 1


# --- Discovery ---

class TestNoveltyScorer:
    def test_novel(self):
        tracks = _make_tracks(5)
        novel = TrackRecord(id="new", bpm=200, valence=0.99, energy=0.99)
        s = NoveltyScorer().score(novel, tracks)
        assert s > 0

    def test_empty_history(self):
        t = TrackRecord()
        assert NoveltyScorer().score(t, []) == 1.0


class TestExplorationExploitation:
    def test_select(self):
        tracks = _make_tracks(10)
        ranked = [(t, 1.0 - i * 0.1) for i, t in enumerate(tracks)]
        ee = ExplorationExploitation(epsilon=0.0)
        sel = ee.select(ranked, tracks[:3], seed=0)
        assert sel.id == tracks[0].id

    def test_rerank(self):
        tracks = _make_tracks(10)
        ranked = [(t, 1.0 - i * 0.1) for i, t in enumerate(tracks)]
        rr = ExplorationExploitation().rerank(ranked, tracks[:3])
        assert len(rr) == 10


class TestGenreBridging:
    def test_find_bridges(self):
        tracks = _make_tracks(20)
        bridges = GenreBridging().find_bridges("Pop", "Rock", tracks, n=3)
        assert len(bridges) <= 3


class TestMoodJourney:
    def test_plan(self):
        tracks = _make_tracks(20)
        journey = MoodJourney().plan((0.1, 0.1, 0.1), (0.9, 0.9, 0.9), tracks, n_steps=5)
        assert len(journey) == 5

    def test_empty(self):
        assert MoodJourney().plan((0, 0, 0), (1, 1, 1), [], n_steps=5) == []


# --- Function-based Similarity ---

class TestFunctionSimilarity:
    def test_cosine_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)

    def test_euclidean_same(self):
        v = np.array([1.0, 2.0])
        assert euclidean_distance(v, v) == pytest.approx(0.0)

    def test_euclidean_known(self):
        assert euclidean_distance(np.array([0, 0]), np.array([3, 4])) == pytest.approx(5.0)

    def test_weighted_feature_full_match(self):
        f = {"a": 0.5, "b": 0.8}
        assert weighted_feature_similarity(f, f) == pytest.approx(1.0)

    def test_weighted_feature_no_common(self):
        assert weighted_feature_similarity({"a": 1}, {"b": 1}) == 0.0

    def test_mood_identical(self):
        assert mood_similarity((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) == pytest.approx(1.0)

    def test_mood_far(self):
        s = mood_similarity((0, 0, 0), (1, 1, 1))
        assert s == pytest.approx(0.0, abs=0.01)

    def test_harmonic_same_key(self):
        assert harmonic_compatibility("C major", "C major") == 1.0

    def test_harmonic_unknown(self):
        assert harmonic_compatibility("Z#", "Fb") == 0.0

    def test_rhythm_same(self):
        assert rhythm_similarity(120, 120) == pytest.approx(1.0)

    def test_rhythm_zero_bpm(self):
        assert rhythm_similarity(0, 120) == 0.0


# --- Embeddings ---

class TestEmbeddings:
    def test_build_embedding(self):
        t = _make_tracks(1)[0]
        d = t.to_feature_dict()
        emb = build_embedding(d)
        assert emb.shape == (16,)
        assert np.isfinite(emb).all()

    def test_build_embedding_normalized(self):
        d = _make_tracks(1)[0].to_feature_dict()
        emb = build_embedding(d)
        norm = np.linalg.norm(emb)
        if norm > 0:
            assert norm == pytest.approx(1.0, abs=0.01)

    def test_pca_reduce(self):
        vecs = np.random.default_rng(0).normal(size=(20, 16))
        reduced = pca_reduce(vecs, n_dims=3)
        assert reduced.shape == (20, 3)

    def test_embedding_space_add_query(self):
        space = EmbeddingSpace()
        tracks = _make_tracks(10)
        for t in tracks:
            emb = TrackEmbedding(track_id=t.id, vector=build_embedding(t.to_feature_dict()))
            space.add(emb)
        results = space.query(build_embedding(tracks[0].to_feature_dict()), n=3)
        assert len(results) == 3

    def test_embedding_space_nearest(self):
        space = EmbeddingSpace()
        tracks = _make_tracks(5)
        for t in tracks:
            space.add(TrackEmbedding(track_id=t.id, vector=build_embedding(t.to_feature_dict())))
        results = space.nearest(tracks[0].id, n=2)
        assert len(results) == 2

    def test_embedding_space_remove(self):
        space = EmbeddingSpace()
        space.add(TrackEmbedding(track_id="a", vector=np.ones(16)))
        space.remove("a")
        mat, ids = space.to_matrix()
        assert mat.shape[0] == 0

    def test_embedding_space_to_matrix(self):
        space = EmbeddingSpace()
        tracks = _make_tracks(5)
        for t in tracks:
            space.add(TrackEmbedding(track_id=t.id, vector=build_embedding(t.to_feature_dict())))
        mat, ids = space.to_matrix()
        assert mat.shape == (5, 16)
        assert len(ids) == 5


# --- Filters ---

class TestFilters:
    def _dict_tracks(self):
        return [
            {"id": "t1", "bpm": 100, "genre": "Pop", "key": "C major",
             "duration_s": 180, "energy": 0.5},
            {"id": "t2", "bpm": 140, "genre": "Rock", "key": "A minor",
             "duration_s": 300, "energy": 0.9},
            {"id": "t3", "bpm": 80, "genre": "Jazz", "key": "G major",
             "duration_s": 240, "energy": 0.3},
        ]

    def test_bpm_filter(self):
        f = BPMFilter(90, 110)
        assert f.accept(self._dict_tracks()[0])
        assert not f.accept(self._dict_tracks()[1])

    def test_genre_filter(self):
        f = GenreFilter(["Rock"])
        assert f.accept(self._dict_tracks()[1])
        assert not f.accept(self._dict_tracks()[0])

    def test_key_filter(self):
        f = KeyFilter(["C major"])
        assert f.accept(self._dict_tracks()[0])
        assert not f.accept(self._dict_tracks()[1])

    def test_duration_filter(self):
        f = DurationFilter(min_s=200, max_s=350)
        assert not f.accept(self._dict_tracks()[0])
        assert f.accept(self._dict_tracks()[1])

    def test_energy_filter(self):
        f = EnergyFilter(0.4, 0.6)
        assert f.accept(self._dict_tracks()[0])
        assert not f.accept(self._dict_tracks()[2])

    def test_exclusion_filter(self):
        f = ExclusionFilter({"t1", "t3"})
        assert not f.accept(self._dict_tracks()[0])
        assert f.accept(self._dict_tracks()[1])

    def test_composite_filter(self):
        cf = CompositeFilter([BPMFilter(90, 150), GenreFilter(["Pop"])])
        result = cf.apply(self._dict_tracks())
        assert len(result) == 1
        assert result[0]["id"] == "t1"

    def test_composite_empty(self):
        cf = CompositeFilter([])
        assert len(cf.apply(self._dict_tracks())) == 3
