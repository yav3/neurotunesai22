"""Tests for collaborative filtering and session-based recommendation."""
import numpy as np
import pytest

from models.music_recommendation.collaborative import (
    InteractionMatrix,
    MFModel,
    PopularityRecommender,
    als_train,
    bm25_weight,
    implicit_confidence,
    item_knn_recommend,
    user_knn_recommend,
)
from models.music_recommendation.session_rec import (
    SessionMomentumTracker,
    SessionState,
    SequentialPattern,
    exponential_decay_centroid,
    find_sequential_patterns,
    markov_predict,
    markov_transition_matrix,
    session_centroid,
    session_coherence_score,
    session_diversity_score,
    session_knn_recommend,
)


# ---- Collaborative filtering ----

class TestInteractionMatrix:
    def test_from_dense(self):
        mat = np.array([[5, 0, 3], [0, 4, 0], [1, 0, 2]])
        im = InteractionMatrix.from_dense(mat)
        assert im.n_users == 3
        assert im.n_items == 3
        assert len(im.values) == 5

    def test_to_dense_roundtrip(self):
        mat = np.array([[5.0, 0, 3], [0, 4, 0]])
        im = InteractionMatrix.from_dense(mat)
        recovered = im.to_dense()
        np.testing.assert_array_equal(recovered, mat)

    def test_empty_matrix(self):
        im = InteractionMatrix(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        assert im.n_users == 0
        assert im.n_items == 0


class TestALSTrain:
    @pytest.fixture
    def interactions(self):
        mat = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [0, 0, 5, 4],
            [0, 1, 5, 0],
        ], dtype=float)
        return InteractionMatrix.from_dense(mat)

    def test_returns_model(self, interactions):
        model = als_train(interactions, k=4, n_iter=5)
        assert isinstance(model, MFModel)
        assert model.user_factors.shape == (5, 4)
        assert model.item_factors.shape == (4, 4)

    def test_predictions_finite(self, interactions):
        model = als_train(interactions, k=4, n_iter=10)
        pred = model.predict(0, 0)
        assert np.isfinite(pred)

    def test_predict_user_shape(self, interactions):
        model = als_train(interactions, k=4, n_iter=5)
        preds = model.predict_user(0)
        assert preds.shape == (4,)

    def test_recommend_excludes_known(self, interactions):
        model = als_train(interactions, k=4, n_iter=10)
        recs = model.recommend(0, known_items={0, 1, 3}, n=5)
        rec_ids = {r[0] for r in recs}
        assert 0 not in rec_ids
        assert 1 not in rec_ids

    def test_recommend_returns_n(self, interactions):
        model = als_train(interactions, k=4, n_iter=5)
        recs = model.recommend(0, n=2)
        assert len(recs) <= 2


class TestUserKNN:
    def test_basic_recommendation(self):
        mat = np.array([
            [5, 3, 0, 0],
            [4, 0, 0, 1],
            [5, 4, 0, 0],
        ], dtype=float)
        im = InteractionMatrix.from_dense(mat)
        recs = user_knn_recommend(im, user_id=1, k_neighbors=2, n=2)
        assert len(recs) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)

    def test_excludes_known(self):
        mat = np.array([
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ], dtype=float)
        im = InteractionMatrix.from_dense(mat)
        recs = user_knn_recommend(im, user_id=0, n=5)
        rec_ids = {r[0] for r in recs}
        assert 0 not in rec_ids
        assert 2 not in rec_ids

    def test_empty_user(self):
        mat = np.array([[0, 0, 0], [1, 1, 0]], dtype=float)
        im = InteractionMatrix.from_dense(mat)
        recs = user_knn_recommend(im, user_id=0, n=5)
        assert isinstance(recs, list)


class TestItemKNN:
    def test_basic_recommendation(self):
        mat = np.array([
            [5, 0, 3],
            [4, 0, 0],
            [0, 5, 4],
        ], dtype=float)
        im = InteractionMatrix.from_dense(mat)
        recs = item_knn_recommend(im, user_id=0, k_neighbors=2, n=3)
        assert isinstance(recs, list)

    def test_empty_user(self):
        mat = np.array([[0, 0], [1, 1]], dtype=float)
        im = InteractionMatrix.from_dense(mat)
        recs = item_knn_recommend(im, user_id=0, n=5)
        assert recs == []


class TestPopularity:
    def test_popularity_order(self):
        mat = np.array([
            [1, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
        ], dtype=float)
        im = InteractionMatrix.from_dense(mat)
        pop = PopularityRecommender.from_interactions(im)
        recs = pop.recommend(n=3)
        assert recs[0][0] == 2  # item 2 has 3 users

    def test_excludes_known(self):
        mat = np.ones((3, 3))
        im = InteractionMatrix.from_dense(mat)
        pop = PopularityRecommender.from_interactions(im)
        recs = pop.recommend(known_items={0, 1}, n=3)
        assert recs[0][0] == 2


class TestImplicitConfidence:
    def test_formula(self):
        counts = np.array([0.0, 1.0, 5.0])
        c = implicit_confidence(counts, alpha=40.0)
        np.testing.assert_array_almost_equal(c, [1.0, 41.0, 201.0])


class TestBM25:
    def test_nonzero(self):
        mat = np.array([[3, 0, 1], [0, 2, 4]], dtype=float)
        im = InteractionMatrix.from_dense(mat)
        weighted = bm25_weight(im)
        dense = weighted.to_dense()
        assert dense[0, 0] > 0
        assert dense[0, 1] == 0.0


# ---- Session-based recommendation ----

class TestSessionState:
    def test_add_and_len(self):
        s = SessionState()
        s.add(1, np.array([0.5, 0.3]), 0.0)
        s.add(2, np.array([0.7, 0.1]), 1.0)
        assert len(s) == 2

    def test_feature_matrix(self):
        s = SessionState()
        s.add(0, np.array([1.0, 2.0]))
        s.add(1, np.array([3.0, 4.0]))
        mat = s.feature_matrix
        assert mat.shape == (2, 2)


class TestSessionCentroid:
    def test_centroid(self):
        s = SessionState()
        s.add(0, np.array([1.0, 0.0]))
        s.add(1, np.array([0.0, 1.0]))
        c = session_centroid(s)
        np.testing.assert_array_almost_equal(c, [0.5, 0.5])

    def test_empty(self):
        s = SessionState()
        c = session_centroid(s)
        assert len(c) == 0


class TestDecayCentroid:
    def test_recency_weight(self):
        s = SessionState()
        s.add(0, np.array([0.0, 0.0]))
        s.add(1, np.array([1.0, 1.0]))
        c = exponential_decay_centroid(s, half_life=1.0)
        assert c[0] > 0.5  # recent track weighted more


class TestMarkov:
    def test_transition_matrix(self):
        s = SessionState()
        for tid in [0, 1, 2, 1, 0]:
            s.add(tid, np.zeros(2))
        T = markov_transition_matrix(s, n_items=3)
        assert T.shape == (3, 3)
        np.testing.assert_array_almost_equal(T.sum(axis=1), [1.0, 1.0, 1.0])

    def test_predict(self):
        T = np.array([[0.0, 0.8, 0.2], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]])
        preds = markov_predict(T, current_item=0, n=2)
        assert preds[0][0] == 1  # highest transition prob from 0


class TestSequentialPatterns:
    def test_finds_patterns(self):
        sessions = [[0, 1, 2], [0, 1, 3], [0, 1, 2]]
        patterns = find_sequential_patterns(sessions, min_support=2, max_len=2)
        assert any(p.items == (0, 1) for p in patterns)

    def test_empty(self):
        patterns = find_sequential_patterns([], min_support=1)
        assert patterns == []


class TestSessionKNN:
    def test_basic(self):
        current = SessionState()
        current.add(0, np.array([1.0, 0.0]))
        h1 = SessionState()
        h1.add(1, np.array([0.9, 0.1]))
        h1.add(2, np.array([0.8, 0.2]))
        recs = session_knn_recommend(current, [h1], np.zeros((3, 2)), k=1, n=5)
        assert isinstance(recs, list)

    def test_empty_session(self):
        recs = session_knn_recommend(SessionState(), [], np.zeros((5, 2)))
        assert recs == []


class TestMomentumTracker:
    def test_momentum_direction(self):
        s = SessionState()
        s.add(0, np.array([0.0, 0.0]))
        s.add(1, np.array([1.0, 0.0]))
        s.add(2, np.array([2.0, 0.0]))
        tracker = SessionMomentumTracker(window=3)
        m = tracker.momentum(s)
        assert m[0] > 0

    def test_predict_next(self):
        s = SessionState()
        s.add(0, np.array([0.0]))
        s.add(1, np.array([1.0]))
        s.add(2, np.array([2.0]))
        tracker = SessionMomentumTracker()
        nxt = tracker.predict_next_feature(s)
        assert nxt[0] > 2.0

    def test_recommend(self):
        s = SessionState()
        s.add(0, np.array([0.0, 0.0]))
        s.add(1, np.array([1.0, 0.0]))
        candidates = np.array([[2.0, 0.0], [0.0, 5.0], [1.5, 0.0]])
        tracker = SessionMomentumTracker()
        recs = tracker.recommend(s, candidates, n=2)
        assert len(recs) == 2


class TestSessionMetrics:
    def test_diversity(self):
        s = SessionState()
        s.add(0, np.array([0.0, 0.0]))
        s.add(1, np.array([1.0, 1.0]))
        div = session_diversity_score(s)
        assert div > 0

    def test_diversity_single(self):
        s = SessionState()
        s.add(0, np.array([1.0]))
        assert session_diversity_score(s) == 0.0

    def test_coherence_identical(self):
        s = SessionState()
        s.add(0, np.array([1.0, 0.0]))
        s.add(1, np.array([1.0, 0.0]))
        assert session_coherence_score(s) == pytest.approx(1.0)

    def test_coherence_orthogonal(self):
        s = SessionState()
        s.add(0, np.array([1.0, 0.0]))
        s.add(1, np.array([0.0, 1.0]))
        assert session_coherence_score(s) == pytest.approx(0.0)
