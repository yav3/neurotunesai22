"""Tests for the music foundation model evaluation harness."""
import pytest
import numpy as np

from models.music_foundation.evaluation import (
    cosine_similarity,
    pairwise_cosine,
    evaluate_retrieval,
    evaluate_classification,
    evaluate_embeddings,
    ModelEvaluator,
    RetrievalMetrics,
    ClassificationMetrics,
    EmbeddingMetrics,
    EvalReport,
)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(5)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cosine_similarity(a, b) == 0.0


class TestPairwiseCosine:
    def test_identity_diagonal(self):
        emb = np.random.randn(5, 10)
        sim = pairwise_cosine(emb)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_symmetry(self):
        emb = np.random.randn(4, 8)
        sim = pairwise_cosine(emb)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)

    def test_shape(self):
        emb = np.random.randn(7, 16)
        sim = pairwise_cosine(emb)
        assert sim.shape == (7, 7)


class TestRetrievalMetrics:
    def test_perfect_retrieval(self):
        rng = np.random.RandomState(42)
        emb = np.zeros((6, 8))
        emb[:3] = rng.randn(3, 8) + np.array([5.0] * 8)
        emb[3:] = rng.randn(3, 8) + np.array([-5.0] * 8)
        labels = np.array([0, 0, 0, 1, 1, 1])
        m = evaluate_retrieval(emb, labels)
        assert m.mrr > 0.9
        assert m.recall_at_1 > 0.9
        assert m.recall_at_5 > 0.9

    def test_single_sample_returns_empty(self):
        m = evaluate_retrieval(np.array([[1.0, 2.0]]), np.array([0]))
        assert m.mrr == 0.0

    def test_all_same_label(self):
        emb = np.random.randn(5, 4)
        labels = np.array([0, 0, 0, 0, 0])
        m = evaluate_retrieval(emb, labels)
        assert m.recall_at_5 > 0.0

    def test_metrics_bounded(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 8)
        labels = np.array([i % 4 for i in range(20)])
        m = evaluate_retrieval(emb, labels)
        assert 0.0 <= m.mrr <= 1.0
        assert 0.0 <= m.recall_at_1 <= 1.0
        assert 0.0 <= m.recall_at_10 <= 1.0
        assert 0.0 <= m.map_at_k <= 1.0


class TestClassificationMetrics:
    def test_perfect_classification(self):
        logits = np.eye(4)
        labels = np.array([0, 1, 2, 3])
        m = evaluate_classification(logits, labels)
        assert m.accuracy == 1.0
        assert m.top_3_accuracy == 1.0

    def test_random_classification(self):
        rng = np.random.RandomState(42)
        logits = rng.randn(100, 5)
        labels = np.array([i % 5 for i in range(100)])
        m = evaluate_classification(logits, labels)
        assert 0.0 <= m.accuracy <= 1.0
        assert m.top_3_accuracy >= m.accuracy

    def test_empty_returns_defaults(self):
        m = evaluate_classification(np.array([]).reshape(0, 3), np.array([]))
        assert m.accuracy == 0.0

    def test_per_class_accuracy(self):
        logits = np.array([[10, 0], [10, 0], [0, 10], [0, 10]])
        labels = np.array([0, 0, 1, 1])
        m = evaluate_classification(logits, labels)
        assert m.per_class_accuracy[0] == 1.0
        assert m.per_class_accuracy[1] == 1.0
        assert len(m.confusion_diagonal) == 2


class TestEmbeddingMetrics:
    def test_well_separated_clusters(self):
        rng = np.random.RandomState(42)
        emb = np.vstack([
            rng.randn(10, 8) + 10,
            rng.randn(10, 8) - 10,
        ])
        labels = np.array([0] * 10 + [1] * 10)
        m = evaluate_embeddings(emb, labels)
        assert m.silhouette > 0.5
        assert m.inter_class_distance > m.intra_class_distance

    def test_too_few_samples(self):
        m = evaluate_embeddings(np.random.randn(2, 4), np.array([0, 1]))
        assert m.silhouette == 0.0

    def test_single_class_returns_empty(self):
        m = evaluate_embeddings(np.random.randn(5, 4), np.array([0, 0, 0, 0, 0]))
        assert m.silhouette == 0.0

    def test_alignment_negative_for_similar_pairs(self):
        rng = np.random.RandomState(42)
        emb = np.vstack([
            rng.randn(5, 4) * 0.01 + np.array([1, 0, 0, 0]),
            rng.randn(5, 4) * 0.01 + np.array([0, 1, 0, 0]),
        ])
        labels = np.array([0] * 5 + [1] * 5)
        m = evaluate_embeddings(emb, labels)
        assert m.alignment < 0.0


class TestEvalReport:
    def test_summary_keys(self):
        report = EvalReport()
        s = report.summary()
        assert "mrr" in s
        assert "recall@1" in s
        assert "accuracy" in s
        assert "silhouette" in s
        assert "alignment" in s
        assert "uniformity" in s

    def test_summary_reflects_values(self):
        report = EvalReport(
            retrieval=RetrievalMetrics(mrr=0.8),
            classification=ClassificationMetrics(accuracy=0.9),
        )
        s = report.summary()
        assert s["mrr"] == 0.8
        assert s["accuracy"] == 0.9


class TestModelEvaluator:
    def test_evaluate_from_embeddings(self):
        rng = np.random.RandomState(42)
        emb = np.vstack([
            rng.randn(10, 16) + 5,
            rng.randn(10, 16) - 5,
        ])
        labels = np.array([0] * 10 + [1] * 10)
        logits = np.zeros((20, 2))
        logits[:10, 0] = 1.0
        logits[10:, 1] = 1.0

        from models.music_foundation.inference import MusicInferencePipeline, InferenceConfig
        from models.music_foundation.encoder import EncoderConfig

        cfg = InferenceConfig(
            encoder=EncoderConfig(latent_dim=16, hidden_dim=32, audio_dim=16, semantic_dim=8),
        )
        pipe = MusicInferencePipeline(cfg)
        evaluator = ModelEvaluator(pipe)

        report = evaluator.evaluate_from_embeddings(emb, labels, logits)
        assert report.n_samples == 20
        assert report.retrieval.mrr > 0.0
        assert report.classification.accuracy > 0.0
        assert report.embedding.silhouette > 0.0

    def test_evaluate_from_embeddings_no_logits(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(10, 8)
        labels = np.array([0] * 5 + [1] * 5)

        from models.music_foundation.inference import MusicInferencePipeline, InferenceConfig
        from models.music_foundation.encoder import EncoderConfig

        cfg = InferenceConfig(
            encoder=EncoderConfig(latent_dim=8, hidden_dim=16, audio_dim=8, semantic_dim=4),
        )
        pipe = MusicInferencePipeline(cfg)
        evaluator = ModelEvaluator(pipe)

        report = evaluator.evaluate_from_embeddings(emb, labels)
        assert report.n_samples == 10
        assert report.classification.accuracy == 0.0
