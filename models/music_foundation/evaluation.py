"""Evaluation harness for the music foundation model.

Computes retrieval, classification, and embedding quality metrics
on held-out data to assess model quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .encoder import MusicFoundationModel, EncoderConfig
from .inference import MusicInferencePipeline, InferenceConfig, InferenceResult


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    return normed @ normed.T


@dataclass
class RetrievalMetrics:
    mrr: float = 0.0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    map_at_k: float = 0.0


@dataclass
class ClassificationMetrics:
    accuracy: float = 0.0
    top_3_accuracy: float = 0.0
    per_class_accuracy: dict[int, float] = field(default_factory=dict)
    confusion_diagonal: list[float] = field(default_factory=list)


@dataclass
class EmbeddingMetrics:
    silhouette: float = 0.0
    inter_class_distance: float = 0.0
    intra_class_distance: float = 0.0
    alignment: float = 0.0
    uniformity: float = 0.0


@dataclass
class EvalReport:
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    classification: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    embedding: EmbeddingMetrics = field(default_factory=EmbeddingMetrics)
    n_samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        return {
            "mrr": self.retrieval.mrr,
            "recall@1": self.retrieval.recall_at_1,
            "recall@5": self.retrieval.recall_at_5,
            "accuracy": self.classification.accuracy,
            "top3_accuracy": self.classification.top_3_accuracy,
            "silhouette": self.embedding.silhouette,
            "alignment": self.embedding.alignment,
            "uniformity": self.embedding.uniformity,
        }


def evaluate_retrieval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> RetrievalMetrics:
    n = len(embeddings)
    if n < 2:
        return RetrievalMetrics()

    sim_matrix = pairwise_cosine(embeddings)
    np.fill_diagonal(sim_matrix, -1.0)

    reciprocal_ranks = []
    recall_1 = recall_5 = recall_10 = 0
    ap_sum = 0.0

    for i in range(n):
        ranked = np.argsort(-sim_matrix[i])
        relevant = set(np.where(labels == labels[i])[0]) - {i}
        if not relevant:
            continue

        first_hit = None
        hits = 0
        precision_sum = 0.0

        for rank, idx in enumerate(ranked[:k]):
            if idx in relevant:
                hits += 1
                precision_sum += hits / (rank + 1)
                if first_hit is None:
                    first_hit = rank

        rr = 1.0 / (first_hit + 1) if first_hit is not None else 0.0
        reciprocal_ranks.append(rr)

        if first_hit is not None and first_hit < 1:
            recall_1 += 1
        if first_hit is not None and first_hit < 5:
            recall_5 += 1
        if first_hit is not None and first_hit < 10:
            recall_10 += 1

        ap_sum += precision_sum / min(len(relevant), k) if relevant else 0.0

    total = len(reciprocal_ranks) or 1
    return RetrievalMetrics(
        mrr=float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        recall_at_1=recall_1 / total,
        recall_at_5=recall_5 / total,
        recall_at_10=recall_10 / total,
        map_at_k=ap_sum / total,
    )


def evaluate_classification(
    logits: np.ndarray,
    labels: np.ndarray,
) -> ClassificationMetrics:
    n = len(logits)
    if n == 0:
        return ClassificationMetrics()

    preds = np.argmax(logits, axis=1)
    accuracy = float(np.mean(preds == labels))

    top3_correct = 0
    for i in range(n):
        top3 = np.argsort(-logits[i])[:3]
        if labels[i] in top3:
            top3_correct += 1
    top3_accuracy = top3_correct / n

    classes = np.unique(labels)
    per_class = {}
    diag = []
    for c in classes:
        mask = labels == c
        if mask.sum() > 0:
            acc = float(np.mean(preds[mask] == c))
            per_class[int(c)] = acc
            diag.append(acc)

    return ClassificationMetrics(
        accuracy=accuracy,
        top_3_accuracy=top3_accuracy,
        per_class_accuracy=per_class,
        confusion_diagonal=diag,
    )


def evaluate_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> EmbeddingMetrics:
    n = len(embeddings)
    if n < 4:
        return EmbeddingMetrics()

    classes = np.unique(labels)
    if len(classes) < 2:
        return EmbeddingMetrics()

    intra_dists = []
    inter_dists = []
    for c in classes:
        mask = labels == c
        class_embs = embeddings[mask]
        other_embs = embeddings[~mask]

        if len(class_embs) > 1:
            for i in range(len(class_embs)):
                for j in range(i + 1, len(class_embs)):
                    intra_dists.append(np.linalg.norm(class_embs[i] - class_embs[j]))

        if len(class_embs) > 0 and len(other_embs) > 0:
            centroid = class_embs.mean(axis=0)
            other_centroid = other_embs.mean(axis=0)
            inter_dists.append(np.linalg.norm(centroid - other_centroid))

    intra = float(np.mean(intra_dists)) if intra_dists else 0.0
    inter = float(np.mean(inter_dists)) if inter_dists else 0.0

    silhouette = 0.0
    if intra > 0 and inter > 0:
        silhouette = (inter - intra) / max(inter, intra)

    sim = pairwise_cosine(embeddings)
    pos_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                pos_sims.append(sim[i, j])
    alignment = -float(np.mean(pos_sims)) if pos_sims else 0.0

    uniformity = float(np.log(np.mean(np.exp(-2 * (1 - sim[np.triu_indices(n, k=1)]))) + 1e-12))

    return EmbeddingMetrics(
        silhouette=silhouette,
        inter_class_distance=inter,
        intra_class_distance=intra,
        alignment=alignment,
        uniformity=uniformity,
    )


class ModelEvaluator:
    def __init__(self, pipeline: MusicInferencePipeline) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        audios: list[np.ndarray],
        labels: np.ndarray,
        sr: int = 22050,
    ) -> EvalReport:
        results = self.pipeline.infer_batch(audios, sr)
        embeddings = np.vstack([r.embedding.reshape(1, -1) for r in results])

        genre_logits = np.vstack([r.genre_logits.reshape(1, -1) for r in results])

        return EvalReport(
            retrieval=evaluate_retrieval(embeddings, labels),
            classification=evaluate_classification(genre_logits, labels),
            embedding=evaluate_embeddings(embeddings, labels),
            n_samples=len(audios),
        )

    def evaluate_from_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        logits: np.ndarray | None = None,
    ) -> EvalReport:
        report = EvalReport(
            retrieval=evaluate_retrieval(embeddings, labels),
            embedding=evaluate_embeddings(embeddings, labels),
            n_samples=len(embeddings),
        )
        if logits is not None:
            report.classification = evaluate_classification(logits, labels)
        return report
