"""Metric learning for music similarity: distance functions, triplet sampling,
contrastive loss, margin-based ranking, and embedding calibration.  Pure numpy."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class DistanceMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    MAHALANOBIS = "mahalanobis"


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def mahalanobis(a: np.ndarray, b: np.ndarray, cov_inv: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(diff @ cov_inv @ diff))


METRIC_FNS = {
    DistanceMetric.EUCLIDEAN: euclidean,
    DistanceMetric.COSINE: cosine_distance,
    DistanceMetric.MANHATTAN: manhattan,
    DistanceMetric.CHEBYSHEV: chebyshev,
}


def pairwise_distances(
    X: np.ndarray, metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n))
    fn = METRIC_FNS[metric]
    for i in range(n):
        for j in range(i + 1, n):
            d = fn(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return D


@dataclass
class Triplet:
    anchor: int
    positive: int
    negative: int
    margin: float = 0.0


def triplet_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray,
    margin: float = 1.0,
) -> float:
    dp = euclidean(anchor, positive)
    dn = euclidean(anchor, negative)
    return float(max(0.0, dp - dn + margin))


def batch_triplet_loss(
    embeddings: np.ndarray, triplets: List[Triplet], margin: float = 1.0,
) -> float:
    if not triplets:
        return 0.0
    total = 0.0
    for t in triplets:
        total += triplet_loss(embeddings[t.anchor], embeddings[t.positive],
                              embeddings[t.negative], margin)
    return total / len(triplets)


def contrastive_loss(
    a: np.ndarray, b: np.ndarray, label: int, margin: float = 1.0,
) -> float:
    d = euclidean(a, b)
    if label == 1:
        return 0.5 * d ** 2
    return 0.5 * max(0.0, margin - d) ** 2


def hard_negative_mining(
    embeddings: np.ndarray, labels: np.ndarray, n_triplets: int = 100,
) -> List[Triplet]:
    n = len(labels)
    D = pairwise_distances(embeddings)
    triplets: List[Triplet] = []
    for _ in range(n_triplets):
        anchor = np.random.randint(n)
        same_mask = labels == labels[anchor]
        diff_mask = ~same_mask
        same_idx = np.where(same_mask)[0]
        diff_idx = np.where(diff_mask)[0]
        if len(same_idx) < 2 or len(diff_idx) == 0:
            continue
        positives = same_idx[same_idx != anchor]
        pos = positives[np.argmax(D[anchor, positives])]
        neg = diff_idx[np.argmin(D[anchor, diff_idx])]
        triplets.append(Triplet(anchor, int(pos), int(neg)))
    return triplets


def semi_hard_negative_mining(
    embeddings: np.ndarray, labels: np.ndarray, margin: float = 1.0,
    n_triplets: int = 100,
) -> List[Triplet]:
    n = len(labels)
    D = pairwise_distances(embeddings)
    triplets: List[Triplet] = []
    for _ in range(n_triplets * 3):
        if len(triplets) >= n_triplets:
            break
        anchor = np.random.randint(n)
        same_mask = labels == labels[anchor]
        diff_mask = ~same_mask
        same_idx = np.where(same_mask)[0]
        diff_idx = np.where(diff_mask)[0]
        if len(same_idx) < 2 or len(diff_idx) == 0:
            continue
        positives = same_idx[same_idx != anchor]
        pos = int(positives[np.random.randint(len(positives))])
        dp = D[anchor, pos]
        candidates = diff_idx[(D[anchor, diff_idx] > dp) &
                              (D[anchor, diff_idx] < dp + margin)]
        if len(candidates) == 0:
            continue
        neg = int(candidates[np.random.randint(len(candidates))])
        triplets.append(Triplet(anchor, pos, neg, margin))
    return triplets


def recall_at_k(
    embeddings: np.ndarray, labels: np.ndarray, k: int = 5,
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
) -> float:
    D = pairwise_distances(embeddings, metric)
    n = len(labels)
    hits = 0
    for i in range(n):
        dists = D[i].copy()
        dists[i] = np.inf
        nearest = np.argsort(dists)[:k]
        if np.any(labels[nearest] == labels[i]):
            hits += 1
    return hits / n


def mean_average_precision(
    embeddings: np.ndarray, labels: np.ndarray,
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
) -> float:
    D = pairwise_distances(embeddings, metric)
    n = len(labels)
    aps = []
    for i in range(n):
        dists = D[i].copy()
        dists[i] = np.inf
        ranked = np.argsort(dists)
        relevant = labels[ranked] == labels[i]
        if not np.any(relevant):
            continue
        cumsum = np.cumsum(relevant)
        precisions = cumsum / (np.arange(len(relevant)) + 1)
        ap = float(np.sum(precisions * relevant) / np.sum(relevant))
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def normalized_discounted_cumulative_gain(
    embeddings: np.ndarray, relevance: np.ndarray, k: int = 10,
    metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
) -> float:
    D = pairwise_distances(embeddings, metric)
    n = len(relevance)
    ndcgs = []
    for i in range(n):
        dists = D[i].copy()
        dists[i] = np.inf
        ranked = np.argsort(dists)[:k]
        gains = relevance[i, ranked] if relevance.ndim == 2 else np.zeros(k)
        discounts = np.log2(np.arange(k) + 2)
        dcg = float(np.sum(gains / discounts))
        ideal = np.sort(gains)[::-1]
        idcg = float(np.sum(ideal / discounts))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def embedding_calibration(
    embeddings: np.ndarray, target_mean: float = 0.0, target_std: float = 1.0,
) -> np.ndarray:
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (embeddings - mean) / std * target_std + target_mean


def embedding_whitening(embeddings: np.ndarray) -> np.ndarray:
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return centered @ W


def proxy_nca_loss(
    embeddings: np.ndarray, labels: np.ndarray, proxies: np.ndarray,
    temperature: float = 0.1,
) -> float:
    n = len(labels)
    unique_labels = np.unique(labels)
    total = 0.0
    for i in range(n):
        pos_proxy = proxies[labels[i]]
        sim_pos = -euclidean(embeddings[i], pos_proxy) / temperature
        log_denom_parts = []
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            sim_neg = -euclidean(embeddings[i], proxies[lbl]) / temperature
            log_denom_parts.append(sim_neg)
        if not log_denom_parts:
            continue
        max_neg = max(log_denom_parts)
        log_sum_exp = max_neg + np.log(sum(np.exp(s - max_neg) for s in log_denom_parts))
        total += -sim_pos + log_sum_exp
    return float(total / max(n, 1))
