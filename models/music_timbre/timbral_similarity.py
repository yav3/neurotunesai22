"""Timbral similarity: perceptual distance metrics, embedding spaces, and retrieval."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class DistanceMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MAHALANOBIS = "mahalanobis"
    EARTH_MOVER = "earth_mover"


@dataclass
class TimbreDescriptor:
    """Multi-dimensional timbre descriptor vector."""
    spectral_centroid: float = 0.0
    spectral_spread: float = 0.0
    spectral_skewness: float = 0.0
    spectral_kurtosis: float = 0.0
    spectral_flux: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_flatness: float = 0.0
    zero_crossing_rate: float = 0.0
    mfcc: np.ndarray = field(default_factory=lambda: np.zeros(13))
    attack_time: float = 0.0
    decay_time: float = 0.0
    temporal_centroid: float = 0.0
    roughness: float = 0.0
    brightness: float = 0.0

    def to_vector(self) -> np.ndarray:
        scalars = np.array([
            self.spectral_centroid, self.spectral_spread,
            self.spectral_skewness, self.spectral_kurtosis,
            self.spectral_flux, self.spectral_rolloff,
            self.spectral_flatness, self.zero_crossing_rate,
            self.attack_time, self.decay_time,
            self.temporal_centroid, self.roughness, self.brightness,
        ])
        return np.concatenate([scalars, self.mfcc])


@dataclass
class SimilarityResult:
    distance: float
    metric: DistanceMetric
    rank: int = 0
    query_id: str = ""
    target_id: str = ""


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def mahalanobis_distance(a: np.ndarray, b: np.ndarray,
                         cov_inv: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(diff @ cov_inv @ diff))


def earth_mover_distance(p: np.ndarray, q: np.ndarray) -> float:
    """1-D Earth mover's distance (Wasserstein-1)."""
    p_norm = p / (np.sum(p) + 1e-12)
    q_norm = q / (np.sum(q) + 1e-12)
    return float(np.sum(np.abs(np.cumsum(p_norm) - np.cumsum(q_norm))))


def compute_distance(a: np.ndarray, b: np.ndarray,
                     metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
                     cov_inv: Optional[np.ndarray] = None) -> float:
    if metric == DistanceMetric.EUCLIDEAN:
        return euclidean_distance(a, b)
    elif metric == DistanceMetric.COSINE:
        return cosine_distance(a, b)
    elif metric == DistanceMetric.MAHALANOBIS:
        if cov_inv is None:
            return euclidean_distance(a, b)
        return mahalanobis_distance(a, b, cov_inv)
    elif metric == DistanceMetric.EARTH_MOVER:
        return earth_mover_distance(a, b)
    return euclidean_distance(a, b)


def descriptor_distance(a: TimbreDescriptor, b: TimbreDescriptor,
                        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
                        weights: Optional[np.ndarray] = None) -> float:
    va, vb = a.to_vector(), b.to_vector()
    if weights is not None:
        va = va * weights
        vb = vb * weights
    return compute_distance(va, vb, metric)


def build_distance_matrix(descriptors: List[TimbreDescriptor],
                          metric: DistanceMetric = DistanceMetric.EUCLIDEAN
                          ) -> np.ndarray:
    n = len(descriptors)
    vecs = [d.to_vector() for d in descriptors]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_distance(vecs[i], vecs[j], metric)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def knn_search(query: TimbreDescriptor,
               database: List[TimbreDescriptor],
               k: int = 5,
               metric: DistanceMetric = DistanceMetric.EUCLIDEAN
               ) -> List[SimilarityResult]:
    qv = query.to_vector()
    dists = []
    for i, item in enumerate(database):
        d = compute_distance(qv, item.to_vector(), metric)
        dists.append((i, d))
    dists.sort(key=lambda x: x[1])
    results = []
    for rank, (idx, dist) in enumerate(dists[:k]):
        results.append(SimilarityResult(
            distance=dist, metric=metric, rank=rank,
            target_id=str(idx),
        ))
    return results


def timbre_embedding(descriptor: TimbreDescriptor,
                     projection: np.ndarray) -> np.ndarray:
    v = descriptor.to_vector()
    return v @ projection


def fit_projection(descriptors: List[TimbreDescriptor],
                   target_dim: int = 8) -> np.ndarray:
    """PCA-based dimensionality reduction for timbre space."""
    vecs = np.array([d.to_vector() for d in descriptors])
    mean = vecs.mean(axis=0)
    centered = vecs - mean
    cov = centered.T @ centered / max(len(vecs) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:target_dim]
    return eigvecs[:, idx]


def perceptual_timbre_distance(a: TimbreDescriptor,
                               b: TimbreDescriptor) -> float:
    """Weighted distance emphasizing perceptually salient features."""
    weights = np.ones(a.to_vector().shape[0])
    weights[0] = 2.0   # spectral centroid
    weights[5] = 1.5   # rolloff
    weights[6] = 1.5   # flatness
    weights[9] = 1.8   # attack time
    weights[12] = 1.5  # brightness
    weights[13:] = 1.2  # MFCCs
    return descriptor_distance(a, b, DistanceMetric.EUCLIDEAN, weights)


def temporal_timbre_distance(seq_a: List[TimbreDescriptor],
                             seq_b: List[TimbreDescriptor]) -> float:
    """DTW-based distance between temporal timbre sequences."""
    na, nb = len(seq_a), len(seq_b)
    if na == 0 or nb == 0:
        return float('inf')
    cost = np.full((na + 1, nb + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            d = descriptor_distance(seq_a[i-1], seq_b[j-1])
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
    return float(cost[na, nb] / (na + nb))


def cluster_timbres(descriptors: List[TimbreDescriptor],
                    n_clusters: int = 4,
                    max_iter: int = 50) -> np.ndarray:
    """K-means clustering in timbre space."""
    vecs = np.array([d.to_vector() for d in descriptors])
    n = len(vecs)
    if n <= n_clusters:
        return np.arange(n)
    rng = np.random.RandomState(42)
    centers = vecs[rng.choice(n, n_clusters, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.array([
            [euclidean_distance(v, c) for c in centers] for v in vecs
        ])
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = vecs[mask].mean(axis=0)
    return labels


def timbre_centroid(descriptors: List[TimbreDescriptor]) -> TimbreDescriptor:
    """Compute the centroid descriptor of a collection."""
    if not descriptors:
        return TimbreDescriptor()
    vecs = np.array([d.to_vector() for d in descriptors])
    mean = vecs.mean(axis=0)
    result = TimbreDescriptor()
    result.spectral_centroid = float(mean[0])
    result.spectral_spread = float(mean[1])
    result.spectral_skewness = float(mean[2])
    result.spectral_kurtosis = float(mean[3])
    result.spectral_flux = float(mean[4])
    result.spectral_rolloff = float(mean[5])
    result.spectral_flatness = float(mean[6])
    result.zero_crossing_rate = float(mean[7])
    result.attack_time = float(mean[8])
    result.decay_time = float(mean[9])
    result.temporal_centroid = float(mean[10])
    result.roughness = float(mean[11])
    result.brightness = float(mean[12])
    result.mfcc = mean[13:]
    return result
