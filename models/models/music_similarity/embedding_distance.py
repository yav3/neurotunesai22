"""Distance metrics for music embeddings and distributions."""
from __future__ import annotations

import numpy as np


class EmbeddingDistanceMetrics:
    """Pairwise and distributional distance functions for music feature vectors."""

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-30 or nb < 1e-30:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))

    @staticmethod
    def mahalanobis_distance(
        a: np.ndarray, b: np.ndarray, cov_inv: np.ndarray
    ) -> float:
        d = np.asarray(a) - np.asarray(b)
        return float(np.sqrt(d @ cov_inv @ d))

    @staticmethod
    def earth_movers_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        a = np.asarray(hist_a, dtype=float)
        b = np.asarray(hist_b, dtype=float)
        sa, sb = a.sum(), b.sum()
        if sa > 0:
            a = a / sa
        if sb > 0:
            b = b / sb
        return float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))))

    @staticmethod
    def kl_divergence(
        p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10
    ) -> float:
        p_ = np.asarray(p, dtype=float) + epsilon
        q_ = np.asarray(q, dtype=float) + epsilon
        p_ = p_ / p_.sum()
        q_ = q_ / q_.sum()
        return float(np.sum(p_ * np.log(p_ / q_)))

    @staticmethod
    def symmetric_kl(
        p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10
    ) -> float:
        return (
            EmbeddingDistanceMetrics.kl_divergence(p, q, epsilon)
            + EmbeddingDistanceMetrics.kl_divergence(q, p, epsilon)
        ) / 2

    @staticmethod
    def bhattacharyya_distance(
        p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10
    ) -> float:
        p_ = np.asarray(p, dtype=float) + epsilon
        q_ = np.asarray(q, dtype=float) + epsilon
        p_ = p_ / p_.sum()
        q_ = q_ / q_.sum()
        bc = np.sum(np.sqrt(p_ * q_))
        return float(-np.log(max(bc, 1e-30)))

    @staticmethod
    def pairwise_distance_matrix(
        embeddings: np.ndarray, metric: str = "cosine"
    ) -> np.ndarray:
        n = embeddings.shape[0]
        mat = np.zeros((n, n))
        fn = {
            "cosine": EmbeddingDistanceMetrics.cosine_distance,
            "euclidean": EmbeddingDistanceMetrics.euclidean_distance,
            "manhattan": EmbeddingDistanceMetrics.manhattan_distance,
        }.get(metric, EmbeddingDistanceMetrics.cosine_distance)
        for i in range(n):
            for j in range(i + 1, n):
                d = fn(embeddings[i], embeddings[j])
                mat[i, j] = d
                mat[j, i] = d
        return mat


class StructuralSimilarity:
    """Sequence-level similarity using SSM, DTW, recurrence."""

    @staticmethod
    def self_similarity_matrix(feature_sequence: np.ndarray) -> np.ndarray:
        if feature_sequence.ndim == 1:
            feature_sequence = feature_sequence.reshape(-1, 1)
        norms = np.linalg.norm(feature_sequence, axis=1, keepdims=True)
        norms = np.where(norms < 1e-30, 1.0, norms)
        normed = feature_sequence / norms
        return normed @ normed.T

    @staticmethod
    def novelty_curve(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
        n = ssm.shape[0]
        if n < 2:
            return np.array([0.0])
        k = min(kernel_size, n)
        kernel = np.ones((k, k))
        half = k // 2
        kernel[:half, half:] = -1
        kernel[half:, :half] = -1
        novelty = np.zeros(n)
        for i in range(n):
            r0 = max(0, i - half)
            r1 = min(n, i + half)
            c0, c1 = r0, r1
            patch = ssm[r0:r1, c0:c1]
            ks = min(patch.shape[0], patch.shape[1])
            if ks < 2:
                continue
            novelty[i] = abs(np.sum(patch * kernel[:ks, :ks]))
        return novelty

    @staticmethod
    def recurrence_matrix(
        feature_sequence: np.ndarray,
        threshold: float | None = None,
        k_nearest: int | None = None,
    ) -> np.ndarray:
        ssm = StructuralSimilarity.self_similarity_matrix(feature_sequence)
        if k_nearest is not None:
            rec = np.zeros_like(ssm, dtype=bool)
            for i in range(ssm.shape[0]):
                idx = np.argsort(-ssm[i])[:k_nearest]
                rec[i, idx] = True
            return rec
        if threshold is None:
            threshold = np.median(ssm)
        return ssm >= threshold

    @staticmethod
    def dtw_distance(
        seq_a: np.ndarray, seq_b: np.ndarray, metric: str = "euclidean"
    ) -> float:
        if seq_a.ndim == 1:
            seq_a = seq_a.reshape(-1, 1)
        if seq_b.ndim == 1:
            seq_b = seq_b.reshape(-1, 1)
        n, m = len(seq_a), len(seq_b)
        if n == 0 or m == 0:
            return 0.0
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return float(cost[n, m])

    @staticmethod
    def cover_song_similarity(
        chroma_a: np.ndarray, chroma_b: np.ndarray
    ) -> float:
        if chroma_a.ndim == 1:
            chroma_a = chroma_a.reshape(1, -1)
        if chroma_b.ndim == 1:
            chroma_b = chroma_b.reshape(1, -1)
        best = np.inf
        mean_a = chroma_a.mean(axis=0)
        mean_b = chroma_b.mean(axis=0)
        for shift in range(12):
            shifted = np.roll(mean_b, shift)
            d = np.linalg.norm(mean_a - shifted)
            if d < best:
                best = d
        return float(best)

    @staticmethod
    def harmonic_similarity(
        chroma_a: np.ndarray, chroma_b: np.ndarray
    ) -> float:
        if len(chroma_a) != 12 or len(chroma_b) != 12:
            return float(np.linalg.norm(np.asarray(chroma_a) - np.asarray(chroma_b)))
        r1, r2 = 1.0, 1.0
        angles = np.arange(12) * (7 * 2 * np.pi / 12)
        angles_m3 = np.arange(12) * (3 * 2 * np.pi / 12)
        angles_M3 = np.arange(12) * (4 * 2 * np.pi / 12)
        def tonal_centroid(c):
            c = np.asarray(c, dtype=float)
            s = c.sum()
            if s < 1e-30:
                c = np.ones(12) / 12
            else:
                c = c / s
            return np.array([
                r1 * np.sum(c * np.sin(angles)),
                r1 * np.sum(c * np.cos(angles)),
                r2 * np.sum(c * np.sin(angles_m3)),
                r2 * np.sum(c * np.cos(angles_m3)),
                r2 * np.sum(c * np.sin(angles_M3)),
                r2 * np.sum(c * np.cos(angles_M3)),
            ])
        tc_a = tonal_centroid(chroma_a)
        tc_b = tonal_centroid(chroma_b)
        return float(np.linalg.norm(tc_a - tc_b))
