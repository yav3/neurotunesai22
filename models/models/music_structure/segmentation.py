"""Self-similarity and recurrence-based music segmentation."""
from __future__ import annotations

import numpy as np


class SelfSimilarityMatrix:
    """Compute self-similarity matrix from a feature sequence."""

    def compute(
        self,
        features: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        n = features.shape[0]
        if n == 0:
            return np.zeros((0, 0))

        if metric == "cosine":
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            normed = features / norms
            sim = normed @ normed.T
            sim = np.clip(sim, -1.0, 1.0)
        elif metric == "euclidean":
            dists = np.zeros((n, n))
            for i in range(n):
                diff = features - features[i]
                dists[i] = np.sqrt(np.sum(diff ** 2, axis=1))
            max_d = dists.max() if dists.max() > 0 else 1.0
            sim = 1.0 - dists / max_d
        elif metric == "dot":
            sim = features @ features.T
            max_val = np.abs(sim).max()
            if max_val > 0:
                sim = sim / max_val
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return sim


class RecurrenceMatrix:
    """Thresholded self-similarity for recurrence/repetition detection."""

    def compute(
        self,
        features: np.ndarray,
        k_nearest: int = 5,
        metric: str = "cosine",
    ) -> np.ndarray:
        ssm = SelfSimilarityMatrix()
        sim = ssm.compute(features, metric=metric)
        n = sim.shape[0]
        R = np.zeros((n, n), dtype=bool)
        for i in range(n):
            row = sim[i].copy()
            row[i] = -np.inf
            if n <= k_nearest:
                R[i, :] = True
            else:
                indices = np.argpartition(row, -k_nearest)[-k_nearest:]
                R[i, indices] = True
        R = R | R.T
        return R.astype(float)

    @staticmethod
    def transitive_closure(R: np.ndarray) -> np.ndarray:
        n = R.shape[0]
        C = R.astype(bool).copy()
        changed = True
        max_iter = n
        for _ in range(max_iter):
            if not changed:
                break
            changed = False
            new_C = C | (C @ C).astype(bool)
            if not np.array_equal(new_C, C):
                changed = True
                C = new_C
        return C.astype(float)


def novelty_curve(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    n = ssm.shape[0]
    if n == 0:
        return np.array([])

    half = kernel_size // 2
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = 1.0
    kernel[half:, half:] = 1.0
    kernel[:half, half:] = -1.0
    kernel[half:, :half] = -1.0

    novelty = np.zeros(n)
    for i in range(half, n - half):
        region = ssm[i - half : i + half, i - half : i + half]
        if region.shape == kernel.shape:
            novelty[i] = np.sum(region * kernel)

    novelty = np.maximum(novelty, 0)
    max_val = novelty.max()
    if max_val > 0:
        novelty = novelty / max_val
    return novelty


def segment_boundaries(
    novelty: np.ndarray,
    threshold: float = 0.3,
    min_segment_length: int = 10,
) -> list[int]:
    boundaries = [0]
    for i in range(1, len(novelty) - 1):
        if novelty[i] > threshold:
            if novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
                if i - boundaries[-1] >= min_segment_length:
                    boundaries.append(i)
    return boundaries
