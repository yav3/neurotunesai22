"""Self-similarity matrix analysis: SSM computation, repetition detection, segmentation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StructuralSegment:
    """A labeled segment of musical structure."""
    label: str
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


def self_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Compute the self-similarity matrix (SSM) from a feature matrix (n_frames x n_features).

    SSM[i,j] = cosine similarity between frame i and frame j.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = features / norms
    return normed @ normed.T


def recurrence_matrix(ssm: np.ndarray, k_nearest: int = 10) -> np.ndarray:
    """Binary recurrence matrix: R[i,j] = 1 iff j is among the k nearest neighbors of i."""
    n = ssm.shape[0]
    R = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        indices = np.argsort(-ssm[i])[:k_nearest]
        R[i, indices] = 1.0
    return R


def diagonal_score(ssm: np.ndarray, lag: int, min_length: int = 4) -> float:
    """Score of a diagonal stripe at a given lag. High score = repeating pattern with period lag."""
    n = ssm.shape[0]
    if lag <= 0 or lag >= n:
        return 0.0
    diag = np.array([ssm[i, i + lag] for i in range(n - lag)])
    if len(diag) < min_length:
        return 0.0
    return float(np.mean(diag))


def find_repetitions(ssm: np.ndarray, threshold: float = 0.7, min_lag: int = 4,
                     max_lag: int | None = None) -> list[tuple[int, float]]:
    """Find lag values with high diagonal scores (repetitions). Returns [(lag, score)] desc."""
    n = ssm.shape[0]
    if max_lag is None:
        max_lag = n // 2
    results: list[tuple[int, float]] = []
    for lag in range(min_lag, min(max_lag + 1, n)):
        score = diagonal_score(ssm, lag)
        if score >= threshold:
            results.append((lag, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def novelty_curve(ssm: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    """Novelty curve from the SSM using a checkerboard kernel. Peaks = structural boundaries."""
    n = ssm.shape[0]
    half = kernel_size // 2
    novelty = np.zeros(n)
    for i in range(half, n - half):
        tl = ssm[i - half:i, i - half:i].mean()
        br = ssm[i:i + half, i:i + half].mean()
        tr = ssm[i - half:i, i:i + half].mean()
        bl = ssm[i:i + half, i - half:i].mean()
        novelty[i] = max(0.0, (tl + br) / 2.0 - (tr + bl) / 2.0)
    return novelty


def segment_from_novelty(novelty: np.ndarray, threshold: float | None = None) -> list[int]:
    """Find boundary frames from a novelty curve (peak picking)."""
    if threshold is None:
        threshold = float(np.mean(novelty) + np.std(novelty))
    boundaries = [0]
    for i in range(1, len(novelty) - 1):
        if novelty[i] > novelty[i - 1] and novelty[i] > novelty[i + 1] and novelty[i] >= threshold:
            boundaries.append(i)
    return boundaries


def label_segments(boundaries: list[int], total_frames: int,
                   ssm: np.ndarray | None = None) -> list[StructuralSegment]:
    """Assign labels (A, B, C, ...) to segments. Similar segments share labels if ssm provided."""
    segments: list[StructuralSegment] = []
    ends = boundaries[1:] + [total_frames]
    label_map: dict[int, str] = {}
    next_label = 0

    for idx, (start, end) in enumerate(zip(boundaries, ends)):
        if ssm is not None and idx > 0:
            best_match = -1
            best_sim = 0.0
            for prev_idx in range(idx):
                ps = segments[prev_idx].start_frame
                pe = segments[prev_idx].end_frame
                seg_len = min(end - start, pe - ps)
                if seg_len < 2:
                    continue
                sim = float(np.mean([ssm[start + k, ps + k] for k in range(seg_len)]))
                if sim > best_sim and sim > 0.6:
                    best_sim = sim
                    best_match = prev_idx
            if best_match >= 0:
                label_map[idx] = segments[best_match].label
            else:
                label_map[idx] = chr(ord("A") + next_label)
                next_label += 1
        else:
            label_map[idx] = chr(ord("A") + next_label)
            next_label += 1
        segments.append(StructuralSegment(label=label_map[idx], start_frame=start, end_frame=end))
    return segments


def form_string(segments: list[StructuralSegment]) -> str:
    """Compact form representation, e.g. 'A B A C A'."""
    return " ".join(s.label for s in segments)


def form_similarity(form_a: str, form_b: str) -> float:
    """Similarity between two form strings via normalized edit distance."""
    a_parts = form_a.split()
    b_parts = form_b.split()
    m, n = len(a_parts), len(b_parts)
    if m == 0 and n == 0:
        return 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a_parts[i - 1] == b_parts[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    max_len = max(m, n)
    return 1.0 - dp[m][n] / max_len if max_len > 0 else 1.0


def symmetry_score(ssm: np.ndarray) -> float:
    """How symmetric the SSM is (1.0 for perfect cosine similarity SSMs)."""
    return float(1.0 - np.mean(np.abs(ssm - ssm.T)))


def repetitiveness(ssm: np.ndarray, exclude_diagonal: bool = True) -> float:
    """Global repetitiveness: mean off-diagonal SSM value."""
    n = ssm.shape[0]
    if n < 2:
        return 0.0
    if exclude_diagonal:
        mask = ~np.eye(n, dtype=bool)
        return float(np.mean(ssm[mask]))
    return float(np.mean(ssm))
