"""Advanced playlist coherence: Laplacian smoothness, spectral flow, segment detection.

Measures how well a playlist hangs together as a listening experience beyond
simple pairwise similarity — graph-Laplacian energy quantifies feature-space
smoothness of the ordering, spectral flow tracks gradual timbral drift, and
segment detection finds natural "chapters" in the sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cosine_sim_matrix(features: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix (n×n)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = features / norms
    return normed @ normed.T


def laplacian_smoothness(features: np.ndarray, sigma: float = 0.5) -> float:
    """Graph-Laplacian smoothness of the playlist ordering.

    Builds a Gaussian similarity graph over the feature vectors, then
    evaluates f^T L f where f is the ordering index.  Low values indicate
    that adjacent-in-playlist tracks are also close in feature space.

    Returns a value in [0, 1] where 1 = perfectly smooth ordering and
    0 = maximally random with respect to the similarity graph.
    """
    n = features.shape[0]
    if n < 2:
        return 1.0

    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(features[i] - features[j]))
            dists[i, j] = d
            dists[j, i] = d

    W = np.exp(-dists ** 2 / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    D = np.diag(W.sum(axis=1))
    L = D - W

    f = np.arange(n, dtype=float)
    f = f - f.mean()
    norm = f @ f
    if norm < 1e-12:
        return 1.0

    quadratic = float(f @ L @ f)
    d_max = float(np.max(np.diag(D)))
    upper_bound = d_max * norm if d_max > 0 else 1.0
    raw = quadratic / max(upper_bound, 1e-12)
    return float(np.clip(1.0 - raw, 0.0, 1.0))


def spectral_flow(features: np.ndarray) -> np.ndarray:
    """Per-transition spectral flux in feature space.

    Returns an (n-1,) array of Euclidean distances between consecutive
    track feature vectors.  Spikes indicate abrupt timbral/stylistic shifts.
    """
    n = features.shape[0]
    if n < 2:
        return np.array([])
    return np.array([
        float(np.linalg.norm(features[i + 1] - features[i]))
        for i in range(n - 1)
    ])


def spectral_flow_score(features: np.ndarray) -> float:
    """Aggregate spectral flow into a single smoothness score (0-1).

    Combines mean flow (lower is smoother) with flow variance (lower is
    more consistent).
    """
    flow = spectral_flow(features)
    if len(flow) == 0:
        return 1.0
    mean_f = float(np.mean(flow))
    std_f = float(np.std(flow))
    smoothness = np.exp(-mean_f)
    consistency = np.exp(-std_f)
    return float(0.6 * smoothness + 0.4 * consistency)


def detect_segments(
    features: np.ndarray,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Detect natural playlist segments (chapters) from feature jumps.

    A segment boundary occurs where the spectral flow exceeds
    ``threshold × max(flow)``.  Returns a list of ``(start, end)``
    index pairs (inclusive start, exclusive end).
    """
    flow = spectral_flow(features)
    if len(flow) == 0:
        return [(0, max(features.shape[0], 1))]
    peak = float(np.max(flow))
    if peak < 1e-12:
        return [(0, features.shape[0])]

    boundaries = [0]
    for i, f in enumerate(flow):
        if f > threshold * peak:
            boundaries.append(i + 1)
    boundaries.append(features.shape[0])

    segments = []
    for i in range(len(boundaries) - 1):
        if boundaries[i] < boundaries[i + 1]:
            segments.append((boundaries[i], boundaries[i + 1]))
    return segments


def segment_coherence(features: np.ndarray, segments: list[tuple[int, int]]) -> list[float]:
    """Per-segment mean cosine similarity (intra-segment cohesion)."""
    scores = []
    for start, end in segments:
        seg = features[start:end]
        if seg.shape[0] < 2:
            scores.append(1.0)
            continue
        sim = _cosine_sim_matrix(seg)
        n = seg.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores.append(float(sim[mask].mean()))
    return scores


@dataclass(frozen=True)
class CoherenceReport:
    laplacian_smoothness: float
    spectral_flow_score: float
    n_segments: int
    segment_sizes: list[int]
    segment_coherences: list[float]
    overall: float


def full_coherence_analysis(
    features: np.ndarray,
    sigma: float = 0.5,
    segment_threshold: float = 0.5,
) -> CoherenceReport:
    """Complete playlist coherence analysis."""
    lap = laplacian_smoothness(features, sigma)
    sfs = spectral_flow_score(features)
    segs = detect_segments(features, segment_threshold)
    seg_coh = segment_coherence(features, segs)
    sizes = [e - s for s, e in segs]

    overall = 0.4 * lap + 0.3 * sfs + 0.3 * (sum(seg_coh) / max(len(seg_coh), 1))
    return CoherenceReport(
        laplacian_smoothness=lap,
        spectral_flow_score=sfs,
        n_segments=len(segs),
        segment_sizes=sizes,
        segment_coherences=seg_coh,
        overall=float(overall),
    )
