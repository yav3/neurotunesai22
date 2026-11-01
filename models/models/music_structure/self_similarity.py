"""Self-similarity matrix: structure analysis via recurrence and novelty.

Computes self-similarity matrices (SSM) from feature sequences and extracts
structural boundaries using checkerboard kernel novelty (Foote 2000),
diagonal enhancement for repeated sections, and path-based segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normed = features / (norms + 1e-12)
    return normed @ normed.T


def euclidean_distance_matrix(features: np.ndarray) -> np.ndarray:
    sq = np.sum(features ** 2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2 * features @ features.T
    return np.sqrt(np.maximum(dist_sq, 0))


def gaussian_similarity_matrix(features: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    dist = euclidean_distance_matrix(features)
    return np.exp(-dist ** 2 / (2 * sigma ** 2))


def recurrence_matrix(features: np.ndarray, threshold: float = 0.5,
                      metric: str = "cosine") -> np.ndarray:
    if metric == "cosine":
        sim = cosine_similarity_matrix(features)
        return (sim >= threshold).astype(np.float64)
    dist = euclidean_distance_matrix(features)
    max_dist = np.max(dist)
    return (dist <= threshold * max_dist).astype(np.float64)


def _checkerboard_kernel(size: int) -> np.ndarray:
    kernel = np.ones((size, size))
    half = size // 2
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1
    return kernel


def novelty_curve(ssm: np.ndarray, kernel_size: int = 64) -> np.ndarray:
    n = ssm.shape[0]
    kernel = _checkerboard_kernel(kernel_size)
    half = kernel_size // 2
    novelty = np.zeros(n)
    for i in range(half, n - half):
        region = ssm[i - half : i + half, i - half : i + half]
        if region.shape == kernel.shape:
            novelty[i] = np.sum(region * kernel)
    novelty = np.maximum(novelty, 0)
    peak = np.max(novelty)
    if peak > 1e-12:
        novelty /= peak
    return novelty


def find_boundaries(novelty: np.ndarray, threshold: float = 0.3,
                    min_distance: int = 10) -> np.ndarray:
    peaks = []
    for i in range(1, len(novelty) - 1):
        if novelty[i] > novelty[i - 1] and novelty[i] > novelty[i + 1]:
            if novelty[i] >= threshold:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
    return np.array(peaks, dtype=int)


def lag_similarity_matrix(ssm: np.ndarray) -> np.ndarray:
    n = ssm.shape[0]
    lag_max = n
    lag_ssm = np.zeros((n, lag_max))
    for lag in range(lag_max):
        for i in range(n - lag):
            lag_ssm[i, lag] = ssm[i, i + lag]
    return lag_ssm


def enhance_diagonals(ssm: np.ndarray, length: int = 8) -> np.ndarray:
    n = ssm.shape[0]
    enhanced = np.zeros_like(ssm)
    for i in range(n):
        for j in range(n):
            count = 0
            for d in range(length):
                if i + d < n and j + d < n:
                    count += ssm[i + d, j + d]
            enhanced[i, j] = count / length
    return enhanced


@dataclass
class StructureSegment:
    start: int
    end: int
    label: str
    confidence: float


def label_segments(boundaries: np.ndarray, ssm: np.ndarray,
                   n_frames: int) -> list[StructureSegment]:
    if len(boundaries) == 0:
        return [StructureSegment(0, n_frames, "A", 1.0)]
    all_bounds = np.concatenate([[0], boundaries, [n_frames]])
    segments: list[StructureSegment] = []
    label_map: dict[int, str] = {}
    next_label = 0
    for idx in range(len(all_bounds) - 1):
        start = int(all_bounds[idx])
        end = int(all_bounds[idx + 1])
        if end <= start:
            continue
        mid = (start + end) // 2
        best_match = -1
        best_sim = 0.0
        for prev_idx, seg in enumerate(segments):
            prev_mid = (seg.start + seg.end) // 2
            if mid < ssm.shape[0] and prev_mid < ssm.shape[1]:
                sim = ssm[mid, prev_mid]
                if sim > best_sim and sim > 0.5:
                    best_sim = sim
                    best_match = prev_idx
        if best_match >= 0:
            label = segments[best_match].label
        else:
            label = chr(ord("A") + next_label % 26)
            next_label += 1
        segments.append(StructureSegment(start, end, label, max(best_sim, 0.5)))
    return segments


@dataclass
class StructureAnalysis:
    ssm: np.ndarray
    novelty: np.ndarray
    boundaries: np.ndarray
    segments: list[StructureSegment]


def analyze_structure(features: np.ndarray, kernel_size: int = 64,
                      boundary_threshold: float = 0.3,
                      min_segment_frames: int = 10) -> StructureAnalysis:
    ssm = cosine_similarity_matrix(features)
    nov = novelty_curve(ssm, kernel_size)
    bounds = find_boundaries(nov, boundary_threshold, min_segment_frames)
    segs = label_segments(bounds, ssm, features.shape[0])
    return StructureAnalysis(ssm, nov, bounds, segs)
