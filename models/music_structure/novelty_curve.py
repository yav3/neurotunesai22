"""Novelty-based segmentation from self-similarity matrices.

Implements the Foote (2000) checkerboard-kernel novelty function and
multi-scale boundary detection for structural segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class NoveltyPeak:
    time_idx: int
    strength: float
    boundary_type: str = "unknown"


def compute_ssm(features: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Self-similarity matrix from frame-level features (n_frames × n_features)."""
    n = features.shape[0]
    if metric == "cosine":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = features / norms
        ssm = normed @ normed.T
    elif metric == "euclidean":
        diff = features[:, None, :] - features[None, :, :]
        ssm = 1.0 / (1.0 + np.sqrt(np.sum(diff ** 2, axis=-1)))
    else:
        ssm = features @ features.T
    return ssm


def checkerboard_kernel(size: int) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel for novelty detection."""
    half = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            sign = 1 if (i < half) == (j < half) else -1
            kernel[i, j] = sign
    sigma = size / 4.0
    y, x = np.mgrid[-half:half + (size % 2), -half:half + (size % 2)]
    y = y[:size, :size]
    x = x[:size, :size]
    gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel * gaussian


def novelty_from_ssm(ssm: np.ndarray, kernel_size: int = 64) -> np.ndarray:
    """Convolve the SSM diagonal with a checkerboard kernel → novelty curve."""
    n = ssm.shape[0]
    kernel = checkerboard_kernel(kernel_size)
    half = kernel_size // 2
    novelty = np.zeros(n)
    for i in range(half, n - half):
        patch = ssm[i - half:i + half, i - half:i + half]
        if patch.shape == kernel.shape:
            novelty[i] = np.sum(patch * kernel)
    novelty = np.maximum(novelty, 0)
    return novelty


def detect_boundaries(
    novelty_curve: np.ndarray,
    threshold: float = 0.5,
    min_segment_frames: int = 10,
) -> List[NoveltyPeak]:
    """Peak-pick boundaries from a novelty curve."""
    if len(novelty_curve) == 0:
        return []
    norm_max = np.max(novelty_curve)
    if norm_max > 0:
        normed = novelty_curve / norm_max
    else:
        return []

    peaks: list[NoveltyPeak] = []
    last_peak = -min_segment_frames
    for i in range(1, len(normed) - 1):
        if normed[i] > threshold and normed[i] > normed[i - 1] and normed[i] > normed[i + 1]:
            if i - last_peak >= min_segment_frames:
                peaks.append(NoveltyPeak(time_idx=i, strength=float(normed[i])))
                last_peak = i
    return peaks


def lag_ssm(ssm: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Time-lag representation: L[i,l] = SSM[i, i+l] for repeat structure."""
    n = ssm.shape[0]
    ml = max_lag if max_lag else n
    lag = np.zeros((n, ml))
    for i in range(n):
        for l in range(min(ml, n - i)):
            lag[i, l] = ssm[i, i + l]
    return lag


def recurrence_plot(features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binary recurrence matrix from pairwise distances."""
    ssm = compute_ssm(features, metric="cosine")
    return (ssm >= threshold).astype(int)


def segment_from_novelty(
    features: np.ndarray,
    sr: float = 22050.0,
    hop_size: int = 512,
    kernel_sizes: Optional[List[int]] = None,
    threshold: float = 0.5,
) -> List[float]:
    """Multi-scale novelty segmentation pipeline. Returns boundary times in seconds."""
    if kernel_sizes is None:
        kernel_sizes = [32, 64, 128]
    ssm = compute_ssm(features)
    combined = np.zeros(features.shape[0])
    for ks in kernel_sizes:
        nc = novelty_from_ssm(ssm, kernel_size=ks)
        if np.max(nc) > 0:
            combined += nc / np.max(nc)
    combined /= len(kernel_sizes)
    peaks = detect_boundaries(combined, threshold=threshold)
    frame_dur = hop_size / sr
    return [p.time_idx * frame_dur for p in peaks]


def transition_matrix(labels: List[int]) -> np.ndarray:
    """Compute section-label transition counts."""
    if not labels:
        return np.zeros((0, 0))
    n_labels = max(labels) + 1
    mat = np.zeros((n_labels, n_labels), dtype=int)
    for i in range(len(labels) - 1):
        mat[labels[i], labels[i + 1]] += 1
    return mat
