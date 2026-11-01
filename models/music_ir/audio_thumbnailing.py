"""Automatic music thumbnailing — structure-aware summarization.

Finds the most representative segment of a piece using self-similarity
matrix analysis and repetition detection (Bartsch & Wakefield 2005,
Cooper & Foote 2003).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class Thumbnail:
    start_frame: int
    end_frame: int
    duration_frames: int
    quality_score: float
    method: str


@dataclass(frozen=True)
class SSMAnalysis:
    ssm: np.ndarray
    novelty_curve: np.ndarray
    segment_boundaries: List[int]
    thumbnail: Thumbnail


# ── Self-similarity matrix ───────────────────────────────────────────────

def compute_ssm(features: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Self-similarity matrix from frame-level features (n_frames × n_dims)."""
    F = np.asarray(features, dtype=np.float64)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    F_normed = F / norms
    if metric == "cosine":
        ssm = F_normed @ F_normed.T
    elif metric == "euclidean":
        sq = np.sum(F ** 2, axis=1)
        ssm = -np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * F @ F.T, 0, None))
        ssm = (ssm - ssm.min()) / max(ssm.max() - ssm.min(), 1e-12)
    else:
        ssm = F_normed @ F_normed.T
    return ssm


def enhance_ssm(ssm: np.ndarray, filter_size: int = 5) -> np.ndarray:
    """Enhance diagonal structures via median filtering along diagonals."""
    n = ssm.shape[0]
    enhanced = np.copy(ssm)
    for k in range(-n + 1, n):
        diag = np.diag(ssm, k)
        if len(diag) >= filter_size:
            filtered = medfilt(diag, kernel_size=min(filter_size, len(diag) if len(diag) % 2 == 1 else len(diag) - 1))
            for i, val in enumerate(filtered):
                r, c = (i - k, i) if k < 0 else (i, i + k)
                if 0 <= r < n and 0 <= c < n:
                    enhanced[r, c] = val
    return enhanced


# ── Novelty detection ────────────────────────────────────────────────────

def novelty_curve(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """Foote (2000) novelty function via checkerboard kernel convolution."""
    n = ssm.shape[0]
    half = kernel_size // 2
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    novelty = np.zeros(n)
    for i in range(half, n - half):
        patch = ssm[i - half : i + half, i - half : i + half]
        if patch.shape == kernel.shape:
            novelty[i] = float(np.sum(patch * kernel))

    novelty = np.clip(novelty, 0, None)
    if novelty.max() > 0:
        novelty /= novelty.max()
    return novelty


def detect_boundaries(
    novelty: np.ndarray,
    threshold: float = 0.3,
    min_segment_frames: int = 10,
) -> List[int]:
    """Peak-pick the novelty curve to find segment boundaries."""
    boundaries = [0]
    for i in range(1, len(novelty) - 1):
        if novelty[i] > threshold and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
            if i - boundaries[-1] >= min_segment_frames:
                boundaries.append(i)
    if boundaries[-1] != len(novelty) - 1:
        boundaries.append(len(novelty) - 1)
    return boundaries


# ── Repetition-based thumbnail ───────────────────────────────────────────

def repetition_score(ssm: np.ndarray, start: int, length: int) -> float:
    """How often a segment [start, start+length) repeats elsewhere in the SSM."""
    n = ssm.shape[0]
    if length <= 0 or start + length > n:
        return 0.0
    segment_sim = ssm[start : start + length, :]
    scores = []
    for j in range(0, n - length, max(length // 2, 1)):
        if abs(j - start) < length:
            continue
        block = segment_sim[:, j : j + length]
        if block.shape[1] == length:
            scores.append(float(np.mean(np.diag(block))))
    return float(np.mean(scores)) if scores else 0.0


def find_thumbnail(
    ssm: np.ndarray,
    target_length: int = 30,
    step: int = 5,
) -> Thumbnail:
    """Find the most-repeated segment of approximately target_length frames."""
    n = ssm.shape[0]
    target_length = min(target_length, n)
    best_start = 0
    best_score = -1.0

    for start in range(0, n - target_length, step):
        score = repetition_score(ssm, start, target_length)
        if score > best_score:
            best_score = score
            best_start = start

    return Thumbnail(
        start_frame=best_start,
        end_frame=best_start + target_length,
        duration_frames=target_length,
        quality_score=best_score,
        method="repetition",
    )


# ── Structure-aware thumbnail ────────────────────────────────────────────

def structure_thumbnail(
    ssm: np.ndarray,
    boundaries: List[int],
    target_length: int = 30,
) -> Thumbnail:
    """Select the most-repeated structural segment closest to target length."""
    segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        segments.append((s, e))

    if not segments:
        return find_thumbnail(ssm, target_length)

    best: Optional[Thumbnail] = None
    for s, e in segments:
        length = e - s
        score = repetition_score(ssm, s, length)
        length_penalty = 1.0 / (1.0 + abs(length - target_length) / max(target_length, 1))
        combined = score * 0.7 + length_penalty * 0.3
        if best is None or combined > best.quality_score:
            best = Thumbnail(
                start_frame=s, end_frame=e,
                duration_frames=length,
                quality_score=combined,
                method="structure",
            )
    return best or find_thumbnail(ssm, target_length)


# ── Information density ──────────────────────────────────────────────────

def information_density(features: np.ndarray, start: int, length: int) -> float:
    """Spectral entropy-based information density of a segment."""
    F = np.asarray(features, dtype=np.float64)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    seg = F[start : start + length]
    if seg.shape[0] == 0:
        return 0.0
    var = np.var(seg, axis=0)
    entropy = float(np.mean(np.log1p(var)))
    return entropy


def representativeness(ssm: np.ndarray, start: int, length: int) -> float:
    """Mean similarity of the segment to the entire piece."""
    n = ssm.shape[0]
    if length <= 0 or start + length > n:
        return 0.0
    block = ssm[start : start + length, :]
    return float(np.mean(block))


# ── Fade envelope ────────────────────────────────────────────────────────

def fade_envelope(
    n_samples: int,
    fade_in: int = 0,
    fade_out: int = 0,
) -> np.ndarray:
    """Generate a linear fade-in / fade-out amplitude envelope."""
    env = np.ones(n_samples)
    if fade_in > 0:
        env[:fade_in] = np.linspace(0, 1, fade_in)
    if fade_out > 0:
        env[-fade_out:] = np.linspace(1, 0, fade_out)
    return env


# ── Full pipeline ────────────────────────────────────────────────────────

def analyze_and_thumbnail(
    features: np.ndarray,
    target_length: int = 30,
    novelty_kernel: int = 16,
    boundary_threshold: float = 0.3,
) -> SSMAnalysis:
    """Full pipeline: SSM → novelty → boundaries → thumbnail."""
    ssm = compute_ssm(features)
    ssm = enhance_ssm(ssm)
    nov = novelty_curve(ssm, novelty_kernel)
    bounds = detect_boundaries(nov, boundary_threshold)
    thumb = structure_thumbnail(ssm, bounds, target_length)
    return SSMAnalysis(
        ssm=ssm,
        novelty_curve=nov,
        segment_boundaries=bounds,
        thumbnail=thumb,
    )
