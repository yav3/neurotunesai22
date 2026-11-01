"""Structural similarity: form-level comparison, section alignment, cover detection.

Implements music similarity measures based on structural form (verse-chorus
patterns), harmonic sequences, melodic contour matching, and cover song
identification using chroma-based and rhythmic features.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class SectionType(Enum):
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    PRE_CHORUS = "pre_chorus"
    OUTRO = "outro"
    SOLO = "solo"
    INTERLUDE = "interlude"


@dataclass(frozen=True)
class FormSection:
    section_type: SectionType
    start_time: float
    end_time: float
    chroma_profile: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True)
class StructuralMatch:
    section_a: FormSection
    section_b: FormSection
    similarity: float
    alignment_offset: float


def chroma_cosine_similarity(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """Cosine similarity between two chroma vectors or averaged chroma matrices."""
    if chroma_a.ndim > 1:
        chroma_a = np.mean(chroma_a, axis=1)
    if chroma_b.ndim > 1:
        chroma_b = np.mean(chroma_b, axis=1)
    norm_a = np.linalg.norm(chroma_a)
    norm_b = np.linalg.norm(chroma_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(chroma_a, chroma_b) / (norm_a * norm_b))


def optimal_transposition_index(chroma_a: np.ndarray, chroma_b: np.ndarray) -> int:
    """Find the transposition (0-11 semitones) that maximizes chroma similarity.

    Essential for cover song detection where the key may differ.
    """
    if chroma_a.ndim > 1:
        chroma_a = np.mean(chroma_a, axis=1)
    if chroma_b.ndim > 1:
        chroma_b = np.mean(chroma_b, axis=1)
    best_shift = 0
    best_sim = -1.0
    for shift in range(12):
        shifted = np.roll(chroma_b, shift)
        sim = chroma_cosine_similarity(chroma_a, shifted)
        if sim > best_sim:
            best_sim = sim
            best_shift = shift
    return best_shift


def form_edit_distance(
    form_a: list[SectionType],
    form_b: list[SectionType],
) -> int:
    """Levenshtein edit distance between two song forms (section sequences)."""
    m, n = len(form_a), len(form_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if form_a[i - 1] == form_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def normalized_form_similarity(
    form_a: list[SectionType],
    form_b: list[SectionType],
) -> float:
    """Normalized form similarity: 1 - edit_distance / max(len(a), len(b))."""
    if not form_a and not form_b:
        return 1.0
    max_len = max(len(form_a), len(form_b))
    if max_len == 0:
        return 1.0
    dist = form_edit_distance(form_a, form_b)
    return 1.0 - dist / max_len


def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Dynamic Time Warping distance between two feature sequences.

    seq_a, seq_b: (T, D) feature matrices.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if metric == "cosine":
                d = 1.0 - chroma_cosine_similarity(seq_a[i - 1], seq_b[j - 1])
            else:
                d = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def qmax_similarity(
    cross_similarity: np.ndarray,
    min_run_length: int = 4,
) -> float:
    """Qmax cover song similarity (Serra et al., 2009).

    Finds the longest locally-optimal path in a binary cross-similarity matrix.
    """
    n, m = cross_similarity.shape
    if n < min_run_length or m < min_run_length:
        return 0.0
    binary = (cross_similarity > np.median(cross_similarity)).astype(np.float64)
    q_max = 0.0
    for diag_offset in range(-n + min_run_length, m - min_run_length + 1):
        run_length = 0
        for k in range(max(n, m)):
            i = k
            j = k + diag_offset
            if i >= n or j >= m or i < 0 or j < 0:
                continue
            if binary[i, j] > 0.5:
                run_length += 1
                q_max = max(q_max, run_length)
            else:
                run_length = 0
    normalizer = min(n, m)
    return q_max / normalizer if normalizer > 0 else 0.0


def melodic_contour_similarity(
    pitches_a: np.ndarray,
    pitches_b: np.ndarray,
) -> float:
    """Compare melodic contours (up/down/same sequences).

    Converts pitch sequences to direction vectors (+1, -1, 0)
    and computes normalized correlation.
    """
    if len(pitches_a) < 2 or len(pitches_b) < 2:
        return 0.0
    contour_a = np.sign(np.diff(pitches_a))
    contour_b = np.sign(np.diff(pitches_b))
    min_len = min(len(contour_a), len(contour_b))
    if min_len == 0:
        return 0.0
    matches = np.sum(contour_a[:min_len] == contour_b[:min_len])
    return float(matches / min_len)


def rhythmic_pattern_similarity(
    onsets_a: np.ndarray,
    onsets_b: np.ndarray,
    resolution: float = 0.01,
    max_duration: float = 4.0,
) -> float:
    """Compare rhythmic patterns by onset histogram correlation."""
    n_bins = int(max_duration / resolution)
    hist_a = np.zeros(n_bins)
    hist_b = np.zeros(n_bins)
    for t in onsets_a:
        idx = int((t % max_duration) / resolution)
        if 0 <= idx < n_bins:
            hist_a[idx] = 1.0
    for t in onsets_b:
        idx = int((t % max_duration) / resolution)
        if 0 <= idx < n_bins:
            hist_b[idx] = 1.0
    norm_a = np.linalg.norm(hist_a)
    norm_b = np.linalg.norm(hist_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(hist_a, hist_b) / (norm_a * norm_b))


def cover_song_score(
    chroma_a: np.ndarray,
    chroma_b: np.ndarray,
    form_a: list[SectionType] | None = None,
    form_b: list[SectionType] | None = None,
    w_chroma: float = 0.6,
    w_form: float = 0.2,
    w_dtw: float = 0.2,
) -> float:
    """Composite cover song detection score.

    Combines transposition-invariant chroma similarity, form similarity,
    and DTW alignment distance.
    """
    shift = optimal_transposition_index(chroma_a, chroma_b)
    if chroma_b.ndim > 1:
        shifted_b = np.roll(chroma_b, shift, axis=0)
    else:
        shifted_b = np.roll(chroma_b, shift)
    chroma_sim = chroma_cosine_similarity(chroma_a, shifted_b)
    form_sim = 0.5
    if form_a is not None and form_b is not None:
        form_sim = normalized_form_similarity(form_a, form_b)
    else:
        w_chroma += w_form / 2
        w_dtw += w_form / 2
        w_form = 0.0
    if chroma_a.ndim > 1 and chroma_b.ndim > 1:
        dtw_raw = dtw_distance(chroma_a.T, shifted_b.T)
        max_dtw = max(chroma_a.shape[1], chroma_b.shape[1])
        dtw_sim = max(0.0, 1.0 - dtw_raw / max(max_dtw, 1))
    else:
        dtw_sim = chroma_sim
    return w_chroma * chroma_sim + w_form * form_sim + w_dtw * dtw_sim
