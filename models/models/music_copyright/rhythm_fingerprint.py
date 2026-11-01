"""Rhythmic pattern fingerprinting for copyright comparison.

Structured rhythm representations beyond quantized durations:
onset-grid patterns, metric weight vectors, rhythmic motif extraction,
and cross-rhythm similarity.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Onset grid ─────────────────────────────────────────────────────

def onsets_to_grid(
    onset_times: np.ndarray,
    grid_resolution: int = 16,
    beats: int = 4,
    beat_duration: float = 0.5,
) -> np.ndarray:
    """Quantise continuous onset times onto a binary grid.

    Returns a binary array of length ``grid_resolution * beats``.
    Each cell represents one sub-beat division.
    """
    total_dur = beats * beat_duration
    n_cells = grid_resolution * beats
    grid = np.zeros(n_cells, dtype=int)
    for t in onset_times:
        if t < 0 or t >= total_dur:
            continue
        idx = int(round(t / total_dur * n_cells)) % n_cells
        grid[idx] = 1
    return grid


def grid_hamming_similarity(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """1 − normalised Hamming distance between two onset grids."""
    if len(grid_a) != len(grid_b):
        return 0.0
    n = len(grid_a)
    if n == 0:
        return 1.0
    return 1.0 - float(np.sum(grid_a != grid_b)) / n


# ── Metric weight vector ──────────────────────────────────────────

def metric_weight_vector(grid_resolution: int = 16, beats: int = 4) -> np.ndarray:
    """Hierarchical metric weight for each grid cell (Lerdahl–Jackendoff style).

    Weight = number of metric levels a position participates in.
    """
    n = grid_resolution * beats
    weights = np.ones(n)
    level = 2
    step = n
    while step >= 2:
        step //= 2
        for i in range(0, n, step):
            weights[i] += 1
        level += 1
    return weights / weights.max()


def weighted_onset_similarity(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Metric-weight-aware cosine similarity between two onset grids."""
    if len(grid_a) != len(grid_b):
        return 0.0
    n = len(grid_a)
    if n == 0:
        return 1.0
    if weights is None:
        weights = metric_weight_vector(n, 1)
    a = grid_a.astype(float) * weights[:n]
    b = grid_b.astype(float) * weights[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Rhythmic motif extraction ─────────────────────────────────────

@dataclass
class RhythmicMotif:
    """A contiguous onset sub-pattern."""
    pattern: tuple[int, ...]
    start: int
    length: int


def extract_motifs(grid: np.ndarray, min_len: int = 4, max_len: int = 16) -> list[RhythmicMotif]:
    """Extract all contiguous sub-patterns of the onset grid."""
    motifs: list[RhythmicMotif] = []
    n = len(grid)
    for length in range(min_len, min(max_len, n) + 1):
        for start in range(n - length + 1):
            pat = tuple(grid[start:start + length])
            motifs.append(RhythmicMotif(pat, start, length))
    return motifs


def motif_jaccard(motifs_a: list[RhythmicMotif], motifs_b: list[RhythmicMotif]) -> float:
    """Jaccard similarity of motif pattern sets."""
    set_a = {m.pattern for m in motifs_a}
    set_b = {m.pattern for m in motifs_b}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── IOI (Inter-Onset Interval) fingerprint ─────────────────────────

def ioi_vector(onset_times: np.ndarray) -> np.ndarray:
    """Inter-onset interval sequence (durations between consecutive onsets)."""
    if len(onset_times) < 2:
        return np.array([])
    sorted_t = np.sort(onset_times)
    return np.diff(sorted_t)


def ioi_ratio_vector(onset_times: np.ndarray) -> np.ndarray:
    """Ratio of consecutive IOIs — tempo-invariant rhythmic fingerprint.

    Each element is IOI[i] / IOI[i+1]; a straight rhythm yields all 1.0,
    a swung rhythm yields alternating 2.0 / 0.5, etc.
    """
    ioi = ioi_vector(onset_times)
    if len(ioi) < 2:
        return np.array([])
    ratios = np.zeros(len(ioi) - 1)
    for i in range(len(ioi) - 1):
        if ioi[i + 1] > 1e-12:
            ratios[i] = ioi[i] / ioi[i + 1]
        else:
            ratios[i] = 0.0
    return ratios


def ioi_ratio_similarity(ratios_a: np.ndarray, ratios_b: np.ndarray) -> float:
    """Cosine similarity between two IOI-ratio vectors (tempo-invariant)."""
    min_len = min(len(ratios_a), len(ratios_b))
    if min_len == 0:
        return 0.0
    a, b = ratios_a[:min_len], ratios_b[:min_len]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Cross-rhythm comparison ────────────────────────────────────────

def cross_rhythm_correlation(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Normalised circular cross-correlation peak between two onset grids.

    Finds the rotation of grid_b that best aligns with grid_a.
    """
    if len(grid_a) != len(grid_b) or len(grid_a) == 0:
        return 0.0
    a = grid_a.astype(float)
    b = grid_b.astype(float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    xcorr = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).real
    return float(np.max(xcorr) / (na * nb))


# ── Composite rhythmic similarity ─────────────────────────────────

@dataclass
class RhythmComparisonResult:
    grid_similarity: float
    weighted_similarity: float
    motif_overlap: float
    ioi_ratio_similarity: float
    cross_correlation: float
    composite: float


def compare_rhythms(
    onsets_a: np.ndarray,
    onsets_b: np.ndarray,
    grid_resolution: int = 16,
    beats: int = 4,
    beat_duration: float = 0.5,
    weights: tuple[float, ...] = (0.2, 0.25, 0.2, 0.2, 0.15),
) -> RhythmComparisonResult:
    """Multi-feature rhythmic similarity between two onset sequences."""
    g_a = onsets_to_grid(onsets_a, grid_resolution, beats, beat_duration)
    g_b = onsets_to_grid(onsets_b, grid_resolution, beats, beat_duration)
    mw = metric_weight_vector(grid_resolution, beats)

    gs = grid_hamming_similarity(g_a, g_b)
    ws = weighted_onset_similarity(g_a, g_b, mw)
    mo_a = extract_motifs(g_a, min_len=4, max_len=8)
    mo_b = extract_motifs(g_b, min_len=4, max_len=8)
    mo = motif_jaccard(mo_a, mo_b)
    ra = ioi_ratio_vector(onsets_a)
    rb = ioi_ratio_vector(onsets_b)
    ir = ioi_ratio_similarity(ra, rb)
    xc = cross_rhythm_correlation(g_a, g_b)

    w = np.array(weights[:5])
    w = w / w.sum()
    composite = float(w[0] * gs + w[1] * ws + w[2] * mo + w[3] * ir + w[4] * xc)

    return RhythmComparisonResult(gs, ws, mo, ir, xc, composite)
