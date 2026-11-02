"""Melodic, harmonic, and rhythmic plagiarism/similarity detection.

Quantitative comparison metrics for assessing musical similarity at multiple
structural levels — melody (pitch contour, interval sequence), harmony
(chord progression comparison), rhythm (onset pattern correlation), and
combined weighted scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np


class SimilarityDimension(Enum):
    MELODY = auto()
    HARMONY = auto()
    RHYTHM = auto()
    STRUCTURE = auto()
    TIMBRE = auto()


@dataclass
class SimilarityScore:
    dimension: SimilarityDimension
    score: float  # 0-1
    detail: str


@dataclass
class PlagiarismReport:
    overall_score: float  # 0-1
    dimension_scores: List[SimilarityScore]
    flagged: bool
    threshold: float
    best_alignment_offset: int
    aligned_segments: List[Tuple[int, int, float]]  # (offset_a, offset_b, local_sim)


def pitch_contour(midi_pitches: np.ndarray) -> np.ndarray:
    """Convert absolute pitches to relative contour: +1 (up), 0 (repeat), -1 (down)."""
    if len(midi_pitches) < 2:
        return np.array([], dtype=float)
    diff = np.diff(midi_pitches.astype(float))
    return np.sign(diff)


def interval_sequence(midi_pitches: np.ndarray) -> np.ndarray:
    """Interval sequence in semitones."""
    if len(midi_pitches) < 2:
        return np.array([], dtype=float)
    return np.diff(midi_pitches.astype(float))


def contour_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized contour match ratio using sliding-window best alignment."""
    ca, cb = pitch_contour(a), pitch_contour(b)
    if len(ca) == 0 or len(cb) == 0:
        return 0.0
    short, long = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    best = 0.0
    for offset in range(len(long) - len(short) + 1):
        match = np.mean(short == long[offset:offset + len(short)])
        best = max(best, float(match))
    return best


def interval_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Interval sequence cosine similarity using best sliding alignment."""
    ia, ib = interval_sequence(a), interval_sequence(b)
    if len(ia) == 0 or len(ib) == 0:
        return 0.0
    short, long = (ia, ib) if len(ia) <= len(ib) else (ib, ia)
    best = 0.0
    for offset in range(len(long) - len(short) + 1):
        seg = long[offset:offset + len(short)]
        dot = float(np.dot(short, seg))
        norms = np.linalg.norm(short) * np.linalg.norm(seg)
        if norms > 0:
            sim = dot / norms
            best = max(best, (sim + 1) / 2)  # map [-1,1] to [0,1]
    return best


def melodic_edit_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Edit distance on interval sequences (insert/delete/substitute)."""
    ia = interval_sequence(a).astype(int)
    ib = interval_sequence(b).astype(int)
    n, m = len(ia), len(ib)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ia[i - 1] == ib[j - 1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
    return int(dp[n, m])


def melodic_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Combined melodic similarity: 0.5·contour + 0.3·interval_cosine + 0.2·(1 - norm_edit)."""
    cs = contour_similarity(a, b)
    isim = interval_similarity(a, b)
    ed = melodic_edit_distance(a, b)
    max_len = max(len(a), len(b)) - 1
    norm_ed = ed / max(max_len, 1)
    return 0.5 * cs + 0.3 * isim + 0.2 * (1.0 - norm_ed)


def chord_progression_similarity(
    chords_a: List[str],
    chords_b: List[str],
) -> float:
    """Longest common subsequence ratio for chord progressions."""
    n, m = len(chords_a), len(chords_b)
    if n == 0 or m == 0:
        return 0.0
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if chords_a[i - 1] == chords_b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    lcs_len = int(dp[n, m])
    return 2.0 * lcs_len / (n + m)


def rhythm_onset_similarity(
    onsets_a: np.ndarray,
    onsets_b: np.ndarray,
    tolerance: float = 0.05,
) -> float:
    """Onset pattern similarity via cross-correlation of quantized onset grids."""
    if len(onsets_a) == 0 or len(onsets_b) == 0:
        return 0.0
    # Quantize to grid
    max_t = max(onsets_a.max(), onsets_b.max())
    resolution = tolerance
    n_bins = int(max_t / resolution) + 1
    grid_a = np.zeros(n_bins)
    grid_b = np.zeros(n_bins)
    for t in onsets_a:
        idx = min(int(t / resolution), n_bins - 1)
        grid_a[idx] = 1.0
    for t in onsets_b:
        idx = min(int(t / resolution), n_bins - 1)
        grid_b[idx] = 1.0
    # Normalized cross-correlation
    na = np.linalg.norm(grid_a)
    nb = np.linalg.norm(grid_b)
    if na == 0 or nb == 0:
        return 0.0
    corr = np.correlate(grid_a, grid_b, mode="full")
    return float(corr.max() / (na * nb))


def ioi_similarity(
    onsets_a: np.ndarray,
    onsets_b: np.ndarray,
) -> float:
    """Inter-onset interval ratio similarity."""
    if len(onsets_a) < 2 or len(onsets_b) < 2:
        return 0.0
    ioi_a = np.diff(np.sort(onsets_a))
    ioi_b = np.diff(np.sort(onsets_b))
    # Normalize to ratios
    if ioi_a.sum() > 0:
        ioi_a = ioi_a / ioi_a.sum()
    if ioi_b.sum() > 0:
        ioi_b = ioi_b / ioi_b.sum()
    # DTW on IOI ratios
    n, m = len(ioi_a), len(ioi_b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(ioi_a[i - 1] - ioi_b[j - 1])
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    raw = float(dp[n, m])
    return max(0.0, 1.0 - raw)


def structural_similarity(
    form_a: List[str],
    form_b: List[str],
) -> float:
    """Song form similarity (e.g. ['V','C','V','C','B','C'] vs ['V','V','C','C','B','C'])."""
    return chord_progression_similarity(form_a, form_b)


def combined_plagiarism_score(
    melody_a: np.ndarray,
    melody_b: np.ndarray,
    chords_a: Optional[List[str]] = None,
    chords_b: Optional[List[str]] = None,
    onsets_a: Optional[np.ndarray] = None,
    onsets_b: Optional[np.ndarray] = None,
    form_a: Optional[List[str]] = None,
    form_b: Optional[List[str]] = None,
    weights: Optional[dict] = None,
    threshold: float = 0.75,
) -> PlagiarismReport:
    """Full plagiarism analysis combining all dimensions."""
    if weights is None:
        weights = {"melody": 0.45, "harmony": 0.25, "rhythm": 0.20, "structure": 0.10}

    scores: List[SimilarityScore] = []

    mel_score = melodic_similarity(melody_a, melody_b)
    scores.append(SimilarityScore(SimilarityDimension.MELODY, mel_score, "contour+interval+edit"))

    harm_score = 0.0
    if chords_a is not None and chords_b is not None:
        harm_score = chord_progression_similarity(chords_a, chords_b)
    scores.append(SimilarityScore(SimilarityDimension.HARMONY, harm_score, "LCS ratio"))

    rhythm_score = 0.0
    if onsets_a is not None and onsets_b is not None:
        r1 = rhythm_onset_similarity(onsets_a, onsets_b)
        r2 = ioi_similarity(onsets_a, onsets_b)
        rhythm_score = 0.6 * r1 + 0.4 * r2
    scores.append(SimilarityScore(SimilarityDimension.RHYTHM, rhythm_score, "xcorr+IOI-DTW"))

    struct_score = 0.0
    if form_a is not None and form_b is not None:
        struct_score = structural_similarity(form_a, form_b)
    scores.append(SimilarityScore(SimilarityDimension.STRUCTURE, struct_score, "form LCS"))

    overall = (
        weights.get("melody", 0) * mel_score
        + weights.get("harmony", 0) * harm_score
        + weights.get("rhythm", 0) * rhythm_score
        + weights.get("structure", 0) * struct_score
    )

    # Find best melodic alignment offset
    ca = pitch_contour(melody_a)
    cb = pitch_contour(melody_b)
    best_off = 0
    if len(ca) > 0 and len(cb) > 0:
        short, long = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
        best_sim = 0.0
        for off in range(len(long) - len(short) + 1):
            sim = float(np.mean(short == long[off:off + len(short)]))
            if sim > best_sim:
                best_sim = sim
                best_off = off

    return PlagiarismReport(
        overall_score=float(np.clip(overall, 0, 1)),
        dimension_scores=scores,
        flagged=overall >= threshold,
        threshold=threshold,
        best_alignment_offset=best_off,
        aligned_segments=[],
    )
