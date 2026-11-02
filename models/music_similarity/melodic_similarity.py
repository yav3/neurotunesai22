"""Melodic similarity metrics: contour, interval, rhythm, and combined.

Implements Müllensiefen & Frieler (2004) melodic similarity dimensions,
edit-distance on interval sequences, Mongeau-Sankoff weighted alignment,
and contour-reduction (Marvin & Laprade 1987) comparison.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
from enum import Enum


class SimilarityMetric(Enum):
    PITCH = "pitch"
    INTERVAL = "interval"
    CONTOUR = "contour"
    RHYTHM = "rhythm"
    COMBINED = "combined"


@dataclass
class MelodicSimilarityResult:
    pitch_similarity: float
    interval_similarity: float
    contour_similarity: float
    rhythm_similarity: float
    combined: float
    alignment_cost: float


def _pitch_sequence(midi_notes: Sequence[float]) -> np.ndarray:
    return np.asarray(midi_notes, dtype=float)


def _interval_sequence(midi_notes: Sequence[float]) -> np.ndarray:
    p = _pitch_sequence(midi_notes)
    return np.diff(p) if len(p) > 1 else np.array([0.0])


def _contour_sequence(midi_notes: Sequence[float]) -> np.ndarray:
    p = _pitch_sequence(midi_notes)
    if len(p) < 2:
        return np.array([0])
    d = np.diff(p)
    return np.sign(d).astype(int)


def edit_distance_normalised(a: Sequence[float], b: Sequence[float]) -> float:
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0.0 if a[i - 1] == b[j - 1] else 1.0
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[n, m]) / max(n, m)


def pitch_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, 1.0 - edit_distance_normalised(list(a), list(b)))


def interval_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    ia = _interval_sequence(a)
    ib = _interval_sequence(b)
    return max(0.0, 1.0 - edit_distance_normalised(ia.tolist(), ib.tolist()))


def contour_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    ca = _contour_sequence(a)
    cb = _contour_sequence(b)
    n = min(len(ca), len(cb))
    if n == 0:
        return 1.0
    return float(np.sum(ca[:n] == cb[:n])) / n


def rhythm_similarity(a_onsets: Sequence[float], b_onsets: Sequence[float],
                       tolerance: float = 0.1) -> float:
    if len(a_onsets) == 0 and len(b_onsets) == 0:
        return 1.0
    a_ioi = np.diff(np.asarray(a_onsets, dtype=float))
    b_ioi = np.diff(np.asarray(b_onsets, dtype=float))
    if len(a_ioi) == 0 or len(b_ioi) == 0:
        return 0.0
    a_norm = a_ioi / max(a_ioi.sum(), 1e-10)
    b_norm = b_ioi / max(b_ioi.sum(), 1e-10)
    n = min(len(a_norm), len(b_norm))
    matches = np.sum(np.abs(a_norm[:n] - b_norm[:n]) < tolerance)
    return float(matches) / max(len(a_norm), len(b_norm))


def mongeau_sankoff(a: Sequence[float], b: Sequence[float],
                     w_sub: float = 1.0, w_ins: float = 0.5,
                     w_del: float = 0.5) -> float:
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + w_del
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + w_ins
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pitch_diff = abs(float(a[i - 1]) - float(b[j - 1]))
            sub_cost = dp[i - 1, j - 1] + w_sub * pitch_diff / 12.0
            del_cost = dp[i - 1, j] + w_del
            ins_cost = dp[i, j - 1] + w_ins
            dp[i, j] = min(sub_cost, del_cost, ins_cost)
    return float(dp[n, m])


def mongeau_sankoff_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    cost = mongeau_sankoff(a, b)
    max_cost = max(len(a), len(b)) * 0.5
    if max_cost < 1e-10:
        return 1.0
    return max(0.0, 1.0 - cost / max_cost)


def melodic_similarity(
    a_pitches: Sequence[float],
    b_pitches: Sequence[float],
    a_onsets: Optional[Sequence[float]] = None,
    b_onsets: Optional[Sequence[float]] = None,
    weights: Optional[Tuple[float, float, float, float]] = None,
) -> MelodicSimilarityResult:
    if weights is None:
        weights = (0.3, 0.3, 0.2, 0.2)
    wp, wi, wc, wr = weights
    ps = pitch_similarity(a_pitches, b_pitches)
    isim = interval_similarity(a_pitches, b_pitches)
    cs = contour_similarity(a_pitches, b_pitches)
    if a_onsets is not None and b_onsets is not None:
        rs = rhythm_similarity(a_onsets, b_onsets)
    else:
        rs = 0.0
        wr = 0.0
    total_w = wp + wi + wc + wr
    combined = (wp * ps + wi * isim + wc * cs + wr * rs) / max(total_w, 1e-10)
    alignment_cost = mongeau_sankoff(a_pitches, b_pitches)
    return MelodicSimilarityResult(
        pitch_similarity=ps, interval_similarity=isim,
        contour_similarity=cs, rhythm_similarity=rs,
        combined=combined, alignment_cost=alignment_cost,
    )


def contour_reduction(midi_notes: Sequence[float], levels: int = 3) -> List[int]:
    pitches = list(midi_notes)
    for _ in range(levels):
        if len(pitches) <= 3:
            break
        reduced = [pitches[0]]
        for i in range(1, len(pitches) - 1):
            if (pitches[i] >= pitches[i - 1] and pitches[i] >= pitches[i + 1]) or \
               (pitches[i] <= pitches[i - 1] and pitches[i] <= pitches[i + 1]):
                reduced.append(pitches[i])
        reduced.append(pitches[-1])
        pitches = reduced
    return [int(p) for p in pitches]


def contour_reduction_similarity(a: Sequence[float], b: Sequence[float],
                                  levels: int = 3) -> float:
    ra = contour_reduction(a, levels)
    rb = contour_reduction(b, levels)
    return contour_similarity(ra, rb)


def transposition_invariant_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return interval_similarity(a, b)


def inversion_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    b_inv = [2 * b[0] - p for p in b] if len(b) > 0 else []
    return interval_similarity(a, b_inv)


def retrograde_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return pitch_similarity(a, list(reversed(b)))


def pairwise_melodic_similarity(
    melodies: List[Sequence[float]],
    metric: SimilarityMetric = SimilarityMetric.COMBINED,
) -> np.ndarray:
    n = len(melodies)
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            if metric == SimilarityMetric.PITCH:
                s = pitch_similarity(melodies[i], melodies[j])
            elif metric == SimilarityMetric.INTERVAL:
                s = interval_similarity(melodies[i], melodies[j])
            elif metric == SimilarityMetric.CONTOUR:
                s = contour_similarity(melodies[i], melodies[j])
            else:
                r = melodic_similarity(melodies[i], melodies[j])
                s = r.combined
            mat[i, j] = mat[j, i] = s
    return mat


def most_similar_pair(melodies: List[Sequence[float]]) -> Tuple[int, int, float]:
    mat = pairwise_melodic_similarity(melodies)
    np.fill_diagonal(mat, -1)
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    return int(idx[0]), int(idx[1]), float(mat[idx])
