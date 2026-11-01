"""Tonal similarity: key-distance, harmonic similarity, chroma-based comparison.

Compares tonal content at multiple levels: key-level (circle-of-fifths,
parallel/relative), chord-progression level (functional edit distance),
and frame-level (chroma cosine, Tonal Centroid Distance per Harte et al. 2006).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence


@dataclass
class TonalSimilarityResult:
    chroma_cosine: float
    tonal_centroid_distance: float
    key_distance: float
    harmonic_rhythm_similarity: float
    combined: float


MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chroma_from_pitches(midi_notes: Sequence[int], normalize: bool = True) -> np.ndarray:
    chroma = np.zeros(12)
    for n in midi_notes:
        chroma[n % 12] += 1.0
    if normalize and chroma.sum() > 0:
        chroma = chroma / chroma.sum()
    return chroma


def chroma_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tonal_centroid_transform(chroma: np.ndarray) -> np.ndarray:
    r1, r2, r3 = 1.0, 1.0, 0.5
    k = np.arange(12)
    c = np.zeros(6)
    c[0] = r1 * np.sum(chroma * np.sin(k * 7 * np.pi / 6))
    c[1] = r1 * np.sum(chroma * np.cos(k * 7 * np.pi / 6))
    c[2] = r2 * np.sum(chroma * np.sin(k * 3 * np.pi / 2))
    c[3] = r2 * np.sum(chroma * np.cos(k * 3 * np.pi / 2))
    c[4] = r3 * np.sum(chroma * np.sin(k * 2 * np.pi / 3))
    c[5] = r3 * np.sum(chroma * np.cos(k * 2 * np.pi / 3))
    return c


def tonal_centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / max(a.sum(), 1e-10)
    b_norm = b / max(b.sum(), 1e-10)
    ca = _tonal_centroid_transform(a_norm)
    cb = _tonal_centroid_transform(b_norm)
    return float(np.linalg.norm(ca - cb))


def detect_key(chroma: np.ndarray) -> Tuple[int, str, float]:
    best_corr = -np.inf
    best_key = 0
    best_mode = "major"
    for shift in range(12):
        shifted = np.roll(chroma, -shift)
        corr_maj = float(np.corrcoef(shifted, MAJOR_PROFILE)[0, 1])
        corr_min = float(np.corrcoef(shifted, MINOR_PROFILE)[0, 1])
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = shift
            best_mode = "major"
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = shift
            best_mode = "minor"
    return best_key, best_mode, best_corr


def key_distance_fifths(key_a: int, mode_a: str, key_b: int, mode_b: str) -> float:
    fifths_a = (key_a * 7) % 12
    fifths_b = (key_b * 7) % 12
    raw = abs(fifths_a - fifths_b)
    circle_dist = min(raw, 12 - raw)
    mode_penalty = 0.5 if mode_a != mode_b else 0.0
    return circle_dist + mode_penalty


def key_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ka, ma, _ = detect_key(a)
    kb, mb, _ = detect_key(b)
    d = key_distance_fifths(ka, ma, kb, mb)
    return max(0.0, 1.0 - d / 7.0)


def harmonic_rhythm_similarity(a_durations: Sequence[float],
                                 b_durations: Sequence[float]) -> float:
    if len(a_durations) == 0 or len(b_durations) == 0:
        return 0.0
    a_arr = np.array(a_durations, dtype=float)
    b_arr = np.array(b_durations, dtype=float)
    a_rate = len(a_arr) / max(a_arr.sum(), 1e-10)
    b_rate = len(b_arr) / max(b_arr.sum(), 1e-10)
    max_rate = max(a_rate, b_rate)
    if max_rate < 1e-10:
        return 1.0
    return 1.0 - abs(a_rate - b_rate) / max_rate


def tonal_similarity(
    a_chroma: np.ndarray,
    b_chroma: np.ndarray,
    a_chord_durations: Optional[Sequence[float]] = None,
    b_chord_durations: Optional[Sequence[float]] = None,
    weights: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2),
) -> TonalSimilarityResult:
    cc = chroma_cosine_similarity(a_chroma, b_chroma)
    tcd = tonal_centroid_distance(a_chroma, b_chroma)
    tcd_sim = max(0.0, 1.0 - tcd / 3.0)
    ks = key_similarity(a_chroma, b_chroma)
    if a_chord_durations is not None and b_chord_durations is not None:
        hrs = harmonic_rhythm_similarity(a_chord_durations, b_chord_durations)
    else:
        hrs = 0.0
    wc, wt, wk, wh = weights
    total = wc + wt + wk + (wh if a_chord_durations is not None else 0)
    combined = (wc * cc + wt * tcd_sim + wk * ks + wh * hrs) / max(total, 1e-10)
    return TonalSimilarityResult(
        chroma_cosine=cc, tonal_centroid_distance=tcd,
        key_distance=1.0 - ks, harmonic_rhythm_similarity=hrs,
        combined=combined,
    )


def chroma_cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    correlations = np.zeros(12)
    for shift in range(12):
        b_shifted = np.roll(b, shift)
        correlations[shift] = chroma_cosine_similarity(a, b_shifted)
    return correlations


def optimal_transposition(a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
    corr = chroma_cross_correlation(a, b)
    best = int(np.argmax(corr))
    return best, float(corr[best])


def pairwise_tonal_similarity(
    chroma_list: List[np.ndarray],
) -> np.ndarray:
    n = len(chroma_list)
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            r = tonal_similarity(chroma_list[i], chroma_list[j])
            mat[i, j] = mat[j, i] = r.combined
    return mat
