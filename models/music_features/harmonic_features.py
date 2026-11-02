"""Harmonic and tonal feature extraction from chroma representations.

Implements Krumhansl-Schmuckler key finding, tonnetz projection,
harmonic change detection (HCDF), chord change rate, and tonal complexity.
"""
from __future__ import annotations

import numpy as np


# ── Key profiles (Krumhansl & Kessler 1982) ─────────────────────────────

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


def _correlate_profile(chroma: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Pearson correlation of chroma with all 12 rotations of profile."""
    scores = np.zeros(12)
    for shift in range(12):
        rotated = np.roll(profile, shift)
        r = np.corrcoef(chroma, rotated)[0, 1]
        scores[shift] = r if np.isfinite(r) else 0.0
    return scores


def key_strength(chroma: np.ndarray) -> tuple[str, str, float]:
    """Krumhansl-Schmuckler key estimation.

    Returns (key_name, mode, correlation).
    *chroma* is a 12-element pitch-class distribution.
    """
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 2:
        chroma = np.mean(chroma, axis=1) if chroma.shape[0] == 12 else np.mean(chroma, axis=0)
    if len(chroma) != 12:
        raise ValueError(f"Expected 12-element chroma, got {len(chroma)}")

    major_scores = _correlate_profile(chroma, MAJOR_PROFILE)
    minor_scores = _correlate_profile(chroma, MINOR_PROFILE)

    best_major_idx = int(np.argmax(major_scores))
    best_minor_idx = int(np.argmax(minor_scores))

    if major_scores[best_major_idx] >= minor_scores[best_minor_idx]:
        return NOTE_NAMES[best_major_idx], "major", float(major_scores[best_major_idx])
    return NOTE_NAMES[best_minor_idx], "minor", float(minor_scores[best_minor_idx])


def mode_strength(chroma: np.ndarray) -> float:
    """Ratio of best major to best minor correlation — >1 = major-leaning."""
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 2:
        chroma = np.mean(chroma, axis=0) if chroma.shape[1] == 12 else np.mean(chroma, axis=1)
    major_scores = _correlate_profile(chroma, MAJOR_PROFILE)
    minor_scores = _correlate_profile(chroma, MINOR_PROFILE)
    best_major = np.max(major_scores)
    best_minor = np.max(minor_scores)
    if abs(best_minor) < 1e-10:
        return float('inf') if best_major > 0 else 0.0
    return float(best_major / best_minor)


def key_clarity(chroma: np.ndarray) -> float:
    """Ratio of best to second-best key correlation — higher = more tonal clarity."""
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 2:
        chroma = np.mean(chroma, axis=0) if chroma.shape[1] == 12 else np.mean(chroma, axis=1)
    all_scores = np.concatenate([
        _correlate_profile(chroma, MAJOR_PROFILE),
        _correlate_profile(chroma, MINOR_PROFILE),
    ])
    sorted_scores = np.sort(all_scores)[::-1]
    if abs(sorted_scores[1]) < 1e-10:
        return float('inf') if sorted_scores[0] > 0 else 0.0
    return float(sorted_scores[0] / sorted_scores[1])


# ── Tonnetz ──────────────────────────────────────────────────────────────

def chroma_to_tonnetz(chroma: np.ndarray) -> np.ndarray:
    """Project 12-d chroma vector onto 6-d Euler tonnetz space.

    Dimensions: (fifth_x, fifth_y, minor_third_x, minor_third_y,
                 major_third_x, major_third_y).
    """
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 2:
        return np.array([chroma_to_tonnetz(c) for c in chroma.T]).T

    norm = np.sum(chroma)
    if norm < 1e-12:
        return np.zeros(6)
    c = chroma / norm
    indices = np.arange(12)
    fifth_x = np.sum(c * np.sin(2 * np.pi * indices * 7 / 12))
    fifth_y = np.sum(c * np.cos(2 * np.pi * indices * 7 / 12))
    minor_x = np.sum(c * np.sin(2 * np.pi * indices * 3 / 12))
    minor_y = np.sum(c * np.cos(2 * np.pi * indices * 3 / 12))
    major_x = np.sum(c * np.sin(2 * np.pi * indices * 4 / 12))
    major_y = np.sum(c * np.cos(2 * np.pi * indices * 4 / 12))
    return np.array([fifth_x, fifth_y, minor_x, minor_y, major_x, major_y])


def tonal_centroid(chroma_sequence: np.ndarray) -> np.ndarray:
    """Mean tonnetz position over a chroma sequence (12, T) -> (6,)."""
    tonnetz = chroma_to_tonnetz(chroma_sequence)
    if tonnetz.ndim == 1:
        return tonnetz
    return np.mean(tonnetz, axis=1)


def tonal_radius(chroma: np.ndarray) -> float:
    """L2 norm of tonnetz projection — distance from tonal center."""
    t = chroma_to_tonnetz(chroma)
    if t.ndim == 2:
        t = np.mean(t, axis=1)
    return float(np.linalg.norm(t))


# ── Harmonic change ──────────────────────────────────────────────────────

def hcdf(chroma_sequence: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Harmonic Change Detection Function (Harte et al. 2006).

    Euclidean distance between successive tonnetz frames, Gaussian-smoothed.
    *chroma_sequence* shape: (12, T).
    """
    tonnetz = chroma_to_tonnetz(chroma_sequence)
    if tonnetz.ndim == 1:
        return np.array([0.0])
    T = tonnetz.shape[1]
    if T < 2:
        return np.zeros(T)
    flux = np.zeros(T)
    for t in range(1, T):
        flux[t] = np.linalg.norm(tonnetz[:, t] - tonnetz[:, t - 1])
    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        flux = gaussian_filter1d(flux, sigma)
    return flux


def chord_change_rate(chroma_sequence: np.ndarray, threshold: float = 0.3) -> float:
    """Fraction of frames where HCDF exceeds threshold — harmonic activity rate."""
    flux = hcdf(chroma_sequence, sigma=2.0)
    if len(flux) == 0:
        return 0.0
    return float(np.mean(flux > threshold))


def harmonic_complexity(chroma: np.ndarray) -> float:
    """Shannon entropy of chroma distribution — higher = less tonal focus."""
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 2:
        chroma = np.mean(chroma, axis=1) if chroma.shape[0] == 12 else np.mean(chroma, axis=0)
    total = np.sum(chroma)
    if total < 1e-12:
        return 0.0
    p = chroma / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def harmonic_feature_vector(chroma_sequence: np.ndarray) -> np.ndarray:
    """Aggregate harmonic feature vector from a chroma matrix (12, T)."""
    chroma_mean = np.mean(chroma_sequence, axis=1) if chroma_sequence.ndim == 2 else chroma_sequence
    key_name, mode, key_corr = key_strength(chroma_mean)
    clarity = key_clarity(chroma_mean)
    complexity = harmonic_complexity(chroma_mean)
    tc = tonal_centroid(chroma_sequence)
    tr = tonal_radius(chroma_mean)
    ccr = chord_change_rate(chroma_sequence)

    return np.concatenate([
        [key_corr, 1.0 if mode == "major" else 0.0, clarity, complexity, tr, ccr],
        tc,
    ])
