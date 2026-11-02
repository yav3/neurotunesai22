"""Narrative arc analysis: tension curves, arc classification, emotional trajectory, surprise."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Tension curve
# ---------------------------------------------------------------------------

def tension_curve(
    harmonic_tension: np.ndarray,
    dynamic_level: np.ndarray,
    harmonic_weight: float = 0.6,
    dynamic_weight: float = 0.4,
) -> np.ndarray:
    """Compute a combined tension curve from harmonic and dynamic features.

    Both inputs are 1-D arrays of the same length, values in [0, 1].
    Returns a weighted combination.
    """
    h = np.asarray(harmonic_tension, dtype=float)
    d = np.asarray(dynamic_level, dtype=float)
    n = min(len(h), len(d))
    h, d = h[:n], d[:n]
    return harmonic_weight * h + dynamic_weight * d


def smooth_tension(curve: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing of a tension curve."""
    if len(curve) < window:
        return curve.copy()
    kernel = np.ones(window) / window
    padded = np.pad(curve, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(curve)]


# ---------------------------------------------------------------------------
# Arc classification
# ---------------------------------------------------------------------------

class ArcType(Enum):
    RAMP = "ramp"                   # monotonically increasing
    ARCH = "arch"                   # rise then fall (classic Freytag)
    WAVE = "wave"                   # multiple peaks
    CLIMAX_DENOUEMENT = "climax_denouement"  # single late peak then descent
    FLAT = "flat"                   # minimal variation


@dataclass
class ArcClassification:
    arc_type: ArcType
    peak_position: float   # 0–1, relative position of highest tension
    peak_count: int
    range_ratio: float     # (max - min) / max, 0 = flat


def classify_arc(curve: np.ndarray, peak_prominence: float = 0.15) -> ArcClassification:
    """Classify the narrative arc shape of a tension curve."""
    c = np.asarray(curve, dtype=float)
    if len(c) < 3:
        return ArcClassification(ArcType.FLAT, 0.0, 0, 0.0)

    c_max = float(np.max(c))
    c_min = float(np.min(c))
    rng = c_max - c_min
    range_ratio = rng / c_max if c_max > 1e-12 else 0.0

    if range_ratio < 0.1:
        return ArcClassification(ArcType.FLAT, 0.0, 0, range_ratio)

    peak_idx = int(np.argmax(c))
    peak_pos = peak_idx / (len(c) - 1)

    # count prominent peaks
    peaks = _find_peaks(c, prominence=peak_prominence * rng)

    if len(peaks) >= 3:
        arc = ArcType.WAVE
    elif len(peaks) <= 1:
        # single peak or monotonic
        diffs = np.diff(c)
        frac_rising = float(np.mean(diffs > 0))
        if frac_rising > 0.75:
            arc = ArcType.RAMP
        elif peak_pos > 0.6:
            arc = ArcType.CLIMAX_DENOUEMENT
        else:
            arc = ArcType.ARCH
    else:
        # 2 peaks
        if peak_pos > 0.6:
            arc = ArcType.CLIMAX_DENOUEMENT
        else:
            arc = ArcType.ARCH

    return ArcClassification(
        arc_type=arc,
        peak_position=peak_pos,
        peak_count=len(peaks),
        range_ratio=range_ratio,
    )


def _find_peaks(x: np.ndarray, prominence: float = 0.0) -> list[int]:
    """Simple peak finder: local maxima above neighbours by at least `prominence`."""
    peaks: list[int] = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            left_prom = x[i] - x[i - 1]
            right_prom = x[i] - x[i + 1]
            if min(left_prom, right_prom) >= prominence:
                peaks.append(i)
    return peaks


# ---------------------------------------------------------------------------
# Emotional trajectory (VAD)
# ---------------------------------------------------------------------------

@dataclass
class VADPoint:
    valence: float
    arousal: float
    dominance: float


def emotional_trajectory(
    valence: np.ndarray,
    arousal: np.ndarray,
    dominance: np.ndarray,
) -> list[VADPoint]:
    """Package per-segment VAD values into a trajectory."""
    n = min(len(valence), len(arousal), len(dominance))
    return [
        VADPoint(float(valence[i]), float(arousal[i]), float(dominance[i]))
        for i in range(n)
    ]


def vad_distance(a: VADPoint, b: VADPoint) -> float:
    """Euclidean distance in VAD space."""
    return float(np.sqrt(
        (a.valence - b.valence) ** 2
        + (a.arousal - b.arousal) ** 2
        + (a.dominance - b.dominance) ** 2
    ))


def emotional_range(trajectory: list[VADPoint]) -> float:
    """Maximum pairwise distance in the VAD trajectory."""
    if len(trajectory) < 2:
        return 0.0
    max_d = 0.0
    for i in range(len(trajectory)):
        for j in range(i + 1, len(trajectory)):
            d = vad_distance(trajectory[i], trajectory[j])
            if d > max_d:
                max_d = d
    return max_d


def emotional_velocity(trajectory: list[VADPoint]) -> np.ndarray:
    """Step-wise VAD distance (rate of emotional change)."""
    if len(trajectory) < 2:
        return np.array([], dtype=float)
    dists = np.array([
        vad_distance(trajectory[i], trajectory[i + 1])
        for i in range(len(trajectory) - 1)
    ], dtype=float)
    return dists


# ---------------------------------------------------------------------------
# Structural surprise (information-theoretic)
# ---------------------------------------------------------------------------

def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = probs[probs > 0]
    return -float(np.sum(p * np.log2(p)))


def segment_entropy(features: np.ndarray) -> float:
    """Entropy of a feature vector (discretised into bins).

    Parameters
    ----------
    features : 1-D array
        Feature values for one segment.
    """
    f = np.asarray(features, dtype=float)
    if len(f) < 2:
        return 0.0
    n_bins = max(int(np.sqrt(len(f))), 2)
    counts, _ = np.histogram(f, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return _entropy(probs)


def surprise_curve(segment_features: list[np.ndarray]) -> np.ndarray:
    """Information-theoretic surprise between consecutive segments.

    Returns |H(segment_i) - H(segment_{i-1})| for i = 1..n-1.
    """
    if len(segment_features) < 2:
        return np.array([], dtype=float)
    entropies = np.array([segment_entropy(sf) for sf in segment_features], dtype=float)
    return np.abs(np.diff(entropies))


def structural_surprise_score(segment_features: list[np.ndarray]) -> float:
    """Overall structural surprise: mean entropy change across segments."""
    sc = surprise_curve(segment_features)
    return float(np.mean(sc)) if len(sc) > 0 else 0.0
