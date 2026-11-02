"""Tension and resolution curves derived from musical features.

Models tension as a multidimensional signal combining harmonic tension
(dissonance, distance from tonic), rhythmic tension (syncopation density,
metric instability), and dynamic tension (loudness contour, rate of change).
The Lerdahl (2001) tonal tension model is the theoretical backbone:
tension = f(surface_dissonance, hierarchical_distance, attraction).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# --- Lerdahl tonal tension primitives ---

# Circle-of-fifths distance (semitones -> CoF steps, mod 12)
_COF_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
_COF_DIST = np.zeros(12, dtype=float)
for _i, _pc in enumerate(_COF_ORDER):
    _COF_DIST[_pc] = _i


def circle_of_fifths_distance(pc_a: int, pc_b: int) -> int:
    """Unsigned distance on the circle of fifths (0..6)."""
    idx_a = _COF_ORDER.index(pc_a % 12)
    idx_b = _COF_ORDER.index(pc_b % 12)
    d = abs(idx_a - idx_b)
    return min(d, 12 - d)


def surface_dissonance(chroma: NDArray) -> NDArray:
    """Compute surface dissonance from a chroma time series.

    Uses the roughness model: sum of products of concurrent pitch-class
    energies weighted by semitone dissonance (after Hutchinson & Knopoff 1978).
    """
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 1:
        chroma = chroma.reshape(1, -1)
    n_frames = chroma.shape[0]
    # Dissonance weight matrix (symmetric, 12x12)
    # Approximate Plomp-Levelt curve mapped to semitone intervals
    _diss_weights = np.array([
        0.0, 1.0, 0.8, 0.5, 0.3, 0.2, 0.7, 0.2, 0.3, 0.5, 0.8, 1.0
    ])
    W = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            W[i, j] = _diss_weights[abs(i - j) % 12]
    np.fill_diagonal(W, 0.0)
    result = np.zeros(n_frames)
    for t in range(n_frames):
        c = chroma[t]
        if c.sum() < 1e-12:
            continue
        c = c / (c.sum() + 1e-12)
        result[t] = c @ W @ c
    return result


def hierarchical_tension(root_sequence: Sequence[int], reference_root: int = 0) -> NDArray:
    """Tonal tension from the circle-of-fifths distance of each chord root
    to a reference tonic (Lerdahl's hierarchical component)."""
    roots = np.asarray(root_sequence) % 12
    ref = reference_root % 12
    return np.array([circle_of_fifths_distance(int(r), ref) for r in roots], dtype=float)


def melodic_attraction(pitches: NDArray) -> NDArray:
    """Voice-leading attraction: inverse-square of interval size in semitones,
    weighted by stability of the target (Lerdahl 2001 §4).
    Returns attraction magnitude per transition (length = len(pitches)-1).
    """
    pitches = np.asarray(pitches, dtype=float)
    if len(pitches) < 2:
        return np.array([], dtype=float)
    intervals = np.abs(np.diff(pitches))
    intervals = np.maximum(intervals, 0.5)  # avoid div-by-zero
    # Stability heuristic: diatonic pitch classes (C major) are more stable
    _stable = {0, 2, 4, 5, 7, 9, 11}
    target_pcs = pitches[1:].astype(int) % 12
    stability = np.array([1.5 if int(pc) in _stable else 1.0 for pc in target_pcs])
    return stability / (intervals ** 2)


# --- Rhythmic tension ---

def syncopation_density(onsets: NDArray, meter_grid: NDArray) -> NDArray:
    """Proportion of onsets that fall off the metric grid within each window.

    Parameters
    ----------
    onsets : 1-D array of onset times (seconds or samples).
    meter_grid : 1-D array of strong-beat times.

    Returns
    -------
    syncopation : scalar in [0, 1].
    """
    onsets = np.asarray(onsets, dtype=float)
    meter_grid = np.asarray(meter_grid, dtype=float)
    if len(onsets) == 0 or len(meter_grid) == 0:
        return np.float64(0.0)
    # Tolerance: half the smallest grid interval
    if len(meter_grid) > 1:
        tol = np.min(np.diff(meter_grid)) * 0.5
    else:
        tol = 0.05
    on_grid = 0
    for o in onsets:
        if np.min(np.abs(meter_grid - o)) <= tol:
            on_grid += 1
    return np.float64(1.0 - on_grid / len(onsets))


def metric_instability(beat_positions: NDArray, expected_ioi: float) -> NDArray:
    """Per-beat metric instability as normalised IOI deviation from expected.

    Parameters
    ----------
    beat_positions : 1-D array of beat times.
    expected_ioi : expected inter-onset interval (e.g. 60/BPM).

    Returns
    -------
    instability : 1-D array (length = len(beat_positions) - 1).
    """
    bp = np.asarray(beat_positions, dtype=float)
    if len(bp) < 2:
        return np.array([], dtype=float)
    iois = np.diff(bp)
    return np.abs(iois - expected_ioi) / expected_ioi


# --- Dynamic tension ---

def loudness_rate_of_change(loudness: NDArray, hop_sec: float = 0.01) -> NDArray:
    """First derivative of a loudness contour, in units/sec."""
    L = np.asarray(loudness, dtype=float)
    if len(L) < 2:
        return np.array([], dtype=float)
    return np.diff(L) / hop_sec


# --- Composite tension curve ---

@dataclass
class TensionWeights:
    """Mixing weights for the composite tension curve."""
    harmonic: float = 0.4
    dissonance: float = 0.3
    rhythmic: float = 0.15
    dynamic: float = 0.15


@dataclass
class TensionCurve:
    """Full tension analysis result."""
    composite: NDArray
    harmonic: NDArray
    dissonance: NDArray
    rhythmic: NDArray | None = None
    dynamic: NDArray | None = None
    peaks: NDArray = field(default_factory=lambda: np.array([], dtype=int))
    troughs: NDArray = field(default_factory=lambda: np.array([], dtype=int))


def _normalise_01(x: NDArray) -> NDArray:
    """Min-max normalise to [0, 1]."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def compute_tension_curve(
    chroma: NDArray,
    root_sequence: Sequence[int] | None = None,
    reference_root: int = 0,
    rhythmic_tension: NDArray | None = None,
    dynamic_tension: NDArray | None = None,
    weights: TensionWeights | None = None,
    peak_prominence: float = 0.15,
) -> TensionCurve:
    """Compute a composite tension curve from available features.

    All component curves are normalised to [0, 1] before mixing.
    """
    w = weights or TensionWeights()
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 1:
        chroma = chroma.reshape(1, -1)
    n = chroma.shape[0]

    # Dissonance component
    diss = _normalise_01(surface_dissonance(chroma))

    # Harmonic component
    if root_sequence is not None and len(root_sequence) >= n:
        harm = _normalise_01(hierarchical_tension(root_sequence[:n], reference_root))
    elif root_sequence is not None:
        ht = hierarchical_tension(root_sequence, reference_root)
        harm = _normalise_01(np.interp(np.linspace(0, len(ht) - 1, n), np.arange(len(ht)), ht))
    else:
        # Fallback: use max chroma bin as rough root proxy
        rough_roots = np.argmax(chroma, axis=1)
        harm = _normalise_01(hierarchical_tension(rough_roots.tolist(), reference_root))

    # Rhythmic component (optional)
    r_curve = None
    if rhythmic_tension is not None:
        rt = np.asarray(rhythmic_tension, dtype=float)
        if len(rt) != n:
            rt = np.interp(np.linspace(0, len(rt) - 1, n), np.arange(len(rt)), rt)
        r_curve = _normalise_01(rt)

    # Dynamic component (optional)
    d_curve = None
    if dynamic_tension is not None:
        dt = np.asarray(dynamic_tension, dtype=float)
        if len(dt) != n:
            dt = np.interp(np.linspace(0, len(dt) - 1, n), np.arange(len(dt)), dt)
        d_curve = _normalise_01(dt)

    # Weighted sum
    composite = w.harmonic * harm + w.dissonance * diss
    total_w = w.harmonic + w.dissonance
    if r_curve is not None:
        composite += w.rhythmic * r_curve
        total_w += w.rhythmic
    if d_curve is not None:
        composite += w.dynamic * d_curve
        total_w += w.dynamic
    composite /= max(total_w, 1e-12)

    # Detect peaks and troughs
    from scipy.signal import find_peaks as _find_peaks
    peaks, _ = _find_peaks(composite, prominence=peak_prominence)
    troughs, _ = _find_peaks(-composite, prominence=peak_prominence)

    return TensionCurve(
        composite=composite,
        harmonic=harm,
        dissonance=diss,
        rhythmic=r_curve,
        dynamic=d_curve,
        peaks=peaks,
        troughs=troughs,
    )


def tension_resolution_ratio(curve: TensionCurve) -> float:
    """Ratio of mean tension at peaks vs troughs. Higher = more contrast."""
    if len(curve.peaks) == 0 or len(curve.troughs) == 0:
        return 1.0
    peak_mean = curve.composite[curve.peaks].mean()
    trough_mean = curve.composite[curve.troughs].mean()
    return float(peak_mean / max(trough_mean, 1e-12))


def tension_arc_type(curve: TensionCurve) -> str:
    """Classify the overall tension arc shape.

    Returns one of: 'ascending', 'descending', 'arch', 'valley', 'plateau', 'volatile'.
    """
    c = curve.composite
    if len(c) < 4:
        return "plateau"
    mid = len(c) // 2
    first_half = c[:mid].mean()
    second_half = c[mid:].mean()
    overall_std = c.std()
    peak_loc = c.argmax() / max(len(c) - 1, 1)
    trough_loc = c.argmin() / max(len(c) - 1, 1)

    if overall_std < 0.05:
        return "plateau"
    if 0.25 < peak_loc < 0.75 and first_half < c[mid:].mean() * 0.8:
        return "arch"
    if 0.25 < trough_loc < 0.75 and first_half > c[mid:].mean() * 1.2:
        return "valley"
    if second_half > first_half * 1.3:
        return "ascending"
    if first_half > second_half * 1.3:
        return "descending"
    return "volatile"
