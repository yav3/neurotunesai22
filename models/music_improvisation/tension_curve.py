"""Harmonic tension modelling for improvisation planning.

Quantifies tonal tension, sensory dissonance, and voice-leading distance
following Lerdahl (2001) / Bigand et al. hierarchical-attraction principles.
Pure numpy — no symbolic music libraries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Interval-class dissonance vector (Hutchinson & Knopoff 1978 / Mashinter 2006).
# Index = interval class 0..11 in semitones; value = empirical roughness weight.
_IC_DISSONANCE = np.array([
    0.0, 1.0, 0.8, 0.4, 0.25, 0.15, 0.7,
    0.15, 0.25, 0.4, 0.8, 1.0,
])


def sensory_dissonance(pitch_classes: list[int]) -> float:
    """Sum of pairwise interval-class dissonance for a simultaneity.

    Parameters
    ----------
    pitch_classes : list[int]
        MIDI pitch classes (0-11) sounding together.

    Returns
    -------
    float
        Aggregate roughness, 0 = consonant.
    """
    pcs = list(set(int(p) % 12 for p in pitch_classes))
    if len(pcs) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            ic = abs(pcs[i] - pcs[j]) % 12
            total += float(_IC_DISSONANCE[ic])
    return total


def voice_leading_distance(
    chord_a: list[int], chord_b: list[int]
) -> float:
    """Minimal total semitone motion between two pitch-class sets (mod 12).

    Uses a greedy closest-voice assignment (not full Hungarian, but sufficient
    for typical chord sizes ≤ 6 voices).
    """
    if not chord_a or not chord_b:
        return 0.0
    a = sorted(set(int(p) % 12 for p in chord_a))
    b = sorted(set(int(p) % 12 for p in chord_b))
    used_b: set[int] = set()
    total = 0.0
    for pa in a:
        best_d = 12
        best_j = -1
        for j, pb in enumerate(b):
            if j in used_b:
                continue
            d = min(abs(pa - pb) % 12, abs(pb - pa) % 12)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used_b.add(best_j)
        total += best_d
    return total


# Circle-of-fifths distance (Krumhansl & Kessler 1982).
_COF_DISTANCE = np.zeros(12)
for _i in range(12):
    _COF_DISTANCE[_i] = min(_i * 7 % 12, (12 - _i) * 7 % 12)


def tonal_distance(key_a: int, key_b: int) -> float:
    """Circle-of-fifths distance between two key roots (0-11)."""
    diff = abs(int(key_a) - int(key_b)) % 12
    return float(_COF_DISTANCE[diff])


@dataclass
class TensionPoint:
    """Tension at a single time-point."""
    time: float
    dissonance: float
    voice_leading: float
    tonal: float
    total: float


def tension_curve(
    chord_sequence: list[list[int]],
    key_roots: list[int] | None = None,
    weights: tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> list[TensionPoint]:
    """Compute a tension curve over a chord progression.

    Parameters
    ----------
    chord_sequence : list[list[int]]
        Each element is a list of MIDI pitch classes sounding at that beat.
    key_roots : list[int] | None
        Per-beat key root (0-11).  If None, the first chord's lowest PC is used.
    weights : (float, float, float)
        (dissonance, voice_leading, tonal) mixing weights.  Sum need not be 1.

    Returns
    -------
    list[TensionPoint]
    """
    if not chord_sequence:
        return []
    n = len(chord_sequence)
    if key_roots is None:
        default_key = min(chord_sequence[0]) % 12 if chord_sequence[0] else 0
        key_roots = [default_key] * n

    w_d, w_v, w_t = weights
    points: list[TensionPoint] = []
    for i, chord in enumerate(chord_sequence):
        d = sensory_dissonance(chord)
        v = voice_leading_distance(chord_sequence[max(0, i - 1)], chord) if i > 0 else 0.0
        t = tonal_distance(key_roots[max(0, i - 1)], key_roots[i]) if i > 0 else 0.0
        total = w_d * d + w_v * v + w_t * t
        points.append(TensionPoint(time=float(i), dissonance=d, voice_leading=v, tonal=t, total=total))
    return points


def tension_release_ratio(curve: list[TensionPoint]) -> float:
    """Ratio of ascending-tension beats to descending-tension beats.

    A balanced solo aims for ≈ 1.0; purely escalating → ∞; purely relaxing → 0.
    """
    if len(curve) < 2:
        return 1.0
    ups = sum(1 for i in range(1, len(curve)) if curve[i].total > curve[i - 1].total)
    downs = sum(1 for i in range(1, len(curve)) if curve[i].total < curve[i - 1].total)
    return ups / max(downs, 1)


def peak_tension_position(curve: list[TensionPoint]) -> float:
    """Fractional position [0,1] of the highest-tension point in the curve."""
    if not curve:
        return 0.0
    idx = max(range(len(curve)), key=lambda i: curve[i].total)
    return idx / max(len(curve) - 1, 1)


def mean_tension(curve: list[TensionPoint]) -> float:
    """Average total tension across the curve."""
    if not curve:
        return 0.0
    return sum(p.total for p in curve) / len(curve)


def tension_variance(curve: list[TensionPoint]) -> float:
    """Variance of total tension — higher = more dynamic phrasing."""
    if len(curve) < 2:
        return 0.0
    vals = np.array([p.total for p in curve])
    return float(np.var(vals))
