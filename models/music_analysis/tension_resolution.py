"""Harmonic tension and resolution analysis: Lerdahl-style tonal tension, dissonance curves."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


INTERVAL_TENSION = {
    0: 0.0,   1: 1.0,  2: 0.5,  3: 0.2,  4: 0.2,  5: 0.3,
    6: 0.9,   7: 0.1,  8: 0.2,  9: 0.2,  10: 0.5, 11: 1.0,
}

TONAL_HIERARCHY = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88]


@dataclass
class TensionPoint:
    time_index: int
    harmonic_tension: float
    dissonance: float
    tonal_distance: float
    combined: float


def interval_dissonance(interval: int) -> float:
    return INTERVAL_TENSION.get(interval % 12, 0.5)


def chord_dissonance(pitch_classes: list[int]) -> float:
    if len(pitch_classes) < 2:
        return 0.0
    tensions = []
    for i in range(len(pitch_classes)):
        for j in range(i + 1, len(pitch_classes)):
            iv = abs(pitch_classes[j] - pitch_classes[i]) % 12
            tensions.append(interval_dissonance(iv))
    return float(np.mean(tensions))


def tonal_distance(chord_pcs: list[int], key_root: int = 0) -> float:
    if not chord_pcs:
        return 0.0
    hierarchy = np.array(TONAL_HIERARCHY)
    max_val = float(np.max(hierarchy))
    total = 0.0
    for pc in chord_pcs:
        idx = (pc - key_root) % 12
        total += 1.0 - hierarchy[idx] / max_val
    return total / len(chord_pcs)


def harmonic_tension(chord_pcs: list[int], key_root: int = 0,
                      dissonance_weight: float = 0.5,
                      distance_weight: float = 0.5) -> float:
    d = chord_dissonance(chord_pcs)
    t = tonal_distance(chord_pcs, key_root)
    return dissonance_weight * d + distance_weight * t


def tension_curve(chord_sequence: list[list[int]], key_root: int = 0) -> np.ndarray:
    return np.array([harmonic_tension(ch, key_root) for ch in chord_sequence])


def resolution_strength(tension_before: float, tension_after: float) -> float:
    if tension_before <= 0:
        return 0.0
    drop = tension_before - tension_after
    return float(np.clip(drop / tension_before, 0.0, 1.0))


def detect_resolutions(curve: np.ndarray, threshold: float = 0.3) -> list[int]:
    resolutions = []
    for i in range(1, len(curve)):
        strength = resolution_strength(curve[i - 1], curve[i])
        if strength >= threshold:
            resolutions.append(i)
    return resolutions


def tension_profile(chord_sequence: list[list[int]], key_root: int = 0) -> list[TensionPoint]:
    points = []
    for i, ch in enumerate(chord_sequence):
        d = chord_dissonance(ch)
        t = tonal_distance(ch, key_root)
        combined = 0.5 * d + 0.5 * t
        points.append(TensionPoint(
            time_index=i, harmonic_tension=combined,
            dissonance=d, tonal_distance=t, combined=combined
        ))
    return points


def cadence_strength(penultimate: list[int], final: list[int],
                      key_root: int = 0) -> float:
    t_pen = harmonic_tension(penultimate, key_root)
    t_fin = harmonic_tension(final, key_root)
    return resolution_strength(t_pen, t_fin)


def prolongational_reduction(curve: np.ndarray, levels: int = 3) -> list[np.ndarray]:
    reductions = [curve.copy()]
    current = curve.copy()
    for _ in range(levels - 1):
        n = len(current)
        if n < 3:
            break
        reduced = np.zeros((n + 1) // 2)
        for j in range(len(reduced)):
            start = j * 2
            end = min(start + 2, n)
            reduced[j] = float(np.max(current[start:end]))
        reductions.append(reduced)
        current = reduced
    return reductions
