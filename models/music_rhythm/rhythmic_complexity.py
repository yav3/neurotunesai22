"""Rhythmic complexity metrics: syncopation, entropy, nPVI, WNBD, Toussaint.

Implements Toussaint (2013) metrically-weighted syncopation, Witek et al.
(2014) groove syncopation profile, normalised pairwise variability index
(nPVI, Patel & Daniele 2003), weighted note-to-beat distance (WNBD,
Gómez et al. 2007), and Shannon/Rényi entropy on onset quantisation grids.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence


@dataclass
class RhythmicComplexityProfile:
    syncopation: float
    entropy: float
    npvi: float
    wnbd: float
    density: float
    evenness: float
    toussaint_complexity: int


METRIC_WEIGHTS_4_4 = np.array([
    5, 1, 2, 1, 3, 1, 2, 1,
    4, 1, 2, 1, 3, 1, 2, 1,
], dtype=float)

METRIC_WEIGHTS_3_4 = np.array([
    5, 1, 2, 1, 3, 1,
    4, 1, 2, 1, 3, 1,
], dtype=float)


def _get_metric_weights(meter: Tuple[int, int] = (4, 4),
                        subdivisions: int = 16) -> np.ndarray:
    if meter == (4, 4) and subdivisions == 16:
        return METRIC_WEIGHTS_4_4.copy()
    if meter == (3, 4) and subdivisions == 12:
        return METRIC_WEIGHTS_3_4.copy()
    weights = np.ones(subdivisions)
    beats = meter[0]
    steps_per_beat = subdivisions // beats
    for b in range(beats):
        weights[b * steps_per_beat] = 4 if b == 0 else 3
        if steps_per_beat >= 2:
            weights[b * steps_per_beat + steps_per_beat // 2] = 2
    return weights


def metrically_weighted_syncopation(
    onsets: Sequence[int],
    meter: Tuple[int, int] = (4, 4),
    subdivisions: int = 16,
) -> float:
    weights = _get_metric_weights(meter, subdivisions)
    grid = np.zeros(subdivisions, dtype=int)
    for o in onsets:
        if 0 <= o < subdivisions:
            grid[o] = 1
    syncopation = 0.0
    for i in range(subdivisions):
        if grid[i] == 1:
            next_i = (i + 1) % subdivisions
            if grid[next_i] == 0 and weights[next_i] > weights[i]:
                syncopation += weights[next_i] - weights[i]
    max_possible = float(np.sum(weights)) * 0.5
    return syncopation / max(max_possible, 1.0)


def onset_entropy(onsets: Sequence[int], subdivisions: int = 16) -> float:
    grid = np.zeros(subdivisions)
    for o in onsets:
        if 0 <= o < subdivisions:
            grid[o] = 1.0
    n = int(np.sum(grid))
    if n <= 1:
        return 0.0
    ioi = []
    positions = [i for i in range(subdivisions) if grid[i] > 0]
    for i in range(1, len(positions)):
        ioi.append(positions[i] - positions[i - 1])
    if positions:
        ioi.append(subdivisions - positions[-1] + positions[0])
    ioi_arr = np.array(ioi, dtype=float)
    total = ioi_arr.sum()
    if total < 1e-10:
        return 0.0
    probs = ioi_arr / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def renyi_entropy(onsets: Sequence[int], subdivisions: int = 16,
                   alpha: float = 2.0) -> float:
    grid = np.zeros(subdivisions)
    for o in onsets:
        if 0 <= o < subdivisions:
            grid[o] = 1.0
    positions = [i for i in range(subdivisions) if grid[i] > 0]
    if len(positions) <= 1:
        return 0.0
    ioi = []
    for i in range(1, len(positions)):
        ioi.append(positions[i] - positions[i - 1])
    ioi.append(subdivisions - positions[-1] + positions[0])
    probs = np.array(ioi, dtype=float)
    probs = probs / probs.sum()
    if abs(alpha - 1.0) < 1e-10:
        return float(-np.sum(probs * np.log2(probs + 1e-12)))
    return float(np.log2(np.sum(probs ** alpha)) / (1 - alpha))


def npvi(durations: Sequence[float]) -> float:
    d = np.asarray(durations, dtype=float)
    if len(d) < 2:
        return 0.0
    contrasts = []
    for i in range(len(d) - 1):
        s = d[i] + d[i + 1]
        if s > 1e-10:
            contrasts.append(abs(d[i] - d[i + 1]) / (s / 2.0))
    return float(np.mean(contrasts)) * 100.0 if contrasts else 0.0


def wnbd(onsets: Sequence[float], beats: Sequence[float]) -> float:
    if len(onsets) == 0 or len(beats) == 0:
        return 0.0
    onsets_a = np.asarray(onsets, dtype=float)
    beats_a = np.asarray(beats, dtype=float)
    total_dist = 0.0
    for o in onsets_a:
        dists = np.abs(beats_a - o)
        min_dist = float(np.min(dists))
        nearest_beat = beats_a[np.argmin(dists)]
        weight = 1.0
        for i, b in enumerate(beats_a):
            if abs(b - nearest_beat) < 1e-10:
                weight = 1.0 / (i % 4 + 1) if i % 4 != 0 else 1.0
                break
        total_dist += min_dist * weight
    return total_dist / len(onsets_a)


def onset_density(onsets: Sequence[int], subdivisions: int = 16) -> float:
    unique = len(set(o for o in onsets if 0 <= o < subdivisions))
    return unique / subdivisions


def evenness(onsets: Sequence[int], subdivisions: int = 16) -> float:
    positions = sorted(set(o for o in onsets if 0 <= o < subdivisions))
    n = len(positions)
    if n <= 1:
        return 1.0
    ideal_spacing = subdivisions / n
    ioi = []
    for i in range(n):
        gap = (positions[(i + 1) % n] - positions[i]) % subdivisions
        ioi.append(gap)
    deviations = [abs(g - ideal_spacing) for g in ioi]
    return max(0.0, 1.0 - float(np.mean(deviations)) / ideal_spacing)


def toussaint_complexity(onsets: Sequence[int], subdivisions: int = 16) -> int:
    grid = [1 if i in onsets else 0 for i in range(subdivisions)]
    complexity = 0
    for k in range(1, subdivisions):
        rotated = grid[k:] + grid[:k]
        diff = sum(a != b for a, b in zip(grid, rotated))
        complexity += diff
    return complexity


def groove_syncopation_profile(onsets: Sequence[int],
                                meter: Tuple[int, int] = (4, 4),
                                subdivisions: int = 16) -> np.ndarray:
    weights = _get_metric_weights(meter, subdivisions)
    profile = np.zeros(subdivisions)
    grid = np.zeros(subdivisions, dtype=int)
    for o in onsets:
        if 0 <= o < subdivisions:
            grid[o] = 1
    for i in range(subdivisions):
        if grid[i] == 1:
            next_i = (i + 1) % subdivisions
            if grid[next_i] == 0 and weights[next_i] > weights[i]:
                profile[i] = weights[next_i] - weights[i]
    return profile


def rhythmic_complexity_profile(
    onsets: Sequence[int],
    durations: Optional[Sequence[float]] = None,
    meter: Tuple[int, int] = (4, 4),
    subdivisions: int = 16,
) -> RhythmicComplexityProfile:
    sync = metrically_weighted_syncopation(onsets, meter, subdivisions)
    ent = onset_entropy(onsets, subdivisions)
    n = npvi(durations) if durations is not None else 0.0
    dens = onset_density(onsets, subdivisions)
    even = evenness(onsets, subdivisions)
    tc = toussaint_complexity(onsets, subdivisions)
    return RhythmicComplexityProfile(
        syncopation=sync, entropy=ent, npvi=n,
        wnbd=0.0, density=dens, evenness=even,
        toussaint_complexity=tc,
    )


def compare_rhythmic_complexity(a: RhythmicComplexityProfile,
                                 b: RhythmicComplexityProfile) -> float:
    va = np.array([a.syncopation, a.entropy, a.npvi / 100, a.density, a.evenness])
    vb = np.array([b.syncopation, b.entropy, b.npvi / 100, b.density, b.evenness])
    return float(np.sqrt(np.sum((va - vb) ** 2)))
