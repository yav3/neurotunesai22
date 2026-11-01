"""Polyrhythm analysis: cross-rhythms, Euclidean rhythms, rhythmic dissonance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "PolyrhythmPattern", "EuclideanRhythm",
    "polyrhythm_grid", "resultant_rhythm", "cross_rhythm_detect",
    "euclidean_rhythm", "rhythmic_dissonance", "polymetric_superposition",
    "rhythm_augment", "rhythm_diminish", "rhythm_retrograde", "rhythm_rotate",
    "rhythmic_complexity", "onset_density",
]


@dataclass
class PolyrhythmPattern:
    ratio_a: int
    ratio_b: int
    lcm_period: int
    onset_grid_a: np.ndarray
    onset_grid_b: np.ndarray
    resultant: np.ndarray


@dataclass
class EuclideanRhythm:
    n_slots: int
    n_onsets: int
    pattern: np.ndarray
    rotation: int = 0


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // _gcd(a, b)


def polyrhythm_grid(ratio_a: int, ratio_b: int, resolution: int = 0) -> PolyrhythmPattern:
    period = _lcm(ratio_a, ratio_b)
    if resolution <= 0:
        resolution = period
    grid_a = np.zeros(resolution, dtype=np.float64)
    grid_b = np.zeros(resolution, dtype=np.float64)
    step_a = resolution / ratio_a
    step_b = resolution / ratio_b
    for i in range(ratio_a):
        idx = int(round(i * step_a)) % resolution
        grid_a[idx] = 1.0
    for i in range(ratio_b):
        idx = int(round(i * step_b)) % resolution
        grid_b[idx] = 1.0
    resultant = np.clip(grid_a + grid_b, 0.0, 1.0)
    return PolyrhythmPattern(
        ratio_a=ratio_a, ratio_b=ratio_b, lcm_period=period,
        onset_grid_a=grid_a, onset_grid_b=grid_b, resultant=resultant,
    )


def resultant_rhythm(onsets_a, onsets_b) -> list:
    if isinstance(onsets_a, (int, np.integer)) and isinstance(onsets_b, (int, np.integer)):
        a, b = int(onsets_a), int(onsets_b)
        n = _lcm(a, b)
        grid = [0] * n
        step_a = n / a if a > 0 else n
        step_b = n / b if b > 0 else n
        for i in range(a):
            grid[int(round(i * step_a)) % n] = 1
        for i in range(b):
            grid[int(round(i * step_b)) % n] = 1
        return grid
    arr_a, arr_b = np.asarray(onsets_a), np.asarray(onsets_b)
    n = max(len(arr_a), len(arr_b))
    result = np.zeros(n)
    result[:len(arr_a)] += arr_a[:n]
    result[:len(arr_b)] += arr_b[:n]
    return list(np.clip(result, 0.0, 1.0))


def cross_rhythm_detect(
    onsets_a: np.ndarray, onsets_b: np.ndarray, tolerance: float = 0.05
) -> Optional[Tuple[int, int]]:
    hits_a = np.where(onsets_a > 0.5)[0]
    hits_b = np.where(onsets_b > 0.5)[0]
    if len(hits_a) < 2 or len(hits_b) < 2:
        return None
    n_a = len(hits_a)
    n_b = len(hits_b)
    g = _gcd(n_a, n_b)
    ratio = (n_a // g, n_b // g)
    if ratio[0] == ratio[1]:
        return None
    span_a = float(hits_a[-1] - hits_a[0])
    span_b = float(hits_b[-1] - hits_b[0])
    if span_a == 0 or span_b == 0:
        return ratio
    ioi_a = span_a / (n_a - 1)
    ioi_b = span_b / (n_b - 1)
    expected_ratio = ioi_b / ioi_a
    actual_ratio = ratio[0] / ratio[1]
    if abs(expected_ratio - actual_ratio) / max(actual_ratio, 1e-9) < tolerance:
        return ratio
    return None


def euclidean_rhythm(n_slots: int, n_onsets: int, rotation: int = 0) -> EuclideanRhythm:
    if n_onsets <= 0:
        return EuclideanRhythm(n_slots, 0, np.zeros(n_slots), rotation)
    if n_onsets >= n_slots:
        return EuclideanRhythm(n_slots, n_slots, np.ones(n_slots), rotation)
    pattern = [0] * n_slots
    bucket = 0
    for i in range(n_slots):
        bucket += n_onsets
        if bucket >= n_slots:
            bucket -= n_slots
            pattern[i] = 1
    arr = np.array(pattern, dtype=np.float64)
    if rotation != 0:
        arr = np.roll(arr, rotation)
    return EuclideanRhythm(n_slots, n_onsets, arr, rotation)


def rhythmic_dissonance(onsets_a: np.ndarray, onsets_b: np.ndarray) -> float:
    if len(onsets_a) == 0 or len(onsets_b) == 0:
        return 0.0
    n = min(len(onsets_a), len(onsets_b))
    a = onsets_a[:n]
    b = onsets_b[:n]
    simultaneous = np.sum((a > 0.5) & (b > 0.5))
    total_onsets = np.sum(a > 0.5) + np.sum(b > 0.5)
    if total_onsets == 0:
        return 0.0
    overlap_ratio = 2.0 * simultaneous / total_onsets
    return 1.0 - overlap_ratio


def polymetric_superposition(
    meter_a: int, meter_b: int, n_cycles: int = 1, resolution: int = 0
) -> np.ndarray:
    period = _lcm(meter_a, meter_b) * n_cycles
    if resolution <= 0:
        resolution = period
    grid = np.zeros(resolution)
    step_a = resolution / (meter_a * n_cycles)
    step_b = resolution / (meter_b * n_cycles)
    for i in range(meter_a * n_cycles + 1):
        idx = min(int(round(i * step_a)), resolution - 1)
        grid[idx] = 1.0
    for i in range(meter_b * n_cycles + 1):
        idx = min(int(round(i * step_b)), resolution - 1)
        grid[idx] = 1.0
    return np.clip(grid, 0.0, 1.0)


def rhythm_augment(pattern: np.ndarray, factor: int = 2) -> np.ndarray:
    return np.repeat(pattern, factor)


def rhythm_diminish(pattern: np.ndarray, factor: int = 2) -> np.ndarray:
    if factor <= 1:
        return pattern.copy()
    return pattern[::factor]


def rhythm_retrograde(pattern: np.ndarray) -> np.ndarray:
    return pattern[::-1].copy()


def rhythm_rotate(pattern: np.ndarray, steps: int = 1) -> np.ndarray:
    return np.roll(pattern, steps)


def rhythmic_complexity(pattern: np.ndarray) -> float:
    if len(pattern) == 0:
        return 0.0
    onsets = np.where(pattern > 0.5)[0]
    if len(onsets) <= 1:
        return 0.0
    ioi = np.diff(onsets).astype(float)
    mean_ioi = np.mean(ioi)
    if mean_ioi == 0:
        return 0.0
    cv = float(np.std(ioi) / mean_ioi)
    density = len(onsets) / len(pattern)
    return float(cv * density)


def onset_density(pattern: np.ndarray) -> float:
    if len(pattern) == 0:
        return 0.0
    return float(np.sum(pattern > 0.5) / len(pattern))


# ---------------------------------------------------------------------------
# Extended API — Bjorklund, clave patterns, polyrhythm generators
# ---------------------------------------------------------------------------

def compute_lcm_cycle(a: int, b: int) -> int:
    return _lcm(a, b)


@dataclass
class _PolyVoices:
    voices: List[List[int]]
    cycle_length: int


def generate_polyrhythm(pulse_a: int, pulse_b: int, total_steps: int) -> _PolyVoices:
    va = [0] * total_steps
    vb = [0] * total_steps
    step_a = total_steps / pulse_a if pulse_a > 0 else total_steps
    step_b = total_steps / pulse_b if pulse_b > 0 else total_steps
    for i in range(pulse_a):
        va[int(round(i * step_a)) % total_steps] = 1
    for i in range(pulse_b):
        vb[int(round(i * step_b)) % total_steps] = 1
    return _PolyVoices(voices=[va, vb], cycle_length=total_steps)


def euclidean_rhythm_bjorklund(k: int, n: int) -> List[int]:
    if n <= 0:
        return []
    if k <= 0:
        return [0] * n
    if k >= n:
        return [1] * n
    groups: List[List[int]] = [[1] for _ in range(k)] + [[0] for _ in range(n - k)]
    while True:
        remainder = len(groups) - k
        if remainder <= 1:
            break
        new_groups: List[List[int]] = []
        pairs = min(k, remainder)
        for i in range(pairs):
            new_groups.append(groups[i] + groups[k + i])
        for i in range(pairs, k):
            new_groups.append(groups[i])
        for i in range(k + pairs, len(groups)):
            new_groups.append(groups[i])
        groups = new_groups
        k = pairs + max(0, k - pairs)
    pattern: List[int] = []
    for g in groups:
        pattern.extend(g)
    return pattern


def cross_rhythm_density(pr: _PolyVoices) -> float:
    combined = set()
    for v in pr.voices:
        for i, val in enumerate(v):
            if val:
                combined.add(i)
    return len(combined) / pr.cycle_length if pr.cycle_length > 0 else 0.0


def polymetric_offset(meter_a: int, meter_b: int, step: int) -> float:
    if step == 0:
        return 0.0
    pos_a = (step % meter_a) / meter_a
    pos_b = (step % meter_b) / meter_b
    diff = abs(pos_a - pos_b)
    return min(diff, 1.0 - diff)


def rhythmic_oddity(pattern: List[int]) -> bool:
    n = len(pattern)
    if n == 0 or n % 2 != 0:
        return False
    onsets = [i for i, v in enumerate(pattern) if v]
    if len(onsets) < 2:
        return False
    half = n // 2
    for i in range(len(onsets)):
        for j in range(i + 1, len(onsets)):
            gap = (onsets[j] - onsets[i]) % n
            if gap == half:
                return False
    return True


def maximal_evenness(k: int, n: int) -> List[int]:
    return euclidean_rhythm_bjorklund(k, n)


def polyrhythm_complexity(a: int, b: int) -> float:
    lcm_val = _lcm(a, b)
    return float(lcm_val / max(a, b))


def isochronous_deviation(onsets: np.ndarray, period: float) -> np.ndarray:
    if len(onsets) == 0:
        return np.array([])
    expected = np.arange(len(onsets)) * period + onsets[0]
    return onsets - expected


def aksak_pattern(groups: List[int]) -> List[int]:
    total = sum(groups)
    pat = [0] * total
    pos = 0
    for g in groups:
        pat[pos] = 1
        pos += g
    return pat


def tresillo() -> List[int]:
    return aksak_pattern([3, 3, 2])


def son_clave_3_2() -> List[int]:
    pat = [0] * 16
    for i in [0, 3, 7, 11, 13]:
        pat[i] = 1
    return pat


def rumba_clave_3_2() -> List[int]:
    pat = [0] * 16
    for i in [0, 3, 8, 11, 13]:
        pat[i] = 1
    return pat
