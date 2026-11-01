"""Polyrhythmic pattern analysis and generation.

Implements Bjorklund/Toussaint euclidean rhythms, polyrhythm generation,
cross-rhythm density, and rhythmic oddity analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class PolyrhythmPattern:
    voices: List[List[int]]
    cycle_length: int
    lcm_beats: int


def compute_lcm_cycle(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def generate_polyrhythm(p: int, q: int, beats: int | None = None) -> PolyrhythmPattern:
    lcm = compute_lcm_cycle(p, q)
    n = beats if beats is not None else lcm
    voice_p = [0] * n
    voice_q = [0] * n
    step_p = n / p
    step_q = n / q
    for i in range(p):
        idx = int(round(i * step_p)) % n
        voice_p[idx] = 1
    for i in range(q):
        idx = int(round(i * step_q)) % n
        voice_q[idx] = 1
    return PolyrhythmPattern(voices=[voice_p, voice_q], cycle_length=n, lcm_beats=lcm)


def euclidean_rhythm_bjorklund(pulses: int, steps: int) -> List[int]:
    """Bjorklund's algorithm for maximally-even distribution of pulses in steps."""
    if pulses <= 0:
        return [0] * steps
    if pulses >= steps:
        return [1] * steps
    groups: list = [[1] for _ in range(pulses)] + [[0] for _ in range(steps - pulses)]
    while True:
        remainder = len(groups) - pulses
        if remainder <= 1:
            break
        new_groups = []
        take = min(pulses, remainder)
        for i in range(take):
            new_groups.append(groups[i] + groups[pulses + i])
        leftover = groups[take:pulses] + groups[pulses + take:]
        groups = new_groups + leftover
        pulses = len(new_groups)
    result: list[int] = []
    for g in groups:
        result.extend(g)
    return result


def cross_rhythm_density(pattern: PolyrhythmPattern) -> float:
    if not pattern.voices or pattern.cycle_length == 0:
        return 0.0
    coincidences = 0
    for i in range(pattern.cycle_length):
        if all(v[i % len(v)] for v in pattern.voices):
            coincidences += 1
    return coincidences / pattern.cycle_length


def polymetric_offset(meter_a: int, meter_b: int, position: int) -> float:
    if meter_a == 0 or meter_b == 0:
        return 0.0
    phase_a = (position % meter_a) / meter_a
    phase_b = (position % meter_b) / meter_b
    return abs(phase_a - phase_b)


def rhythmic_oddity(pattern: List[int]) -> bool:
    """Test Pressing's rhythmic oddity property: no two inter-onset intervals
    that sum to exactly half the cycle length are adjacent."""
    onsets = [i for i, v in enumerate(pattern) if v]
    if len(onsets) < 2:
        return False
    n = len(pattern)
    half = n / 2.0
    iois = []
    for i in range(len(onsets)):
        nxt = onsets[(i + 1) % len(onsets)]
        cur = onsets[i]
        ioi = (nxt - cur) % n
        iois.append(ioi)
    for i in range(len(iois)):
        pair_sum = iois[i] + iois[(i + 1) % len(iois)]
        if abs(pair_sum - half) < 1e-9:
            return False
    return True


def maximal_evenness(k: int, n: int) -> List[int]:
    """Toussaint's maximally-even distribution — equivalent to euclidean rhythm."""
    return euclidean_rhythm_bjorklund(k, n)


def polyrhythm_complexity(p: int, q: int) -> float:
    """Measure complexity of a p:q polyrhythm.
    Higher when the ratio is farther from simple integer relationships."""
    if p == 0 or q == 0:
        return 0.0
    g = math.gcd(p, q)
    reduced_p, reduced_q = p // g, q // g
    return math.log2(reduced_p * reduced_q)


def isochronous_deviation(onsets: np.ndarray, period: float) -> np.ndarray:
    """Measure how much each onset deviates from a perfectly isochronous grid."""
    if len(onsets) == 0 or period <= 0:
        return np.array([])
    grid_positions = np.round(onsets / period) * period
    return onsets - grid_positions


def resultant_rhythm(p: int, q: int) -> List[int]:
    """Schillinger's resultant rhythm: the union of p-pulse and q-pulse patterns."""
    lcm = compute_lcm_cycle(p, q)
    result = [0] * lcm
    for i in range(lcm):
        if i % (lcm // p) == 0 or i % (lcm // q) == 0:
            result[i] = 1
    return result


def aksak_pattern(groups: List[int]) -> List[int]:
    """Generate an aksak (additive) rhythm from a list of beat groups.
    E.g., [2,2,3] for 7/8 in Balkan music."""
    pattern: list[int] = []
    for g in groups:
        pattern.append(1)
        pattern.extend([0] * (g - 1))
    return pattern


def tresillo() -> List[int]:
    """The 3+3+2 tresillo pattern — foundational Afro-Cuban rhythm."""
    return aksak_pattern([3, 3, 2])


def son_clave_3_2() -> List[int]:
    """Son clave 3-2 pattern in 16 pulses."""
    pattern = [0] * 16
    for i in [0, 3, 7, 11, 13]:
        pattern[i] = 1
    return pattern


def rumba_clave_3_2() -> List[int]:
    """Rumba clave 3-2 pattern in 16 pulses."""
    pattern = [0] * 16
    for i in [0, 3, 7, 10, 13]:
        pattern[i] = 1
    return pattern


def onset_density(pattern: List[int]) -> float:
    if not pattern:
        return 0.0
    return sum(pattern) / len(pattern)
