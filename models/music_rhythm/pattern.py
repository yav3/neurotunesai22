"""Rhythm patterns: Euclidean rhythms, necklace equivalence, complexity metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RhythmPattern:
    steps: int
    onsets: tuple[int, ...]

    @property
    def density(self) -> float:
        return len(self.onsets) / max(self.steps, 1)

    @property
    def binary(self) -> tuple[int, ...]:
        b = [0] * self.steps
        for o in self.onsets:
            if 0 <= o < self.steps:
                b[o] = 1
        return tuple(b)

    @property
    def ioi(self) -> tuple[int, ...]:
        if len(self.onsets) < 2:
            return ()
        sorted_o = sorted(self.onsets)
        intervals = []
        for i in range(len(sorted_o) - 1):
            intervals.append(sorted_o[i + 1] - sorted_o[i])
        intervals.append(self.steps - sorted_o[-1] + sorted_o[0])
        return tuple(intervals)


def euclidean_rhythm(pulses: int, steps: int) -> RhythmPattern:
    if pulses <= 0 or steps <= 0:
        return RhythmPattern(steps=max(steps, 0), onsets=())
    pulses = min(pulses, steps)
    pattern = [0] * steps
    bucket = 0.0
    increment = pulses / steps
    onsets = []
    for i in range(steps):
        bucket += increment
        if bucket >= 1.0:
            bucket -= 1.0
            pattern[i] = 1
            onsets.append(i)
    if len(onsets) < pulses:
        for i in range(steps):
            if pattern[i] == 0 and len(onsets) < pulses:
                pattern[i] = 1
                onsets.append(i)
    return RhythmPattern(steps=steps, onsets=tuple(sorted(onsets[:pulses])))


def _rotate(seq: tuple[int, ...], k: int) -> tuple[int, ...]:
    n = len(seq)
    if n == 0:
        return seq
    k = k % n
    return seq[k:] + seq[:k]


def necklace_equivalence(a: RhythmPattern, b: RhythmPattern) -> bool:
    if a.steps != b.steps or len(a.onsets) != len(b.onsets):
        return False
    ba = a.binary
    bb = b.binary
    for k in range(a.steps):
        if _rotate(ba, k) == bb:
            return True
    return False


def pattern_similarity(a: RhythmPattern, b: RhythmPattern) -> float:
    if a.steps != b.steps:
        max_steps = max(a.steps, b.steps)
        ba = list(a.binary) + [0] * (max_steps - a.steps)
        bb = list(b.binary) + [0] * (max_steps - b.steps)
    else:
        ba = list(a.binary)
        bb = list(b.binary)
    matches = sum(1 for x, y in zip(ba, bb) if x == y)
    return matches / max(len(ba), 1)


def rhythm_complexity(pattern: RhythmPattern) -> float:
    intervals = pattern.ioi
    if len(intervals) < 2:
        return 0.0
    arr = np.array(intervals, dtype=float)
    mean_ioi = np.mean(arr)
    if mean_ioi < 1e-10:
        return 0.0
    cv = float(np.std(arr) / mean_ioi)
    non_simple = sum(1 for i in intervals if i not in (1, 2, 3, 4))
    ratio = non_simple / len(intervals)
    return float(np.clip(0.5 * cv + 0.5 * ratio, 0.0, 1.0))


def pattern_from_ioi(ioi: Sequence[int]) -> RhythmPattern:
    if not ioi:
        return RhythmPattern(steps=0, onsets=())
    steps = sum(ioi)
    onsets = []
    pos = 0
    for interval in ioi:
        onsets.append(pos)
        pos += interval
    return RhythmPattern(steps=steps, onsets=tuple(onsets))


def pattern_to_ioi(pattern: RhythmPattern) -> tuple[int, ...]:
    return pattern.ioi
