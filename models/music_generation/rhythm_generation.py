"""Algorithmic rhythm generation: Euclidean rhythms, probability grids, mutation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "RhythmGrid", "TimeSignature", "RhythmPattern",
    "euclidean_rhythm", "density", "syncopation_index",
    "generate_kick_pattern", "generate_hihat_pattern",
    "combine_patterns", "apply_groove", "pattern_similarity",
    "mutate_rhythm",
]


@dataclass
class RhythmGrid:
    steps: int = 16
    pulses: List[bool] = field(default_factory=list)
    swing: float = 0.0


@dataclass
class TimeSignature:
    beats: int = 4
    beat_value: int = 4


@dataclass
class RhythmPattern:
    grid: RhythmGrid = field(default_factory=RhythmGrid)
    velocity_pattern: List[float] = field(default_factory=list)
    time_sig: TimeSignature = field(default_factory=TimeSignature)


def euclidean_rhythm(pulses: int, steps: int) -> List[bool]:
    if steps <= 0:
        return []
    pulses = max(0, min(pulses, steps))
    if pulses == 0:
        return [False] * steps
    if pulses == steps:
        return [True] * steps

    groups: List[List[bool]] = [[True] for _ in range(pulses)] + [
        [False] for _ in range(steps - pulses)
    ]

    while True:
        remainder_start = pulses
        remainder_count = len(groups) - remainder_start
        if remainder_count <= 1:
            break
        merge_count = min(pulses, remainder_count)
        new_groups = []
        for i in range(merge_count):
            new_groups.append(groups[i] + groups[remainder_start + i])
        for i in range(merge_count, remainder_start):
            new_groups.append(groups[i])
        for i in range(remainder_start + merge_count, len(groups)):
            new_groups.append(groups[i])
        groups = new_groups
        pulses = merge_count + (remainder_start - merge_count)

    result: List[bool] = []
    for g in groups:
        result.extend(g)
    return result


def density(grid: RhythmGrid) -> float:
    if not grid.pulses:
        return 0.0
    return sum(1 for p in grid.pulses if p) / len(grid.pulses)


def syncopation_index(grid: RhythmGrid, meter_weights: Optional[List[float]] = None) -> float:
    if not grid.pulses:
        return 0.0
    n = len(grid.pulses)
    if meter_weights is None:
        meter_weights = [1.0 if i % 4 == 0 else (0.5 if i % 2 == 0 else 0.25) for i in range(n)]
    if len(meter_weights) < n:
        meter_weights = meter_weights + [0.25] * (n - len(meter_weights))

    sync = 0.0
    for i in range(n):
        if grid.pulses[i]:
            next_idx = (i + 1) % n
            if not grid.pulses[next_idx] or True:
                if meter_weights[i] < meter_weights[next_idx if next_idx < len(meter_weights) else 0]:
                    sync += meter_weights[next_idx] - meter_weights[i]
    max_possible = sum(meter_weights) or 1.0
    return sync / max_possible


def generate_kick_pattern(
    steps: int = 16,
    target_density: float = 0.25,
    seed: Optional[int] = None,
) -> RhythmGrid:
    rng = random.Random(seed)
    num_pulses = max(1, round(target_density * steps))
    base = euclidean_rhythm(num_pulses, steps)
    return RhythmGrid(steps=steps, pulses=base)


def generate_hihat_pattern(
    steps: int = 16,
    open_prob: float = 0.2,
    seed: Optional[int] = None,
) -> RhythmGrid:
    rng = random.Random(seed)
    pulses = []
    for i in range(steps):
        if i % 2 == 0:
            pulses.append(True)
        else:
            pulses.append(rng.random() < open_prob)
    return RhythmGrid(steps=steps, pulses=pulses)


def combine_patterns(patterns: List[RhythmGrid]) -> List[List[bool]]:
    if not patterns:
        return []
    max_steps = max(len(p.pulses) for p in patterns)
    result = []
    for p in patterns:
        row = list(p.pulses) + [False] * (max_steps - len(p.pulses))
        result.append(row)
    return result


def apply_groove(
    grid: RhythmGrid,
    groove_template: Optional[List[float]] = None,
) -> List[float]:
    if not grid.pulses:
        return []
    n = len(grid.pulses)
    step_dur = 1.0 / max(n, 1)

    if groove_template is None:
        groove_template = [0.0] * n

    onset_times = []
    for i, hit in enumerate(grid.pulses):
        if hit:
            base_time = i * step_dur
            offset = groove_template[i % len(groove_template)] if groove_template else 0.0
            if grid.swing > 0 and i % 2 == 1:
                offset += grid.swing * step_dur * 0.5
            onset_times.append(base_time + offset)
    return onset_times


def pattern_similarity(a: RhythmGrid, b: RhythmGrid) -> float:
    if not a.pulses or not b.pulses:
        return 0.0
    n = max(len(a.pulses), len(b.pulses))
    pa = list(a.pulses) + [False] * (n - len(a.pulses))
    pb = list(b.pulses) + [False] * (n - len(b.pulses))
    matches = sum(1 for x, y in zip(pa, pb) if x == y)
    return matches / n


def mutate_rhythm(
    grid: RhythmGrid,
    mutation_rate: float = 0.1,
    seed: Optional[int] = None,
) -> RhythmGrid:
    rng = random.Random(seed)
    new_pulses = []
    for p in grid.pulses:
        if rng.random() < mutation_rate:
            new_pulses.append(not p)
        else:
            new_pulses.append(p)
    return RhythmGrid(steps=grid.steps, pulses=new_pulses, swing=grid.swing)
