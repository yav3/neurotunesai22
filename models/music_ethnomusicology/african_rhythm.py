"""Sub-Saharan African polyrhythm: timelines, Euclidean rhythms, cross-rhythms."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimelinePattern:
    name: str
    pulses: tuple[int, ...]

    @property
    def cycle_length(self) -> int:
        return len(self.pulses)

    @property
    def onset_count(self) -> int:
        return sum(self.pulses)


STANDARD_PATTERNS: dict[str, TimelinePattern] = {
    "son_clave_32":   TimelinePattern("Son Clave 3-2", (1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0)),
    "son_clave_23":   TimelinePattern("Son Clave 2-3", (0,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0)),
    "rumba_clave":    TimelinePattern("Rumba Clave", (1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0)),
    "bembe":          TimelinePattern("Bembe", (1,0,1,0,1,1,0,1,0,1,0,1)),
    "shiko":          TimelinePattern("Shiko", (1,0,0,1,0,1,0,0,1,0,1,0)),
    "gahu":           TimelinePattern("Gahu", (1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,0)),
    "agbadza":        TimelinePattern("Agbadza", (1,0,1,0,1,1,0,1,0,1,1,0)),
    "fume_fume":      TimelinePattern("Fume-Fume", (1,0,1,1,0,1,0,1,1,0,1,0)),
    "kpanlogo":       TimelinePattern("Kpanlogo", (1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0)),
    "bossa_nova":     TimelinePattern("Bossa Nova Clave", (1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0)),
    "tresillo":       TimelinePattern("Tresillo", (1,0,0,1,0,0,1,0)),
    "cinquillo":      TimelinePattern("Cinquillo", (1,0,1,1,0,1,1,0)),
}


def euclidean_rhythm(pulses: int, steps: int) -> tuple[int, ...]:
    if pulses <= 0 or steps <= 0:
        return (0,) * max(steps, 0)
    pulses = min(pulses, steps)
    groups: list[list[int]] = [[1] for _ in range(pulses)] + [[0] for _ in range(steps - pulses)]
    while True:
        remainder = [g for g in groups if len(g) != len(groups[0])] if groups else []
        if len(remainder) <= 1:
            break
        main = [g for g in groups if g not in remainder]
        if not main:
            break
        new_groups: list[list[int]] = []
        n = min(len(main), len(remainder))
        for i in range(n):
            new_groups.append(main[i] + remainder[i])
        new_groups.extend(main[n:])
        new_groups.extend(remainder[n:])
        groups = new_groups
    result: list[int] = []
    for g in groups:
        result.extend(g)
    return tuple(result)


def cross_rhythm(pulses_a: int, pulses_b: int, cycle_length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    a = euclidean_rhythm(pulses_a, cycle_length)
    b = euclidean_rhythm(pulses_b, cycle_length)
    return a, b


def rotate_pattern(pattern: tuple[int, ...], offset: int) -> tuple[int, ...]:
    if not pattern:
        return pattern
    n = len(pattern)
    offset = offset % n
    return pattern[offset:] + pattern[:offset]


def pattern_density(pattern: tuple[int, ...]) -> float:
    if not pattern:
        return 0.0
    return sum(pattern) / len(pattern)


def syncopation_index(pattern: tuple[int, ...]) -> float:
    if not pattern or len(pattern) < 2:
        return 0.0
    n = len(pattern)
    meter_weights = np.zeros(n)
    for i in range(n):
        if i == 0:
            meter_weights[i] = 4.0
        elif i == n // 2:
            meter_weights[i] = 3.0
        elif i % (n // 4) == 0 if n >= 4 else False:
            meter_weights[i] = 2.0
        else:
            meter_weights[i] = 1.0
    syncopation = 0.0
    for i in range(n):
        if pattern[i] == 1:
            next_idx = (i + 1) % n
            if meter_weights[next_idx] > meter_weights[i]:
                syncopation += meter_weights[next_idx] - meter_weights[i]
    max_syncopation = sum(meter_weights) if sum(pattern) > 0 else 1.0
    return syncopation / max_syncopation
