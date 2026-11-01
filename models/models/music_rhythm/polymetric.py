"""Polymetric and polymeter analysis — simultaneous conflicting meters."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Meter:
    beats: int
    subdivision: int = 1

    @property
    def pulse_count(self) -> int:
        return self.beats * self.subdivision


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def polymetric_cycle_length(m1: Meter, m2: Meter) -> int:
    return lcm(m1.pulse_count, m2.pulse_count)


def coincidence_points(m1: Meter, m2: Meter) -> list[int]:
    cycle = polymetric_cycle_length(m1, m2)
    p1, p2 = m1.pulse_count, m2.pulse_count
    return [i for i in range(cycle) if i % p1 == 0 and i % p2 == 0]


def coincidence_density(m1: Meter, m2: Meter) -> float:
    cycle = polymetric_cycle_length(m1, m2)
    points = coincidence_points(m1, m2)
    if cycle == 0:
        return 1.0
    return len(points) / cycle


def metric_dissonance(m1: Meter, m2: Meter) -> float:
    cd = coincidence_density(m1, m2)
    return 1.0 - cd


def hemiola_ratio(m1: Meter, m2: Meter) -> tuple[int, int]:
    g = math.gcd(m1.beats, m2.beats)
    return m1.beats // g, m2.beats // g


def is_hemiola(m1: Meter, m2: Meter) -> bool:
    r = hemiola_ratio(m1, m2)
    return r in [(3, 2), (2, 3)]


def beat_pattern(meter: Meter, n_pulses: int) -> list[bool]:
    return [i % meter.pulse_count == 0 for i in range(n_pulses)]


def combined_pattern(m1: Meter, m2: Meter) -> list[tuple[bool, bool]]:
    cycle = polymetric_cycle_length(m1, m2)
    p1 = beat_pattern(m1, cycle)
    p2 = beat_pattern(m2, cycle)
    return list(zip(p1, p2))


def metric_complexity(meters: list[Meter]) -> float:
    if len(meters) < 2:
        return 0.0
    total_dissonance = 0.0
    count = 0
    for i in range(len(meters)):
        for j in range(i + 1, len(meters)):
            total_dissonance += metric_dissonance(meters[i], meters[j])
            count += 1
    return total_dissonance / count


def grouping_conflict(onsets_a: list[float], onsets_b: list[float], tolerance: float = 0.05) -> float:
    if not onsets_a or not onsets_b:
        return 0.0
    conflicts = 0
    total = 0
    for oa in onsets_a:
        for ob in onsets_b:
            total += 1
            if abs(oa - ob) > tolerance:
                conflicts += 1
    return conflicts / max(total, 1)
