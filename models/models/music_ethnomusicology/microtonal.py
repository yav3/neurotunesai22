"""Microtonal systems — cent-based pitch, EDO subdivisions, just-intonation lattices, quarter-tone notation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class MicroPitch:
    cents: float

    @property
    def frequency_ratio(self) -> float:
        return 2.0 ** (self.cents / 1200.0)

    def to_hz(self, reference_hz: float = 440.0) -> float:
        return reference_hz * self.frequency_ratio

    @classmethod
    def from_ratio(cls, ratio: float) -> "MicroPitch":
        return cls(1200.0 * math.log2(ratio))

    @classmethod
    def from_hz(cls, hz: float, reference_hz: float = 440.0) -> "MicroPitch":
        return cls(1200.0 * math.log2(hz / reference_hz))

    def nearest_12tet(self) -> int:
        return round(self.cents / 100.0)

    def deviation_cents(self) -> float:
        return self.cents - self.nearest_12tet() * 100.0


def edo_scale(n: int) -> list[MicroPitch]:
    step = 1200.0 / n
    return [MicroPitch(i * step) for i in range(n)]


def just_interval(numerator: int, denominator: int) -> MicroPitch:
    return MicroPitch.from_ratio(numerator / denominator)


JUST_INTERVALS: dict[str, MicroPitch] = {
    "unison": just_interval(1, 1),
    "minor_second": just_interval(16, 15),
    "major_second": just_interval(9, 8),
    "minor_third": just_interval(6, 5),
    "major_third": just_interval(5, 4),
    "perfect_fourth": just_interval(4, 3),
    "tritone": just_interval(45, 32),
    "perfect_fifth": just_interval(3, 2),
    "minor_sixth": just_interval(8, 5),
    "major_sixth": just_interval(5, 3),
    "minor_seventh": just_interval(9, 5),
    "major_seventh": just_interval(15, 8),
    "octave": just_interval(2, 1),
}


def syntonic_comma() -> MicroPitch:
    return MicroPitch.from_ratio(81 / 80)


def pythagorean_comma() -> MicroPitch:
    return MicroPitch.from_ratio(3**12 / 2**19)


def quarter_tone_scale() -> list[MicroPitch]:
    return edo_scale(24)


@dataclass
class LatticeNode:
    fifths: int
    thirds: int

    @property
    def ratio(self) -> float:
        return (3.0 / 2.0) ** self.fifths * (5.0 / 4.0) ** self.thirds

    @property
    def pitch(self) -> MicroPitch:
        return MicroPitch.from_ratio(self.ratio)


def ji_lattice(max_fifths: int = 2, max_thirds: int = 1) -> list[LatticeNode]:
    nodes: list[LatticeNode] = []
    for f in range(-max_fifths, max_fifths + 1):
        for t in range(-max_thirds, max_thirds + 1):
            nodes.append(LatticeNode(f, t))
    return nodes


def lattice_distance(a: LatticeNode, b: LatticeNode) -> float:
    return math.sqrt((a.fifths - b.fifths) ** 2 + (a.thirds - b.thirds) ** 2)


def cents_distance(a: MicroPitch, b: MicroPitch) -> float:
    return abs(a.cents - b.cents)


def closest_edo_pitch(target: MicroPitch, edo_n: int) -> MicroPitch:
    scale = edo_scale(edo_n)
    target_mod = target.cents % 1200.0
    return min(scale, key=lambda p: abs(p.cents - target_mod))


def edo_approximation_error(ji_intervals: list[MicroPitch], edo_n: int) -> float:
    if not ji_intervals:
        return 0.0
    errors = []
    for ji in ji_intervals:
        closest = closest_edo_pitch(ji, edo_n)
        errors.append(abs((ji.cents % 1200.0) - closest.cents))
    return sum(errors) / len(errors)


def best_edo_for_intervals(intervals: list[MicroPitch], candidates: list[int] | None = None) -> int:
    if candidates is None:
        candidates = [12, 17, 19, 22, 24, 31, 34, 41, 53, 72]
    return min(candidates, key=lambda n: edo_approximation_error(intervals, n))
