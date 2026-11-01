"""Tuning systems: just intonation, Pythagorean, meantone, non-Western scales."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TuningSystem:
    name: str
    intervals: tuple[float, ...]
    octave_ratio: float = 2.0
    description: str = ""

    @property
    def n_tones(self) -> int:
        return len(self.intervals)

    @property
    def cents(self) -> tuple[float, ...]:
        return tuple(ratio_to_cents(r) for r in self.intervals)

    def frequencies(self, base_freq: float = 261.63) -> tuple[float, ...]:
        return tuple(base_freq * r for r in self.intervals)


def cents_to_ratio(cents: float) -> float:
    return 2.0 ** (cents / 1200.0)


def ratio_to_cents(ratio: float) -> float:
    if ratio <= 0:
        return 0.0
    return 1200.0 * np.log2(ratio)


def compare_tunings(a: TuningSystem, b: TuningSystem) -> float:
    ca = np.array(a.cents)
    cb = np.array(b.cents)
    min_len = min(len(ca), len(cb))
    if min_len == 0:
        return 0.0
    return float(np.mean(np.abs(ca[:min_len] - cb[:min_len])))


def equal_temperament(n_divisions: int = 12) -> TuningSystem:
    intervals = tuple(2.0 ** (i / n_divisions) for i in range(n_divisions))
    return TuningSystem(
        name=f"{n_divisions}-TET",
        intervals=intervals,
        description=f"Equal division of octave into {n_divisions} parts",
    )


def pythagorean_tuning() -> TuningSystem:
    ratios = [1.0]
    r = 1.0
    for _ in range(11):
        r *= 3.0 / 2.0
        while r >= 2.0:
            r /= 2.0
        ratios.append(r)
    ratios.sort()
    return TuningSystem(
        name="Pythagorean",
        intervals=tuple(ratios),
        description="3-limit tuning via pure fifths",
    )


def just_intonation(limit: int = 5) -> TuningSystem:
    if limit == 5:
        intervals = (1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8)
    elif limit == 7:
        intervals = (1/1, 16/15, 9/8, 7/6, 5/4, 4/3, 7/5, 3/2, 8/5, 5/3, 7/4, 15/8)
    else:
        intervals = (1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8)
    return TuningSystem(
        name=f"Just Intonation ({limit}-limit)",
        intervals=intervals,
        description=f"{limit}-limit just intonation",
    )


def meantone_temperament(comma_fraction: float = 0.25) -> TuningSystem:
    fifth = (3.0 / 2.0) * (81.0 / 80.0) ** (-comma_fraction)
    ratios = [1.0]
    r = 1.0
    for _ in range(11):
        r *= fifth
        while r >= 2.0:
            r /= 2.0
        ratios.append(r)
    ratios.sort()
    return TuningSystem(
        name=f"Meantone (1/{int(1/comma_fraction)}-comma)",
        intervals=tuple(ratios),
        description=f"Meantone with {comma_fraction} syntonic comma tempering",
    )


_MAQAM_CENTS: dict[str, tuple[float, ...]] = {
    "rast": (0, 200, 350, 500, 700, 900, 1050, 1200),
    "bayati": (0, 150, 300, 500, 700, 850, 1000, 1200),
    "hijaz": (0, 100, 350, 500, 700, 850, 1000, 1200),
    "saba": (0, 150, 300, 450, 700, 800, 1000, 1200),
    "nahawand": (0, 200, 300, 500, 700, 800, 1000, 1200),
}


def maqam_intervals(name: str) -> TuningSystem:
    name_lower = name.lower()
    if name_lower not in _MAQAM_CENTS:
        raise ValueError(f"Unknown maqam: {name}. Available: {list(_MAQAM_CENTS.keys())}")
    cents = _MAQAM_CENTS[name_lower]
    intervals = tuple(cents_to_ratio(c) for c in cents)
    return TuningSystem(
        name=f"Maqam {name.capitalize()}",
        intervals=intervals,
        description=f"Arabic maqam {name}",
    )


_RAGA_CENTS: dict[str, tuple[float, ...]] = {
    "bilawal": (0, 200, 400, 500, 700, 900, 1100),
    "kafi": (0, 200, 300, 500, 700, 900, 1000),
    "bhairavi": (0, 100, 300, 500, 700, 800, 1000),
    "todi": (0, 100, 300, 500, 600, 900, 1100),
    "marwa": (0, 100, 400, 600, 700, 900, 1100),
}


def raga_intervals(name: str) -> TuningSystem:
    name_lower = name.lower()
    if name_lower not in _RAGA_CENTS:
        raise ValueError(f"Unknown raga: {name}. Available: {list(_RAGA_CENTS.keys())}")
    cents = _RAGA_CENTS[name_lower]
    intervals = tuple(cents_to_ratio(c) for c in cents)
    return TuningSystem(
        name=f"Raga {name.capitalize()}",
        intervals=intervals,
        description=f"Indian raga {name}",
    )


def pelog_scale() -> TuningSystem:
    cents = (0, 120, 270, 540, 680, 780, 940)
    return TuningSystem(
        name="Pelog",
        intervals=tuple(cents_to_ratio(c) for c in cents),
        description="Javanese pelog (7-tone)",
    )


def slendro_scale() -> TuningSystem:
    cents = (0, 240, 480, 720, 960)
    return TuningSystem(
        name="Slendro",
        intervals=tuple(cents_to_ratio(c) for c in cents),
        description="Javanese slendro (5-tone near-equal)",
    )
