"""Non-Western and historical tuning systems: Pythagorean, meantone, maqam, gamelan, raga."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TuningSystem", "TuningComparison",
    "pythagorean_tuning", "just_intonation", "equal_temperament",
    "meantone_tuning", "maqam_quarter_tones", "slendro_tuning",
    "pelog_tuning", "shruti_tuning", "cents_from_ratio",
    "ratio_from_cents", "tuning_deviation", "compare_tunings",
    "frequency_from_cents", "nearest_12tet_note",
]


@dataclass
class TuningSystem:
    name: str
    intervals_cents: np.ndarray
    reference_hz: float = 440.0
    note_names: List[str] = field(default_factory=list)


@dataclass
class TuningComparison:
    system_a: str
    system_b: str
    mean_deviation_cents: float
    max_deviation_cents: float
    per_degree_deviation: np.ndarray


def cents_from_ratio(ratio: float) -> float:
    if ratio <= 0:
        return 0.0
    return 1200.0 * math.log2(ratio)


def ratio_from_cents(cents: float) -> float:
    return 2.0 ** (cents / 1200.0)


def frequency_from_cents(reference_hz: float, cents: float) -> float:
    return reference_hz * ratio_from_cents(cents)


def nearest_12tet_note(cents: float) -> Tuple[int, float]:
    semitone = round(cents / 100.0)
    deviation = cents - semitone * 100.0
    return semitone, deviation


def equal_temperament(n_divisions: int = 12, reference_hz: float = 440.0) -> TuningSystem:
    intervals = np.array([i * 1200.0 / n_divisions for i in range(n_divisions + 1)])
    names_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C'"]
    names = names_12 if n_divisions == 12 else [str(i) for i in range(n_divisions + 1)]
    return TuningSystem("equal_temperament", intervals, reference_hz, names)


def pythagorean_tuning(reference_hz: float = 440.0) -> TuningSystem:
    fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    ratios = {}
    for i, degree in enumerate(fifths_order):
        ratio = (3.0 / 2.0) ** i
        while ratio >= 2.0:
            ratio /= 2.0
        ratios[degree] = ratio
    intervals = sorted(cents_from_ratio(ratios[d]) for d in range(12))
    intervals.append(1200.0)
    return TuningSystem("pythagorean", np.array(intervals), reference_hz)


def just_intonation(reference_hz: float = 440.0) -> TuningSystem:
    ratios = [1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8, 2]
    intervals = np.array([cents_from_ratio(r) for r in ratios])
    return TuningSystem("just_intonation", intervals, reference_hz)


def meantone_tuning(comma_fraction: float = 0.25, reference_hz: float = 440.0) -> TuningSystem:
    syntonic_comma = cents_from_ratio(81 / 80)
    fifth = 700.0 + (1.0 - 4.0 * comma_fraction) * syntonic_comma / 4.0
    intervals = [0.0]
    for i in range(1, 12):
        val = (i * fifth) % 1200.0
        intervals.append(val)
    intervals.sort()
    intervals.append(1200.0)
    return TuningSystem("meantone", np.array(intervals), reference_hz)


def maqam_quarter_tones(reference_hz: float = 440.0) -> TuningSystem:
    intervals = np.array([i * 50.0 for i in range(25)])
    return TuningSystem("maqam_24tet", intervals, reference_hz)


def slendro_tuning(reference_hz: float = 440.0) -> TuningSystem:
    intervals = np.array([0.0, 240.0, 480.0, 720.0, 960.0, 1200.0])
    return TuningSystem("slendro", intervals, reference_hz)


def pelog_tuning(reference_hz: float = 440.0) -> TuningSystem:
    intervals = np.array([0.0, 120.0, 270.0, 540.0, 670.0, 790.0, 950.0, 1200.0])
    return TuningSystem("pelog", intervals, reference_hz)


def shruti_tuning(reference_hz: float = 440.0) -> TuningSystem:
    shruti_ratios = [
        1, 256/243, 16/15, 10/9, 9/8, 32/27, 6/5, 5/4, 81/64,
        4/3, 27/20, 45/32, 729/512, 3/2, 128/81, 8/5, 5/3,
        27/16, 16/9, 9/5, 15/8, 243/128, 2
    ]
    intervals = np.array([cents_from_ratio(r) for r in shruti_ratios])
    return TuningSystem("shruti_22", intervals, reference_hz)


def tuning_deviation(system: TuningSystem, reference: Optional[TuningSystem] = None) -> np.ndarray:
    if reference is None:
        n = len(system.intervals_cents) - 1
        reference = equal_temperament(n, system.reference_hz)
    n = min(len(system.intervals_cents), len(reference.intervals_cents))
    return system.intervals_cents[:n] - reference.intervals_cents[:n]


def compare_tunings(a: TuningSystem, b: TuningSystem) -> TuningComparison:
    n = min(len(a.intervals_cents), len(b.intervals_cents))
    dev = a.intervals_cents[:n] - b.intervals_cents[:n]
    abs_dev = np.abs(dev)
    return TuningComparison(
        system_a=a.name, system_b=b.name,
        mean_deviation_cents=float(np.mean(abs_dev)),
        max_deviation_cents=float(np.max(abs_dev)),
        per_degree_deviation=dev,
    )
