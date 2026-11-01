"""Tuning & temperament: equal, just, meantone, well, historical temperaments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional

import numpy as np


class TemperamentType(str, Enum):
    EQUAL = "equal"
    JUST_MAJOR = "just_major"
    JUST_MINOR = "just_minor"
    PYTHAGOREAN = "pythagorean"
    MEANTONE_QUARTER = "quarter_comma_meantone"
    WERCKMEISTER_III = "werckmeister_iii"
    KIRNBERGER_III = "kirnberger_iii"
    YOUNG = "young"
    VALLOTTI = "vallotti"


@dataclass
class TemperamentSpec:
    name: str
    cents: np.ndarray          # 12 deviations from 12-TET in cents
    reference_freq: float = 440.0
    reference_midi: int = 69

    def freq(self, midi: int) -> float:
        pc = midi % 12
        octave_offset = (midi - self.reference_midi) / 12.0
        cent_dev = self.cents[pc] - self.cents[self.reference_midi % 12]
        return self.reference_freq * (2.0 ** (octave_offset + cent_dev / 1200.0))


@dataclass
class IntervalQuality:
    ratio: Tuple[int, int]
    cents: float
    beat_rate_hz: float
    name: str


JUST_RATIOS_MAJOR: Dict[int, Tuple[int, int]] = {
    0: (1, 1), 1: (16, 15), 2: (9, 8), 3: (6, 5), 4: (5, 4),
    5: (4, 3), 6: (45, 32), 7: (3, 2), 8: (8, 5), 9: (5, 3),
    10: (9, 5), 11: (15, 8),
}

JUST_RATIOS_MINOR: Dict[int, Tuple[int, int]] = {
    0: (1, 1), 1: (16, 15), 2: (9, 8), 3: (6, 5), 4: (5, 4),
    5: (4, 3), 6: (7, 5), 7: (3, 2), 8: (8, 5), 9: (5, 3),
    10: (7, 4), 11: (15, 8),
}

SYNTONIC_COMMA_CENTS = 1200.0 * math.log2(81 / 80)
PYTHAGOREAN_COMMA_CENTS = 1200.0 * math.log2(3**12 / 2**19)
DIESIS_CENTS = 1200.0 * math.log2(128 / 125)


def ratio_to_cents(num: int, den: int) -> float:
    if num <= 0 or den <= 0:
        raise ValueError("ratio components must be positive")
    return 1200.0 * math.log2(num / den)


def cents_to_ratio_approx(c: float, limit: int = 128) -> Tuple[int, int]:
    target = 2.0 ** (c / 1200.0)
    best_n, best_d, best_err = 1, 1, abs(target - 1.0)
    for d in range(1, limit + 1):
        n = round(target * d)
        if 1 <= n <= limit * 2:
            err = abs(n / d - target)
            if err < best_err:
                best_n, best_d, best_err = n, d, err
    return (best_n, best_d)


def beat_frequency(f1: float, f2: float) -> float:
    return abs(f1 - f2)


def _just_cents(ratios: Dict[int, Tuple[int, int]]) -> np.ndarray:
    return np.array([ratio_to_cents(*ratios[i]) for i in range(12)])


def equal_temperament_cents() -> np.ndarray:
    return np.array([i * 100.0 for i in range(12)])


def pythagorean_cents() -> np.ndarray:
    fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    cents = np.zeros(12)
    pure_fifth = 1200.0 * math.log2(3 / 2)
    for i, pc in enumerate(fifths_order):
        c = (i * pure_fifth) % 1200.0
        cents[pc] = c
    return cents


def quarter_comma_meantone_cents() -> np.ndarray:
    fifth = 1200.0 * math.log2(3 / 2) - SYNTONIC_COMMA_CENTS / 4.0
    cents = np.zeros(12)
    order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    for i, pc in enumerate(order):
        cents[pc] = (i * fifth) % 1200.0
    return cents


def werckmeister_iii_cents() -> np.ndarray:
    pc = PYTHAGOREAN_COMMA_CENTS
    return np.array([
        0.0, 90.225, 192.18, 294.135, 390.225,
        498.045, 588.27, 696.09, 792.18, 888.27,
        996.09, 1092.18,
    ])


def kirnberger_iii_cents() -> np.ndarray:
    sc = SYNTONIC_COMMA_CENTS
    return np.array([
        0.0, 90.225, 193.157, 294.135, 386.314,
        498.045, 590.224, 696.578, 792.180, 889.735,
        996.090, 1088.269,
    ])


def young_temperament_cents() -> np.ndarray:
    return np.array([
        0.0, 93.9, 195.8, 297.8, 391.7,
        499.9, 591.9, 697.9, 795.8, 893.8,
        999.8, 1091.8,
    ])


def vallotti_cents() -> np.ndarray:
    return np.array([
        0.0, 94.135, 196.09, 298.045, 392.18,
        501.955, 592.18, 698.045, 796.09, 894.135,
        1000.0, 1092.18,
    ])


def build_temperament(typ: TemperamentType,
                       ref_freq: float = 440.0,
                       ref_midi: int = 69) -> TemperamentSpec:
    builders = {
        TemperamentType.EQUAL: equal_temperament_cents,
        TemperamentType.JUST_MAJOR: lambda: _just_cents(JUST_RATIOS_MAJOR),
        TemperamentType.JUST_MINOR: lambda: _just_cents(JUST_RATIOS_MINOR),
        TemperamentType.PYTHAGOREAN: pythagorean_cents,
        TemperamentType.MEANTONE_QUARTER: quarter_comma_meantone_cents,
        TemperamentType.WERCKMEISTER_III: werckmeister_iii_cents,
        TemperamentType.KIRNBERGER_III: kirnberger_iii_cents,
        TemperamentType.YOUNG: young_temperament_cents,
        TemperamentType.VALLOTTI: vallotti_cents,
    }
    return TemperamentSpec(
        name=typ.value,
        cents=builders[typ](),
        reference_freq=ref_freq,
        reference_midi=ref_midi,
    )


def deviation_from_equal(spec: TemperamentSpec) -> np.ndarray:
    et = equal_temperament_cents()
    return spec.cents - et


def wolf_interval_cents(spec: TemperamentSpec) -> List[Tuple[int, int, float]]:
    pure_fifth = ratio_to_cents(3, 2)
    wolves: List[Tuple[int, int, float]] = []
    for i in range(12):
        j = (i + 7) % 12
        interval = (spec.cents[j] - spec.cents[i]) % 1200.0
        dev = abs(interval - pure_fifth)
        if dev > 10.0:
            wolves.append((i, j, interval))
    return wolves


def key_color_profile(spec: TemperamentSpec) -> np.ndarray:
    et = equal_temperament_cents()
    profile = np.zeros(12)
    for root in range(12):
        major_thirds = [(root + 4) % 12]
        fifths = [(root + 7) % 12]
        dev = 0.0
        for m in major_thirds:
            actual = (spec.cents[m] - spec.cents[root]) % 1200.0
            pure = ratio_to_cents(5, 4)
            dev += abs(actual - pure)
        for f in fifths:
            actual = (spec.cents[f] - spec.cents[root]) % 1200.0
            pure = ratio_to_cents(3, 2)
            dev += abs(actual - pure)
        profile[root] = dev
    return profile


def compare_temperaments(a: TemperamentSpec, b: TemperamentSpec) -> float:
    return float(np.sqrt(np.mean((a.cents - b.cents) ** 2)))


def interval_quality(spec: TemperamentSpec, root: int, target: int,
                     ref_freq: float = 440.0) -> IntervalQuality:
    actual_cents = (spec.cents[target % 12] - spec.cents[root % 12]) % 1200.0
    approx = cents_to_ratio_approx(actual_cents)
    f1 = spec.freq(60 + root)
    f2 = spec.freq(60 + target)
    return IntervalQuality(
        ratio=approx,
        cents=actual_cents,
        beat_rate_hz=beat_frequency(f1, f2),
        name=f"{root}->{target}",
    )


def temperament_spectrum(spec: TemperamentSpec, root_midi: int = 60,
                         n_partials: int = 8) -> np.ndarray:
    f0 = spec.freq(root_midi)
    return np.array([f0 * (k + 1) for k in range(n_partials)])
