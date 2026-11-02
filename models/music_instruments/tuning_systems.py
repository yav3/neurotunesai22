"""Musical tuning systems: equal temperament, Pythagorean, just intonation, meantone."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class TuningSystem(Enum):
    EQUAL_TEMPERAMENT = "equal_temperament"
    PYTHAGOREAN = "pythagorean"
    JUST_INTONATION = "just_intonation"
    MEANTONE = "meantone"
    WELL_TEMPERAMENT = "well_temperament"


def equal_temperament(semitones: int, reference: float = 440.0) -> float:
    return reference * (2.0 ** (semitones / 12.0))


def pythagorean(degree: int, reference: float = 440.0) -> float:
    fifths_map = {0: 0, 1: -5, 2: 2, 3: -3, 4: 4, 5: -1, 6: 6, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}
    n_fifths = fifths_map.get(degree % 12, 0)
    ratio = (3.0 / 2.0) ** n_fifths
    while ratio < 1.0:
        ratio *= 2.0
    while ratio >= 2.0:
        ratio /= 2.0
    octave = degree // 12
    return reference * ratio * (2.0 ** octave)


JUST_RATIOS = [
    1.0, 16/15, 9/8, 6/5, 5/4, 4/3,
    45/32, 3/2, 8/5, 5/3, 9/5, 15/8,
]


def just_intonation(degree: int, reference: float = 440.0) -> float:
    octave = degree // 12
    semitone = degree % 12
    return reference * JUST_RATIOS[semitone] * (2.0 ** octave)


def meantone(degree: int, reference: float = 440.0, comma_fraction: float = 0.25) -> float:
    fifth = 3.0 / 2.0 * (81/80) ** (-comma_fraction)
    fifths_map = {0: 0, 1: -5, 2: 2, 3: -3, 4: 4, 5: -1, 6: 6, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}
    n_fifths = fifths_map.get(degree % 12, 0)
    ratio = fifth ** n_fifths
    while ratio < 1.0:
        ratio *= 2.0
    while ratio >= 2.0:
        ratio /= 2.0
    octave = degree // 12
    return reference * ratio * (2.0 ** octave)


WELL_TEMPERAMENT_CENTS = [
    0, 90, 192, 294, 390, 498,
    588, 696, 792, 888, 996, 1092,
]


def well_temperament(degree: int, reference: float = 440.0) -> float:
    octave = degree // 12
    semitone = degree % 12
    cents = WELL_TEMPERAMENT_CENTS[semitone]
    return reference * (2.0 ** (cents / 1200.0)) * (2.0 ** octave)


def cents_deviation(freq: float, reference: float) -> float:
    if freq <= 0 or reference <= 0:
        return 0.0
    return 1200.0 * math.log2(freq / reference)


def frequency_ratio(f1: float, f2: float) -> float:
    if f2 <= 0:
        return 0.0
    return f1 / f2


def compare_tunings(
    n_notes: int = 12,
    reference: float = 440.0,
) -> dict[str, list[float]]:
    return {
        "equal": [equal_temperament(i, reference) for i in range(n_notes)],
        "pythagorean": [pythagorean(i, reference) for i in range(n_notes)],
        "just": [just_intonation(i, reference) for i in range(n_notes)],
        "meantone": [meantone(i, reference) for i in range(n_notes)],
        "well": [well_temperament(i, reference) for i in range(n_notes)],
    }
