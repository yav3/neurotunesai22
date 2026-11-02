"""Interval logic: classification, inversion, compound intervals, consonance.

Formal interval arithmetic following Forte (1973) and Rahn (1980).
Consonance/dissonance ratings per Helmholtz (1863) roughness model and
Plomp-Levelt (1965) critical-band theory. Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence
from enum import Enum


class IntervalQuality(Enum):
    PERFECT = "P"
    MAJOR = "M"
    MINOR = "m"
    AUGMENTED = "A"
    DIMINISHED = "d"
    DOUBLY_AUGMENTED = "AA"
    DOUBLY_DIMINISHED = "dd"


class ConsonanceClass(Enum):
    PERFECT_CONSONANCE = "perfect_consonance"
    IMPERFECT_CONSONANCE = "imperfect_consonance"
    MILD_DISSONANCE = "mild_dissonance"
    SHARP_DISSONANCE = "sharp_dissonance"


@dataclass
class Interval:
    semitones: int
    generic: int          # 1-based (unison=1, second=2, ...)
    quality: IntervalQuality
    direction: int        # 1=ascending, -1=descending, 0=unison
    compound: bool

    @property
    def simple_semitones(self) -> int:
        return self.semitones % 12

    @property
    def octaves(self) -> int:
        return self.semitones // 12


PERFECT_INTERVALS = {0: 1, 5: 4, 7: 5}  # semitones -> generic
MAJOR_INTERVALS = {2: 2, 4: 3, 9: 6, 11: 7}
MINOR_INTERVALS = {1: 2, 3: 3, 8: 6, 10: 7}

INTERVAL_NAMES = {
    (1, "P"): "P1", (2, "m"): "m2", (2, "M"): "M2", (3, "m"): "m3",
    (3, "M"): "M3", (4, "P"): "P4", (4, "A"): "A4",
    (5, "d"): "d5", (5, "P"): "P5", (6, "m"): "m6", (6, "M"): "M6",
    (7, "m"): "m7", (7, "M"): "M7", (8, "P"): "P8",
}

CONSONANCE_RATING = {
    0: 1.0,   # P1
    1: 0.05,  # m2
    2: 0.3,   # M2
    3: 0.7,   # m3
    4: 0.75,  # M3
    5: 0.85,  # P4
    6: 0.15,  # tritone
    7: 0.95,  # P5
    8: 0.7,   # m6
    9: 0.75,  # M6
    10: 0.35, # m7
    11: 0.1,  # M7
}


def classify_interval(semitones: int) -> Interval:
    direction = 1 if semitones > 0 else (-1 if semitones < 0 else 0)
    abs_semi = abs(semitones)
    compound = abs_semi > 12
    simple = abs_semi % 12
    octaves = abs_semi // 12

    if simple in PERFECT_INTERVALS:
        generic = PERFECT_INTERVALS[simple]
        quality = IntervalQuality.PERFECT
    elif simple in MAJOR_INTERVALS:
        generic = MAJOR_INTERVALS[simple]
        quality = IntervalQuality.MAJOR
    elif simple in MINOR_INTERVALS:
        generic = MINOR_INTERVALS[simple]
        quality = IntervalQuality.MINOR
    elif simple == 6:
        generic = 4
        quality = IntervalQuality.AUGMENTED
    else:
        generic = 1
        quality = IntervalQuality.PERFECT

    if compound:
        generic += 7 * octaves

    return Interval(abs_semi, generic, quality, direction, compound)


def interval_name(interval: Interval) -> str:
    simple = interval.simple_semitones
    g = ((interval.generic - 1) % 7) + 1
    q = interval.quality.value
    key = (g, q)
    base = INTERVAL_NAMES.get(key, f"{q}{g}")
    if interval.compound:
        return f"{base} (compound, +{interval.octaves} oct)"
    return base


def invert_interval(interval: Interval) -> Interval:
    if interval.compound:
        simple_semi = interval.simple_semitones
    else:
        simple_semi = interval.semitones
    inverted_semi = 12 - simple_semi
    return classify_interval(inverted_semi * (-interval.direction if interval.direction else 1))


def consonance_rating(semitones: int) -> float:
    return CONSONANCE_RATING.get(abs(semitones) % 12, 0.5)


def classify_consonance(semitones: int) -> ConsonanceClass:
    r = consonance_rating(semitones)
    if r >= 0.85:
        return ConsonanceClass.PERFECT_CONSONANCE
    elif r >= 0.6:
        return ConsonanceClass.IMPERFECT_CONSONANCE
    elif r >= 0.25:
        return ConsonanceClass.MILD_DISSONANCE
    return ConsonanceClass.SHARP_DISSONANCE


def plomp_levelt_dissonance(f1: float, f2: float) -> float:
    if f1 > f2:
        f1, f2 = f2, f1
    s = 0.24 * min(f1, f2) / (0.021 * min(f1, f2) + 19)
    x = (f2 - f1) / max(s, 1e-10)
    return float(max(np.exp(-3.5 * x) - np.exp(-5.75 * x), 0.0))


def chord_dissonance(frequencies: Sequence[float]) -> float:
    freqs = list(frequencies)
    n = len(freqs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += plomp_levelt_dissonance(freqs[i], freqs[j])
            count += 1
    return total / max(count, 1)


def frequency_ratio(semitones: int) -> float:
    return 2.0 ** (semitones / 12.0)


def cents_from_ratio(ratio: float) -> float:
    if ratio <= 0:
        return 0.0
    return float(1200.0 * np.log2(ratio))


def ratio_from_cents(cents: float) -> float:
    return float(2.0 ** (cents / 1200.0))


def interval_vector_from_pitches(pitches: Sequence[int]) -> np.ndarray:
    pcs = sorted(set(p % 12 for p in pitches))
    vec = np.zeros(6, dtype=int)
    n = len(pcs)
    for i in range(n):
        for j in range(i + 1, n):
            ic = min((pcs[j] - pcs[i]) % 12, (pcs[i] - pcs[j]) % 12)
            if 1 <= ic <= 6:
                vec[ic - 1] += 1
    return vec


def all_intervals(pitches: Sequence[int]) -> List[Interval]:
    result = []
    for i in range(len(pitches) - 1):
        semi = pitches[i + 1] - pitches[i]
        result.append(classify_interval(semi))
    return result


def interval_histogram(pitches: Sequence[int]) -> np.ndarray:
    hist = np.zeros(25)  # -12 to +12
    for i in range(len(pitches) - 1):
        semi = pitches[i + 1] - pitches[i]
        idx = semi + 12
        if 0 <= idx < 25:
            hist[idx] += 1
    return hist


def melodic_interval_profile(pitches: Sequence[int]) -> np.ndarray:
    if len(pitches) < 2:
        return np.zeros(12)
    intervals = [abs(pitches[i + 1] - pitches[i]) % 12 for i in range(len(pitches) - 1)]
    profile = np.zeros(12)
    for iv in intervals:
        profile[iv] += 1
    total = profile.sum()
    if total > 0:
        profile /= total
    return profile


def consonance_curve(pitches: Sequence[int]) -> np.ndarray:
    if len(pitches) < 2:
        return np.array([])
    return np.array([consonance_rating(pitches[i + 1] - pitches[i])
                      for i in range(len(pitches) - 1)])


def average_consonance(pitches: Sequence[int]) -> float:
    curve = consonance_curve(pitches)
    if len(curve) == 0:
        return 0.0
    return float(np.mean(curve))


def interval_entropy(pitches: Sequence[int]) -> float:
    hist = melodic_interval_profile(pitches)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def leap_step_ratio(pitches: Sequence[int], leap_threshold: int = 3) -> float:
    if len(pitches) < 2:
        return 0.0
    steps = 0
    leaps = 0
    for i in range(len(pitches) - 1):
        if abs(pitches[i + 1] - pitches[i]) >= leap_threshold:
            leaps += 1
        else:
            steps += 1
    return leaps / max(steps + leaps, 1)


def directional_change_rate(pitches: Sequence[int]) -> float:
    if len(pitches) < 3:
        return 0.0
    changes = 0
    for i in range(1, len(pitches) - 1):
        d1 = pitches[i] - pitches[i - 1]
        d2 = pitches[i + 1] - pitches[i]
        if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
            changes += 1
    return changes / (len(pitches) - 2)
