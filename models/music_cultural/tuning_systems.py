"""Tuning systems: equal temperament, just intonation, Pythagorean, meantone, well temperament."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import math

import numpy as np


class CentConverter:
    @staticmethod
    def freq_to_cents(f: float, ref: float = 440.0) -> float:
        if f <= 0 or ref <= 0:
            return 0.0
        return 1200.0 * math.log2(f / ref)

    @staticmethod
    def cents_to_freq(cents: float, ref: float = 440.0) -> float:
        return ref * (2.0 ** (cents / 1200.0))

    @staticmethod
    def ratio_to_cents(ratio: float) -> float:
        if ratio <= 0:
            return 0.0
        return 1200.0 * math.log2(ratio)

    @staticmethod
    def cents_to_ratio(cents: float) -> float:
        return 2.0 ** (cents / 1200.0)


@dataclass
class TuningSystem:
    name: str
    pitch_cents: list[float] = field(default_factory=list)

    def n_pitches(self) -> int:
        return len(self.pitch_cents)

    def frequencies(self, ref_hz: float = 440.0, ref_index: int = 9) -> list[float]:
        return [ref_hz * (2.0 ** ((c - self.pitch_cents[ref_index]) / 1200.0))
                for c in self.pitch_cents]

    def interval(self, i: int, j: int) -> float:
        return self.pitch_cents[j % len(self.pitch_cents)] - self.pitch_cents[i % len(self.pitch_cents)]


class EqualTemperament(TuningSystem):
    def __init__(self, divisions: int = 12):
        step = 1200.0 / divisions
        cents = [i * step for i in range(divisions)]
        super().__init__(name=f"{divisions}-TET", pitch_cents=cents)
        self.divisions = divisions

    def step_size(self) -> float:
        return 1200.0 / self.divisions


# Pre-built standard temperaments
TET_12 = EqualTemperament(12)
TET_19 = EqualTemperament(19)
TET_24 = EqualTemperament(24)
TET_31 = EqualTemperament(31)
TET_53 = EqualTemperament(53)


class JustIntonation(TuningSystem):
    # 5-limit ratios for 12 pitch classes (C to B)
    FIVE_LIMIT: list[tuple[int, int]] = [
        (1, 1), (16, 15), (9, 8), (6, 5), (5, 4), (4, 3),
        (45, 32), (3, 2), (8, 5), (5, 3), (9, 5), (15, 8),
    ]
    SEVEN_LIMIT: list[tuple[int, int]] = [
        (1, 1), (15, 14), (9, 8), (6, 5), (5, 4), (4, 3),
        (7, 5), (3, 2), (8, 5), (5, 3), (7, 4), (15, 8),
    ]
    ELEVEN_LIMIT: list[tuple[int, int]] = [
        (1, 1), (16, 15), (9, 8), (6, 5), (5, 4), (4, 3),
        (11, 8), (3, 2), (8, 5), (5, 3), (11, 6), (15, 8),
    ]

    def __init__(self, limit: int = 5):
        if limit == 7:
            ratios = self.SEVEN_LIMIT
        elif limit == 11:
            ratios = self.ELEVEN_LIMIT
        else:
            ratios = self.FIVE_LIMIT
        cents = [CentConverter.ratio_to_cents(n / d) for n, d in ratios]
        super().__init__(name=f"Just Intonation ({limit}-limit)", pitch_cents=cents)
        self.ratios = ratios
        self.limit = limit


class PythagoreanTuning(TuningSystem):
    def __init__(self):
        # Chain of pure fifths (3/2): F-C-G-D-A-E-B
        # Starting from C, go up in fifths and reduce to one octave
        fifth_ratio = 3.0 / 2.0
        fifth_cents = CentConverter.ratio_to_cents(fifth_ratio)
        # Generate 12 notes via fifth chain: start at F (index -1 in fifths from C)
        chain_positions = [-1, 0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2]
        cents = []
        for pos in chain_positions:
            c = (pos * fifth_cents) % 1200.0
            cents.append(c)
        cents.sort()
        super().__init__(name="Pythagorean", pitch_cents=cents)

    def wolf_fifth(self) -> float:
        # Diminished sixth: 12 pure fifths overshoot 7 octaves by Pythagorean comma
        pythagorean_comma = CentConverter.ratio_to_cents((3.0 ** 12) / (2.0 ** 19))
        return CentConverter.ratio_to_cents(3.0 / 2.0) - pythagorean_comma


class MeantoneTemperament(TuningSystem):
    def __init__(self, comma_fraction: float = 0.25):
        # Each fifth is narrowed by comma_fraction of syntonic comma
        syntonic_comma = CentConverter.ratio_to_cents(81.0 / 80.0)  # ~21.5 cents
        narrowed_fifth = CentConverter.ratio_to_cents(3.0 / 2.0) - comma_fraction * syntonic_comma
        chain_positions = [-1, 0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2]
        cents = []
        for pos in chain_positions:
            c = (pos * narrowed_fifth) % 1200.0
            cents.append(c)
        cents.sort()
        frac_name = {0.25: "Quarter-comma", 1/3: "Third-comma", 1/6: "Sixth-comma"}
        name = frac_name.get(comma_fraction, f"{comma_fraction}-comma")
        super().__init__(name=f"Meantone ({name})", pitch_cents=cents)
        self.comma_fraction = comma_fraction


QUARTER_COMMA_MEANTONE = MeantoneTemperament(0.25)
THIRD_COMMA_MEANTONE = MeantoneTemperament(1.0 / 3.0)
SIXTH_COMMA_MEANTONE = MeantoneTemperament(1.0 / 6.0)


class WellTemperament(TuningSystem):
    # Cent offsets from 12-TET for each pitch class C through B
    WERCKMEISTER_III: list[float] = [
        0.0, -10.3, -7.8, -5.9, -3.9, -2.0, -12.3, -3.9, -8.2, -5.9, -3.9, -2.0,
    ]
    KIRNBERGER_III: list[float] = [
        0.0, -10.3, -6.8, -5.9, -3.4, -2.0, -12.3, -3.9, -8.2, -5.4, -3.9, -1.5,
    ]
    VALLOTTI: list[float] = [
        0.0, -5.9, -3.9, -2.0, -5.9, -2.0, -7.8, -2.0, -3.9, -5.9, 0.0, -3.9,
    ]

    def __init__(self, variant: str = "werckmeister_iii"):
        if variant.lower() in ("kirnberger", "kirnberger_iii"):
            offsets = self.KIRNBERGER_III
            name = "Kirnberger III"
        elif variant.lower() == "vallotti":
            offsets = self.VALLOTTI
            name = "Vallotti"
        else:
            offsets = self.WERCKMEISTER_III
            name = "Werckmeister III"
        tet = EqualTemperament(12)
        cents = [tet.pitch_cents[i] + offsets[i] for i in range(12)]
        super().__init__(name=name, pitch_cents=cents)
        self.variant = variant


class TuningComparator:
    @staticmethod
    def compare(a: TuningSystem, b: TuningSystem) -> list[float]:
        n = min(a.n_pitches(), b.n_pitches())
        return [a.pitch_cents[i] - b.pitch_cents[i] for i in range(n)]

    @staticmethod
    def max_deviation(a: TuningSystem, b: TuningSystem) -> float:
        diffs = TuningComparator.compare(a, b)
        if not diffs:
            return 0.0
        return float(max(abs(d) for d in diffs))

    @staticmethod
    def identify_commas(a: TuningSystem, b: TuningSystem,
                        threshold: float = 1.0) -> list[tuple[int, float, str]]:
        diffs = TuningComparator.compare(a, b)
        results = []
        for i, d in enumerate(diffs):
            ad = abs(d)
            if ad < threshold:
                continue
            if 19.0 < ad < 24.0:
                label = "syntonic comma"
            elif 1.5 < ad < 2.5:
                label = "schisma"
            elif 40.0 < ad < 45.0:
                label = "diesis"
            elif 22.0 < ad < 25.0:
                label = "Pythagorean comma"
            else:
                label = "deviation"
            results.append((i, d, label))
        return results
