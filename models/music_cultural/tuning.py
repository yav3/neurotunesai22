"""Tuning systems: equal temperament, just intonation, Pythagorean, meantone, well temperament."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


def cents_to_ratio(cents: float) -> float:
    return 2 ** (cents / 1200.0)


def ratio_to_cents(ratio: float) -> float:
    if ratio <= 0:
        return 0.0
    return 1200.0 * np.log2(ratio)


def frequency_ratio(f1: float, f2: float) -> float:
    if f1 <= 0:
        return 0.0
    return f2 / f1


class TuningSystem(ABC):
    @abstractmethod
    def frequency(self, degree: int, root: float = 440.0) -> float: ...

    def frequencies(self, degrees: list[int], root: float = 440.0) -> list[float]:
        return [self.frequency(d, root) for d in degrees]

    def interval_cents(self, degree_a: int, degree_b: int) -> float:
        fa = self.frequency(degree_a)
        fb = self.frequency(degree_b)
        return ratio_to_cents(fb / fa) if fa > 0 else 0.0

    def get_ratios(self, n_degrees: int | None = None) -> list[float]:
        n = n_degrees if n_degrees is not None else (getattr(self, 'divisions', 12) + 1)
        return [self.frequency(d, 1.0) for d in range(n)]


class EqualTemperament(TuningSystem):
    def __init__(self, divisions: int = 12):
        self.divisions = divisions

    def frequency(self, degree: int, root: float = 440.0) -> float:
        return root * (2 ** (degree / self.divisions))

    def cent_per_step(self) -> float:
        return 1200.0 / self.divisions


class JustIntonation(TuningSystem):
    RATIOS = {
        0: 1/1, 1: 16/15, 2: 9/8, 3: 6/5, 4: 5/4,
        5: 4/3, 6: 45/32, 7: 3/2, 8: 8/5, 9: 5/3,
        10: 9/5, 11: 15/8,
    }

    def frequency(self, degree: int, root: float = 440.0) -> float:
        octave = degree // 12
        pc = degree % 12
        ratio = self.RATIOS.get(pc, 2 ** (pc / 12))
        return root * ratio * (2 ** octave)

    def beat_frequency(self, degree_a: int, degree_b: int, root: float = 440.0) -> float:
        fa = self.frequency(degree_a, root)
        fb = self.frequency(degree_b, root)
        return abs(fb - fa)


class PythagoreanTuning(TuningSystem):
    def __init__(self):
        self._ratios = self._build_ratios()

    def _build_ratios(self) -> dict[int, float]:
        ratios = {0: 1.0}
        r = 1.0
        for i in range(1, 12):
            r *= 3/2
            while r >= 2.0:
                r /= 2.0
            ratios[i] = r
        return dict(sorted(ratios.items(), key=lambda x: x[1]))

    def frequency(self, degree: int, root: float = 440.0) -> float:
        octave = degree // 12
        pc = degree % 12
        ordered = sorted(self._ratios.values())
        ratio = ordered[pc] if pc < len(ordered) else 2 ** (pc / 12)
        return root * ratio * (2 ** octave)

    @property
    def pythagorean_comma(self) -> float:
        return ratio_to_cents((3/2)**12 / (2**7))


class MeantoneTemperament(TuningSystem):
    def __init__(self, comma_fraction: float = 0.25):
        self._fifth = 3/2 * (81/80) ** (-comma_fraction)
        self._ratios = self._build()

    def _build(self) -> dict[int, float]:
        ratios = {0: 1.0}
        r = 1.0
        for i in range(1, 12):
            r *= self._fifth
            while r >= 2.0:
                r /= 2.0
            ratios[i] = r
        return dict(sorted(ratios.items(), key=lambda x: x[1]))

    def frequency(self, degree: int, root: float = 440.0) -> float:
        octave = degree // 12
        pc = degree % 12
        ordered = sorted(self._ratios.values())
        ratio = ordered[pc] if pc < len(ordered) else 2 ** (pc / 12)
        return root * ratio * (2 ** octave)


class WellTemperament(TuningSystem):
    WERCKMEISTER_III = [0, 90.225, 192.18, 294.135, 390.225, 498.045,
                         588.27, 696.09, 792.18, 888.27, 996.09, 1092.18]

    def __init__(self, cents_table: list[float] | None = None):
        self._cents = cents_table or self.WERCKMEISTER_III

    def frequency(self, degree: int, root: float = 440.0) -> float:
        octave = degree // 12
        pc = degree % 12
        if pc < len(self._cents):
            ratio = cents_to_ratio(self._cents[pc])
        else:
            ratio = 2 ** (pc / 12)
        return root * ratio * (2 ** octave)

    def key_color(self, root_pc: int = 0) -> float:
        if root_pc >= len(self._cents):
            return 0.0
        et_cents = root_pc * 100.0
        return abs(self._cents[root_pc] - et_cents)
