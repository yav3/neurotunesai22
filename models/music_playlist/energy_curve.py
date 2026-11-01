from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class EnergyCurveShape(Enum):
    LINEAR_UP = "linear_up"
    LINEAR_DOWN = "linear_down"
    PEAK = "peak"
    VALLEY = "valley"
    PLATEAU = "plateau"
    WAVE = "wave"
    EXPONENTIAL_BUILD = "exponential_build"
    SLOW_BURN = "slow_burn"
    CUSTOM = "custom"


@dataclass
class EnergyCurve:
    shape: EnergyCurveShape
    values: list[float]
    n_slots: int
    min_energy: float = 0.0
    max_energy: float = 1.0

    def at(self, index: int) -> float:
        if index < 0 or index >= self.n_slots:
            raise IndexError(f"Index {index} out of range [0, {self.n_slots})")
        return self.values[index]

    def tolerance_at(self, index: int, tolerance: float = 0.15) -> tuple[float, float]:
        v = self.at(index)
        return (max(self.min_energy, v - tolerance), min(self.max_energy, v + tolerance))


class EnergyCurveDesigner:
    def design(
        self,
        shape: EnergyCurveShape,
        n_slots: int,
        min_energy: float = 0.1,
        max_energy: float = 1.0,
        custom_values: list[float] | None = None,
    ) -> EnergyCurve:
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")

        if shape == EnergyCurveShape.CUSTOM:
            if custom_values is None or len(custom_values) != n_slots:
                raise ValueError("custom_values required with length == n_slots")
            values = [float(np.clip(v, min_energy, max_energy)) for v in custom_values]
            return EnergyCurve(shape=shape, values=values, n_slots=n_slots,
                               min_energy=min_energy, max_energy=max_energy)

        t = np.linspace(0, 1, n_slots)
        rng = max_energy - min_energy

        if shape == EnergyCurveShape.LINEAR_UP:
            raw = t
        elif shape == EnergyCurveShape.LINEAR_DOWN:
            raw = 1.0 - t
        elif shape == EnergyCurveShape.PEAK:
            raw = np.where(t <= 0.5, 2 * t, 2 * (1 - t))
        elif shape == EnergyCurveShape.VALLEY:
            raw = np.where(t <= 0.5, 1 - 2 * t, 2 * t - 1)
        elif shape == EnergyCurveShape.PLATEAU:
            raw = np.where(t < 0.2, t / 0.2,
                           np.where(t > 0.8, (1 - t) / 0.2, 1.0))
        elif shape == EnergyCurveShape.WAVE:
            raw = 0.5 + 0.5 * np.sin(2 * np.pi * t - np.pi / 2)
        elif shape == EnergyCurveShape.EXPONENTIAL_BUILD:
            raw = (np.exp(3 * t) - 1) / (np.e ** 3 - 1)
        elif shape == EnergyCurveShape.SLOW_BURN:
            raw = np.where(t < 0.6, t / 3, (t - 0.6) / 0.4 * 0.8 + 0.2)
        else:
            raw = np.full(n_slots, 0.5)

        values = (min_energy + rng * raw).tolist()
        return EnergyCurve(shape=shape, values=values, n_slots=n_slots,
                           min_energy=min_energy, max_energy=max_energy)

    def fit_curve_to_tracks(
        self,
        energies: list[float],
    ) -> tuple[EnergyCurveShape, float]:
        if len(energies) < 2:
            return EnergyCurveShape.PLATEAU, 1.0

        n = len(energies)
        e = np.array(energies)
        e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)

        best_shape = EnergyCurveShape.PLATEAU
        best_corr = -1.0

        for shape in EnergyCurveShape:
            if shape == EnergyCurveShape.CUSTOM:
                continue
            target = self.design(shape, n, 0.0, 1.0)
            t_arr = np.array(target.values)
            corr = float(np.corrcoef(e_norm, t_arr)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
            if corr > best_corr:
                best_corr = corr
                best_shape = shape

        return best_shape, max(0.0, best_corr)
