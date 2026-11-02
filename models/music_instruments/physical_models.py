"""Physical models of musical instruments: strings, tubes, membranes, bars."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StringModel:
    length_m: float
    tension_n: float
    linear_density_kg_m: float

    @property
    def fundamental(self) -> float:
        return (1 / (2 * self.length_m)) * math.sqrt(self.tension_n / self.linear_density_kg_m)


def string_frequency(length: float, tension: float, density: float, harmonic: int = 1) -> float:
    f0 = (1 / (2 * length)) * math.sqrt(tension / density)
    return f0 * harmonic


def string_harmonics(model: StringModel, n: int = 8) -> list[float]:
    f0 = model.fundamental
    return [f0 * k for k in range(1, n + 1)]


@dataclass
class TubeModel:
    length_m: float
    is_open: bool = True
    speed_of_sound: float = 343.0
    end_correction: float = 0.6
    radius_m: float = 0.01

    @property
    def effective_length(self) -> float:
        corr = self.end_correction * self.radius_m
        return self.length_m + (2 * corr if self.is_open else corr)


def tube_resonance(model: TubeModel, mode: int = 1) -> float:
    L = model.effective_length
    if model.is_open:
        return mode * model.speed_of_sound / (2 * L)
    return (2 * mode - 1) * model.speed_of_sound / (4 * L)


def open_pipe_harmonics(length: float, n: int = 8, c: float = 343.0) -> list[float]:
    f0 = c / (2 * length)
    return [f0 * k for k in range(1, n + 1)]


def closed_pipe_harmonics(length: float, n: int = 8, c: float = 343.0) -> list[float]:
    f0 = c / (4 * length)
    return [f0 * (2 * k - 1) for k in range(1, n + 1)]


@dataclass
class MembraneModel:
    radius_m: float
    tension_per_length: float
    surface_density: float

    @property
    def fundamental(self) -> float:
        return 0.3827 / self.radius_m * math.sqrt(self.tension_per_length / self.surface_density)


def membrane_modes(model: MembraneModel, n_modes: int = 6) -> list[float]:
    bessel_zeros = [2.405, 3.832, 5.136, 5.520, 6.380, 7.016, 7.588, 8.417]
    f_scale = math.sqrt(model.tension_per_length / model.surface_density) / (2 * math.pi * model.radius_m)
    return [f_scale * bessel_zeros[i] for i in range(min(n_modes, len(bessel_zeros)))]


@dataclass
class BarModel:
    length_m: float
    thickness_m: float
    density: float
    youngs_modulus: float

    @property
    def radius_of_gyration(self) -> float:
        return self.thickness_m / math.sqrt(12)


def bar_frequency(model: BarModel, mode: int = 1) -> float:
    kappa = model.radius_of_gyration
    free_free_lambdas = [4.730, 7.853, 10.996, 14.137]
    if mode < 1 or mode > len(free_free_lambdas):
        mode = 1
    lam = free_free_lambdas[mode - 1]
    c_bar = math.sqrt(model.youngs_modulus / model.density)
    return (lam ** 2 * kappa * c_bar) / (2 * math.pi * model.length_m ** 2)
