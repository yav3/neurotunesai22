"""Physical models of instrument sound production: strings, pipes, membranes, bars."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StringModel:
    length_m: float
    tension_N: float
    density_kg_m: float
    damping: float = 0.001
    inharmonicity_B: float = 0.0

    @property
    def fundamental(self) -> float:
        if self.length_m <= 0 or self.density_kg_m <= 0:
            return 0.0
        return 0.5 / self.length_m * np.sqrt(self.tension_N / self.density_kg_m)


@dataclass
class PipeModel:
    length_m: float
    radius_m: float
    open_end: bool = True
    temperature_c: float = 20.0

    @property
    def speed_of_sound(self) -> float:
        return 331.3 * np.sqrt(1.0 + self.temperature_c / 273.15)

    @property
    def end_correction(self) -> float:
        return 0.6133 * self.radius_m

    @property
    def effective_length(self) -> float:
        corrections = 2 if self.open_end else 1
        return self.length_m + corrections * self.end_correction


@dataclass
class MembraneModel:
    radius_m: float
    tension_N_m: float
    density_kg_m2: float


@dataclass
class BarModel:
    length_m: float
    thickness_m: float
    youngs_modulus: float
    density: float


BESSEL_ZEROS_01: list[float] = [
    2.4048, 5.5201, 8.6537, 11.7915, 14.9309,
    18.0711, 21.2116, 24.3525, 27.4935, 30.6346,
]

BESSEL_ZEROS_11: list[float] = [
    3.8317, 7.0156, 10.1735, 13.3237, 16.4706,
    19.6159, 22.7601, 25.9037, 29.0468, 32.1897,
]

FREE_BAR_NODES: list[float] = [
    4.730, 7.853, 10.996, 14.137, 17.279,
]


def string_overtone_series(model: StringModel, n_harmonics: int = 16) -> list[float]:
    f0 = model.fundamental
    if f0 <= 0:
        return []
    freqs: list[float] = []
    for n in range(1, n_harmonics + 1):
        correction = np.sqrt(1.0 + model.inharmonicity_B * n * n)
        freqs.append(float(f0 * n * correction))
    return freqs


def string_decay_time(model: StringModel, harmonic: int = 1) -> float:
    if model.damping <= 0:
        return float("inf")
    f = model.fundamental * harmonic
    if f <= 0:
        return float("inf")
    return 1.0 / (model.damping * (2 * np.pi * f))


def pipe_resonance_frequencies(model: PipeModel, n_modes: int = 8) -> list[float]:
    c = model.speed_of_sound
    L_eff = model.effective_length
    if L_eff <= 0:
        return []
    freqs: list[float] = []
    if model.open_end:
        for n in range(1, n_modes + 1):
            freqs.append(float(n * c / (2.0 * L_eff)))
    else:
        for n in range(n_modes):
            freqs.append(float((2 * n + 1) * c / (4.0 * L_eff)))
    return freqs


def membrane_modes(model: MembraneModel, n_modes: int = 10) -> list[float]:
    if model.radius_m <= 0 or model.density_kg_m2 <= 0 or model.tension_N_m <= 0:
        return []
    c = np.sqrt(model.tension_N_m / model.density_kg_m2)
    freqs: list[float] = []
    all_zeros = sorted(BESSEL_ZEROS_01[:n_modes] + BESSEL_ZEROS_11[:n_modes])
    for z in all_zeros[:n_modes]:
        freqs.append(float(z * c / (2.0 * np.pi * model.radius_m)))
    return freqs


def bar_frequencies(model: BarModel, n_modes: int = 5) -> list[float]:
    if model.length_m <= 0 or model.density <= 0 or model.thickness_m <= 0:
        return []
    kappa = model.thickness_m / np.sqrt(12.0)
    c_bar = np.sqrt(model.youngs_modulus / model.density)
    freqs: list[float] = []
    for i in range(min(n_modes, len(FREE_BAR_NODES))):
        beta_n = FREE_BAR_NODES[i] / model.length_m
        f = (beta_n ** 2 * kappa * c_bar) / (2.0 * np.pi)
        freqs.append(float(f))
    return freqs


def radiation_impedance(frequency: float, source_radius: float) -> complex:
    if frequency <= 0 or source_radius <= 0:
        return 0j
    k = 2.0 * np.pi * frequency / 343.0
    ka = k * source_radius
    rho_c = 413.0
    R = rho_c * ka ** 2 / (1.0 + ka ** 2)
    X = rho_c * ka / (1.0 + ka ** 2)
    return complex(R, X)


def radiation_efficiency(frequency: float, source_radius: float) -> float:
    Z = radiation_impedance(frequency, source_radius)
    rho_c = 413.0
    if rho_c <= 0:
        return 0.0
    return float(Z.real / rho_c)


def coupling_coefficient(
    driver_impedance: complex,
    resonator_impedance: complex,
) -> float:
    z_total = abs(driver_impedance) + abs(resonator_impedance)
    if z_total < 1e-12:
        return 0.0
    z_match = 2.0 * abs(driver_impedance) * abs(resonator_impedance) / (z_total ** 2)
    return float(4.0 * z_match)


def inharmonicity_factor(
    string_diameter_m: float,
    string_length_m: float,
    youngs_modulus: float = 2e11,
) -> float:
    if string_length_m <= 0 or string_diameter_m <= 0:
        return 0.0
    r = string_diameter_m / 2.0
    I = np.pi * r ** 4 / 4.0
    B = (np.pi ** 2 * youngs_modulus * I) / (string_length_m ** 2)
    return float(B)
