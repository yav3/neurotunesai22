"""Absorption coefficient database and NRC calculations for acoustic treatment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000]


@dataclass
class MaterialCoeffs:
    name: str
    coeffs_125: float
    coeffs_250: float
    coeffs_500: float
    coeffs_1000: float
    coeffs_2000: float
    coeffs_4000: float

    def as_array(self) -> np.ndarray:
        return np.array([
            self.coeffs_125, self.coeffs_250, self.coeffs_500,
            self.coeffs_1000, self.coeffs_2000, self.coeffs_4000,
        ])

    @property
    def nrc(self) -> float:
        return float(np.mean([self.coeffs_250, self.coeffs_500,
                              self.coeffs_1000, self.coeffs_2000]))


MATERIALS: dict[str, MaterialCoeffs] = {
    "concrete_painted": MaterialCoeffs("Painted concrete", 0.01, 0.01, 0.02, 0.02, 0.02, 0.03),
    "concrete_rough": MaterialCoeffs("Rough concrete", 0.02, 0.03, 0.03, 0.03, 0.04, 0.07),
    "brick_unglazed": MaterialCoeffs("Unglazed brick", 0.03, 0.03, 0.03, 0.04, 0.05, 0.07),
    "plaster_smooth": MaterialCoeffs("Smooth plaster", 0.01, 0.02, 0.02, 0.03, 0.04, 0.05),
    "glass_window": MaterialCoeffs("Window glass", 0.35, 0.25, 0.18, 0.12, 0.07, 0.04),
    "glass_heavy": MaterialCoeffs("Heavy plate glass", 0.18, 0.06, 0.04, 0.03, 0.02, 0.02),
    "gypsum_board": MaterialCoeffs("Gypsum board (drywall)", 0.29, 0.10, 0.05, 0.04, 0.07, 0.09),
    "plywood_thin": MaterialCoeffs("Thin plywood panel", 0.28, 0.22, 0.17, 0.09, 0.10, 0.11),
    "wood_floor": MaterialCoeffs("Wood floor on joists", 0.15, 0.11, 0.10, 0.07, 0.06, 0.07),
    "parquet_on_concrete": MaterialCoeffs("Parquet on concrete", 0.04, 0.04, 0.07, 0.06, 0.06, 0.07),
    "carpet_thin": MaterialCoeffs("Thin carpet on concrete", 0.02, 0.06, 0.14, 0.37, 0.60, 0.65),
    "carpet_heavy": MaterialCoeffs("Heavy carpet on pad", 0.08, 0.24, 0.57, 0.69, 0.71, 0.73),
    "curtain_light": MaterialCoeffs("Light curtain", 0.03, 0.04, 0.11, 0.17, 0.24, 0.35),
    "curtain_heavy": MaterialCoeffs("Heavy velour curtain", 0.14, 0.35, 0.55, 0.72, 0.70, 0.65),
    "acoustic_tile": MaterialCoeffs("Acoustic ceiling tile", 0.20, 0.40, 0.70, 0.80, 0.60, 0.40),
    "acoustic_foam_2in": MaterialCoeffs("2-inch acoustic foam", 0.11, 0.30, 0.68, 0.97, 0.98, 0.95),
    "acoustic_foam_4in": MaterialCoeffs("4-inch acoustic foam", 0.30, 0.70, 0.97, 1.00, 1.00, 1.00),
    "fiberglass_2in": MaterialCoeffs("2-inch fiberglass panel", 0.22, 0.55, 0.82, 0.97, 0.99, 1.00),
    "fiberglass_4in": MaterialCoeffs("4-inch fiberglass panel", 0.60, 0.90, 0.97, 1.00, 1.00, 1.00),
    "perforated_panel": MaterialCoeffs("Perforated wood panel", 0.40, 0.70, 0.80, 0.50, 0.40, 0.30),
    "bass_trap_corner": MaterialCoeffs("Corner bass trap", 0.70, 0.80, 0.60, 0.40, 0.30, 0.25),
    "rpg_diffusor": MaterialCoeffs("RPG-type diffusor", 0.15, 0.30, 0.55, 0.65, 0.60, 0.55),
    "open_window": MaterialCoeffs("Open window (100%)", 1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "audience_seated": MaterialCoeffs("Seated audience", 0.39, 0.57, 0.80, 0.94, 0.92, 0.87),
    "upholstered_seat": MaterialCoeffs("Upholstered seat (empty)", 0.19, 0.37, 0.56, 0.67, 0.61, 0.59),
    "water_surface": MaterialCoeffs("Water surface", 0.01, 0.01, 0.01, 0.02, 0.02, 0.03),
}


@dataclass
class Surface:
    material: str
    area_m2: float


def total_absorption(surfaces: list[Surface]) -> np.ndarray:
    a_total = np.zeros(6)
    for s in surfaces:
        mat = MATERIALS.get(s.material)
        if mat is None:
            continue
        a_total += mat.as_array() * s.area_m2
    return a_total


def weighted_nrc(surfaces: list[Surface]) -> float:
    total_area = sum(s.area_m2 for s in surfaces)
    if total_area == 0:
        return 0.0
    weighted = 0.0
    for s in surfaces:
        mat = MATERIALS.get(s.material)
        if mat:
            weighted += mat.nrc * s.area_m2
    return weighted / total_area


def sabine_rt60(volume_m3: float, surfaces: list[Surface]) -> np.ndarray:
    a = total_absorption(surfaces)
    a = np.maximum(a, 1e-6)
    return 0.161 * volume_m3 / a


def eyring_rt60(volume_m3: float, surfaces: list[Surface]) -> np.ndarray:
    total_area = sum(s.area_m2 for s in surfaces)
    if total_area == 0:
        return np.zeros(6)
    a = total_absorption(surfaces) / total_area
    a = np.minimum(a, 0.999)
    return -0.161 * volume_m3 / (total_area * np.log(1 - a))


def optimal_rt60(volume_m3: float, use: str = "music") -> float:
    targets = {
        "speech": 0.4 + 0.05 * volume_m3 ** (1 / 3),
        "music": 0.8 + 0.12 * volume_m3 ** (1 / 3),
        "rehearsal": 0.5 + 0.08 * volume_m3 ** (1 / 3),
        "recording": 0.3 + 0.03 * volume_m3 ** (1 / 3),
        "worship": 1.2 + 0.15 * volume_m3 ** (1 / 3),
    }
    return targets.get(use, 0.8)


def treatment_area_needed(
    current_absorption: float, target_absorption: float,
    treatment_material: str, band_index: int = 2,
) -> float:
    mat = MATERIALS.get(treatment_material)
    if mat is None:
        return 0.0
    coeffs = mat.as_array()
    if band_index < 0 or band_index >= len(coeffs) or coeffs[band_index] < 0.01:
        return 0.0
    needed = target_absorption - current_absorption
    if needed <= 0:
        return 0.0
    return needed / coeffs[band_index]
