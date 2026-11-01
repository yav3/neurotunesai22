"""Room acoustics: RT60, clarity, definition, Schroeder integration."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RoomParameters:
    volume_m3: float
    surface_area_m2: float
    avg_absorption: float


@dataclass
class RT60Result:
    rt60_sabine: float
    rt60_eyring: float
    mean_free_path: float


def rt60_sabine(volume: float, surface_area: float, avg_absorption: float) -> float:
    absorbing_area = surface_area * avg_absorption
    if absorbing_area <= 0:
        return float("inf")
    return 0.161 * volume / absorbing_area


def rt60_eyring(volume: float, surface_area: float, avg_absorption: float) -> float:
    if avg_absorption <= 0:
        return float("inf")
    if avg_absorption >= 1.0:
        return 0.0
    return -0.161 * volume / (surface_area * np.log(1.0 - avg_absorption))


def mean_free_path(volume: float, surface_area: float) -> float:
    if surface_area <= 0:
        return 0.0
    return 4.0 * volume / surface_area


def compute_rt60(room: RoomParameters) -> RT60Result:
    return RT60Result(
        rt60_sabine=rt60_sabine(room.volume_m3, room.surface_area_m2, room.avg_absorption),
        rt60_eyring=rt60_eyring(room.volume_m3, room.surface_area_m2, room.avg_absorption),
        mean_free_path=mean_free_path(room.volume_m3, room.surface_area_m2),
    )


@dataclass
class ClarityResult:
    c50: float
    c80: float
    d50: float
    d80: float
    center_time_ms: float


def clarity(impulse_response: np.ndarray, sample_rate: int) -> ClarityResult:
    ir = np.asarray(impulse_response, dtype=float)
    ir2 = ir ** 2
    n = len(ir2)
    t = np.arange(n) / sample_rate
    n50 = min(int(0.05 * sample_rate), n)
    n80 = min(int(0.08 * sample_rate), n)
    e_total = np.sum(ir2)
    if e_total < 1e-30:
        return ClarityResult(0.0, 0.0, 0.0, 0.0, 0.0)
    e_early_50 = np.sum(ir2[:n50])
    e_late_50 = np.sum(ir2[n50:])
    e_early_80 = np.sum(ir2[:n80])
    e_late_80 = np.sum(ir2[n80:])
    c50 = 10.0 * np.log10(e_early_50 / max(e_late_50, 1e-30))
    c80 = 10.0 * np.log10(e_early_80 / max(e_late_80, 1e-30))
    d50 = e_early_50 / e_total
    d80 = e_early_80 / e_total
    ts = float(np.sum(t * ir2) / e_total * 1000.0)
    return ClarityResult(c50=float(c50), c80=float(c80), d50=float(d50),
                         d80=float(d80), center_time_ms=ts)


def early_decay_time(impulse_response: np.ndarray, sample_rate: int) -> float:
    ir = np.asarray(impulse_response, dtype=float)
    schroeder = schroeder_curve(ir)
    if len(schroeder) < 2:
        return 0.0
    db = 10.0 * np.log10(np.maximum(schroeder / max(schroeder[0], 1e-30), 1e-30))
    idx_0 = 0
    idx_10 = np.searchsorted(-db, 10.0)
    if idx_10 <= idx_0 or idx_10 >= len(db):
        return 0.0
    t_10 = idx_10 / sample_rate
    return float(6.0 * t_10)


def schroeder_curve(impulse_response: np.ndarray) -> np.ndarray:
    ir2 = np.asarray(impulse_response, dtype=float) ** 2
    return np.cumsum(ir2[::-1])[::-1]


def mixing_time_estimate(volume: float) -> float:
    return np.sqrt(volume) * 2.0


@dataclass
class AbsorptionCoefficients:
    freq_125: float = 0.0
    freq_250: float = 0.0
    freq_500: float = 0.0
    freq_1000: float = 0.0
    freq_2000: float = 0.0
    freq_4000: float = 0.0

    def at_freq(self, freq: float) -> float:
        bands = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
        values = np.array([self.freq_125, self.freq_250, self.freq_500,
                           self.freq_1000, self.freq_2000, self.freq_4000])
        return float(np.interp(freq, bands, values))

    def nrc(self) -> float:
        return (self.freq_250 + self.freq_500 + self.freq_1000 + self.freq_2000) / 4.0


CONCRETE = AbsorptionCoefficients(0.01, 0.01, 0.015, 0.02, 0.02, 0.02)
CARPET = AbsorptionCoefficients(0.02, 0.06, 0.14, 0.37, 0.60, 0.65)
ACOUSTIC_TILE = AbsorptionCoefficients(0.20, 0.40, 0.70, 0.80, 0.60, 0.40)
CURTAIN_HEAVY = AbsorptionCoefficients(0.07, 0.31, 0.49, 0.75, 0.70, 0.60)
GLASS = AbsorptionCoefficients(0.35, 0.25, 0.18, 0.12, 0.07, 0.04)
