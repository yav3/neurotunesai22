"""Room acoustics: RT60, early reflections, diffusion, absorption modeling.

Implements room impulse response analysis, Sabine/Eyring RT60 estimation,
early reflection detection, diffusion modeling, and acoustic parameter
computation for music production and spatial audio.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoomParameters:
    length_m: float
    width_m: float
    height_m: float

    @property
    def volume_m3(self) -> float:
        return self.length_m * self.width_m * self.height_m

    @property
    def surface_area_m2(self) -> float:
        l, w, h = self.length_m, self.width_m, self.height_m
        return 2 * (l * w + l * h + w * h)

    @property
    def mean_free_path_m(self) -> float:
        """Average distance between reflections: 4V/S."""
        sa = self.surface_area_m2
        if sa < 1e-12:
            return 0.0
        return 4 * self.volume_m3 / sa


@dataclass(frozen=True)
class AcousticParameters:
    rt60: float
    edt: float  # early decay time
    c50: float  # clarity (dB)
    c80: float  # clarity (dB)
    d50: float  # definition (0-1)
    ts: float   # center time (ms)
    br: float   # bass ratio


def sabine_rt60(volume_m3: float, total_absorption_m2: float) -> float:
    """Sabine reverberation time: T60 = 0.161 * V / A."""
    if total_absorption_m2 < 1e-12:
        return float("inf")
    return 0.161 * volume_m3 / total_absorption_m2


def eyring_rt60(
    volume_m3: float,
    surface_area_m2: float,
    avg_absorption_coeff: float,
) -> float:
    """Eyring reverberation time for high absorption.

    T60 = 0.161 * V / (-S * ln(1 - alpha_avg))
    """
    if avg_absorption_coeff >= 1.0:
        return 0.0
    if avg_absorption_coeff <= 0.0:
        return float("inf")
    denominator = -surface_area_m2 * math.log(1.0 - avg_absorption_coeff)
    if denominator < 1e-12:
        return float("inf")
    return 0.161 * volume_m3 / denominator


def total_absorption(
    surface_areas: np.ndarray,
    absorption_coefficients: np.ndarray,
) -> float:
    """Compute total absorption A = sum(S_i * alpha_i)."""
    return float(np.sum(surface_areas * absorption_coefficients))


def schroeder_curve(impulse_response: np.ndarray) -> np.ndarray:
    """Compute Schroeder backward integration curve from an impulse response.

    C(t) = integral from t to infinity of h^2(tau) dtau
    Normalized so C(0) = 1.
    """
    h2 = impulse_response ** 2
    curve = np.cumsum(h2[::-1])[::-1]
    if curve[0] > 0:
        curve = curve / curve[0]
    return curve


def estimate_rt60_from_ir(
    impulse_response: np.ndarray,
    sr: float,
    method: str = "T30",
) -> float:
    """Estimate RT60 from an impulse response using Schroeder curve.

    method: "T20" uses -5 to -25 dB range (extrapolate to -60),
            "T30" uses -5 to -35 dB range.
    """
    curve = schroeder_curve(impulse_response)
    curve_db = 10 * np.log10(np.maximum(curve, 1e-20))
    if method == "T20":
        upper, lower, factor = -5.0, -25.0, 3.0
    else:
        upper, lower, factor = -5.0, -35.0, 2.0
    upper_idx = np.searchsorted(-curve_db, -upper)
    lower_idx = np.searchsorted(-curve_db, -lower)
    if upper_idx >= lower_idx or lower_idx >= len(curve_db):
        return 0.0
    t = np.arange(upper_idx, lower_idx) / sr
    y = curve_db[upper_idx:lower_idx]
    if len(t) < 2:
        return 0.0
    coeffs = np.polyfit(t, y, 1)
    slope = coeffs[0]
    if slope >= 0:
        return 0.0
    decay_range = abs(lower - upper)
    t_decay = decay_range / abs(slope)
    return float(t_decay * factor)


def early_decay_time(impulse_response: np.ndarray, sr: float) -> float:
    """EDT: time for the Schroeder curve to drop from 0 to -10 dB, × 6."""
    curve = schroeder_curve(impulse_response)
    curve_db = 10 * np.log10(np.maximum(curve, 1e-20))
    idx_10 = np.searchsorted(-curve_db, 10.0)
    if idx_10 == 0 or idx_10 >= len(curve_db):
        return 0.0
    return float(idx_10 / sr * 6.0)


def clarity(impulse_response: np.ndarray, sr: float, limit_ms: float = 80.0) -> float:
    """Clarity C_t: 10*log10(early_energy / late_energy).

    limit_ms: boundary between early and late (50 for C50, 80 for C80).
    """
    limit_samples = int(limit_ms * sr / 1000.0)
    h2 = impulse_response ** 2
    early = float(np.sum(h2[:limit_samples]))
    late = float(np.sum(h2[limit_samples:]))
    if late < 1e-20:
        return 60.0
    return float(10 * math.log10(early / late))


def definition_d50(impulse_response: np.ndarray, sr: float) -> float:
    """Definition D50: ratio of early (0-50ms) to total energy."""
    limit = int(50 * sr / 1000.0)
    h2 = impulse_response ** 2
    total = float(np.sum(h2))
    if total < 1e-20:
        return 0.0
    return float(np.sum(h2[:limit]) / total)


def center_time(impulse_response: np.ndarray, sr: float) -> float:
    """Center time Ts: first moment of the squared IR, in ms."""
    h2 = impulse_response ** 2
    total = float(np.sum(h2))
    if total < 1e-20:
        return 0.0
    t = np.arange(len(h2)) / sr
    return float(np.sum(t * h2) / total * 1000.0)


def bass_ratio(rt60_125: float, rt60_250: float, rt60_500: float, rt60_1000: float) -> float:
    """Bass ratio BR = (T_125 + T_250) / (T_500 + T_1000)."""
    denom = rt60_500 + rt60_1000
    if denom < 1e-12:
        return 0.0
    return (rt60_125 + rt60_250) / denom


def image_source_reflections(
    source: np.ndarray,
    receiver: np.ndarray,
    room: RoomParameters,
    max_order: int = 2,
    speed_of_sound: float = 343.0,
) -> list[tuple[float, float]]:
    """Compute early reflections using the image source method.

    Returns list of (delay_seconds, attenuation) tuples.
    """
    reflections: list[tuple[float, float]] = []
    dims = np.array([room.length_m, room.width_m, room.height_m])
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                image = source.copy()
                if nx % 2 != 0:
                    image[0] = 2 * dims[0] * ((nx + 1) // 2) - source[0]
                else:
                    image[0] = source[0] + nx * dims[0]
                if ny % 2 != 0:
                    image[1] = 2 * dims[1] * ((ny + 1) // 2) - source[1]
                else:
                    image[1] = source[1] + ny * dims[1]
                if nz % 2 != 0:
                    image[2] = 2 * dims[2] * ((nz + 1) // 2) - source[2]
                else:
                    image[2] = source[2] + nz * dims[2]
                dist = float(np.linalg.norm(image - receiver))
                if dist < 1e-6:
                    continue
                delay = dist / speed_of_sound
                order = abs(nx) + abs(ny) + abs(nz)
                atten = 1.0 / dist * (0.9 ** order)
                reflections.append((delay, atten))
    reflections.sort(key=lambda x: x[0])
    return reflections


def compute_acoustic_parameters(
    impulse_response: np.ndarray,
    sr: float,
    rt60_bands: dict[int, float] | None = None,
) -> AcousticParameters:
    """Compute standard acoustic parameters from an impulse response."""
    rt = estimate_rt60_from_ir(impulse_response, sr)
    edt_val = early_decay_time(impulse_response, sr)
    c50_val = clarity(impulse_response, sr, 50.0)
    c80_val = clarity(impulse_response, sr, 80.0)
    d50_val = definition_d50(impulse_response, sr)
    ts_val = center_time(impulse_response, sr)
    br_val = 1.0
    if rt60_bands and all(k in rt60_bands for k in [125, 250, 500, 1000]):
        br_val = bass_ratio(rt60_bands[125], rt60_bands[250], rt60_bands[500], rt60_bands[1000])
    return AcousticParameters(
        rt60=rt, edt=edt_val, c50=c50_val, c80=c80_val,
        d50=d50_val, ts=ts_val, br=br_val,
    )
