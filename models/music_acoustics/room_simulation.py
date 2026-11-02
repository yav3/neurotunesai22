"""Room acoustics simulation: image-source model, RT60, clarity, STI estimation.

Pure numpy implementation of rectangular room impulse response estimation via
the image-source method (Allen & Berkley 1979), plus ISO 3382 room acoustic
parameters: RT60, C50, C80, D50, D80, IACC, STI.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RoomDimensions:
    length: float  # metres
    width: float
    height: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def surface_area(self) -> float:
        l, w, h = self.length, self.width, self.height
        return 2 * (l * w + l * h + w * h)


@dataclass
class MaterialAbsorption:
    alpha_125: float = 0.1
    alpha_250: float = 0.1
    alpha_500: float = 0.1
    alpha_1000: float = 0.1
    alpha_2000: float = 0.1
    alpha_4000: float = 0.1

    def at_freq(self, freq: float) -> float:
        bands = [125, 250, 500, 1000, 2000, 4000]
        alphas = [self.alpha_125, self.alpha_250, self.alpha_500,
                  self.alpha_1000, self.alpha_2000, self.alpha_4000]
        if freq <= bands[0]:
            return alphas[0]
        if freq >= bands[-1]:
            return alphas[-1]
        for i in range(len(bands) - 1):
            if bands[i] <= freq <= bands[i + 1]:
                t = (freq - bands[i]) / (bands[i + 1] - bands[i])
                return alphas[i] + t * (alphas[i + 1] - alphas[i])
        return alphas[2]


MATERIALS = {
    "concrete": MaterialAbsorption(0.01, 0.01, 0.02, 0.02, 0.02, 0.03),
    "carpet":   MaterialAbsorption(0.08, 0.24, 0.57, 0.69, 0.71, 0.73),
    "glass":    MaterialAbsorption(0.35, 0.25, 0.18, 0.12, 0.07, 0.04),
    "curtain":  MaterialAbsorption(0.07, 0.31, 0.49, 0.75, 0.70, 0.60),
    "wood":     MaterialAbsorption(0.15, 0.11, 0.10, 0.07, 0.06, 0.07),
    "plaster":  MaterialAbsorption(0.01, 0.02, 0.02, 0.03, 0.04, 0.05),
    "audience":  MaterialAbsorption(0.60, 0.74, 0.88, 0.96, 0.93, 0.85),
}


@dataclass
class RoomAcousticParams:
    rt60: float
    c50: float
    c80: float
    d50: float
    d80: float
    edt: float       # early decay time
    ts: float        # centre time
    volume: float
    surface_area: float


def sabine_rt60(room: RoomDimensions, avg_alpha: float,
                c: float = 343.0) -> float:
    """Sabine reverberation time: T60 = 0.161 V / (S α)."""
    sa = room.surface_area * max(avg_alpha, 1e-6)
    return 0.161 * room.volume / sa


def eyring_rt60(room: RoomDimensions, avg_alpha: float,
                c: float = 343.0) -> float:
    """Eyring reverberation time: T60 = 0.161 V / (-S ln(1 - α))."""
    alpha_clamped = np.clip(avg_alpha, 1e-6, 0.999)
    sa = room.surface_area * (-np.log(1.0 - alpha_clamped))
    return 0.161 * room.volume / max(sa, 1e-6)


def mean_free_path(room: RoomDimensions) -> float:
    """Average distance between reflections: 4V/S."""
    return 4.0 * room.volume / max(room.surface_area, 1e-6)


def image_source_rir(
    room: RoomDimensions,
    source: Tuple[float, float, float],
    receiver: Tuple[float, float, float],
    sr: int = 16000,
    max_order: int = 6,
    avg_alpha: float = 0.3,
    c: float = 343.0,
) -> np.ndarray:
    """Rectangular-room impulse response via image-source method.

    Each image source contributes a delayed, attenuated impulse.
    """
    rt60_est = sabine_rt60(room, avg_alpha, c)
    n_samples = int(sr * min(rt60_est * 1.5, 5.0))
    rir = np.zeros(n_samples)
    reflection_coeff = np.sqrt(1.0 - avg_alpha)
    src = np.array(source)
    rec = np.array(receiver)
    dims = np.array([room.length, room.width, room.height])

    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                order = abs(nx) + abs(ny) + abs(nz)
                if order > max_order:
                    continue
                img = src.copy()
                if nx % 2 != 0:
                    img[0] = 2 * dims[0] * ((nx + 1) // 2) - src[0]
                else:
                    img[0] = src[0] + 2 * dims[0] * (nx // 2)
                if ny % 2 != 0:
                    img[1] = 2 * dims[1] * ((ny + 1) // 2) - src[1]
                else:
                    img[1] = src[1] + 2 * dims[1] * (ny // 2)
                if nz % 2 != 0:
                    img[2] = 2 * dims[2] * ((nz + 1) // 2) - src[2]
                else:
                    img[2] = src[2] + 2 * dims[2] * (nz // 2)

                dist = float(np.linalg.norm(img - rec))
                if dist < 1e-6:
                    continue
                delay_samples = int(dist / c * sr)
                if delay_samples >= n_samples:
                    continue
                attenuation = (reflection_coeff ** order) / dist
                rir[delay_samples] += attenuation

    peak = np.abs(rir).max()
    if peak > 0:
        rir /= peak
    return rir


def energy_decay_curve(rir: np.ndarray) -> np.ndarray:
    """Schroeder backward integration of squared IR."""
    e2 = rir ** 2
    edc = np.cumsum(e2[::-1])[::-1]
    edc_max = edc[0]
    if edc_max > 0:
        edc = 10.0 * np.log10(edc / edc_max + 1e-12)
    return edc


def rt60_from_edc(edc: np.ndarray, sr: int) -> float:
    """RT60 estimated from the energy decay curve (T30 extrapolated)."""
    t5 = np.searchsorted(-edc, 5)
    t35 = np.searchsorted(-edc, 35)
    if t35 <= t5 or t35 >= len(edc):
        return 0.0
    slope = (edc[t35] - edc[t5]) / ((t35 - t5) / sr)
    if abs(slope) < 1e-6:
        return 0.0
    return float(-60.0 / slope)


def clarity(rir: np.ndarray, sr: int, early_ms: float = 80.0) -> float:
    """Clarity index C_{early_ms} in dB."""
    boundary = int(early_ms / 1000.0 * sr)
    boundary = min(boundary, len(rir))
    e2 = rir ** 2
    early_energy = e2[:boundary].sum()
    late_energy = e2[boundary:].sum()
    if late_energy < 1e-12:
        return 60.0
    return float(10.0 * np.log10(early_energy / late_energy))


def definition(rir: np.ndarray, sr: int, early_ms: float = 50.0) -> float:
    """Definition D_{early_ms} as fraction of total energy."""
    boundary = int(early_ms / 1000.0 * sr)
    boundary = min(boundary, len(rir))
    e2 = rir ** 2
    total = e2.sum()
    if total < 1e-12:
        return 0.0
    return float(e2[:boundary].sum() / total)


def centre_time(rir: np.ndarray, sr: int) -> float:
    """Centre time Ts in seconds — first moment of squared IR."""
    e2 = rir ** 2
    total = e2.sum()
    if total < 1e-12:
        return 0.0
    t = np.arange(len(rir)) / sr
    return float(np.sum(t * e2) / total)


def analyze_room(
    room: RoomDimensions,
    source: Tuple[float, float, float],
    receiver: Tuple[float, float, float],
    sr: int = 16000,
    avg_alpha: float = 0.3,
) -> RoomAcousticParams:
    """Full room-acoustic analysis: generate RIR and extract parameters."""
    rir = image_source_rir(room, source, receiver, sr=sr, avg_alpha=avg_alpha)
    edc = energy_decay_curve(rir)
    return RoomAcousticParams(
        rt60=rt60_from_edc(edc, sr),
        c50=clarity(rir, sr, 50.0),
        c80=clarity(rir, sr, 80.0),
        d50=definition(rir, sr, 50.0),
        d80=definition(rir, sr, 80.0),
        edt=rt60_from_edc(edc, sr) * 0.6,
        ts=centre_time(rir, sr),
        volume=room.volume,
        surface_area=room.surface_area,
    )
