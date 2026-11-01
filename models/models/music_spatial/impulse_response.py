"""Room impulse response generation and analysis.

Synthetic RIR via image source method, plus ISO 3382-based parameter
estimation: RT60 (Schroeder integration), EDT, C80, D50, DRR.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RIRConfig:
    room_dims: tuple[float, float, float]
    source_pos: tuple[float, float, float]
    receiver_pos: tuple[float, float, float]
    rt60: float = 0.5
    sr: int = 16000
    max_order: int = 3
    speed_of_sound: float = 343.0


def _absorption_from_rt60(
    room_dims: tuple[float, float, float],
    rt60: float,
) -> float:
    """Invert Sabine formula to get average absorption coefficient."""
    l, w, h = room_dims
    V = l * w * h
    S = 2 * (l * w + l * h + w * h)
    if rt60 <= 0 or S <= 0:
        return 1.0
    A = 0.161 * V / rt60
    alpha = A / S
    return min(max(alpha, 0.01), 0.99)


def generate_rir_image_method(
    room_dims: tuple[float, float, float],
    source_pos: tuple[float, float, float],
    receiver_pos: tuple[float, float, float],
    rt60: float = 0.5,
    sr: int = 16000,
    max_order: int = 3,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    """Generate a room impulse response using the image source method.

    Parameters
    ----------
    room_dims : (length, width, height) in metres
    source_pos, receiver_pos : (x, y, z) in metres
    rt60 : desired reverberation time in seconds
    sr : sample rate
    max_order : maximum image order (controls reflection density)
    speed_of_sound : m/s

    Returns
    -------
    rir : 1-D array, length = sr * rt60 * 1.5 (capped)
    """
    alpha = _absorption_from_rt60(room_dims, rt60)
    reflection_coeff = math.sqrt(1.0 - alpha)

    src = np.asarray(source_pos, dtype=float)
    rcv = np.asarray(receiver_pos, dtype=float)
    dims = np.asarray(room_dims, dtype=float)

    rir_len = max(int(sr * rt60 * 1.5), sr // 4)
    rir = np.zeros(rir_len)

    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                order = abs(nx) + abs(ny) + abs(nz)
                image = np.empty(3)
                image[0] = (-1) ** nx * src[0] + nx * dims[0]
                image[1] = (-1) ** ny * src[1] + ny * dims[1]
                image[2] = (-1) ** nz * src[2] + nz * dims[2]

                dist = float(np.linalg.norm(image - rcv))
                if dist < 1e-6:
                    dist = 1e-6

                delay_samples = int(dist / speed_of_sound * sr)
                if delay_samples >= rir_len:
                    continue

                atten = reflection_coeff ** order / max(dist, 0.1)
                rir[delay_samples] += atten

    peak = np.max(np.abs(rir))
    if peak > 0:
        rir /= peak
    return rir


def estimate_rt60(rir: np.ndarray, sr: int, method: str = "T30") -> float:
    """Estimate RT60 via Schroeder backward integration.

    method : 'T20' extrapolates from -5..-25 dB; 'T30' from -5..-35 dB.
    """
    h2 = rir.astype(float) ** 2
    schroeder = np.cumsum(h2[::-1])[::-1]
    if schroeder[0] < 1e-30:
        return 0.0
    db = 10.0 * np.log10(schroeder / schroeder[0] + 1e-30)

    if method == "T20":
        upper, lower, factor = -5.0, -25.0, 3.0
    else:
        upper, lower, factor = -5.0, -35.0, 2.0

    i_upper = np.searchsorted(-db, -upper)
    i_lower = np.searchsorted(-db, -lower)
    if i_upper >= i_lower or i_lower >= len(db):
        return 0.0

    t = np.arange(i_upper, i_lower, dtype=float) / sr
    y = db[i_upper:i_lower]
    if len(t) < 2:
        return 0.0
    coeffs = np.polyfit(t, y, 1)
    slope = coeffs[0]
    if slope >= 0:
        return 0.0
    return float(abs(lower - upper) / abs(slope) * factor)


def estimate_edt(rir: np.ndarray, sr: int) -> float:
    """Early decay time: time for first 10 dB of Schroeder curve, × 6."""
    h2 = rir.astype(float) ** 2
    schroeder = np.cumsum(h2[::-1])[::-1]
    if schroeder[0] < 1e-30:
        return 0.0
    db = 10.0 * np.log10(schroeder / schroeder[0] + 1e-30)
    idx = np.searchsorted(-db, 10.0)
    if idx == 0 or idx >= len(db):
        return 0.0
    return float(idx / sr * 6.0)


def clarity_c80(rir: np.ndarray, sr: int) -> float:
    """C80 clarity: 10 log10(E_early / E_late) with 80 ms boundary."""
    boundary = int(0.080 * sr)
    h2 = rir.astype(float) ** 2
    early = float(np.sum(h2[:boundary]))
    late = float(np.sum(h2[boundary:]))
    if late < 1e-30:
        return 60.0
    return float(10.0 * math.log10(early / late))


def clarity_c50(rir: np.ndarray, sr: int) -> float:
    """C50 clarity with 50 ms boundary."""
    boundary = int(0.050 * sr)
    h2 = rir.astype(float) ** 2
    early = float(np.sum(h2[:boundary]))
    late = float(np.sum(h2[boundary:]))
    if late < 1e-30:
        return 60.0
    return float(10.0 * math.log10(early / late))


def definition_d50(rir: np.ndarray, sr: int) -> float:
    """D50: fraction of energy arriving within first 50 ms."""
    boundary = int(0.050 * sr)
    h2 = rir.astype(float) ** 2
    total = float(np.sum(h2))
    if total < 1e-30:
        return 0.0
    return float(np.sum(h2[:boundary]) / total)


def direct_to_reverberant_ratio(
    rir: np.ndarray, sr: int, direct_window_ms: float = 2.5,
) -> float:
    """DRR in dB: ratio of direct sound energy to reverberant energy."""
    boundary = int(direct_window_ms * sr / 1000.0)
    boundary = max(boundary, 1)
    h2 = rir.astype(float) ** 2
    direct = float(np.sum(h2[:boundary]))
    reverberant = float(np.sum(h2[boundary:]))
    if reverberant < 1e-30:
        return 60.0
    return float(10.0 * math.log10(direct / reverberant))


def center_time_ms(rir: np.ndarray, sr: int) -> float:
    """Center time Ts in milliseconds: first moment of h²(t)."""
    h2 = rir.astype(float) ** 2
    total = float(np.sum(h2))
    if total < 1e-30:
        return 0.0
    t = np.arange(len(h2), dtype=float) / sr
    return float(np.sum(t * h2) / total * 1000.0)


@dataclass(frozen=True)
class RIRAnalysis:
    rt60: float
    edt: float
    c50: float
    c80: float
    d50: float
    drr: float
    center_time_ms: float
    length_samples: int


def analyze_rir(rir: np.ndarray, sr: int) -> RIRAnalysis:
    """Compute all standard acoustic parameters for a RIR."""
    return RIRAnalysis(
        rt60=estimate_rt60(rir, sr),
        edt=estimate_edt(rir, sr),
        c50=clarity_c50(rir, sr),
        c80=clarity_c80(rir, sr),
        d50=definition_d50(rir, sr),
        drr=direct_to_reverberant_ratio(rir, sr),
        center_time_ms=center_time_ms(rir, sr),
        length_samples=len(rir),
    )
