"""Stage acoustics: RT60, clarity, definition, venue profiling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VenueProfile:
    rt60: float
    clarity_c80: float
    definition_d50: float
    center_time_ts: float
    bass_ratio: float
    brilliance: float


def estimate_rt60(impulse_response: np.ndarray, sr: int = 44100) -> float:
    if len(impulse_response) < 2:
        return 0.0
    ir2 = impulse_response ** 2
    cumsum_rev = np.cumsum(ir2[::-1])[::-1]
    total = cumsum_rev[0]
    if total < 1e-30:
        return 0.0
    decay_db = 10.0 * np.log10(np.maximum(cumsum_rev / total, 1e-30))

    i5 = np.searchsorted(-decay_db, 5.0)
    i25 = np.searchsorted(-decay_db, 25.0)
    if i25 <= i5 or i25 >= len(decay_db):
        return 0.0
    t5 = i5 / sr
    t25 = i25 / sr
    slope = (decay_db[i25] - decay_db[i5]) / max(t25 - t5, 1e-12)
    if slope >= 0:
        return 0.0
    return float(-60.0 / slope)


def clarity_c80(impulse_response: np.ndarray, sr: int = 44100) -> float:
    if len(impulse_response) < 2:
        return 0.0
    ir2 = impulse_response ** 2
    boundary = int(0.08 * sr)
    early = float(np.sum(ir2[:boundary]))
    late = float(np.sum(ir2[boundary:]))
    if late < 1e-30:
        return 50.0
    return float(10.0 * np.log10(max(early / late, 1e-30)))


def definition_d50(impulse_response: np.ndarray, sr: int = 44100) -> float:
    if len(impulse_response) < 2:
        return 0.0
    ir2 = impulse_response ** 2
    boundary = int(0.05 * sr)
    early = float(np.sum(ir2[:boundary]))
    total = float(np.sum(ir2))
    if total < 1e-30:
        return 0.0
    return float(early / total)


def center_time(impulse_response: np.ndarray, sr: int = 44100) -> float:
    if len(impulse_response) < 2:
        return 0.0
    ir2 = impulse_response ** 2
    total = float(np.sum(ir2))
    if total < 1e-30:
        return 0.0
    t = np.arange(len(ir2)) / sr
    return float(np.sum(t * ir2) / total)


def bass_ratio(rt60_low: float, rt60_mid: float) -> float:
    if rt60_mid < 1e-12:
        return 0.0
    return rt60_low / rt60_mid


def stage_support(
    impulse_response: np.ndarray,
    sr: int = 44100,
    early_ms: float = 100.0,
    late_ms: float = 1000.0,
) -> float:
    if len(impulse_response) < 2:
        return 0.0
    ir2 = impulse_response ** 2
    direct_end = int(0.01 * sr)
    early_end = int(early_ms / 1000.0 * sr)
    direct_energy = float(np.sum(ir2[:direct_end]))
    early_energy = float(np.sum(ir2[direct_end:early_end]))
    if direct_energy < 1e-30:
        return -np.inf
    return float(10.0 * np.log10(max(early_energy / direct_energy, 1e-30)))


def analyze_venue(impulse_response: np.ndarray, sr: int = 44100) -> VenueProfile:
    rt = estimate_rt60(impulse_response, sr)
    c80 = clarity_c80(impulse_response, sr)
    d50 = definition_d50(impulse_response, sr)
    ts = center_time(impulse_response, sr)
    return VenueProfile(
        rt60=rt,
        clarity_c80=c80,
        definition_d50=d50,
        center_time_ts=ts,
        bass_ratio=1.0,
        brilliance=0.5,
    )
