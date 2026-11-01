"""Reverberation modeling: early reflections, late diffuse field, and reverb design."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


SPEED_OF_SOUND = 343.0  # m/s at 20 C


@dataclass
class ReverbParameters:
    """Configuration for a synthetic reverb."""
    room_volume_m3: float = 200.0
    surface_area_m2: float = 200.0
    mean_absorption: float = 0.3
    pre_delay_ms: float = 10.0
    early_reflection_count: int = 6
    density: float = 0.8  # 0 sparse, 1 dense
    diffusion: float = 0.7
    hf_damping: float = 0.5  # high-frequency rolloff factor


@dataclass
class EarlyReflection:
    """A single early reflection."""
    delay_ms: float
    amplitude: float
    azimuth_deg: float = 0.0


def rt60_sabine(volume: float, surface_area: float, absorption: float) -> float:
    """Sabine RT60 in seconds: T = 0.161 * V / (S * alpha)."""
    if surface_area * absorption <= 0:
        return float("inf")
    return 0.161 * volume / (surface_area * absorption)


def rt60_eyring(volume: float, surface_area: float, absorption: float) -> float:
    """Eyring RT60: T = 0.161 * V / (-S * ln(1 - alpha))."""
    if absorption >= 1.0 or absorption <= 0 or surface_area <= 0:
        return 0.0
    return 0.161 * volume / (-surface_area * math.log(1.0 - absorption))


def mean_free_path(volume: float, surface_area: float) -> float:
    """Mean free path in metres: 4V/S."""
    if surface_area <= 0:
        return 0.0
    return 4.0 * volume / surface_area


def mixing_time_ms(volume: float) -> float:
    """Approximate mixing time in ms (Polack): t_mix ~ 4 * sqrt(V / c)."""
    return 4.0 * math.sqrt(volume / SPEED_OF_SOUND) * 1000.0


def generate_early_reflections(params: ReverbParameters) -> list[EarlyReflection]:
    """Generate early reflection pattern based on room geometry heuristic."""
    mfp = mean_free_path(params.room_volume_m3, params.surface_area_m2)
    reflections: list[EarlyReflection] = []
    for i in range(1, params.early_reflection_count + 1):
        path_length = mfp * i * 0.7
        delay = params.pre_delay_ms + (path_length / SPEED_OF_SOUND) * 1000.0
        amp = (1.0 - params.mean_absorption) ** i
        azimuth = ((i * 137.508) % 360.0) - 180.0  # golden angle spread
        reflections.append(EarlyReflection(delay_ms=delay, amplitude=amp, azimuth_deg=azimuth))
    return reflections


def late_field_density(rt60: float, time_ms: float) -> float:
    """Normalized energy density of the late diffuse field at a given time.

    Exponential decay: E(t) = exp(-6.9 * t / RT60).
    """
    if rt60 <= 0:
        return 0.0
    t_s = time_ms / 1000.0
    return math.exp(-6.9 * t_s / rt60)


def energy_decay_curve(rt60: float, duration_ms: float, sample_rate: int = 1000) -> np.ndarray:
    """Compute the energy decay curve (EDC) as a numpy array."""
    n = int(duration_ms * sample_rate / 1000.0)
    t = np.linspace(0, duration_ms / 1000.0, n)
    if rt60 <= 0:
        return np.zeros(n)
    return np.exp(-6.9 * t / rt60)


def clarity_c80(early_energy: float, late_energy: float) -> float:
    """Clarity C80 in dB: 10*log10(E_early / E_late)."""
    if late_energy <= 0:
        return float("inf")
    if early_energy <= 0:
        return float("-inf")
    return 10.0 * math.log10(early_energy / late_energy)


def definition_d50(early_energy: float, total_energy: float) -> float:
    """Definition D50: ratio of early (0-50ms) to total energy."""
    if total_energy <= 0:
        return 0.0
    return min(1.0, early_energy / total_energy)


def center_time_ts(reflections: list[EarlyReflection], rt60: float) -> float:
    """Center time Ts (ms): energy-weighted mean arrival time."""
    if not reflections and rt60 <= 0:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for r in reflections:
        e = r.amplitude ** 2
        numerator += r.delay_ms * e
        denominator += e
    if rt60 > 0:
        late_center = rt60 * 1000.0 / 6.9
        late_energy = 1.0 - sum(r.amplitude ** 2 for r in reflections)
        if late_energy > 0:
            numerator += late_center * late_energy
            denominator += late_energy
    if denominator <= 0:
        return 0.0
    return numerator / denominator


@dataclass
class ReverbProfile:
    """Full reverb characterization."""
    rt60_sabine: float
    rt60_eyring: float
    mean_free_path_m: float
    mixing_time_ms: float
    early_reflections: list[EarlyReflection]
    clarity_c80_db: float
    definition_d50: float
    center_time_ms: float


def analyze_reverb(params: ReverbParameters) -> ReverbProfile:
    """Complete reverb analysis from room parameters."""
    rt_s = rt60_sabine(params.room_volume_m3, params.surface_area_m2, params.mean_absorption)
    rt_e = rt60_eyring(params.room_volume_m3, params.surface_area_m2, params.mean_absorption)
    mfp = mean_free_path(params.room_volume_m3, params.surface_area_m2)
    mt = mixing_time_ms(params.room_volume_m3)
    ers = generate_early_reflections(params)

    early_e = sum(r.amplitude ** 2 for r in ers if r.delay_ms <= 80.0 + params.pre_delay_ms)
    total_e = sum(r.amplitude ** 2 for r in ers) + 1.0
    late_e = total_e - early_e

    c80 = clarity_c80(early_e, late_e)
    d50_early = sum(r.amplitude ** 2 for r in ers if r.delay_ms <= 50.0 + params.pre_delay_ms)
    d50 = definition_d50(d50_early, total_e)
    ts = center_time_ts(ers, rt_s)

    return ReverbProfile(
        rt60_sabine=rt_s,
        rt60_eyring=rt_e,
        mean_free_path_m=mfp,
        mixing_time_ms=mt,
        early_reflections=ers,
        clarity_c80_db=c80,
        definition_d50=d50,
        center_time_ms=ts,
    )


def reverb_tail_level_db(rt60: float, time_ms: float) -> float:
    """Level of the reverb tail in dB relative to direct sound at a given time."""
    if rt60 <= 0:
        return float("-inf")
    return -60.0 * (time_ms / 1000.0) / rt60
