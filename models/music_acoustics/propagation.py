"""Sound propagation: attenuation, absorption, diffraction, Doppler shift."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Medium:
    speed_of_sound: float = 343.0
    density: float = 1.225
    absorption_coeff_per_m: float = 0.0


AIR_20C = Medium(speed_of_sound=343.0, density=1.225, absorption_coeff_per_m=0.0)
WATER = Medium(speed_of_sound=1482.0, density=998.0, absorption_coeff_per_m=0.0002)


def propagation_delay(distance_m: float, medium: Medium = AIR_20C) -> float:
    if medium.speed_of_sound <= 0 or distance_m < 0:
        return 0.0
    return distance_m / medium.speed_of_sound


def inverse_square_attenuation(distance_m: float, ref_distance: float = 1.0) -> float:
    if distance_m <= 0 or ref_distance <= 0:
        return 0.0
    return float(-20.0 * np.log10(distance_m / ref_distance))


def atmospheric_absorption_coefficient(
    freq_hz: float,
    temperature_c: float = 20.0,
    humidity_pct: float = 50.0,
    pressure_kpa: float = 101.325,
) -> float:
    T = temperature_c + 273.15
    T_ref = 293.15
    T_01 = 273.16
    p_ref = 101.325
    pr = pressure_kpa / p_ref
    T_ratio = T / T_ref

    C_humid = 4.6151 - 6.8346 * (T_01 / T) ** 1.261
    h = humidity_pct * 10.0 ** C_humid * pr

    f_rO = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    f_rN = pr * T_ratio ** (-0.5) * (9.0 + 280.0 * h * np.exp(-4.170 * (T_ratio ** (-1.0 / 3.0) - 1.0)))

    f = freq_hz
    alpha = (
        8.686 * f * f * (
            1.84e-11 * pr ** (-1) * T_ratio ** 0.5
            + T_ratio ** (-2.5) * (
                0.01275 * np.exp(-2239.1 / T) / (f_rO + f * f / f_rO)
                + 0.1068 * np.exp(-3352.0 / T) / (f_rN + f * f / f_rN)
            )
        )
    )
    return float(alpha)


def atmospheric_absorption(
    freq_hz: float,
    distance_m: float,
    temperature_c: float = 20.0,
    humidity_pct: float = 50.0,
) -> float:
    alpha = atmospheric_absorption_coefficient(freq_hz, temperature_c, humidity_pct)
    return float(alpha * distance_m)


def ground_reflection_interference(
    source_height: float,
    receiver_height: float,
    distance: float,
    freq_hz: float,
    ground_reflection_coeff: float = -0.9,
    speed_of_sound: float = 343.0,
) -> float:
    d_direct = np.sqrt(distance ** 2 + (source_height - receiver_height) ** 2)
    d_reflected = np.sqrt(distance ** 2 + (source_height + receiver_height) ** 2)
    path_diff = d_reflected - d_direct
    wavelength = speed_of_sound / freq_hz if freq_hz > 0 else float("inf")
    phase_diff = 2.0 * np.pi * path_diff / wavelength
    R = ground_reflection_coeff
    p_ratio_sq = 1.0 + R ** 2 + 2.0 * R * np.cos(phase_diff)
    if p_ratio_sq <= 0:
        return -100.0
    return float(10.0 * np.log10(p_ratio_sq))


def diffraction_maekawa(
    freq_hz: float,
    barrier_height: float,
    source_distance: float,
    receiver_distance: float,
    source_height: float = 0.0,
    receiver_height: float = 0.0,
    speed_of_sound: float = 343.0,
) -> float:
    d_s = np.sqrt(source_distance ** 2 + (barrier_height - source_height) ** 2)
    d_r = np.sqrt(receiver_distance ** 2 + (barrier_height - receiver_height) ** 2)
    d_direct = np.sqrt((source_distance + receiver_distance) ** 2
                       + (source_height - receiver_height) ** 2)
    delta = d_s + d_r - d_direct
    wavelength = speed_of_sound / freq_hz if freq_hz > 0 else 1.0
    N = 2.0 * delta / wavelength
    if N < -0.2:
        return 0.0
    if N < 0:
        attenuation = 5.0 + 8.0 * N
    else:
        attenuation = 5.0 + 20.0 * np.log10(np.sqrt(2.0 * np.pi * N) / np.tanh(np.sqrt(2.0 * np.pi * N) + 1e-12))
    return float(min(attenuation, 24.0))


def doppler_shift(
    source_velocity: float,
    observer_velocity: float,
    frequency: float,
    speed_of_sound: float = 343.0,
) -> float:
    if speed_of_sound <= 0 or frequency <= 0:
        return frequency
    denom = speed_of_sound - source_velocity
    if abs(denom) < 1e-12:
        return float("inf")
    return float(frequency * (speed_of_sound + observer_velocity) / denom)


def outdoor_spl_at_distance(
    source_spl_db: float,
    distance_m: float,
    freq_hz: float = 1000.0,
    temperature_c: float = 20.0,
    humidity_pct: float = 50.0,
    ref_distance: float = 1.0,
) -> float:
    spreading = inverse_square_attenuation(distance_m, ref_distance)
    absorption = atmospheric_absorption(freq_hz, distance_m, temperature_c, humidity_pct)
    return source_spl_db + spreading - absorption


def reverb_time_sabine(
    volume_m3: float,
    total_absorption_m2: float,
) -> float:
    if total_absorption_m2 <= 0:
        return float("inf")
    return 0.161 * volume_m3 / total_absorption_m2


def reverb_time_eyring(
    volume_m3: float,
    surface_area_m2: float,
    avg_absorption_coeff: float,
) -> float:
    if surface_area_m2 <= 0 or avg_absorption_coeff <= 0 or avg_absorption_coeff >= 1.0:
        return float("inf") if avg_absorption_coeff <= 0 else 0.0
    return 0.161 * volume_m3 / (-surface_area_m2 * np.log(1.0 - avg_absorption_coeff))


def critical_distance(
    volume_m3: float,
    rt60: float,
) -> float:
    if rt60 <= 0:
        return 0.0
    return 0.057 * np.sqrt(volume_m3 / rt60)
