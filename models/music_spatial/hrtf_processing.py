"""Head-Related Transfer Function processing for spatial audio.

Models ITD (Woodworth), ILD (sphere diffraction), pinna filtering,
distance attenuation, air absorption, and HRTF synthesis / application.
Pure numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


SPEED_OF_SOUND = 343.0
HEAD_RADIUS_DEFAULT = 0.0875


@dataclass
class SphericalCoord:
    azimuth: float
    elevation: float
    distance: float = 1.0


@dataclass
class HRTFPair:
    left: np.ndarray
    right: np.ndarray
    position: SphericalCoord
    sr: float


def interaural_time_difference(
    azimuth_deg: float,
    head_radius: float = HEAD_RADIUS_DEFAULT,
) -> float:
    theta = math.radians(azimuth_deg)
    if abs(theta) < 1e-12:
        return 0.0
    return (head_radius / SPEED_OF_SOUND) * (abs(theta) + math.sin(abs(theta))) * np.sign(theta)


def interaural_level_difference(
    azimuth_deg: float,
    frequency: float,
    head_radius: float = HEAD_RADIUS_DEFAULT,
) -> float:
    theta = math.radians(azimuth_deg)
    ka = 2 * math.pi * frequency * head_radius / SPEED_OF_SOUND
    alpha = 1.0 + ka / 2.0
    shadow = alpha * abs(math.sin(theta))
    ild = 20.0 * math.log10(max(1.0 + shadow, 1e-12) / max(1.0, 1e-12))
    return ild * np.sign(theta)


def pinna_filter(
    elevation_deg: float,
    n_taps: int = 128,
    sr: float = 44100.0,
) -> np.ndarray:
    elev = math.radians(elevation_deg)
    notch_freq = 7000.0 + 1500.0 * math.sin(elev)
    boost_freq = 3000.0 + 500.0 * math.sin(elev)
    n = np.arange(n_taps)
    h = np.ones(n_taps, dtype=float)
    t = n / sr
    notch_depth = 0.3
    h -= notch_depth * np.sin(2 * np.pi * notch_freq * t) * np.exp(-50.0 * t)
    boost_gain = 0.15 * (1.0 + math.sin(elev))
    h += boost_gain * np.sin(2 * np.pi * boost_freq * t) * np.exp(-30.0 * t)
    h *= np.hanning(n_taps)
    norm = np.sqrt(np.sum(h**2))
    return h / norm if norm > 0 else h


def distance_attenuation(distance: float, reference_distance: float = 1.0) -> float:
    if distance <= 0:
        return 0.0
    return -20.0 * math.log10(max(distance / reference_distance, 1e-12))


def air_absorption(
    distance: float,
    frequency: float,
    humidity: float = 50.0,
    temperature: float = 20.0,
) -> float:
    base_coeff = 0.01 * (frequency / 1000.0) ** 1.7
    humidity_factor = max(1.0 - (humidity - 50.0) * 0.005, 0.5)
    temp_factor = 1.0 + (temperature - 20.0) * 0.003
    return base_coeff * humidity_factor * temp_factor * distance


def synthesize_hrtf(
    position: SphericalCoord,
    n_taps: int = 256,
    sr: float = 44100.0,
) -> HRTFPair:
    itd = interaural_time_difference(position.azimuth)
    itd_samples = int(abs(itd) * sr)
    ild = interaural_level_difference(position.azimuth, 1000.0)
    pinna_l = pinna_filter(position.elevation, n_taps, sr)
    pinna_r = pinna_filter(position.elevation, n_taps, sr)
    dist_atten_db = distance_attenuation(position.distance)
    dist_gain = 10.0 ** (dist_atten_db / 20.0)
    left = np.zeros(n_taps + itd_samples)
    right = np.zeros(n_taps + itd_samples)
    ild_linear = 10.0 ** (abs(ild) / 20.0)
    if itd >= 0:
        left[:n_taps] = pinna_l * dist_gain * ild_linear
        right[itd_samples:itd_samples + n_taps] = pinna_r * dist_gain / max(ild_linear, 1e-12)
    else:
        right[:n_taps] = pinna_r * dist_gain * ild_linear
        left[itd_samples:itd_samples + n_taps] = pinna_l * dist_gain / max(ild_linear, 1e-12)
    return HRTFPair(left=left, right=right, position=position, sr=sr)


def apply_hrtf(
    mono_signal: np.ndarray,
    hrtf: HRTFPair,
) -> Tuple[np.ndarray, np.ndarray]:
    left = np.convolve(mono_signal, hrtf.left, mode='full')[:len(mono_signal)]
    right = np.convolve(mono_signal, hrtf.right, mode='full')[:len(mono_signal)]
    return left, right


def crossfade_hrtf(hrtf_a: HRTFPair, hrtf_b: HRTFPair, t: float) -> HRTFPair:
    t = max(0.0, min(1.0, t))
    n = max(len(hrtf_a.left), len(hrtf_b.left))
    left_a = np.pad(hrtf_a.left, (0, max(0, n - len(hrtf_a.left))))
    left_b = np.pad(hrtf_b.left, (0, max(0, n - len(hrtf_b.left))))
    right_a = np.pad(hrtf_a.right, (0, max(0, n - len(hrtf_a.right))))
    right_b = np.pad(hrtf_b.right, (0, max(0, n - len(hrtf_b.right))))
    left = left_a * (1 - t) + left_b * t
    right = right_a * (1 - t) + right_b * t
    az = hrtf_a.position.azimuth * (1 - t) + hrtf_b.position.azimuth * t
    el = hrtf_a.position.elevation * (1 - t) + hrtf_b.position.elevation * t
    d = hrtf_a.position.distance * (1 - t) + hrtf_b.position.distance * t
    return HRTFPair(left=left, right=right, position=SphericalCoord(az, el, d), sr=hrtf_a.sr)


def nearfield_correction(distance: float, head_radius: float = HEAD_RADIUS_DEFAULT) -> float:
    if distance <= head_radius:
        return 6.0
    ratio = head_radius / distance
    return 20.0 * math.log10(1.0 + ratio)


def externalization_cues(
    signal_left: np.ndarray,
    signal_right: np.ndarray,
) -> float:
    if len(signal_left) == 0 or len(signal_right) == 0:
        return 0.0
    ic = np.abs(np.corrcoef(signal_left[:min(len(signal_left), len(signal_right))],
                             signal_right[:min(len(signal_left), len(signal_right))])[0, 1])
    ild_var = np.var(20.0 * np.log10(np.maximum(np.abs(signal_left[:len(signal_right)]), 1e-12) /
                                      np.maximum(np.abs(signal_right), 1e-12)))
    decorrelation = 1.0 - ic
    spectral_variety = min(ild_var / 50.0, 1.0)
    return float(0.5 * decorrelation + 0.5 * spectral_variety)
