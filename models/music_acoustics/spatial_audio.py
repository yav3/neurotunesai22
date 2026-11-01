"""Spatial audio: ITD/ILD, ambisonics encoding, binaural rendering."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SPEED_OF_SOUND = 343.0
HEAD_RADIUS = 0.0875


@dataclass
class SpatialSource:
    azimuth_deg: float
    elevation_deg: float
    distance_m: float = 1.0


def itd(azimuth_deg: float, head_radius: float = HEAD_RADIUS) -> float:
    theta = np.radians(azimuth_deg)
    return head_radius / SPEED_OF_SOUND * (np.sin(theta) + theta)


def ild(azimuth_deg: float, frequency: float) -> float:
    theta = np.radians(azimuth_deg)
    ka = 2.0 * np.pi * frequency * HEAD_RADIUS / SPEED_OF_SOUND
    shadow = 1.0 + ka * np.abs(np.sin(theta))
    return float(20.0 * np.log10(max(shadow, 1e-10)))


@dataclass
class AmbisonicsChannels:
    W: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray


def foa_encode(signal: np.ndarray, azimuth_deg: float,
               elevation_deg: float) -> AmbisonicsChannels:
    signal = np.asarray(signal, dtype=float)
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    w = signal * (1.0 / np.sqrt(2.0))
    x = signal * np.cos(az) * np.cos(el)
    y = signal * np.sin(az) * np.cos(el)
    z = signal * np.sin(el)
    return AmbisonicsChannels(W=w, X=x, Y=y, Z=z)


def foa_decode_stereo(amb: AmbisonicsChannels) -> tuple[np.ndarray, np.ndarray]:
    left = amb.W + 0.7071 * amb.X + 0.7071 * amb.Y
    right = amb.W + 0.7071 * amb.X - 0.7071 * amb.Y
    return left, right


def binaural_pan(signal: np.ndarray, azimuth_deg: float,
                 sample_rate: int = 44100) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    delay_s = itd(azimuth_deg)
    delay_samples = int(abs(delay_s) * sample_rate)
    az_norm = azimuth_deg / 90.0
    gain_left = np.cos(np.clip(az_norm, -1, 1) * np.pi / 4.0)
    gain_right = np.sin(np.clip(az_norm, -1, 1) * np.pi / 4.0 + np.pi / 4.0)
    left = signal * gain_left
    right = signal * gain_right
    if delay_samples > 0:
        if azimuth_deg > 0:
            right = np.concatenate([np.zeros(delay_samples), right[:-delay_samples or None]])
        else:
            left = np.concatenate([np.zeros(delay_samples), left[:-delay_samples or None]])
    return left, right


def distance_attenuation(distance: float, reference_distance: float = 1.0) -> float:
    if distance <= 0:
        return 1.0
    return reference_distance / max(distance, reference_distance)


def air_absorption_db(distance: float, frequency: float,
                      humidity_pct: float = 50.0, temp_c: float = 20.0) -> float:
    alpha = 0.01 * (frequency / 1000.0) ** 1.7 * (1.0 + 0.01 * (50.0 - humidity_pct))
    return alpha * distance


@dataclass
class StereoPosition:
    left_gain: float
    right_gain: float


def constant_power_pan(position: float) -> StereoPosition:
    p = np.clip(position, -1.0, 1.0)
    angle = (p + 1.0) / 2.0 * (np.pi / 2.0)
    return StereoPosition(
        left_gain=float(np.cos(angle)),
        right_gain=float(np.sin(angle)),
    )


def hrtf_delay_model(azimuth_deg: float, elevation_deg: float) -> tuple[float, float]:
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    base_delay = HEAD_RADIUS / SPEED_OF_SOUND
    left_delay = base_delay * (1.0 - np.sin(az) * np.cos(el))
    right_delay = base_delay * (1.0 + np.sin(az) * np.cos(el))
    return float(left_delay), float(right_delay)
