"""Head-related transfer function modeling for spatial audio."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SPEED_OF_SOUND: float = 343.0  # m/s


@dataclass
class HRTFParams:
    azimuth: float
    elevation: float
    distance: float = 1.0
    head_radius_cm: float = 8.75


def interaural_time_difference(azimuth_rad: float, head_radius_m: float = 0.0875) -> float:
    """Woodworth spherical-head ITD: (r/c)(theta + sin(theta))."""
    theta = abs(azimuth_rad)
    if theta > math.pi / 2:
        theta = math.pi / 2
    return (head_radius_m / SPEED_OF_SOUND) * (theta + math.sin(theta))


def interaural_level_difference(azimuth_rad: float, frequency: float) -> float:
    """Frequency-dependent ILD: negligible below ~500 Hz, up to ~20 dB above 3 kHz."""
    if frequency < 200:
        return 0.0
    ka = 2 * math.pi * frequency * 0.0875 / SPEED_OF_SOUND
    shadow = min(1.0, ka / 3.0)
    return shadow * 20.0 * abs(math.sin(azimuth_rad))


def hrtf_filter_pair(params: HRTFParams, n_taps: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Generate approximate left/right FIR filters from spherical-head model."""
    az_rad = math.radians(params.azimuth)
    itd = interaural_time_difference(az_rad, params.head_radius_cm / 100.0)
    sr = 44100
    delay_samples = itd * sr

    t = np.arange(n_taps) / sr
    center = n_taps // 2
    atten = distance_attenuation(params.distance)

    left_delay = center + delay_samples / 2 if params.azimuth > 0 else center - delay_samples / 2
    right_delay = center - delay_samples / 2 if params.azimuth > 0 else center + delay_samples / 2

    left_ir = np.sinc(np.arange(n_taps) - left_delay) * np.hanning(n_taps) * atten
    right_ir = np.sinc(np.arange(n_taps) - right_delay) * np.hanning(n_taps) * atten

    el_atten = math.cos(math.radians(params.elevation))
    left_ir *= (1.0 + 0.3 * math.sin(math.radians(params.azimuth))) * el_atten
    right_ir *= (1.0 - 0.3 * math.sin(math.radians(params.azimuth))) * el_atten

    return left_ir / (np.max(np.abs(left_ir)) + 1e-12), right_ir / (np.max(np.abs(right_ir)) + 1e-12)


def distance_attenuation(distance: float, reference: float = 1.0) -> float:
    """Inverse-distance law with near-field correction."""
    if distance <= 0:
        return 1.0
    return reference / max(distance, 0.1)


@dataclass
class HRTFInterpolator:
    """Bilinear interpolation between measured HRTF positions."""
    grid_azimuths: np.ndarray
    grid_elevations: np.ndarray
    filters_left: np.ndarray   # shape (n_az, n_el, n_taps)
    filters_right: np.ndarray

    def interpolate(self, azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray]:
        az_idx = np.searchsorted(self.grid_azimuths, azimuth).clip(1, len(self.grid_azimuths) - 1)
        el_idx = np.searchsorted(self.grid_elevations, elevation).clip(1, len(self.grid_elevations) - 1)

        az0, az1 = self.grid_azimuths[az_idx - 1], self.grid_azimuths[az_idx]
        el0, el1 = self.grid_elevations[el_idx - 1], self.grid_elevations[el_idx]

        t_az = (azimuth - az0) / (az1 - az0 + 1e-12)
        t_el = (elevation - el0) / (el1 - el0 + 1e-12)
        t_az = np.clip(t_az, 0, 1)
        t_el = np.clip(t_el, 0, 1)

        w00 = (1 - t_az) * (1 - t_el)
        w10 = t_az * (1 - t_el)
        w01 = (1 - t_az) * t_el
        w11 = t_az * t_el

        left = (w00 * self.filters_left[az_idx - 1, el_idx - 1] +
                w10 * self.filters_left[az_idx, el_idx - 1] +
                w01 * self.filters_left[az_idx - 1, el_idx] +
                w11 * self.filters_left[az_idx, el_idx])
        right = (w00 * self.filters_right[az_idx - 1, el_idx - 1] +
                 w10 * self.filters_right[az_idx, el_idx - 1] +
                 w01 * self.filters_right[az_idx - 1, el_idx] +
                 w11 * self.filters_right[az_idx, el_idx])
        return left, right
