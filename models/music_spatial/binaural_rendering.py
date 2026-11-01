"""Binaural audio rendering with analytical HRTF modeling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEAD_RADIUS_M = 0.0875
SPEED_OF_SOUND = 343.0


@dataclass
class HRTFCoefficients:
    left_delay_samples: float
    right_delay_samples: float
    left_gain: float
    right_gain: float
    left_filter: np.ndarray
    right_filter: np.ndarray


def woodworth_itd(azimuth_rad: float, head_radius_m: float = HEAD_RADIUS_M, sr: int = 44100) -> tuple[float, float]:
    if abs(azimuth_rad) <= np.pi / 2:
        itd_s = (head_radius_m / SPEED_OF_SOUND) * (azimuth_rad + np.sin(azimuth_rad))
    else:
        sign = 1.0 if azimuth_rad > 0 else -1.0
        itd_s = sign * (head_radius_m / SPEED_OF_SOUND) * (np.pi / 2 + 1.0)
    itd_samples = itd_s * sr
    left_delay = max(0.0, -itd_samples)
    right_delay = max(0.0, itd_samples)
    return left_delay, right_delay


def ild_model(azimuth_rad: float, frequency_hz: float, head_radius_m: float = HEAD_RADIUS_M) -> float:
    ka = 2.0 * np.pi * frequency_hz * head_radius_m / SPEED_OF_SOUND
    if ka < 0.1:
        return 0.0
    shadow = min(ka * abs(np.sin(azimuth_rad)), 10.0)
    ild_db = 1.5 * shadow
    return float(ild_db) if azimuth_rad >= 0 else float(-ild_db)


def simple_hrtf(
    azimuth_rad: float, elevation_rad: float = 0.0,
    sr: int = 44100, head_radius_m: float = HEAD_RADIUS_M,
) -> HRTFCoefficients:
    left_delay, right_delay = woodworth_itd(azimuth_rad, head_radius_m, sr)
    ild_1k = ild_model(azimuth_rad, 1000.0, head_radius_m)
    left_gain = 10.0 ** (-max(0.0, ild_1k) / 20.0)
    right_gain = 10.0 ** (-max(0.0, -ild_1k) / 20.0)
    elev_atten = np.cos(elevation_rad) * 0.3 + 0.7
    left_gain *= elev_atten
    right_gain *= elev_atten
    n_taps = 32
    left_filt = np.zeros(n_taps)
    right_filt = np.zeros(n_taps)
    left_filt[0] = 1.0
    right_filt[0] = 1.0
    shadow_side = 1 if azimuth_rad >= 0 else 0
    for i in range(1, n_taps):
        decay = np.exp(-0.3 * i)
        left_filt[i] = decay * (0.05 if shadow_side == 0 else -0.02)
        right_filt[i] = decay * (0.05 if shadow_side == 1 else -0.02)
    return HRTFCoefficients(
        left_delay_samples=left_delay, right_delay_samples=right_delay,
        left_gain=left_gain, right_gain=right_gain,
        left_filter=left_filt, right_filter=right_filt,
    )


def _apply_fractional_delay(signal: np.ndarray, delay: float) -> np.ndarray:
    n = len(signal)
    int_delay = int(np.floor(delay))
    frac = delay - int_delay
    out = np.zeros(n)
    for i in range(int_delay + 1, n):
        out[i] = signal[i - int_delay] * (1.0 - frac) + signal[i - int_delay - 1] * frac
    return out


def render_binaural(
    mono: np.ndarray, azimuth_rad: float, elevation_rad: float = 0.0,
    distance_m: float = 1.0, sr: int = 44100,
) -> tuple[np.ndarray, np.ndarray]:
    hrtf = simple_hrtf(azimuth_rad, elevation_rad, sr)
    dist_atten = 1.0 / max(distance_m, 0.1)
    left = _apply_fractional_delay(mono, hrtf.left_delay_samples) * hrtf.left_gain * dist_atten
    right = _apply_fractional_delay(mono, hrtf.right_delay_samples) * hrtf.right_gain * dist_atten
    left = np.convolve(left, hrtf.left_filter, mode="same")
    right = np.convolve(right, hrtf.right_filter, mode="same")
    return left, right


def render_binaural_moving(
    mono: np.ndarray, azimuth_trajectory: np.ndarray,
    sr: int = 44100, block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(mono)
    left_out = np.zeros(n)
    right_out = np.zeros(n)
    n_blocks = (n + block_size - 1) // block_size
    traj_interp = np.interp(
        np.arange(n_blocks), np.linspace(0, n_blocks - 1, len(azimuth_trajectory)), azimuth_trajectory,
    )
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = mono[start:end]
        az = float(traj_interp[b])
        l, r = render_binaural(block, az, 0.0, 1.0, sr)
        left_out[start:end] = l[:end - start]
        right_out[start:end] = r[:end - start]
    return left_out, right_out


def ambisonics_to_binaural(
    W: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    sr: int = 44100,
) -> tuple[np.ndarray, np.ndarray]:
    left = 0.5 * W + 0.5 * X + 0.5 * Y + 0.15 * Z
    right = 0.5 * W + 0.5 * X - 0.5 * Y + 0.15 * Z
    itd_samples = HEAD_RADIUS_M / SPEED_OF_SOUND * sr * 0.5
    left = _apply_fractional_delay(left, itd_samples * 0.3)
    right = _apply_fractional_delay(right, itd_samples * 0.3)
    return left, right


def crossfeed(
    left: np.ndarray, right: np.ndarray,
    amount: float = 0.3, delay_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(left), len(right))
    l_out = np.copy(left[:n])
    r_out = np.copy(right[:n])
    for i in range(delay_samples, n):
        l_out[i] += amount * right[i - delay_samples]
        r_out[i] += amount * left[i - delay_samples]
    norm = 1.0 / (1.0 + amount)
    return l_out * norm, r_out * norm
