"""Binaural rendering: synthesise 3D audio for headphone playback.

Implement HRTF interpolation, ITD/ILD computation, distance attenuation,
head-tracking compensation, and crossfeed for stereo-to-binaural conversion.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SourcePosition:
    azimuth_deg: float
    elevation_deg: float
    distance_m: float = 1.0


@dataclass
class BinauralConfig:
    head_radius_m: float = 0.0875
    sr: int = 44100
    speed_of_sound: float = 343.0


def itd_woodworth(azimuth_rad: float, head_radius: float = 0.0875,
                  c: float = 343.0) -> float:
    if abs(azimuth_rad) <= np.pi / 2:
        return head_radius * (azimuth_rad + np.sin(azimuth_rad)) / c
    sign = 1.0 if azimuth_rad > 0 else -1.0
    return sign * head_radius * (np.pi / 2 + 1.0) / c


def ild_simple(azimuth_deg: float, frequency_hz: float) -> float:
    alpha = min(frequency_hz / 1000.0, 6.0)
    return alpha * np.sin(np.radians(azimuth_deg))


def distance_attenuation(distance_m: float, ref_distance_m: float = 1.0) -> float:
    return ref_distance_m / max(distance_m, 0.01)


def air_absorption(distance_m: float, frequency_hz: float,
                   humidity_pct: float = 50.0) -> float:
    alpha = 0.01 * (frequency_hz / 1000.0) ** 1.5 * (1.0 - humidity_pct / 100.0)
    return float(np.exp(-alpha * distance_m))


def hrtf_delay_samples(azimuth_deg: float, config: BinauralConfig) -> Tuple[int, int]:
    azimuth_rad = np.radians(azimuth_deg)
    itd = itd_woodworth(azimuth_rad, config.head_radius_m, config.speed_of_sound)
    delay_samples = int(abs(itd) * config.sr)
    if itd >= 0:
        return 0, delay_samples
    return delay_samples, 0


def apply_itd(signal: np.ndarray, delay_left: int,
              delay_right: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal) + max(delay_left, delay_right)
    left = np.zeros(n)
    right = np.zeros(n)
    left[delay_left:delay_left + len(signal)] = signal
    right[delay_right:delay_right + len(signal)] = signal
    return left[:len(signal)], right[:len(signal)]


def apply_ild(left: np.ndarray, right: np.ndarray,
              ild_db: float) -> Tuple[np.ndarray, np.ndarray]:
    gain = 10.0 ** (abs(ild_db) / 20.0)
    if ild_db > 0:
        return left / gain, right * gain
    return left * gain, right / gain


def spatialise_mono(signal: np.ndarray, position: SourcePosition,
                    config: BinauralConfig) -> Tuple[np.ndarray, np.ndarray]:
    dl, dr = hrtf_delay_samples(position.azimuth_deg, config)
    left, right = apply_itd(signal, dl, dr)
    freq_est = 1000.0
    ild_val = ild_simple(position.azimuth_deg, freq_est)
    left, right = apply_ild(left, right, ild_val)
    atten = distance_attenuation(position.distance_m)
    return left * atten, right * atten


def crossfeed(left: np.ndarray, right: np.ndarray,
              amount: float = 0.3, delay_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    n = len(left)
    out_l = left.copy()
    out_r = right.copy()
    for i in range(delay_samples, n):
        out_l[i] += amount * right[i - delay_samples]
        out_r[i] += amount * left[i - delay_samples]
    return out_l, out_r


def stereo_width(left: np.ndarray, right: np.ndarray,
                 width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    mid = (left + right) / 2
    side = (left - right) / 2
    return mid + width * side, mid - width * side


def multi_source_render(signals: List[np.ndarray],
                        positions: List[SourcePosition],
                        config: BinauralConfig) -> Tuple[np.ndarray, np.ndarray]:
    max_len = max(len(s) for s in signals)
    mix_l = np.zeros(max_len)
    mix_r = np.zeros(max_len)
    for sig, pos in zip(signals, positions):
        l, r = spatialise_mono(sig, pos, config)
        mix_l[:len(l)] += l
        mix_r[:len(r)] += r
    peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-10)
    return mix_l / peak, mix_r / peak
