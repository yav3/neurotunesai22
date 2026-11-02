"""Spatial mixing for music production — panning, stereo imaging, immersive mix placement."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class MixElement:
    name: str
    pan: float = 0.0
    level_db: float = 0.0
    width: float = 0.0
    depth: float = 0.5
    height: float = 0.0
    frequency_range: Tuple[float, float] = (20.0, 20000.0)


@dataclass
class StereoImage:
    correlation: float = 1.0
    width: float = 0.0
    balance: float = 0.0
    center_level: float = 0.0
    side_level: float = -float('inf')


@dataclass
class MixAnalysis:
    elements: List[MixElement]
    frequency_balance: np.ndarray
    stereo_image: StereoImage
    headroom_db: float = 0.0
    crest_factor: float = 0.0


def equal_power_pan(angle: float) -> Tuple[float, float]:
    theta = (angle + 1.0) * math.pi / 4.0
    return (math.cos(theta), math.sin(theta))


def linear_pan(angle: float) -> Tuple[float, float]:
    r = (angle + 1.0) / 2.0
    return (1.0 - r, r)


def pan_to_stereo(
    mono: np.ndarray,
    pan: float,
    law: str = "equal_power",
) -> np.ndarray:
    if law == "equal_power":
        l_gain, r_gain = equal_power_pan(pan)
    else:
        l_gain, r_gain = linear_pan(pan)
    stereo = np.zeros((len(mono), 2))
    stereo[:, 0] = mono * l_gain
    stereo[:, 1] = mono * r_gain
    return stereo


def stereo_width(stereo: np.ndarray, target_width: float) -> np.ndarray:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        return stereo
    mid = (stereo[:, 0] + stereo[:, 1]) / 2.0
    side = (stereo[:, 0] - stereo[:, 1]) / 2.0
    w = np.clip(target_width, 0.0, 2.0)
    out = np.zeros_like(stereo)
    out[:, 0] = mid * (2.0 - w) / 2.0 + side * w / 2.0
    out[:, 1] = mid * (2.0 - w) / 2.0 - side * w / 2.0
    return out


def mid_side_encode(stereo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mid = (stereo[:, 0] + stereo[:, 1]) / 2.0
    side = (stereo[:, 0] - stereo[:, 1]) / 2.0
    return mid, side


def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    stereo = np.zeros((len(mid), 2))
    stereo[:, 0] = mid + side
    stereo[:, 1] = mid - side
    return stereo


def stereo_correlation(stereo: np.ndarray) -> float:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        return 1.0
    l, r = stereo[:, 0], stereo[:, 1]
    norm_l = np.linalg.norm(l)
    norm_r = np.linalg.norm(r)
    if norm_l < 1e-12 or norm_r < 1e-12:
        return 0.0
    return float(np.dot(l, r) / (norm_l * norm_r))


def balance_meter(stereo: np.ndarray) -> float:
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        return 0.0
    l_rms = np.sqrt(np.mean(stereo[:, 0] ** 2))
    r_rms = np.sqrt(np.mean(stereo[:, 1] ** 2))
    total = l_rms + r_rms
    if total < 1e-12:
        return 0.0
    return float((r_rms - l_rms) / total)


def depth_via_predelay(
    signal: np.ndarray,
    depth: float,
    sample_rate: float = 44100.0,
    max_predelay_ms: float = 80.0,
) -> np.ndarray:
    predelay_ms = depth * max_predelay_ms
    predelay_samples = int(predelay_ms * sample_rate / 1000.0)
    if predelay_samples <= 0:
        return signal.copy()
    result = np.zeros(len(signal) + predelay_samples)
    result[predelay_samples:predelay_samples + len(signal)] = signal
    return result[:len(signal)]


def frequency_masking_score(
    elements: List[MixElement],
) -> np.ndarray:
    n = len(elements)
    masking = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lo_i, hi_i = elements[i].frequency_range
            lo_j, hi_j = elements[j].frequency_range
            overlap_lo = max(lo_i, lo_j)
            overlap_hi = min(hi_i, hi_j)
            if overlap_hi <= overlap_lo:
                continue
            range_i = max(hi_i - lo_i, 1.0)
            overlap_ratio = (overlap_hi - overlap_lo) / range_i
            pan_distance = abs(elements[i].pan - elements[j].pan)
            spatial_relief = min(pan_distance / 2.0, 1.0)
            masking[i, j] = overlap_ratio * (1.0 - 0.5 * spatial_relief)
    return masking


def mix_headroom(
    signals: List[np.ndarray],
    levels_db: List[float],
) -> float:
    if not signals:
        return float('inf')
    max_len = max(len(s) for s in signals)
    mix = np.zeros(max_len)
    for sig, db in zip(signals, levels_db):
        gain = 10 ** (db / 20.0)
        mix[:len(sig)] += sig * gain
    peak = np.max(np.abs(mix))
    if peak < 1e-12:
        return float('inf')
    return float(-20.0 * math.log10(peak))


def spatial_density(elements: List[MixElement]) -> float:
    if not elements:
        return 0.0
    n_bins = 20
    bins = np.zeros(n_bins)
    for e in elements:
        idx = int((e.pan + 1.0) / 2.0 * (n_bins - 1))
        idx = max(0, min(n_bins - 1, idx))
        bins[idx] += 1
    occupied = np.sum(bins > 0)
    return float(occupied / n_bins)


def haas_delay(
    stereo: np.ndarray,
    delay_ms: float,
    sample_rate: float = 44100.0,
    side: str = "right",
) -> np.ndarray:
    delay_samples = int(delay_ms * sample_rate / 1000.0)
    result = stereo.copy()
    ch = 1 if side == "right" else 0
    if delay_samples > 0 and delay_samples < len(stereo):
        result[delay_samples:, ch] = stereo[:-delay_samples, ch]
        result[:delay_samples, ch] = 0.0
    return result


def loudness_balance(
    band_levels: Dict[str, float],
) -> Dict[str, float]:
    from typing import Dict
    target = {"low": -6.0, "low_mid": -3.0, "mid": 0.0, "high_mid": -2.0, "high": -5.0}
    deviations = {}
    for band, level in band_levels.items():
        t = target.get(band, 0.0)
        deviations[band] = level - t
    return deviations


def panorama_score(elements: List[MixElement]) -> float:
    if not elements:
        return 0.0
    pans = [e.pan for e in elements]
    spread = max(pans) - min(pans)
    symmetry = 1.0 - abs(np.mean(pans))
    return float(0.6 * spread / 2.0 + 0.4 * symmetry)
