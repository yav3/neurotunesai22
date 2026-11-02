"""Stereo imaging analysis — width, correlation, balance, mono compatibility."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class StereoAnalysis:
    width: float  # 0 (mono) to 1 (full stereo)
    correlation: float  # -1 to 1 (1=mono, 0=uncorrelated, -1=out of phase)
    balance: float  # -1 (left) to 1 (right)
    mono_compatibility: float  # 0 to 1, energy preserved when summed to mono
    mid_side_ratio_db: float  # dB ratio of mid to side energy


def stereo_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation between left and right channels."""
    n = min(len(left), len(right))
    if n < 2:
        return 1.0
    l, r = left[:n].astype(float), right[:n].astype(float)
    l_mean, r_mean = np.mean(l), np.mean(r)
    num = np.sum((l - l_mean) * (r - r_mean))
    den = np.sqrt(np.sum((l - l_mean) ** 2) * np.sum((r - r_mean) ** 2))
    if den < 1e-30:
        return 1.0
    return float(num / den)


def stereo_width(left: np.ndarray, right: np.ndarray) -> float:
    """Stereo width from mid/side energy ratio. 0=mono, 1=full stereo."""
    n = min(len(left), len(right))
    l, r = left[:n].astype(float), right[:n].astype(float)
    mid = (l + r) / 2.0
    side = (l - r) / 2.0
    mid_energy = np.sum(mid ** 2)
    side_energy = np.sum(side ** 2)
    total = mid_energy + side_energy
    if total < 1e-30:
        return 0.0
    return float(side_energy / total)


def stereo_balance(left: np.ndarray, right: np.ndarray) -> float:
    """Pan balance: -1 (left) to 1 (right)."""
    n = min(len(left), len(right))
    l_rms = np.sqrt(np.mean(left[:n].astype(float) ** 2))
    r_rms = np.sqrt(np.mean(right[:n].astype(float) ** 2))
    total = l_rms + r_rms
    if total < 1e-30:
        return 0.0
    return float((r_rms - l_rms) / total)


def mono_compatibility(left: np.ndarray, right: np.ndarray) -> float:
    """Energy preservation ratio when summed to mono."""
    n = min(len(left), len(right))
    l, r = left[:n].astype(float), right[:n].astype(float)
    stereo_energy = np.sum(l ** 2) + np.sum(r ** 2)
    mono = l + r
    mono_energy = np.sum(mono ** 2) / 2.0
    if stereo_energy < 1e-30:
        return 1.0
    return float(min(1.0, mono_energy / stereo_energy))


def mid_side_ratio_db(left: np.ndarray, right: np.ndarray) -> float:
    """Mid-to-side energy ratio in dB."""
    n = min(len(left), len(right))
    l, r = left[:n].astype(float), right[:n].astype(float)
    mid_energy = np.sum(((l + r) / 2.0) ** 2)
    side_energy = np.sum(((l - r) / 2.0) ** 2)
    if side_energy < 1e-30:
        return 60.0
    if mid_energy < 1e-30:
        return -60.0
    return float(10.0 * np.log10(mid_energy / side_energy))


def analyze_stereo(left: np.ndarray, right: np.ndarray) -> StereoAnalysis:
    return StereoAnalysis(
        width=stereo_width(left, right),
        correlation=stereo_correlation(left, right),
        balance=stereo_balance(left, right),
        mono_compatibility=mono_compatibility(left, right),
        mid_side_ratio_db=mid_side_ratio_db(left, right),
    )


def to_mid_side(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert stereo to mid/side representation."""
    l, r = left.astype(float), right.astype(float)
    return (l + r) / 2.0, (l - r) / 2.0


def from_mid_side(mid: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert mid/side back to left/right."""
    return mid + side, mid - side


def widen_stereo(left: np.ndarray, right: np.ndarray, factor: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Adjust stereo width by scaling the side channel."""
    mid, side = to_mid_side(left, right)
    side_scaled = side * factor
    return from_mid_side(mid, side_scaled)
