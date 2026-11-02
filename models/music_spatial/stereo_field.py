"""Stereo field analysis: panning, width, mid-side, balance."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def pan_law(position: float, law: str = "constant_power") -> tuple[float, float]:
    """Pan position in [-1, 1] to (left_gain, right_gain)."""
    p = (position + 1.0) / 2.0  # [0, 1]
    if law == "linear":
        return 1.0 - p, p
    # constant power
    angle = p * math.pi / 2
    return math.cos(angle), math.sin(angle)


def stereo_width(left: np.ndarray, right: np.ndarray) -> float:
    """Correlation-based stereo width: 1 - |r_LR|. Range [0, 1]."""
    if len(left) < 2:
        return 0.0
    cc = np.corrcoef(left, right)[0, 1]
    if np.isnan(cc):
        return 0.0
    return 1.0 - abs(cc)


def mid_side_encode(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    return mid, side


def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = mid + side
    right = mid - side
    return left, right


def stereo_balance(left: np.ndarray, right: np.ndarray) -> float:
    """Balance in [-1, 1]: -1 = full left, +1 = full right."""
    rms_l = np.sqrt(np.mean(left ** 2) + 1e-12)
    rms_r = np.sqrt(np.mean(right ** 2) + 1e-12)
    return (rms_r - rms_l) / (rms_r + rms_l)


def phantom_center_level(left: np.ndarray, right: np.ndarray) -> float:
    """Perceived center level from correlated content (dB relative to mono sum)."""
    mid, _ = mid_side_encode(left, right)
    rms_mid = np.sqrt(np.mean(mid ** 2) + 1e-12)
    rms_mono = np.sqrt(np.mean(((left + right) / 2) ** 2) + 1e-12)
    return 20 * np.log10(rms_mid / (rms_mono + 1e-12))


@dataclass
class StereoField:
    width: float
    balance: float
    correlation: float
    center_level_db: float

    @classmethod
    def analyze(cls, left: np.ndarray, right: np.ndarray) -> StereoField:
        w = stereo_width(left, right)
        b = stereo_balance(left, right)
        cc = np.corrcoef(left, right)[0, 1] if len(left) >= 2 else 1.0
        if np.isnan(cc):
            cc = 0.0
        cl = phantom_center_level(left, right)
        return cls(width=w, balance=b, correlation=float(cc), center_level_db=float(cl))
