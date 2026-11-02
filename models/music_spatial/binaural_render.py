"""Binaural rendering pipeline for spatial audio."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SourcePosition:
    """A sound source in spherical coordinates (degrees)."""
    azimuth: float  # -180 to 180
    elevation: float = 0.0  # -90 to 90
    distance: float = 1.0  # metres

    @property
    def azimuth_rad(self) -> float:
        return math.radians(self.azimuth)

    @property
    def elevation_rad(self) -> float:
        return math.radians(self.elevation)


@dataclass
class BinauralOutput:
    """Stereo binaural output."""
    left: np.ndarray
    right: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.left) / self.sample_rate

    @property
    def ild_db(self) -> float:
        """Interaural level difference in dB (RMS-based)."""
        rms_l = float(np.sqrt(np.mean(self.left ** 2))) + 1e-12
        rms_r = float(np.sqrt(np.mean(self.right ** 2))) + 1e-12
        return 20.0 * math.log10(rms_l / rms_r)


def itd_samples(azimuth_deg: float, sample_rate: int, head_radius: float = 0.0875) -> int:
    """Interaural time difference in samples (Woodworth formula)."""
    c = 343.0  # m/s
    theta = math.radians(azimuth_deg)
    itd_s = (head_radius / c) * (math.sin(abs(theta)) + abs(theta))
    return int(round(itd_s * sample_rate))


def ild_factor(azimuth_deg: float) -> tuple[float, float]:
    """Simple ILD gain factors (left, right) based on azimuth."""
    theta = math.radians(azimuth_deg)
    # panning law: cos/sin crossfade
    angle_norm = (theta + math.pi / 2) / math.pi  # 0 (full left) to 1 (full right)
    angle_norm = max(0.0, min(1.0, angle_norm))
    gain_l = math.cos(angle_norm * math.pi / 2)
    gain_r = math.sin(angle_norm * math.pi / 2)
    return (gain_l, gain_r)


def distance_attenuation(distance: float, reference: float = 1.0) -> float:
    """Inverse-distance attenuation factor."""
    if distance <= 0:
        return 1.0
    return reference / max(distance, reference)


def render_binaural(
    mono: np.ndarray,
    position: SourcePosition,
    sample_rate: int,
) -> BinauralOutput:
    """Render a mono source to binaural stereo given its position."""
    n = len(mono)
    # ITD
    delay = itd_samples(position.azimuth, sample_rate)
    # ILD
    gl, gr = ild_factor(position.azimuth)
    # distance
    atten = distance_attenuation(position.distance)

    left = np.zeros(n + abs(delay), dtype=np.float64)
    right = np.zeros(n + abs(delay), dtype=np.float64)

    if delay >= 0:
        # sound arrives at left ear first (source on left)
        left[:n] = mono * gl * atten
        right[delay:delay + n] = mono * gr * atten
    else:
        d = abs(delay)
        left[d:d + n] = mono * gl * atten
        right[:n] = mono * gr * atten

    return BinauralOutput(left=left, right=right, sample_rate=sample_rate)


def mix_binaural(outputs: list[BinauralOutput]) -> BinauralOutput:
    """Sum multiple binaural outputs into one mix."""
    if not outputs:
        raise ValueError("Need at least one output to mix")
    sr = outputs[0].sample_rate
    max_len = max(len(o.left) for o in outputs)
    left = np.zeros(max_len, dtype=np.float64)
    right = np.zeros(max_len, dtype=np.float64)
    for o in outputs:
        left[:len(o.left)] += o.left
        right[:len(o.right)] += o.right
    return BinauralOutput(left=left, right=right, sample_rate=sr)


def crossfeed(output: BinauralOutput, amount: float = 0.2) -> BinauralOutput:
    """Apply crossfeed blending between L/R channels."""
    amount = max(0.0, min(1.0, amount))
    new_l = output.left * (1 - amount) + output.right * amount
    new_r = output.right * (1 - amount) + output.left * amount
    return BinauralOutput(left=new_l, right=new_r, sample_rate=output.sample_rate)


def perceived_width(output: BinauralOutput) -> float:
    """Estimate stereo width as 1 - |correlation(L,R)|."""
    if len(output.left) < 2:
        return 0.0
    corr = float(np.corrcoef(output.left[:len(output.right)], output.right[:len(output.left)])[0, 1])
    if np.isnan(corr):
        return 0.0
    return 1.0 - abs(corr)
