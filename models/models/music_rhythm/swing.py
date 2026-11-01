"""Swing feel quantification and application.

Models the timing deviations that create swing, shuffle, and groove feels,
with detection and synthesis of swing ratios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SwingProfile:
    ratio: float  # 1.0 = straight, 2.0 = triplet swing, 1.5 = light swing
    depth: float  # 0.0-1.0, how consistently swing is applied
    style: str  # 'jazz', 'blues', 'funk', 'straight', etc.


SWING_TEMPLATES = {
    "straight": 1.0,
    "light": 1.25,
    "medium": 1.5,
    "jazz": 1.67,
    "triplet": 2.0,
    "blues": 1.75,
    "funk": 1.3,
    "bossa": 1.15,
}


def swing_ratio_to_offset(ratio: float, beat_duration: float) -> float:
    """Convert swing ratio to the timing offset applied to the off-beat.
    ratio=1.0 → offset=0 (straight); ratio=2.0 → triplet swing."""
    if ratio <= 0:
        return 0.0
    half = beat_duration / 2.0
    long = beat_duration * ratio / (1.0 + ratio)
    return long - half


def apply_swing(
    onsets: np.ndarray,
    ratio: float,
    beat_duration: float,
    *,
    depth: float = 1.0,
) -> np.ndarray:
    """Shift off-beat onsets according to swing ratio.
    depth in [0,1] controls how much of the full swing offset is applied."""
    if ratio == 1.0 or depth == 0.0:
        return onsets.copy()
    offset = swing_ratio_to_offset(ratio, beat_duration)
    result = onsets.copy().astype(float)
    for i in range(len(result)):
        beat_phase = (result[i] % beat_duration) / beat_duration
        if 0.4 < beat_phase < 0.6:
            result[i] += offset * depth
    return result


def detect_swing_ratio(
    onsets: np.ndarray,
    tempo_bpm: float,
    beats_per_measure: int = 4,
) -> SwingProfile:
    """Estimate swing ratio from performed timing by comparing long/short
    eighth-note pairs."""
    beat_dur = 60.0 / tempo_bpm
    eighth = beat_dur / 2.0
    if len(onsets) < 3:
        return SwingProfile(ratio=1.0, depth=0.0, style="straight")

    iois = np.diff(onsets)
    pairs: list[float] = []
    i = 0
    while i < len(iois) - 1:
        total = iois[i] + iois[i + 1]
        if abs(total - beat_dur) < beat_dur * 0.3:
            if iois[i] > 0 and iois[i + 1] > 0:
                pairs.append(iois[i] / iois[i + 1])
            i += 2
        else:
            i += 1

    if not pairs:
        return SwingProfile(ratio=1.0, depth=0.0, style="straight")

    ratios = np.array(pairs)
    mean_ratio = float(np.median(ratios))
    consistency = 1.0 - float(np.std(ratios) / (np.mean(ratios) + 1e-9))
    consistency = max(0.0, min(1.0, consistency))

    style = "straight"
    if mean_ratio > 1.6:
        style = "jazz"
    elif mean_ratio > 1.4:
        style = "blues"
    elif mean_ratio > 1.2:
        style = "light"
    elif mean_ratio > 1.05:
        style = "funk"

    return SwingProfile(ratio=mean_ratio, depth=consistency, style=style)


def shuffle_feel(
    onsets: np.ndarray, intensity: float, beat_duration: float
) -> np.ndarray:
    """Apply shuffle (triplet-based swing) with variable intensity [0,1]."""
    triplet_ratio = 2.0
    effective_ratio = 1.0 + (triplet_ratio - 1.0) * intensity
    return apply_swing(onsets, effective_ratio, beat_duration, depth=1.0)


def swing_template(style: str) -> float:
    """Return swing ratio for a named style."""
    return SWING_TEMPLATES.get(style.lower(), 1.0)


def microtiming_deviation(
    performed: np.ndarray, quantized: np.ndarray
) -> np.ndarray:
    """Compute per-note timing deviation (performed - quantized) in seconds."""
    n = min(len(performed), len(quantized))
    return performed[:n] - quantized[:n]


def timing_profile(
    deviations: np.ndarray,
) -> dict:
    """Summarize microtiming deviations."""
    if len(deviations) == 0:
        return {"mean": 0.0, "std": 0.0, "range": 0.0, "ahead_pct": 0.0}
    return {
        "mean": float(np.mean(deviations)),
        "std": float(np.std(deviations)),
        "range": float(np.max(deviations) - np.min(deviations)),
        "ahead_pct": float(np.mean(deviations < 0) * 100),
    }


def behind_the_beat(onsets: np.ndarray, grid: np.ndarray, threshold_ms: float = 10.0) -> np.ndarray:
    """Identify notes that are consistently behind the beat grid."""
    devs = microtiming_deviation(onsets, grid)
    return np.where(devs > threshold_ms / 1000.0)[0]


def ahead_of_beat(onsets: np.ndarray, grid: np.ndarray, threshold_ms: float = 10.0) -> np.ndarray:
    """Identify notes that are consistently ahead of the beat grid."""
    devs = microtiming_deviation(onsets, grid)
    return np.where(devs < -threshold_ms / 1000.0)[0]
