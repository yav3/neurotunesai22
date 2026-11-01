"""Groove analysis: microtiming, swing, syncopation, quantization."""
from __future__ import annotations
import numpy as np


def quantize_onsets(
    onsets: np.ndarray,
    grid_size: float,
    strength: float = 1.0,
) -> np.ndarray:
    """Quantize onset times to nearest grid positions.

    Args:
        onsets: onset times.
        grid_size: grid spacing (same units as onsets).
        strength: 1.0 = full quantization, 0.0 = no change.
    """
    if grid_size <= 0 or len(onsets) == 0:
        return onsets.copy()
    quantized = np.round(onsets / grid_size) * grid_size
    return onsets + strength * (quantized - onsets)


def humanize_grid(
    grid_times: np.ndarray,
    amount: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Add controlled random deviations to grid-quantized times."""
    rng = np.random.RandomState(seed)
    deviations = rng.normal(0, amount, len(grid_times))
    return grid_times + deviations


def microtiming_deviations(
    onsets: np.ndarray,
    grid_size: float,
) -> np.ndarray:
    """Compute deviations of onsets from the nearest grid point.

    Positive = ahead (rushed), negative = behind (dragged).
    """
    if grid_size <= 0 or len(onsets) == 0:
        return np.zeros_like(onsets)
    nearest_grid = np.round(onsets / grid_size) * grid_size
    return onsets - nearest_grid


def swing_ratio(
    onsets: np.ndarray,
    beat_period: float,
) -> float:
    """Estimate swing ratio from onset positions.

    Swing ratio = duration of long subdivision / duration of short subdivision.
    Straight = 1.0 (equal 8ths), typical swing ≈ 1.5–2.0.
    """
    if beat_period <= 0 or len(onsets) < 3:
        return 1.0

    half = beat_period / 2.0
    long_durations = []
    short_durations = []

    for i in range(len(onsets) - 1):
        ioi = onsets[i + 1] - onsets[i]
        if ioi <= 0:
            continue
        # classify as "on-beat to off-beat" or "off-beat to on-beat"
        phase = (onsets[i] % beat_period) / beat_period
        if phase < 0.25 or phase > 0.75:
            long_durations.append(ioi)
        else:
            short_durations.append(ioi)

    if not long_durations or not short_durations:
        return 1.0

    return float(np.median(long_durations) / np.median(short_durations))


def syncopation_index(
    onsets: np.ndarray,
    grid_size: float = 1.0,
    meter: int = 4,
) -> float:
    """Compute syncopation as proportion of onsets on weak beats.

    Higher value = more syncopated (0.0 to 1.0).
    """
    if len(onsets) == 0 or grid_size <= 0:
        return 0.0

    positions = np.round(onsets / grid_size).astype(int)
    beat_positions = positions % meter
    strong_beats = {0, meter // 2} if meter >= 2 else {0}

    n_weak = sum(1 for p in beat_positions if p not in strong_beats)
    return float(n_weak / len(positions))


def groove_template_distance(
    onsets: np.ndarray,
    template: np.ndarray,
    beat_period: float,
) -> float:
    """L2 distance between onset microtiming pattern and a groove template.

    Both are normalized to one beat period.
    """
    if len(onsets) == 0 or len(template) == 0 or beat_period <= 0:
        return float("inf")

    # normalize onsets to phase within beat
    phases = (onsets % beat_period) / beat_period
    # histogram of phases
    n_bins = len(template)
    hist, _ = np.histogram(phases, bins=n_bins, range=(0, 1))
    hist = hist / (np.sum(hist) + 1e-12)
    template_norm = template / (np.sum(template) + 1e-12)
    return float(np.sqrt(np.sum((hist - template_norm) ** 2)))


class GrooveAnalyzer:
    """Comprehensive groove analysis."""

    def __init__(self, grid_size: float = 0.125, meter: int = 4) -> None:
        self.grid_size = grid_size
        self.meter = meter

    def analyze(self, onsets: np.ndarray, beat_period: float) -> dict:
        deviations = microtiming_deviations(onsets, self.grid_size)
        sw = swing_ratio(onsets, beat_period)
        sync = syncopation_index(onsets, self.grid_size, self.meter)
        return {
            "microtiming_mean": float(np.mean(deviations)) if len(deviations) > 0 else 0.0,
            "microtiming_std": float(np.std(deviations)) if len(deviations) > 0 else 0.0,
            "swing_ratio": sw,
            "syncopation": sync,
            "n_onsets": len(onsets),
        }
