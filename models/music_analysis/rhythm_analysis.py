"""Rhythm analysis: onset patterns, syncopation, groove, microtiming.

Implements rhythmic feature extraction including inter-onset interval
analysis, syncopation measures, groove quantification, and microtiming
deviation analysis.
"""
from __future__ import annotations


import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class MetricLevel(Enum):
    DOWNBEAT = 0
    BEAT = 1
    SUBDIVISION = 2
    SUB_SUBDIVISION = 3


@dataclass(frozen=True)
class RhythmicEvent:
    onset_time: float
    duration: float
    velocity: float
    metric_position: float
    metric_level: MetricLevel


@dataclass(frozen=True)
class GrooveProfile:
    mean_ioi: float
    ioi_std: float
    swing_ratio: float
    microtiming_deviation_ms: float
    syncopation_index: float
    rhythmic_density: float
    metric_clarity: float


def inter_onset_intervals(onsets: np.ndarray) -> np.ndarray:
    """Compute inter-onset intervals from an array of onset times."""
    if len(onsets) < 2:
        return np.array([], dtype=np.float64)
    sorted_onsets = np.sort(onsets)
    return np.diff(sorted_onsets)


def ioi_statistics(iois: np.ndarray) -> dict[str, float]:
    """Compute IOI statistics: mean, std, cv, npvi."""
    if len(iois) == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "npvi": 0.0}
    mean = float(np.mean(iois))
    std = float(np.std(iois))
    cv = std / mean if mean > 0 else 0.0
    npvi = 0.0
    if len(iois) >= 2:
        pairs = 0.0
        for i in range(len(iois) - 1):
            s = iois[i] + iois[i + 1]
            if s > 0:
                pairs += abs(iois[i] - iois[i + 1]) / (s / 2)
        npvi = 100 * pairs / (len(iois) - 1)
    return {"mean": mean, "std": std, "cv": cv, "npvi": npvi}


def swing_ratio(onsets: np.ndarray, beat_duration: float) -> float:
    """Estimate swing ratio from onset times.

    Swing ratio = duration of long note / duration of short note
    in a swung pair. Perfect swing = 2:1 (ratio 2.0), straight = 1.0.
    """
    if len(onsets) < 3 or beat_duration <= 0:
        return 1.0
    subdivisions = (onsets % beat_duration) / beat_duration
    on_beat = subdivisions[subdivisions < 0.15]
    off_beat = subdivisions[(subdivisions > 0.3) & (subdivisions < 0.85)]
    if len(off_beat) == 0:
        return 1.0
    mean_off = float(np.mean(off_beat))
    long_part = mean_off
    short_part = 1.0 - mean_off
    if short_part < 0.05:
        return 1.0
    return long_part / short_part


def microtiming_deviations(
    onsets: np.ndarray,
    grid_times: np.ndarray,
) -> np.ndarray:
    """Compute deviations of onsets from a quantized grid.

    Returns signed deviations in the same units as onsets.
    """
    deviations = np.empty(len(onsets))
    for i, onset in enumerate(onsets):
        idx = np.argmin(np.abs(grid_times - onset))
        deviations[i] = onset - grid_times[idx]
    return deviations


def syncopation_lhl(
    pattern: np.ndarray,
    meter: list[int],
) -> float:
    """Longuet-Higgins & Lee syncopation measure.

    pattern: binary onset pattern (1=onset, 0=rest)
    meter: metric weight per position (higher = stronger beat)

    Syncopation occurs when an onset on a weak beat is followed by a
    rest on a stronger beat.
    """
    n = len(pattern)
    if n < 2 or len(meter) < n:
        return 0.0
    syncopation = 0.0
    for i in range(n - 1):
        if pattern[i] == 1 and pattern[i + 1] == 0:
            weight_diff = meter[i + 1] - meter[i]
            if weight_diff > 0:
                syncopation += weight_diff
    return syncopation


def metric_weight_pattern(beats_per_bar: int, subdivisions: int = 2) -> list[int]:
    """Generate a hierarchical metric weight pattern.

    Returns weight per subdivision position. Higher = metrically stronger.
    """
    n = beats_per_bar * subdivisions
    weights = [0] * n
    weights[0] = 4
    if beats_per_bar >= 2:
        half = beats_per_bar // 2
        weights[half * subdivisions] = 3
    for b in range(beats_per_bar):
        pos = b * subdivisions
        if weights[pos] == 0:
            weights[pos] = 2
    for i in range(n):
        if weights[i] == 0:
            weights[i] = 1
    return weights


def rhythmic_density(onsets: np.ndarray, duration: float) -> float:
    """Onsets per second."""
    if duration <= 0:
        return 0.0
    return len(onsets) / duration


def onset_entropy(quantized_positions: np.ndarray, n_bins: int) -> float:
    """Shannon entropy of onset distribution across metric positions."""
    counts = np.zeros(n_bins)
    for pos in quantized_positions:
        idx = int(pos) % n_bins
        counts[idx] += 1
    total = np.sum(counts)
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def metric_clarity(onsets: np.ndarray, tempo_bpm: float, sr: float = 1.0) -> float:
    """Strength of the metric pulse, measured by autocorrelation peak ratio.

    Returns ratio of the peak at the beat period to the mean autocorrelation.
    """
    if len(onsets) < 4 or tempo_bpm <= 0:
        return 0.0
    beat_period = 60.0 / tempo_bpm
    iois = inter_onset_intervals(onsets)
    if len(iois) == 0:
        return 0.0
    max_lag = int(beat_period * 4 / np.mean(iois)) if np.mean(iois) > 0 else 1
    max_lag = min(max_lag, len(iois))
    if max_lag < 2:
        return 0.0
    signal = np.zeros(max_lag * 2)
    cum_ioi = 0.0
    for ioi in iois:
        cum_ioi += ioi
        idx = int(cum_ioi / np.mean(iois))
        if 0 <= idx < len(signal):
            signal[idx] = 1.0
    acf = np.correlate(signal, signal, mode="full")
    acf = acf[len(acf) // 2:]
    if len(acf) < 2 or acf[0] < 1e-12:
        return 0.0
    acf = acf / acf[0]
    beat_lag = int(beat_period / np.mean(iois)) if np.mean(iois) > 0 else 1
    if beat_lag < 1 or beat_lag >= len(acf):
        return 0.0
    return float(acf[beat_lag])


def compute_groove_profile(
    onsets: np.ndarray,
    grid_times: np.ndarray,
    beat_duration: float,
    tempo_bpm: float = 120.0,
    total_duration: float | None = None,
) -> GrooveProfile:
    """Compute a full groove profile from onset times and a reference grid."""
    iois = inter_onset_intervals(onsets)
    stats = ioi_statistics(iois)
    sr = swing_ratio(onsets, beat_duration)
    devs = microtiming_deviations(onsets, grid_times)
    mt_dev = float(np.std(devs) * 1000) if len(devs) > 0 else 0.0
    meter = metric_weight_pattern(4, 4)
    pattern = np.zeros(len(meter))
    for onset in onsets:
        pos = int((onset % beat_duration) / beat_duration * len(meter)) % len(meter)
        pattern[pos] = 1
    sync = syncopation_lhl(pattern, meter)
    dur = total_duration if total_duration else (float(np.max(onsets) - np.min(onsets)) if len(onsets) > 1 else 1.0)
    density = rhythmic_density(onsets, dur)
    clarity = metric_clarity(onsets, tempo_bpm)
    return GrooveProfile(
        mean_ioi=stats["mean"],
        ioi_std=stats["std"],
        swing_ratio=sr,
        microtiming_deviation_ms=mt_dev,
        syncopation_index=sync,
        rhythmic_density=density,
        metric_clarity=clarity,
    )
