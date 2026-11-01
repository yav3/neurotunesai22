"""Expressive timing: rubato, swing, groove quantisation, timing deviation analysis.

Models the micro-temporal deviations that distinguish human performance from
mechanical playback: rubato (tempo flexibility), swing feel, groove templates,
timing jitter analysis, and phrase-arc temporal shaping.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class GrooveStyle(str, Enum):
    STRAIGHT = "straight"
    SWING = "swing"
    SHUFFLE = "shuffle"
    BOSSA = "bossa"
    REGGAE = "reggae"
    FUNK = "funk"
    WALTZ = "waltz"


@dataclass
class TimingDeviation:
    onset_ms: float
    grid_ms: float
    deviation_ms: float
    beat_position: float
    is_early: bool


@dataclass
class GrooveTemplate:
    style: GrooveStyle
    subdivisions: int
    offsets_ms: np.ndarray  # per-subdivision offset from grid


SWING_RATIOS = {
    "light":  0.55,
    "medium": 0.60,
    "heavy":  0.67,
    "triplet": 2.0 / 3.0,
}


def quantise_to_grid(onsets_ms: np.ndarray, bpm: float,
                     subdivisions: int = 4) -> np.ndarray:
    """Snap onsets to nearest grid position."""
    beat_ms = 60_000.0 / bpm
    grid_ms = beat_ms / subdivisions
    return np.round(onsets_ms / grid_ms) * grid_ms


def timing_deviations(onsets_ms: np.ndarray, bpm: float,
                      subdivisions: int = 4) -> List[TimingDeviation]:
    """Compute per-onset deviation from the quantised grid."""
    beat_ms = 60_000.0 / bpm
    grid_ms = beat_ms / subdivisions
    quantised = quantise_to_grid(onsets_ms, bpm, subdivisions)
    result = []
    for onset, grid in zip(onsets_ms, quantised):
        dev = onset - grid
        bp = (grid / beat_ms) % 1.0
        result.append(TimingDeviation(
            onset_ms=float(onset), grid_ms=float(grid),
            deviation_ms=float(dev), beat_position=float(bp),
            is_early=dev < 0,
        ))
    return result


def swing_ratio(onsets_ms: np.ndarray, bpm: float) -> float:
    """Estimate swing ratio from eighth-note pairs.

    Returns ratio ∈ [0.5, 0.75]; 0.5 = straight, 2/3 ≈ triplet swing.
    """
    beat_ms = 60_000.0 / bpm
    eighth_ms = beat_ms / 2.0
    pairs: List[Tuple[float, float]] = []
    for i in range(len(onsets_ms) - 1):
        ioi = onsets_ms[i + 1] - onsets_ms[i]
        if 0.3 * eighth_ms < ioi < 1.7 * eighth_ms:
            pairs.append((onsets_ms[i], ioi))
    if len(pairs) < 4:
        return 0.5
    long_short: List[float] = []
    for i in range(0, len(pairs) - 1, 2):
        long_ioi = pairs[i][1]
        short_ioi = pairs[i + 1][1] if i + 1 < len(pairs) else long_ioi
        total = long_ioi + short_ioi
        if total > 0:
            long_short.append(long_ioi / total)
    if not long_short:
        return 0.5
    return float(np.clip(np.median(long_short), 0.5, 0.75))


def rubato_curve(onsets_ms: np.ndarray, bpm: float,
                 window: int = 8) -> np.ndarray:
    """Local tempo curve via windowed IOI inversion.

    Returns array of local BPM values, one per onset (except last).
    """
    if len(onsets_ms) < 2:
        return np.array([bpm])
    ioi = np.diff(onsets_ms)
    ioi = np.maximum(ioi, 1.0)
    local_bpm = 60_000.0 / ioi
    if window > 1 and len(local_bpm) >= window:
        kernel = np.ones(window) / window
        local_bpm = np.convolve(local_bpm, kernel, mode='same')
    return local_bpm


def rubato_strength(onsets_ms: np.ndarray, bpm: float) -> float:
    """Coefficient of variation of local tempo — higher = more rubato."""
    curve = rubato_curve(onsets_ms, bpm)
    if len(curve) < 2:
        return 0.0
    mu = curve.mean()
    if mu < 1e-6:
        return 0.0
    return float(curve.std() / mu)


def groove_template_from_performance(
    onsets_ms: np.ndarray, bpm: float, subdivisions: int = 8,
) -> GrooveTemplate:
    """Extract groove template: average deviation per subdivision slot."""
    beat_ms = 60_000.0 / bpm
    grid_ms = beat_ms / subdivisions
    slots: Dict[int, List[float]] = {i: [] for i in range(subdivisions)}
    for onset in onsets_ms:
        grid_pos = onset / grid_ms
        slot = int(round(grid_pos)) % subdivisions
        dev = onset - round(grid_pos) * grid_ms
        slots[slot].append(dev)
    offsets = np.zeros(subdivisions)
    for slot, devs in slots.items():
        if devs:
            offsets[slot] = float(np.median(devs))
    return GrooveTemplate(
        style=GrooveStyle.STRAIGHT,
        subdivisions=subdivisions,
        offsets_ms=offsets,
    )


def apply_groove(onsets_ms: np.ndarray, template: GrooveTemplate,
                 bpm: float, strength: float = 1.0) -> np.ndarray:
    """Apply a groove template to quantised onsets."""
    beat_ms = 60_000.0 / bpm
    grid_ms = beat_ms / template.subdivisions
    result = onsets_ms.copy()
    for i, onset in enumerate(result):
        slot = int(round(onset / grid_ms)) % template.subdivisions
        result[i] = onset + template.offsets_ms[slot] * strength
    return result


def phrase_timing_arc(n_notes: int, peak_position: float = 0.5,
                      max_deviation_ms: float = 20.0) -> np.ndarray:
    """Generate a phrase-shaped timing deviation arc (ritard at boundaries)."""
    t = np.linspace(0, 1, n_notes)
    arc = -max_deviation_ms * (2 * (t - peak_position)) ** 2 + max_deviation_ms
    return arc


def jitter_rms(onsets_ms: np.ndarray, bpm: float,
               subdivisions: int = 4) -> float:
    """RMS timing jitter relative to grid."""
    devs = timing_deviations(onsets_ms, bpm, subdivisions)
    if not devs:
        return 0.0
    d = np.array([x.deviation_ms for x in devs])
    return float(np.sqrt(np.mean(d ** 2)))


def beat_strength_profile(onsets_ms: np.ndarray, bpm: float,
                          beats_per_bar: int = 4) -> np.ndarray:
    """Fraction of notes landing on each beat of the bar."""
    beat_ms = 60_000.0 / bpm
    bar_ms = beat_ms * beats_per_bar
    counts = np.zeros(beats_per_bar)
    for onset in onsets_ms:
        bar_pos = (onset % bar_ms) / beat_ms
        beat = int(bar_pos) % beats_per_bar
        counts[beat] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def anticipation_ratio(onsets_ms: np.ndarray, bpm: float,
                       subdivisions: int = 4,
                       threshold_ms: float = 10.0) -> float:
    """Fraction of onsets that arrive early (anticipations)."""
    devs = timing_deviations(onsets_ms, bpm, subdivisions)
    if not devs:
        return 0.0
    early = sum(1 for d in devs if d.deviation_ms < -threshold_ms)
    return early / len(devs)
