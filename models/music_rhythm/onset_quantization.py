"""Onset quantization: grid snapping, quantization strength, humanize,
swing quantize, and groove template alignment.  Pure numpy."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class GridResolution(str, Enum):
    WHOLE = "whole"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"
    SIXTEENTH = "sixteenth"
    THIRTY_SECOND = "thirty_second"
    TRIPLET_QUARTER = "triplet_quarter"
    TRIPLET_EIGHTH = "triplet_eighth"
    TRIPLET_SIXTEENTH = "triplet_sixteenth"


GRID_DIVISIONS: Dict[GridResolution, float] = {
    GridResolution.WHOLE: 1.0,
    GridResolution.HALF: 2.0,
    GridResolution.QUARTER: 4.0,
    GridResolution.EIGHTH: 8.0,
    GridResolution.SIXTEENTH: 16.0,
    GridResolution.THIRTY_SECOND: 32.0,
    GridResolution.TRIPLET_QUARTER: 3.0,
    GridResolution.TRIPLET_EIGHTH: 6.0,
    GridResolution.TRIPLET_SIXTEENTH: 12.0,
}


@dataclass
class QuantizationResult:
    original: np.ndarray
    quantized: np.ndarray
    offsets: np.ndarray
    grid_points: np.ndarray
    strength: float


def build_grid(
    n_beats: int, resolution: GridResolution, tempo_bpm: float = 120.0,
) -> np.ndarray:
    divisions = GRID_DIVISIONS[resolution]
    beat_dur = 60.0 / tempo_bpm
    n_points = int(n_beats * divisions) + 1
    return np.linspace(0, n_beats * beat_dur, n_points)


def snap_to_grid(onsets: np.ndarray, grid: np.ndarray) -> np.ndarray:
    result = np.empty_like(onsets)
    for i, t in enumerate(onsets):
        idx = np.argmin(np.abs(grid - t))
        result[i] = grid[idx]
    return result


def quantize(
    onsets: np.ndarray,
    tempo_bpm: float = 120.0,
    resolution: GridResolution = GridResolution.SIXTEENTH,
    strength: float = 1.0,
    n_beats: int = 0,
) -> QuantizationResult:
    if n_beats <= 0:
        beat_dur = 60.0 / tempo_bpm
        n_beats = int(np.ceil(onsets.max() / beat_dur)) + 1 if len(onsets) > 0 else 4
    grid = build_grid(n_beats, resolution, tempo_bpm)
    snapped = snap_to_grid(onsets, grid)
    quantized = onsets + strength * (snapped - onsets)
    offsets = quantized - onsets
    return QuantizationResult(onsets.copy(), quantized, offsets, grid, strength)


def multi_grid_quantize(
    onsets: np.ndarray,
    tempo_bpm: float = 120.0,
    resolutions: Optional[List[GridResolution]] = None,
    strength: float = 1.0,
) -> QuantizationResult:
    if resolutions is None:
        resolutions = [GridResolution.EIGHTH, GridResolution.TRIPLET_EIGHTH]
    beat_dur = 60.0 / tempo_bpm
    n_beats = int(np.ceil(onsets.max() / beat_dur)) + 1 if len(onsets) > 0 else 4
    grids = [build_grid(n_beats, r, tempo_bpm) for r in resolutions]
    combined = np.unique(np.concatenate(grids))
    snapped = snap_to_grid(onsets, combined)
    quantized = onsets + strength * (snapped - onsets)
    offsets = quantized - onsets
    return QuantizationResult(onsets.copy(), quantized, offsets, combined, strength)


def swing_quantize(
    onsets: np.ndarray,
    tempo_bpm: float = 120.0,
    swing_ratio: float = 0.67,
    resolution: GridResolution = GridResolution.EIGHTH,
) -> np.ndarray:
    beat_dur = 60.0 / tempo_bpm
    divisions = GRID_DIVISIONS[resolution]
    sub_dur = beat_dur * (4.0 / divisions)
    result = np.empty_like(onsets)
    for i, t in enumerate(onsets):
        pair_idx = t / sub_dur
        pair_pos = pair_idx % 2
        pair_start = (int(pair_idx) // 2) * 2 * sub_dur
        if pair_pos < 1.0:
            result[i] = pair_start + pair_pos * sub_dur
        else:
            swing_point = pair_start + swing_ratio * 2 * sub_dur
            frac = pair_pos - 1.0
            result[i] = swing_point + frac * (1.0 - swing_ratio) * 2 * sub_dur
    return result


def humanize(
    onsets: np.ndarray, timing_std_ms: float = 10.0, velocity_std: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    timing_offsets = rng.normal(0, timing_std_ms / 1000.0, len(onsets))
    humanized = onsets + timing_offsets
    humanized = np.maximum(humanized, 0.0)
    vel_offsets = rng.normal(0, velocity_std, len(onsets))
    return humanized, vel_offsets


def quantization_error(original: np.ndarray, quantized: np.ndarray) -> float:
    return float(np.sqrt(np.mean((original - quantized) ** 2)))


def groove_template(onsets: np.ndarray, tempo_bpm: float, n_beats: int = 4) -> np.ndarray:
    beat_dur = 60.0 / tempo_bpm
    bar_dur = n_beats * beat_dur
    phases = np.mod(onsets, bar_dur) / bar_dur
    n_bins = n_beats * 4
    template = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for phase in phases:
        idx = int(phase * n_bins) % n_bins
        counts[idx] += 1
    total = counts.sum()
    if total > 0:
        template = counts / total
    return template


def apply_groove_template(
    onsets: np.ndarray, template: np.ndarray, tempo_bpm: float,
    n_beats: int = 4, strength: float = 0.5,
) -> np.ndarray:
    beat_dur = 60.0 / tempo_bpm
    bar_dur = n_beats * beat_dur
    n_bins = len(template)
    bin_dur = bar_dur / n_bins
    result = onsets.copy()
    for i, t in enumerate(onsets):
        bar_pos = np.mod(t, bar_dur)
        current_bin = int(bar_pos / bin_dur) % n_bins
        target_bin = current_bin
        if current_bin > 0 and template[current_bin - 1] > template[current_bin]:
            target_bin = current_bin - 1
        elif current_bin < n_bins - 1 and template[current_bin + 1] > template[current_bin]:
            target_bin = current_bin + 1
        target_time = t - bar_pos + target_bin * bin_dur + bin_dur / 2
        result[i] = t + strength * (target_time - t)
    return result


def detect_swing(onsets: np.ndarray, tempo_bpm: float) -> float:
    beat_dur = 60.0 / tempo_bpm
    eighth_dur = beat_dur / 2
    offbeat_offsets = []
    for t in onsets:
        beat_phase = np.mod(t, beat_dur)
        if abs(beat_phase) < eighth_dur * 0.3:
            continue
        if abs(beat_phase - beat_dur) < eighth_dur * 0.3:
            continue
        offbeat_offsets.append(beat_phase / beat_dur)
    if not offbeat_offsets:
        return 0.5
    return float(np.mean(offbeat_offsets))


def onset_density(onsets: np.ndarray, window_sec: float = 1.0) -> np.ndarray:
    if len(onsets) == 0:
        return np.array([0.0])
    t_max = onsets.max()
    n_windows = max(1, int(np.ceil(t_max / window_sec)))
    density = np.zeros(n_windows)
    for t in onsets:
        idx = min(int(t / window_sec), n_windows - 1)
        density[idx] += 1
    return density / window_sec


def rhythmic_complexity(onsets: np.ndarray, tempo_bpm: float) -> float:
    if len(onsets) < 2:
        return 0.0
    iois = np.diff(onsets)
    beat_dur = 60.0 / tempo_bpm
    ratios = iois / beat_dur
    deviation_from_grid = np.abs(ratios - np.round(ratios * 4) / 4)
    syncopation = float(np.mean(deviation_from_grid > 0.1))
    density = len(onsets) / max(onsets[-1] - onsets[0], 0.01)
    norm_density = min(density / 8.0, 1.0)
    return float(min(1.0, 0.6 * syncopation + 0.4 * norm_density))
