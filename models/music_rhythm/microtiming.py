"""Expressive microtiming analysis — Friberg & Sundberg (1995), Honing (2001).

Models sub-beat timing deviations that define groove, swing, and
expressive performance style.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence

import numpy as np


# ── Data types ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimingProfile:
    mean_deviation_ms: float
    std_deviation_ms: float
    skewness: float
    per_position: np.ndarray  # mean deviation per metric position


@dataclass(frozen=True)
class SwingAnalysis:
    swing_ratio: float       # long/short ratio (1.0 = straight, 2.0 = triplet)
    consistency: float       # std of per-pair ratios
    pairs: List[Tuple[float, float]]  # (long_ms, short_ms) pairs


@dataclass(frozen=True)
class RubatoSegment:
    start_beat: int
    end_beat: int
    type: str  # "accelerando", "ritardando", "agogic_accent", "steady"
    slope: float  # tempo change rate (BPM/beat), 0 for steady


@dataclass(frozen=True)
class TempoEnvelope:
    beat_times_ms: np.ndarray
    instantaneous_bpm: np.ndarray
    mean_bpm: float
    tempo_variance: float
    rubato_segments: List[RubatoSegment]


# ── Grid quantization ────────────────────────────────────────────────────

def quantize_to_grid(
    onsets_ms: np.ndarray,
    grid_ms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Snap onsets to nearest grid point; return (grid_indices, deviations_ms)."""
    onsets = np.asarray(onsets_ms, dtype=np.float64)
    grid = np.asarray(grid_ms, dtype=np.float64)
    indices = np.empty(len(onsets), dtype=int)
    devs = np.empty(len(onsets), dtype=np.float64)
    for i, t in enumerate(onsets):
        dists = np.abs(grid - t)
        idx = int(np.argmin(dists))
        indices[i] = idx
        devs[i] = t - grid[idx]
    return indices, devs


def make_isochronous_grid(
    start_ms: float,
    period_ms: float,
    n_beats: int,
    subdivisions: int = 1,
) -> np.ndarray:
    """Generate an evenly-spaced grid."""
    step = period_ms / subdivisions
    total = n_beats * subdivisions
    return np.array([start_ms + i * step for i in range(total)])


# ── Timing profile extraction ────────────────────────────────────────────

def timing_profile(
    onsets_ms: np.ndarray,
    grid_ms: np.ndarray,
    n_positions: int = 4,
) -> TimingProfile:
    """Compute per-metric-position timing statistics."""
    indices, devs = quantize_to_grid(onsets_ms, grid_ms)
    positions = indices % n_positions
    per_pos = np.zeros(n_positions)
    for p in range(n_positions):
        mask = positions == p
        if np.any(mask):
            per_pos[p] = float(np.mean(devs[mask]))
    mean_dev = float(np.mean(devs))
    std_dev = float(np.std(devs))
    n = len(devs)
    if n > 2 and std_dev > 0:
        skew = float(np.sum(((devs - mean_dev) / std_dev) ** 3) / n)
    else:
        skew = 0.0
    return TimingProfile(
        mean_deviation_ms=mean_dev,
        std_deviation_ms=std_dev,
        skewness=skew,
        per_position=per_pos,
    )


# ── Swing analysis ───────────────────────────────────────────────────────

def analyze_swing(
    onsets_ms: np.ndarray,
    beat_period_ms: float,
) -> SwingAnalysis:
    """Extract swing ratio from pairs of consecutive eighth notes."""
    onsets = np.asarray(onsets_ms, dtype=np.float64)
    iois = np.diff(onsets)
    pairs: List[Tuple[float, float]] = []
    half_beat = beat_period_ms / 2.0

    i = 0
    while i < len(iois) - 1:
        total = iois[i] + iois[i + 1]
        if abs(total - beat_period_ms) < beat_period_ms * 0.3:
            pairs.append((float(iois[i]), float(iois[i + 1])))
            i += 2
        else:
            i += 1

    if not pairs:
        return SwingAnalysis(swing_ratio=1.0, consistency=0.0, pairs=[])

    ratios = [long / max(short, 1e-6) for long, short in pairs]
    return SwingAnalysis(
        swing_ratio=float(np.mean(ratios)),
        consistency=float(np.std(ratios)),
        pairs=pairs,
    )


def swing_ratio_to_percentage(ratio: float) -> float:
    """Convert long/short ratio to swing percentage (50% = straight, 67% = triplet)."""
    return ratio / (1.0 + ratio) * 100.0


# ── Tempo envelope and rubato ────────────────────────────────────────────

def tempo_envelope(beat_times_ms: np.ndarray) -> TempoEnvelope:
    """Extract instantaneous tempo curve and detect rubato."""
    times = np.asarray(beat_times_ms, dtype=np.float64)
    if len(times) < 2:
        return TempoEnvelope(
            beat_times_ms=times,
            instantaneous_bpm=np.array([120.0]),
            mean_bpm=120.0,
            tempo_variance=0.0,
            rubato_segments=[],
        )

    iois_ms = np.diff(times)
    iois_ms = np.clip(iois_ms, 1.0, None)
    inst_bpm = 60000.0 / iois_ms
    mean_bpm = float(np.mean(inst_bpm))
    var_bpm = float(np.var(inst_bpm))

    segments = _detect_rubato(inst_bpm)

    return TempoEnvelope(
        beat_times_ms=times,
        instantaneous_bpm=inst_bpm,
        mean_bpm=mean_bpm,
        tempo_variance=var_bpm,
        rubato_segments=segments,
    )


def _detect_rubato(
    bpm_curve: np.ndarray,
    window: int = 4,
    threshold_bpm_per_beat: float = 2.0,
) -> List[RubatoSegment]:
    """Segment bpm curve into accel/rit/steady regions."""
    if len(bpm_curve) < window:
        return [RubatoSegment(0, len(bpm_curve) - 1, "steady", 0.0)]

    segments: List[RubatoSegment] = []
    i = 0
    while i < len(bpm_curve) - window + 1:
        seg = bpm_curve[i : i + window]
        slope = float(np.polyfit(np.arange(window), seg, 1)[0])
        if slope > threshold_bpm_per_beat:
            seg_type = "accelerando"
        elif slope < -threshold_bpm_per_beat:
            seg_type = "ritardando"
        else:
            seg_type = "steady"
        segments.append(RubatoSegment(i, i + window - 1, seg_type, slope))
        i += window

    if i < len(bpm_curve):
        segments.append(RubatoSegment(i, len(bpm_curve) - 1, "steady", 0.0))

    return _merge_rubato(segments)


def _merge_rubato(segments: List[RubatoSegment]) -> List[RubatoSegment]:
    """Merge adjacent segments of the same type."""
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.type == merged[-1].type:
            merged[-1] = RubatoSegment(
                merged[-1].start_beat, seg.end_beat,
                seg.type, (merged[-1].slope + seg.slope) / 2.0,
            )
        else:
            merged.append(seg)
    return merged


# ── Microtiming transfer ─────────────────────────────────────────────────

def transfer_microtiming(
    source_onsets_ms: np.ndarray,
    source_grid_ms: np.ndarray,
    target_grid_ms: np.ndarray,
) -> np.ndarray:
    """Apply source performance's timing deviations onto target grid."""
    _, devs = quantize_to_grid(source_onsets_ms, source_grid_ms)
    n_target = len(target_grid_ms)
    if len(devs) == 0:
        return np.array(target_grid_ms, dtype=np.float64)
    tiled_devs = np.resize(devs, n_target)
    return np.asarray(target_grid_ms, dtype=np.float64) + tiled_devs


# ── Groove similarity ────────────────────────────────────────────────────

def groove_similarity(
    profile_a: TimingProfile,
    profile_b: TimingProfile,
) -> float:
    """Cosine similarity between per-position deviation profiles."""
    a = profile_a.per_position
    b = profile_b.per_position
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def agogic_accent_positions(
    onsets_ms: np.ndarray,
    grid_ms: np.ndarray,
    threshold_ms: float = 15.0,
) -> List[int]:
    """Find beats where the performer delays significantly (agogic accent)."""
    _, devs = quantize_to_grid(onsets_ms, grid_ms)
    return [int(i) for i, d in enumerate(devs) if d > threshold_ms]
