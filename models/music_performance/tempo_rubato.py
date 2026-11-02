"""Tempo rubato modeling: timing deviations in expressive performance.

Models micro-timing, phrase-level rubato, and tempo curves. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class RubatoType(Enum):
    AGOGIC = "agogic"
    MELODIC = "melodic"
    STRUCTURAL = "structural"
    CADENTIAL = "cadential"
    PHRASE_FINAL = "phrase_final"


@dataclass
class TempoEvent:
    onset_time: float
    nominal_time: float
    deviation_ms: float
    local_tempo_bpm: float
    rubato_type: RubatoType | None = None


@dataclass
class TempoCurve:
    times: np.ndarray
    tempos: np.ndarray
    mean_tempo: float
    tempo_variability: float
    rubato_events: list[TempoEvent] = field(default_factory=list)


def ioi_to_tempo(ioi_seconds: np.ndarray, beat_duration: float = 1.0) -> np.ndarray:
    """Convert inter-onset intervals to instantaneous tempo (BPM)."""
    safe_ioi = np.maximum(ioi_seconds, 1e-6)
    return 60.0 * beat_duration / safe_ioi


def tempo_curve_from_onsets(
    onsets: np.ndarray, nominal_bpm: float, beats_per_note: float = 1.0,
) -> TempoCurve:
    """Build a tempo curve from performed onset times."""
    if len(onsets) < 2:
        return TempoCurve(onsets, np.array([nominal_bpm]), nominal_bpm, 0.0)

    ioi = np.diff(onsets)
    tempos = ioi_to_tempo(ioi, beats_per_note)
    nominal_ioi = 60.0 * beats_per_note / nominal_bpm

    deviations = np.diff(onsets) - nominal_ioi
    events = []
    for i, (dev, tempo) in enumerate(zip(deviations, tempos)):
        events.append(TempoEvent(
            onset_time=float(onsets[i]),
            nominal_time=float(i * nominal_ioi),
            deviation_ms=float(dev * 1000),
            local_tempo_bpm=float(tempo),
        ))

    return TempoCurve(
        times=onsets[:-1],
        tempos=tempos,
        mean_tempo=float(np.mean(tempos)),
        tempo_variability=float(np.std(tempos)),
        rubato_events=events,
    )


def phrase_rubato(
    onsets: np.ndarray, phrase_boundaries: Sequence[int],
) -> list[float]:
    """Compute per-phrase tempo deviation ratio (performed/nominal duration)."""
    if len(onsets) < 2 or not phrase_boundaries:
        return []

    ratios = []
    boundaries = sorted(set([0] + list(phrase_boundaries) + [len(onsets) - 1]))
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end <= start or end >= len(onsets):
            continue
        performed_dur = onsets[end] - onsets[start]
        n_notes = end - start
        if n_notes > 0:
            avg_ioi = performed_dur / n_notes
            overall_avg = (onsets[-1] - onsets[0]) / (len(onsets) - 1)
            ratios.append(float(avg_ioi / max(overall_avg, 1e-6)))
    return ratios


def cadential_ritardando(
    tempos: np.ndarray, window: int = 4,
) -> float:
    """Measure the degree of ritardando in the final *window* beats. Returns tempo ratio."""
    if len(tempos) < window + 1:
        return 1.0
    final_avg = float(np.mean(tempos[-window:]))
    prior_avg = float(np.mean(tempos[-(2 * window):-window]))
    return final_avg / max(prior_avg, 1e-6)


def timing_profile(
    onsets: np.ndarray, nominal_bpm: float, beats_per_note: float = 1.0,
) -> dict[str, float]:
    """Aggregate timing statistics for a performance."""
    nominal_ioi = 60.0 * beats_per_note / nominal_bpm
    ioi = np.diff(onsets)
    deviations = ioi - nominal_ioi

    return {
        "mean_deviation_ms": float(np.mean(deviations) * 1000),
        "std_deviation_ms": float(np.std(deviations) * 1000),
        "max_early_ms": float(np.min(deviations) * 1000),
        "max_late_ms": float(np.max(deviations) * 1000),
        "tempo_variability_bpm": float(np.std(ioi_to_tempo(ioi, beats_per_note))),
        "tempo_drift_bpm": float(
            np.polyfit(np.arange(len(ioi)), ioi_to_tempo(ioi, beats_per_note), 1)[0]
            * len(ioi)
        ),
    }


def micro_timing_grid(
    onsets: np.ndarray, grid_resolution: float = 0.01,
) -> np.ndarray:
    """Snap onsets to a grid and return deviations from nearest grid point."""
    grid_points = np.round(onsets / grid_resolution) * grid_resolution
    return onsets - grid_points


def swing_ratio(onsets: np.ndarray) -> float:
    """Estimate swing ratio from alternating IOI pairs (long-short pattern)."""
    ioi = np.diff(onsets)
    if len(ioi) < 4:
        return 1.0
    long_ioi = ioi[0::2]
    short_ioi = ioi[1::2]
    n = min(len(long_ioi), len(short_ioi))
    if n == 0:
        return 1.0
    ratio = np.mean(long_ioi[:n]) / max(np.mean(short_ioi[:n]), 1e-6)
    return float(ratio)


def expressive_timing_score(
    onsets: np.ndarray, nominal_bpm: float,
) -> float:
    """Heuristic expressiveness score in [0,1]: moderate variation = high, mechanical or chaotic = low."""
    profile = timing_profile(onsets, nominal_bpm)
    std_ms = abs(profile["std_deviation_ms"])
    if std_ms < 5:
        return 0.2
    elif std_ms < 15:
        return 0.5 + (std_ms - 5) / 20
    elif std_ms < 40:
        return 1.0 - (std_ms - 15) / 50
    else:
        return max(0.1, 0.5 - (std_ms - 40) / 100)
