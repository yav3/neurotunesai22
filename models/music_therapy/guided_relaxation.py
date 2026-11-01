"""Guided relaxation protocols using music parameters.

Progressive muscle relaxation timing, iso-principle tempo curves,
breath-synchronised phrase lengths, and tension-release arc design.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RelaxationPhase(Enum):
    INDUCTION = "induction"
    DEEPENING = "deepening"
    PLATEAU = "plateau"
    EMERGENCE = "emergence"


@dataclass
class RelaxationSegment:
    phase: RelaxationPhase
    duration_s: float
    target_bpm: float
    target_loudness_db: float
    harmonic_complexity: float  # 0-1, lower = simpler/more consonant


@dataclass
class RelaxationProtocol:
    segments: List[RelaxationSegment]
    total_duration_s: float
    breath_rate_bpm: float = 6.0  # resonant frequency ~0.1 Hz


def iso_principle_tempo_curve(
    initial_bpm: float,
    target_bpm: float,
    duration_s: float,
    sr_hz: float = 1.0,
    curve: str = "exponential",
) -> np.ndarray:
    """Generate tempo curve following the iso principle (match-then-lead)."""
    n = int(duration_s * sr_hz)
    t = np.linspace(0, 1, n)
    if curve == "exponential":
        values = initial_bpm * (target_bpm / initial_bpm) ** t
    elif curve == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
        values = initial_bpm + (target_bpm - initial_bpm) * s
    else:  # linear
        values = initial_bpm + (target_bpm - initial_bpm) * t
    return values


def breath_sync_phrase_lengths(
    breath_rate_bpm: float = 6.0,
    inhale_ratio: float = 0.4,
    n_phrases: int = 8,
) -> np.ndarray:
    """Compute phrase lengths synchronised to breathing cycle.

    Returns array of phrase durations in seconds.
    """
    cycle_s = 60.0 / breath_rate_bpm
    inhale_s = cycle_s * inhale_ratio
    exhale_s = cycle_s * (1.0 - inhale_ratio)
    lengths = np.empty(n_phrases)
    for i in range(n_phrases):
        lengths[i] = inhale_s if i % 2 == 0 else exhale_s
    return lengths


def progressive_relaxation_schedule(
    total_duration_s: float = 1200.0,
    initial_bpm: float = 80.0,
    target_bpm: float = 60.0,
    muscle_groups: int = 8,
) -> RelaxationProtocol:
    """Build a full progressive relaxation protocol."""
    induction = total_duration_s * 0.15
    deepening = total_duration_s * 0.55
    plateau = total_duration_s * 0.20
    emergence = total_duration_s * 0.10

    mid_bpm = (initial_bpm + target_bpm) / 2
    segments = [
        RelaxationSegment(RelaxationPhase.INDUCTION, induction, initial_bpm, -20, 0.6),
        RelaxationSegment(RelaxationPhase.DEEPENING, deepening, mid_bpm, -30, 0.4),
        RelaxationSegment(RelaxationPhase.PLATEAU, plateau, target_bpm, -35, 0.2),
        RelaxationSegment(RelaxationPhase.EMERGENCE, emergence, mid_bpm, -25, 0.5),
    ]
    return RelaxationProtocol(
        segments=segments,
        total_duration_s=total_duration_s,
        breath_rate_bpm=6.0,
    )


def tension_release_arc(n_points: int = 100, peak_position: float = 0.3) -> np.ndarray:
    """Generate a tension-release arc for guided imagery.

    Peak tension at *peak_position* fraction, then gradual release.
    Returns values in [0, 1].
    """
    t = np.linspace(0, 1, n_points)
    rise = np.where(t <= peak_position, t / peak_position, 0.0)
    fall = np.where(t > peak_position, (1.0 - t) / (1.0 - peak_position), 0.0)
    arc = rise + fall
    return np.clip(arc, 0, 1)


def harmonic_simplification_curve(
    n_points: int = 100,
    initial_complexity: float = 0.7,
    final_complexity: float = 0.15,
) -> np.ndarray:
    """Harmonic complexity reduction over a relaxation session."""
    t = np.linspace(0, 1, n_points)
    return initial_complexity + (final_complexity - initial_complexity) * (1 - np.exp(-3 * t))


def entrainment_effectiveness(
    stimulus_bpm: np.ndarray,
    response_bpm: np.ndarray,
) -> float:
    """Measure how well a physiological response tracks the stimulus tempo.

    Returns correlation in [−1, 1]; higher = better entrainment.
    """
    if len(stimulus_bpm) < 3 or len(response_bpm) < 3:
        return 0.0
    n = min(len(stimulus_bpm), len(response_bpm))
    s = stimulus_bpm[:n]
    r = response_bpm[:n]
    s_std = np.std(s)
    r_std = np.std(r)
    if s_std < 1e-12 or r_std < 1e-12:
        return 0.0
    return float(np.corrcoef(s, r)[0, 1])
