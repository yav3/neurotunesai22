"""Iso Principle: match current emotional state then gradually shift."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class IsoTrajectory:
    values: tuple[float, ...]
    steps: int


def iso_match_tempo(
    current_bpm: float, target_bpm: float, steps: int = 8
) -> list[float]:
    if steps < 2:
        return [target_bpm]
    t = np.linspace(0, 1, steps)
    curve = 1 / (1 + np.exp(-6 * (t - 0.5)))
    curve = (curve - curve[0]) / (curve[-1] - curve[0])
    return [float(current_bpm + (target_bpm - current_bpm) * c) for c in curve]


def iso_match_mode(
    current_mode: str, target_mode: str
) -> list[str]:
    MODE_ORDER = [
        "locrian", "phrygian", "aeolian", "dorian",
        "mixolydian", "ionian", "lydian",
    ]
    try:
        i_start = MODE_ORDER.index(current_mode.lower())
        i_end = MODE_ORDER.index(target_mode.lower())
    except ValueError:
        return [current_mode, target_mode]
    if i_start <= i_end:
        return MODE_ORDER[i_start : i_end + 1]
    return MODE_ORDER[i_end : i_start + 1][::-1]


def emotional_trajectory(
    start_valence: float, end_valence: float, n_steps: int = 10
) -> list[float]:
    if n_steps < 2:
        return [end_valence]
    t = np.linspace(0, 1, n_steps)
    curve = 1 / (1 + np.exp(-8 * (t - 0.5)))
    curve = (curve - curve[0]) / (curve[-1] - curve[0])
    return [float(start_valence + (end_valence - start_valence) * c) for c in curve]


def iso_energy_curve(
    start_energy: float, target_energy: float, duration_min: float, step_min: float = 2.0
) -> list[tuple[float, float]]:
    n = max(2, int(duration_min / step_min))
    vals = emotional_trajectory(start_energy, target_energy, n)
    times = np.linspace(0, duration_min, n).tolist()
    return list(zip(times, vals))
