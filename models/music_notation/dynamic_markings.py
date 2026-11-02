"""Dynamic markings: encode, interpret, and apply loudness instructions.

Map between symbolic dynamic notations (pp, p, mp, mf, f, ff, etc.) and
their MIDI velocity realisations, including crescendo/diminuendo ramps,
subito dynamics, and hairpin parsing.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DynamicLevel(str, Enum):
    PPP = "ppp"
    PP = "pp"
    P = "p"
    MP = "mp"
    MF = "mf"
    F = "f"
    FF = "ff"
    FFF = "fff"
    FP = "fp"
    SFZ = "sfz"
    SFZP = "sfzp"
    RF = "rf"


DYNAMIC_VELOCITY: Dict[DynamicLevel, int] = {
    DynamicLevel.PPP: 16,
    DynamicLevel.PP: 33,
    DynamicLevel.P: 49,
    DynamicLevel.MP: 64,
    DynamicLevel.MF: 80,
    DynamicLevel.F: 96,
    DynamicLevel.FF: 112,
    DynamicLevel.FFF: 127,
    DynamicLevel.SFZ: 120,
}


@dataclass
class DynamicEvent:
    time_beat: float
    level: DynamicLevel
    is_subito: bool = False


@dataclass
class HairpinSpan:
    start_beat: float
    end_beat: float
    start_level: DynamicLevel
    end_level: DynamicLevel


def velocity_at_time(events: List[DynamicEvent],
                     hairpins: List[HairpinSpan],
                     t: float) -> int:
    base_vel = 80
    for e in reversed(events):
        if e.time_beat <= t:
            base_vel = DYNAMIC_VELOCITY.get(e.level, 80)
            break
    for h in hairpins:
        if h.start_beat <= t <= h.end_beat:
            v0 = DYNAMIC_VELOCITY.get(h.start_level, 80)
            v1 = DYNAMIC_VELOCITY.get(h.end_level, 80)
            frac = (t - h.start_beat) / max(h.end_beat - h.start_beat, 0.01)
            base_vel = int(v0 + frac * (v1 - v0))
            break
    return max(1, min(127, base_vel))


def velocity_curve(events: List[DynamicEvent],
                   hairpins: List[HairpinSpan],
                   total_beats: float,
                   resolution: float = 0.25) -> np.ndarray:
    times = np.arange(0, total_beats, resolution)
    return np.array([velocity_at_time(events, hairpins, t) for t in times])


def detect_dynamics_from_velocity(velocities: np.ndarray) -> List[DynamicLevel]:
    result = []
    thresholds = sorted(DYNAMIC_VELOCITY.items(), key=lambda x: x[1])
    for v in velocities:
        best = DynamicLevel.MF
        for level, vel in thresholds:
            if v <= vel + 8:
                best = level
                break
        result.append(best)
    return result


def crescendo_ramp(start_vel: int, end_vel: int,
                   n_notes: int, curve: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 1, n_notes)
    ramp = start_vel + (end_vel - start_vel) * t ** curve
    return np.clip(ramp, 1, 127).astype(int)


def diminuendo_ramp(start_vel: int, end_vel: int,
                    n_notes: int, curve: float = 1.0) -> np.ndarray:
    return crescendo_ramp(start_vel, end_vel, n_notes, curve)


def subito_dynamic(velocities: np.ndarray, at_index: int,
                   new_vel: int) -> np.ndarray:
    result = velocities.copy()
    result[at_index:] = new_vel
    return np.clip(result, 1, 127)


def dynamic_range(velocities: np.ndarray) -> Dict[str, float]:
    if len(velocities) == 0:
        return {"range_db": 0.0, "mean_vel": 0.0, "std_vel": 0.0}
    v = velocities.astype(float)
    db_range = 20 * np.log10(max(v.max(), 1) / max(v.min(), 1))
    return {
        "range_db": float(db_range),
        "mean_vel": float(np.mean(v)),
        "std_vel": float(np.std(v)),
        "min_vel": int(np.min(velocities)),
        "max_vel": int(np.max(velocities)),
    }


def terrace_dynamics(velocities: np.ndarray,
                     n_levels: int = 4) -> np.ndarray:
    edges = np.linspace(0, 128, n_levels + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    result = np.zeros_like(velocities)
    for i, v in enumerate(velocities):
        idx = min(int(v / (128 / n_levels)), n_levels - 1)
        result[i] = int(centres[idx])
    return np.clip(result, 1, 127).astype(int)
