"""Mood-arc-based playlist sequencing using valence-arousal trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class MoodArc(Enum):
    UPLIFTING = "uplifting"
    WINDING_DOWN = "winding_down"
    EMOTIONAL_JOURNEY = "emotional_journey"
    STEADY_STATE = "steady_state"
    CATHARTIC = "cathartic"


@dataclass
class MoodPoint:
    valence: float  # [-1, 1]
    arousal: float  # [-1, 1]
    timestamp_ratio: float  # [0, 1]


ARC_TEMPLATES: dict[MoodArc, list[MoodPoint]] = {
    MoodArc.UPLIFTING: [
        MoodPoint(0.2, 0.3, 0.0),
        MoodPoint(0.5, 0.5, 0.3),
        MoodPoint(0.8, 0.8, 0.7),
        MoodPoint(0.9, 0.7, 1.0),
    ],
    MoodArc.WINDING_DOWN: [
        MoodPoint(0.5, 0.7, 0.0),
        MoodPoint(0.4, 0.4, 0.4),
        MoodPoint(0.3, 0.2, 0.7),
        MoodPoint(0.2, 0.0, 1.0),
    ],
    MoodArc.EMOTIONAL_JOURNEY: [
        MoodPoint(0.3, 0.3, 0.0),
        MoodPoint(-0.2, 0.6, 0.25),
        MoodPoint(-0.5, 0.8, 0.5),
        MoodPoint(0.2, 0.5, 0.75),
        MoodPoint(0.6, 0.4, 1.0),
    ],
    MoodArc.STEADY_STATE: [
        MoodPoint(0.5, 0.5, 0.0),
        MoodPoint(0.5, 0.5, 1.0),
    ],
    MoodArc.CATHARTIC: [
        MoodPoint(0.0, 0.3, 0.0),
        MoodPoint(-0.3, 0.7, 0.3),
        MoodPoint(-0.5, 0.9, 0.5),
        MoodPoint(0.3, 0.6, 0.7),
        MoodPoint(0.7, 0.4, 1.0),
    ],
}


def plan_mood_journey(arc: MoodArc, n_tracks: int) -> list[MoodPoint]:
    """Interpolate arc waypoints to n_tracks positions."""
    template = ARC_TEMPLATES.get(arc, ARC_TEMPLATES[MoodArc.STEADY_STATE])
    if n_tracks <= 1:
        return [template[0]]
    ratios = np.linspace(0.0, 1.0, n_tracks)
    t_ratios = [p.timestamp_ratio for p in template]
    v_vals = [p.valence for p in template]
    a_vals = [p.arousal for p in template]
    points = []
    for r in ratios:
        v = float(np.interp(r, t_ratios, v_vals))
        a = float(np.interp(r, t_ratios, a_vals))
        points.append(MoodPoint(v, a, float(r)))
    return points


def match_track_to_target(track_valence: float, track_arousal: float, target: MoodPoint) -> float:
    """Euclidean distance in VA space (lower = better match)."""
    return math.sqrt((track_valence - target.valence) ** 2 + (track_arousal - target.arousal) ** 2)


def sequence_for_arc(
    tracks: list[dict],
    arc: MoodArc,
) -> list[int]:
    """Order track indices to approximate mood arc (greedy nearest-target)."""
    n = len(tracks)
    if n == 0:
        return []
    targets = plan_mood_journey(arc, n)
    available = list(range(n))
    ordered = []
    for target in targets:
        if not available:
            break
        best = min(available, key=lambda i: match_track_to_target(
            tracks[i].get("valence", 0.0),
            tracks[i].get("arousal", 0.0),
            target,
        ))
        ordered.append(best)
        available.remove(best)
    return ordered
