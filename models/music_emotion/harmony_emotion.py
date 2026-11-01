"""Harmony-emotion mapping: chord quality → affective dimensions.

Maps harmonic content (chord quality, progression tension, mode brightness)
to valence-arousal-dominance coordinates using perceptual weighting models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class AffectPoint:
    valence: float
    arousal: float
    dominance: float = 0.5

    def distance(self, other: AffectPoint) -> float:
        return float(np.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        ))


CHORD_AFFECT: Dict[str, AffectPoint] = {
    "major": AffectPoint(0.75, 0.55, 0.65),
    "minor": AffectPoint(0.30, 0.45, 0.35),
    "diminished": AffectPoint(0.15, 0.70, 0.20),
    "augmented": AffectPoint(0.40, 0.75, 0.50),
    "dom7": AffectPoint(0.60, 0.65, 0.60),
    "min7": AffectPoint(0.35, 0.40, 0.40),
    "maj7": AffectPoint(0.80, 0.45, 0.55),
    "dim7": AffectPoint(0.10, 0.80, 0.15),
    "sus2": AffectPoint(0.55, 0.35, 0.45),
    "sus4": AffectPoint(0.50, 0.50, 0.45),
    "power": AffectPoint(0.50, 0.70, 0.75),
    "add9": AffectPoint(0.70, 0.50, 0.55),
}


def chord_affect(quality: str) -> AffectPoint:
    """Look up the affective profile for a chord quality."""
    return CHORD_AFFECT.get(quality, AffectPoint(0.5, 0.5, 0.5))


def progression_valence(qualities: List[str]) -> np.ndarray:
    """Valence trajectory for a chord progression."""
    return np.array([chord_affect(q).valence for q in qualities])


def progression_arousal(qualities: List[str]) -> np.ndarray:
    """Arousal trajectory for a chord progression."""
    return np.array([chord_affect(q).arousal for q in qualities])


def harmonic_tension(root_intervals: List[int]) -> np.ndarray:
    """Tension estimate from root motion intervals (semitones).

    Tension is higher for tritone and semitone motions, lower for
    perfect fourth/fifth.
    """
    tension_map = {
        0: 0.0, 1: 0.8, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.2,
        6: 0.9, 7: 0.1, 8: 0.35, 9: 0.4, 10: 0.6, 11: 0.75,
    }
    return np.array([tension_map.get(abs(i) % 12, 0.5)
                     for i in root_intervals])


def mode_brightness(scale_intervals: Tuple[int, ...]) -> float:
    """Brightness score based on cumulative pitch height of scale degrees.

    Lydian (raised 4th) → high brightness.
    Locrian (lowered 2nd, 5th) → low brightness.
    """
    if not scale_intervals:
        return 0.5
    major = (2, 2, 1, 2, 2, 2, 1)
    n = min(len(scale_intervals), len(major))
    cum_s = list(np.cumsum(scale_intervals[:n]))
    cum_m = list(np.cumsum(major[:n]))
    diff = sum(a - b for a, b in zip(cum_s, cum_m))
    return float(np.clip(0.5 + diff * 0.05, 0.0, 1.0))


def emotional_trajectory(qualities: List[str],
                         root_intervals: List[int]) -> List[AffectPoint]:
    """Compute per-chord emotional position including tension modulation."""
    tensions = harmonic_tension(root_intervals) if root_intervals else np.zeros(len(qualities))
    trajectory = []
    for i, q in enumerate(qualities):
        base = chord_affect(q)
        t = tensions[i] if i < len(tensions) else 0.0
        trajectory.append(AffectPoint(
            valence=float(np.clip(base.valence - 0.2 * t, 0, 1)),
            arousal=float(np.clip(base.arousal + 0.3 * t, 0, 1)),
            dominance=base.dominance,
        ))
    return trajectory


def emotional_range(trajectory: List[AffectPoint]) -> float:
    """Maximum distance between any two points in an emotional trajectory."""
    if len(trajectory) < 2:
        return 0.0
    max_dist = 0.0
    for i in range(len(trajectory)):
        for j in range(i + 1, len(trajectory)):
            d = trajectory[i].distance(trajectory[j])
            if d > max_dist:
                max_dist = d
    return max_dist


def emotional_centroid(trajectory: List[AffectPoint]) -> AffectPoint:
    """Mean position in affect space."""
    if not trajectory:
        return AffectPoint(0.5, 0.5, 0.5)
    return AffectPoint(
        valence=float(np.mean([p.valence for p in trajectory])),
        arousal=float(np.mean([p.arousal for p in trajectory])),
        dominance=float(np.mean([p.dominance for p in trajectory])),
    )


def emotional_variance(trajectory: List[AffectPoint]) -> float:
    """Variance of affect-space positions (emotional instability)."""
    if len(trajectory) < 2:
        return 0.0
    centroid = emotional_centroid(trajectory)
    dists = [centroid.distance(p) for p in trajectory]
    return float(np.var(dists))


def valence_arousal_quadrant(point: AffectPoint) -> str:
    """Classify an affect point into Russell circumplex quadrant."""
    if point.valence >= 0.5 and point.arousal >= 0.5:
        return "happy/excited"
    elif point.valence < 0.5 and point.arousal >= 0.5:
        return "angry/tense"
    elif point.valence < 0.5 and point.arousal < 0.5:
        return "sad/depressed"
    else:
        return "calm/relaxed"


def major_minor_ratio(qualities: List[str]) -> float:
    """Ratio of major-type to minor-type chords in a progression."""
    major_types = {"major", "maj7", "dom7", "add9", "sus2", "sus4", "power"}
    minor_types = {"minor", "min7", "diminished", "dim7"}
    n_major = sum(1 for q in qualities if q in major_types)
    n_minor = sum(1 for q in qualities if q in minor_types)
    total = n_major + n_minor
    if total == 0:
        return 0.5
    return n_major / total
