"""Articulation modelling for expressive performance.

Note-level articulation descriptors (legato, staccato, accent, tenuto),
articulation detection from MIDI velocity/duration, and performance
rendering with articulation templates.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class ArticulationType(Enum):
    LEGATO = "legato"
    STACCATO = "staccato"
    STACCATISSIMO = "staccatissimo"
    TENUTO = "tenuto"
    ACCENT = "accent"
    MARCATO = "marcato"
    NORMAL = "normal"


@dataclass
class ArticulationEvent:
    onset_s: float
    duration_s: float
    velocity: int  # 0-127
    articulation: ArticulationType
    overlap_ratio: float  # < 0 = gap, > 0 = overlap with next note


# Duration ratio thresholds (performed/notated)
ARTICULATION_THRESHOLDS = {
    ArticulationType.STACCATISSIMO: (0.0, 0.25),
    ArticulationType.STACCATO: (0.25, 0.5),
    ArticulationType.NORMAL: (0.5, 0.85),
    ArticulationType.TENUTO: (0.85, 1.0),
    ArticulationType.LEGATO: (1.0, 1.5),
}


def classify_articulation(
    performed_duration: float,
    notated_duration: float,
    velocity: int = 64,
    accent_threshold: int = 100,
) -> ArticulationType:
    """Classify articulation from performed vs notated duration ratio."""
    if notated_duration <= 0:
        return ArticulationType.NORMAL

    ratio = performed_duration / notated_duration

    if velocity >= accent_threshold:
        return ArticulationType.ACCENT

    for art_type, (lo, hi) in ARTICULATION_THRESHOLDS.items():
        if lo <= ratio < hi:
            return art_type

    if ratio >= 1.5:
        return ArticulationType.LEGATO
    return ArticulationType.NORMAL


def detect_articulations(
    onsets: np.ndarray,
    durations: np.ndarray,
    velocities: np.ndarray,
    notated_durations: np.ndarray,
) -> List[ArticulationEvent]:
    """Detect articulation for each note in a performance."""
    n = min(len(onsets), len(durations), len(velocities), len(notated_durations))
    events = []
    for i in range(n):
        art = classify_articulation(
            durations[i], notated_durations[i], int(velocities[i])
        )
        overlap = 0.0
        if i < n - 1:
            gap = onsets[i + 1] - (onsets[i] + durations[i])
            if notated_durations[i] > 0:
                overlap = -gap / notated_durations[i]
        events.append(ArticulationEvent(
            onset_s=float(onsets[i]),
            duration_s=float(durations[i]),
            velocity=int(velocities[i]),
            articulation=art,
            overlap_ratio=float(overlap),
        ))
    return events


def articulation_template(
    art_type: ArticulationType,
) -> Tuple[float, float]:
    """Return (duration_ratio, velocity_scale) for rendering."""
    templates = {
        ArticulationType.LEGATO: (1.05, 1.0),
        ArticulationType.STACCATO: (0.4, 0.9),
        ArticulationType.STACCATISSIMO: (0.2, 0.85),
        ArticulationType.TENUTO: (0.95, 1.05),
        ArticulationType.ACCENT: (0.8, 1.3),
        ArticulationType.MARCATO: (0.7, 1.4),
        ArticulationType.NORMAL: (0.75, 1.0),
    }
    return templates.get(art_type, (0.75, 1.0))


def render_with_articulation(
    notated_durations: np.ndarray,
    notated_velocities: np.ndarray,
    articulations: List[ArticulationType],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply articulation templates to produce performed durations and velocities."""
    n = min(len(notated_durations), len(notated_velocities), len(articulations))
    performed_dur = np.zeros(n)
    performed_vel = np.zeros(n)
    for i in range(n):
        dr, vs = articulation_template(articulations[i])
        performed_dur[i] = notated_durations[i] * dr
        performed_vel[i] = np.clip(notated_velocities[i] * vs, 1, 127)
    return performed_dur, performed_vel


def legato_ratio(events: List[ArticulationEvent]) -> float:
    """Fraction of notes classified as legato or tenuto."""
    if not events:
        return 0.0
    legato_count = sum(
        1 for e in events
        if e.articulation in (ArticulationType.LEGATO, ArticulationType.TENUTO)
    )
    return legato_count / len(events)


def staccato_ratio(events: List[ArticulationEvent]) -> float:
    """Fraction of notes classified as staccato or staccatissimo."""
    if not events:
        return 0.0
    stacc_count = sum(
        1 for e in events
        if e.articulation in (ArticulationType.STACCATO, ArticulationType.STACCATISSIMO)
    )
    return stacc_count / len(events)


def mean_overlap_ratio(events: List[ArticulationEvent]) -> float:
    """Mean overlap ratio across all note transitions."""
    if len(events) < 2:
        return 0.0
    overlaps = [e.overlap_ratio for e in events[:-1]]
    return float(np.mean(overlaps))
