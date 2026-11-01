"""Articulation analysis: staccato, legato, accent, tenuto detection and scoring.

Analyzes note-level articulation from onset/offset timing and amplitude
envelopes, classifies articulation types, and scores expressive intent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ArticulationType(Enum):
    LEGATO = "legato"
    STACCATO = "staccato"
    STACCATISSIMO = "staccatissimo"
    TENUTO = "tenuto"
    ACCENT = "accent"
    MARCATO = "marcato"
    PORTATO = "portato"
    NORMAL = "normal"


@dataclass(frozen=True)
class NoteArticulation:
    onset: float
    offset: float
    duration: float
    ioi: float
    duty_cycle: float
    peak_velocity: float
    attack_time: float
    articulation: ArticulationType


def classify_articulation(
    duration: float,
    ioi: float,
    peak_velocity: float,
    attack_time_ms: float,
) -> ArticulationType:
    duty = duration / max(ioi, 1e-6)
    if duty < 0.3:
        return ArticulationType.STACCATISSIMO
    if duty < 0.5:
        return ArticulationType.STACCATO
    if duty > 0.95 and attack_time_ms < 10:
        return ArticulationType.LEGATO
    if duty > 0.9 and peak_velocity > 0.8:
        return ArticulationType.TENUTO
    if peak_velocity > 0.85 and attack_time_ms < 15:
        return ArticulationType.MARCATO
    if peak_velocity > 0.7 and attack_time_ms < 20:
        return ArticulationType.ACCENT
    if 0.5 <= duty <= 0.75 and peak_velocity > 0.5:
        return ArticulationType.PORTATO
    return ArticulationType.NORMAL


def analyze_articulations(
    onsets: np.ndarray,
    offsets: np.ndarray,
    velocities: np.ndarray,
    attack_times_ms: np.ndarray,
) -> list[NoteArticulation]:
    n = min(len(onsets), len(offsets), len(velocities), len(attack_times_ms))
    results: list[NoteArticulation] = []
    for i in range(n):
        duration = offsets[i] - onsets[i]
        ioi = onsets[i + 1] - onsets[i] if i < n - 1 else duration * 1.2
        art = classify_articulation(duration, ioi, float(velocities[i]), float(attack_times_ms[i]))
        results.append(NoteArticulation(
            onset=float(onsets[i]),
            offset=float(offsets[i]),
            duration=duration,
            ioi=ioi,
            duty_cycle=duration / max(ioi, 1e-6),
            peak_velocity=float(velocities[i]),
            attack_time=float(attack_times_ms[i]),
            articulation=art,
        ))
    return results


def articulation_consistency(notes: list[NoteArticulation]) -> float:
    if len(notes) < 2:
        return 1.0
    duties = [n.duty_cycle for n in notes]
    velocities = [n.peak_velocity for n in notes]
    duty_cv = float(np.std(duties)) / max(float(np.mean(duties)), 1e-6)
    vel_cv = float(np.std(velocities)) / max(float(np.mean(velocities)), 1e-6)
    return max(0.0, 1.0 - (duty_cv + vel_cv) / 2.0)


def legato_score(notes: list[NoteArticulation]) -> float:
    if not notes:
        return 0.0
    overlaps = []
    for i in range(len(notes) - 1):
        gap = notes[i + 1].onset - notes[i].offset
        overlaps.append(max(0.0, -gap))
    return float(np.mean([o > 0 for o in overlaps])) if overlaps else 0.0


def staccato_ratio(notes: list[NoteArticulation]) -> float:
    if not notes:
        return 0.0
    count = sum(1 for n in notes if n.articulation in (ArticulationType.STACCATO, ArticulationType.STACCATISSIMO))
    return count / len(notes)


def dynamic_accent_positions(
    notes: list[NoteArticulation],
    threshold: float = 0.7,
) -> list[int]:
    return [i for i, n in enumerate(notes) if n.peak_velocity > threshold and n.articulation in (ArticulationType.ACCENT, ArticulationType.MARCATO)]


def phrase_articulation_profile(
    notes: list[NoteArticulation],
) -> dict[str, float]:
    if not notes:
        return {}
    counts: dict[str, int] = {}
    for n in notes:
        key = n.articulation.value
        counts[key] = counts.get(key, 0) + 1
    total = len(notes)
    return {k: v / total for k, v in counts.items()}


def expressive_range(notes: list[NoteArticulation]) -> float:
    if len(notes) < 2:
        return 0.0
    duties = [n.duty_cycle for n in notes]
    velocities = [n.peak_velocity for n in notes]
    duty_range = max(duties) - min(duties)
    vel_range = max(velocities) - min(velocities)
    return (duty_range + vel_range) / 2.0
