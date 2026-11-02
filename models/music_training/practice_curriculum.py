"""Practice curriculum: difficulty estimation, sequencing, mastery tracking.

Adaptive curriculum that estimates piece difficulty along multiple musical
dimensions (rhythmic, harmonic, technical, cognitive) and sequences exercises
for progressive skill development. Pure numpy computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MusicalDimension(str, Enum):
    RHYTHMIC = "rhythmic"
    HARMONIC = "harmonic"
    TECHNICAL = "technical"
    COGNITIVE = "cognitive"
    EXPRESSIVE = "expressive"


@dataclass
class PieceDifficulty:
    piece_id: str
    rhythmic: float = 0.0
    harmonic: float = 0.0
    technical: float = 0.0
    cognitive: float = 0.0
    expressive: float = 0.0

    @property
    def overall(self) -> float:
        dims = [self.rhythmic, self.harmonic, self.technical,
                self.cognitive, self.expressive]
        return float(np.mean(dims))

    def as_vector(self) -> np.ndarray:
        return np.array([self.rhythmic, self.harmonic, self.technical,
                         self.cognitive, self.expressive])


def estimate_rhythmic_difficulty(note_onsets: np.ndarray,
                                 beat_dur: float = 0.5) -> float:
    if len(note_onsets) < 2:
        return 0.0
    ioi = np.diff(note_onsets)
    ratios = ioi / beat_dur
    syncopation = float(np.mean(np.abs(ratios - np.round(ratios))))
    variety = len(np.unique(np.round(ratios, 2))) / max(len(ratios), 1)
    speed = float(np.clip(1.0 / (np.mean(ioi) + 0.01), 0, 10) / 10.0)
    return float(np.clip(0.3 * syncopation + 0.3 * variety + 0.4 * speed, 0, 1))


def estimate_harmonic_difficulty(pitch_classes: np.ndarray) -> float:
    if len(pitch_classes) < 2:
        return 0.0
    pc = pitch_classes % 12
    chromaticism = len(np.unique(pc)) / 12.0
    intervals = np.abs(np.diff(pitch_classes))
    large_leaps = float(np.mean(intervals > 7))
    return float(np.clip(0.5 * chromaticism + 0.5 * large_leaps, 0, 1))


def estimate_technical_difficulty(pitches: np.ndarray,
                                   velocities: np.ndarray | None = None) -> float:
    if len(pitches) < 2:
        return 0.0
    intervals = np.abs(np.diff(pitches))
    range_score = float(np.clip((np.ptp(pitches) - 12) / 48.0, 0, 1))
    leap_score = float(np.clip(np.mean(intervals) / 12.0, 0, 1))
    dyn_score = 0.0
    if velocities is not None and len(velocities) > 1:
        dyn_score = float(np.clip(np.std(velocities) / 40.0, 0, 1))
    return float(np.clip(0.3 * range_score + 0.4 * leap_score + 0.3 * dyn_score, 0, 1))


@dataclass
class MasteryState:
    dimension_scores: np.ndarray = field(default_factory=lambda: np.zeros(5))
    attempts: int = 0
    history: list[float] = field(default_factory=list)

    @property
    def mastery_level(self) -> float:
        return float(np.mean(self.dimension_scores))

    def update(self, performance: np.ndarray, lr: float = 0.1) -> None:
        self.attempts += 1
        self.dimension_scores = (1 - lr) * self.dimension_scores + lr * performance
        self.history.append(float(np.mean(performance)))


class ZPDSequencer:
    """Sequences pieces within the Zone of Proximal Development."""

    def __init__(self, lower: float = 0.6, upper: float = 0.85):
        self.lower = lower
        self.upper = upper

    def select_next(self, mastery: MasteryState,
                    candidates: list[PieceDifficulty],
                    k: int = 3) -> list[PieceDifficulty]:
        if not candidates:
            return []
        center = (self.lower + self.upper) / 2
        target = mastery.dimension_scores * (1.0 + center - 0.5)

        scored = []
        for piece in candidates:
            diff = piece.as_vector() - target
            d = float(np.linalg.norm(diff))
            in_zpd = all(
                self.lower * max(m, 0.05) <= p <= self.upper * (1 + m)
                for m, p in zip(mastery.dimension_scores, piece.as_vector())
            )
            if in_zpd:
                d *= 0.5
            scored.append((d, piece))

        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[:k]]

    def difficulty_curve(self, n_steps: int, start: float = 0.1,
                          end: float = 0.9) -> np.ndarray:
        t = np.linspace(0, 1, n_steps)
        return start + (end - start) * np.log1p(t * (np.e - 1))


def spaced_repetition_interval(n_reviews: int,
                                ease: float = 2.5,
                                last_days: float = 1.0) -> float:
    if n_reviews == 0:
        return 1.0
    if n_reviews == 1:
        return 6.0
    return last_days * ease


def forgetting_curve(t_days: np.ndarray, stability: float = 1.0) -> np.ndarray:
    return np.exp(-t_days / max(stability, 0.01))
