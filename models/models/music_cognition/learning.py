"""Music learning: skill progression, practice scheduling, difficulty estimation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SkillLevel(Enum):
    NOVICE = 1
    BEGINNER = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5
    VIRTUOSO = 6


@dataclass
class SkillProfile:
    technique: float = 0.0
    rhythm: float = 0.0
    pitch_accuracy: float = 0.0
    sight_reading: float = 0.0
    ear_training: float = 0.0
    expression: float = 0.0
    theory: float = 0.0

    @property
    def overall(self) -> float:
        vals = [self.technique, self.rhythm, self.pitch_accuracy,
                self.sight_reading, self.ear_training, self.expression, self.theory]
        return float(np.mean(vals))

    @property
    def level(self) -> SkillLevel:
        o = self.overall
        if o < 0.15:
            return SkillLevel.NOVICE
        if o < 0.30:
            return SkillLevel.BEGINNER
        if o < 0.50:
            return SkillLevel.INTERMEDIATE
        if o < 0.70:
            return SkillLevel.ADVANCED
        if o < 0.90:
            return SkillLevel.EXPERT
        return SkillLevel.VIRTUOSO


def skill_progression(
    practice_hours: float,
    current_skill: float,
    ceiling: float = 1.0,
    learning_rate: float = 0.1,
    deliberate_practice_fraction: float = 0.5,
) -> float:
    if practice_hours <= 0 or current_skill >= ceiling:
        return current_skill
    effective_hours = practice_hours * (0.5 + 0.5 * deliberate_practice_fraction)
    headroom = ceiling - current_skill
    gain = headroom * (1.0 - np.exp(-learning_rate * effective_hours))
    return float(min(current_skill + gain, ceiling))


def optimal_practice_schedule(
    current_skill: float,
    target_skill: float,
    hours_per_week: float = 5.0,
    learning_rate: float = 0.1,
    deliberate_fraction: float = 0.6,
) -> dict[str, float]:
    if current_skill >= target_skill or hours_per_week <= 0:
        return {"weeks_needed": 0.0, "total_hours": 0.0}
    effective_rate = learning_rate * (0.5 + 0.5 * deliberate_fraction)
    if effective_rate <= 0:
        return {"weeks_needed": float("inf"), "total_hours": float("inf")}
    headroom = 1.0 - current_skill
    if headroom <= 0:
        return {"weeks_needed": float("inf"), "total_hours": float("inf")}
    target_gain = target_skill - current_skill
    frac = 1.0 - target_gain / headroom
    if frac <= 0:
        return {"weeks_needed": float("inf"), "total_hours": float("inf")}
    total_hours = -np.log(frac) / effective_rate
    weeks = total_hours / hours_per_week
    return {"weeks_needed": float(weeks), "total_hours": float(total_hours)}


def sight_reading_difficulty(
    pitch_range_semitones: int = 12,
    rhythmic_complexity: float = 0.5,
    key_changes: int = 0,
    accidentals_per_bar: float = 0.0,
    tempo_bpm: float = 120.0,
    n_voices: int = 1,
) -> float:
    raw = (
        0.20 * min(pitch_range_semitones / 36, 1.0)
        + 0.25 * rhythmic_complexity
        + 0.10 * min(key_changes / 5, 1.0)
        + 0.15 * min(accidentals_per_bar / 4, 1.0)
        + 0.15 * np.clip((tempo_bpm - 60) / 180, 0, 1)
        + 0.15 * min((n_voices - 1) / 4, 1.0)
    )
    return float(np.clip(raw, 0.0, 1.0))


class IntervalType(Enum):
    UNISON = 0
    MINOR_SECOND = 1
    MAJOR_SECOND = 2
    MINOR_THIRD = 3
    MAJOR_THIRD = 4
    PERFECT_FOURTH = 5
    TRITONE = 6
    PERFECT_FIFTH = 7
    MINOR_SIXTH = 8
    MAJOR_SIXTH = 9
    MINOR_SEVENTH = 10
    MAJOR_SEVENTH = 11
    OCTAVE = 12


INTERVAL_DIFFICULTY: dict[IntervalType, float] = {
    IntervalType.UNISON: 0.0,
    IntervalType.OCTAVE: 0.15,
    IntervalType.PERFECT_FIFTH: 0.20,
    IntervalType.PERFECT_FOURTH: 0.25,
    IntervalType.MAJOR_THIRD: 0.30,
    IntervalType.MINOR_THIRD: 0.35,
    IntervalType.MAJOR_SIXTH: 0.40,
    IntervalType.MINOR_SIXTH: 0.45,
    IntervalType.MAJOR_SECOND: 0.50,
    IntervalType.MINOR_SECOND: 0.55,
    IntervalType.MAJOR_SEVENTH: 0.70,
    IntervalType.MINOR_SEVENTH: 0.65,
    IntervalType.TRITONE: 0.80,
}


def ear_training_accuracy(
    interval: IntervalType,
    skill_level: float,
    is_ascending: bool = True,
) -> float:
    difficulty = INTERVAL_DIFFICULTY.get(interval, 0.5)
    direction_bonus = 0.05 if is_ascending else 0.0
    raw = skill_level - difficulty + direction_bonus + 0.3
    return float(np.clip(1.0 / (1.0 + np.exp(-5.0 * raw)), 0.0, 1.0))


def spaced_repetition_interval(
    previous_interval_days: float,
    quality: int,
    easiness_factor: float = 2.5,
) -> tuple[float, float]:
    ef = easiness_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    ef = max(ef, 1.3)
    if quality < 3:
        next_interval = 1.0
    elif previous_interval_days <= 0:
        next_interval = 1.0
    elif previous_interval_days <= 1:
        next_interval = 6.0
    else:
        next_interval = previous_interval_days * ef
    return next_interval, ef


def practice_efficiency(
    minutes_practiced: float,
    errors_before: int,
    errors_after: int,
    passages_covered: int = 1,
) -> float:
    if minutes_practiced <= 0 or passages_covered <= 0:
        return 0.0
    error_reduction = max(errors_before - errors_after, 0)
    return float(error_reduction / (minutes_practiced * passages_covered) * 10.0)
