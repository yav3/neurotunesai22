"""Interval training: ear training exercises for interval recognition.

Generates interval identification exercises, tracks accuracy by interval type,
adapts difficulty based on learner performance, and provides spaced-repetition
scheduling for weak intervals.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class IntervalQuality(Enum):
    PERFECT = "perfect"
    MAJOR = "major"
    MINOR = "minor"
    AUGMENTED = "augmented"
    DIMINISHED = "diminished"


@dataclass(frozen=True)
class Interval:
    semitones: int
    name: str
    quality: IntervalQuality
    direction: str = "ascending"

    @property
    def difficulty(self) -> float:
        easy = {0, 12, 7, 5}
        medium = {4, 3, 8, 9}
        if self.semitones % 12 in easy:
            return 0.2
        if self.semitones % 12 in medium:
            return 0.5
        return 0.8


INTERVALS = [
    Interval(0, "unison", IntervalQuality.PERFECT),
    Interval(1, "minor 2nd", IntervalQuality.MINOR),
    Interval(2, "major 2nd", IntervalQuality.MAJOR),
    Interval(3, "minor 3rd", IntervalQuality.MINOR),
    Interval(4, "major 3rd", IntervalQuality.MAJOR),
    Interval(5, "perfect 4th", IntervalQuality.PERFECT),
    Interval(6, "tritone", IntervalQuality.AUGMENTED),
    Interval(7, "perfect 5th", IntervalQuality.PERFECT),
    Interval(8, "minor 6th", IntervalQuality.MINOR),
    Interval(9, "major 6th", IntervalQuality.MAJOR),
    Interval(10, "minor 7th", IntervalQuality.MINOR),
    Interval(11, "major 7th", IntervalQuality.MAJOR),
    Interval(12, "octave", IntervalQuality.PERFECT),
]


@dataclass
class ExerciseResult:
    interval: Interval
    correct: bool
    response_time_ms: float = 0.0


@dataclass
class LearnerState:
    accuracy_by_interval: dict[int, list[bool]] = field(default_factory=dict)
    exposure_count: dict[int, int] = field(default_factory=dict)
    last_seen: dict[int, int] = field(default_factory=dict)
    session_index: int = 0

    def record(self, result: ExerciseResult) -> None:
        st = result.interval.semitones
        self.accuracy_by_interval.setdefault(st, []).append(result.correct)
        self.exposure_count[st] = self.exposure_count.get(st, 0) + 1
        self.last_seen[st] = self.session_index

    def accuracy(self, semitones: int, window: int = 10) -> float:
        history = self.accuracy_by_interval.get(semitones, [])
        recent = history[-window:] if history else []
        return sum(recent) / max(len(recent), 1)

    def weakest_intervals(self, top_n: int = 3) -> list[int]:
        accs = [(st, self.accuracy(st)) for st in self.accuracy_by_interval]
        accs.sort(key=lambda x: x[1])
        return [st for st, _ in accs[:top_n]]


def generate_exercise(
    state: LearnerState,
    available: list[Interval] | None = None,
    bias_weak: float = 0.6,
    rng: random.Random | None = None,
) -> Interval:
    rng = rng or random.Random()
    pool = available or INTERVALS
    if not pool:
        return INTERVALS[7]
    weak = state.weakest_intervals(3)
    weak_pool = [iv for iv in pool if iv.semitones in weak]
    if weak_pool and rng.random() < bias_weak:
        return rng.choice(weak_pool)
    return rng.choice(pool)


def spaced_repetition_priority(
    state: LearnerState,
    current_session: int,
) -> list[tuple[int, float]]:
    priorities: list[tuple[int, float]] = []
    for iv in INTERVALS:
        st = iv.semitones
        acc = state.accuracy(st)
        gap = current_session - state.last_seen.get(st, 0)
        exposure = state.exposure_count.get(st, 0)
        need = (1.0 - acc) * 0.5 + min(1.0, gap / 10.0) * 0.3 + (1.0 / (exposure + 1)) * 0.2
        priorities.append((st, need))
    priorities.sort(key=lambda x: -x[1])
    return priorities


def confusion_matrix(state: LearnerState) -> np.ndarray:
    n = 13
    mat = np.zeros((n, n))
    for st, history in state.accuracy_by_interval.items():
        if 0 <= st <= 12:
            correct = sum(history)
            total = len(history)
            mat[st, st] = correct
            wrong = total - correct
            if wrong > 0:
                others = [i for i in range(n) if i != st]
                per_other = wrong / len(others)
                for o in others:
                    mat[st, o] += per_other
    return mat


def mastery_level(state: LearnerState) -> float:
    if not state.accuracy_by_interval:
        return 0.0
    accs = [state.accuracy(st) for st in range(13)]
    return float(np.mean(accs))


def recommend_session_plan(
    state: LearnerState,
    n_exercises: int = 20,
) -> list[Interval]:
    priorities = spaced_repetition_priority(state, state.session_index)
    plan: list[Interval] = []
    interval_map = {iv.semitones: iv for iv in INTERVALS}
    for st, _ in priorities:
        if len(plan) >= n_exercises:
            break
        iv = interval_map.get(st)
        if iv:
            reps = max(1, int(n_exercises * 0.15))
            plan.extend([iv] * reps)
    return plan[:n_exercises]
