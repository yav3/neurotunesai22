"""Ear training models: interval/chord/melodic/harmonic recognition and adaptive difficulty."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


INTERVAL_NAMES = [
    "P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7", "P8",
]

INTERVAL_SEMITONES = {name: i for i, name in enumerate(INTERVAL_NAMES)}


class ChordQuality(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    DOMINANT_7 = "dom7"
    MAJOR_7 = "maj7"
    MINOR_7 = "min7"
    HALF_DIM_7 = "half_dim7"
    DIM_7 = "dim7"
    MINOR_MAJOR_7 = "min_maj7"
    ADD9 = "add9"


CHORD_INTERVALS: Dict[ChordQuality, List[int]] = {
    ChordQuality.MAJOR: [0, 4, 7],
    ChordQuality.MINOR: [0, 3, 7],
    ChordQuality.DIMINISHED: [0, 3, 6],
    ChordQuality.AUGMENTED: [0, 4, 8],
    ChordQuality.DOMINANT_7: [0, 4, 7, 10],
    ChordQuality.MAJOR_7: [0, 4, 7, 11],
    ChordQuality.MINOR_7: [0, 3, 7, 10],
    ChordQuality.HALF_DIM_7: [0, 3, 6, 10],
    ChordQuality.DIM_7: [0, 3, 6, 9],
    ChordQuality.MINOR_MAJOR_7: [0, 3, 7, 11],
    ChordQuality.ADD9: [0, 4, 7, 14],
}


@dataclass
class IntervalResponse:
    presented: int
    answered: str
    correct: bool
    response_time_ms: float


@dataclass
class ChordResponse:
    presented: ChordQuality
    answered: ChordQuality
    correct: bool
    response_time_ms: float


def score_interval_recognition(responses: List[IntervalResponse]) -> Dict[str, float]:
    if not responses:
        return {"accuracy": 0.0, "avg_time_ms": 0.0}
    correct = sum(1 for r in responses if r.correct)
    avg_time = sum(r.response_time_ms for r in responses) / len(responses)
    return {
        "accuracy": correct / len(responses),
        "avg_time_ms": avg_time,
        "n_trials": float(len(responses)),
    }


def interval_confusion_matrix(responses: List[IntervalResponse]) -> np.ndarray:
    n = len(INTERVAL_NAMES)
    matrix = np.zeros((n, n))
    for r in responses:
        presented_idx = r.presented % n
        answered_idx = INTERVAL_SEMITONES.get(r.answered, -1)
        if 0 <= answered_idx < n:
            matrix[presented_idx, answered_idx] += 1
    return matrix


def identify_chord_quality(intervals: List[int]) -> Optional[ChordQuality]:
    normalized = sorted(set(i % 12 for i in intervals))
    if 0 not in normalized:
        return None
    for quality, template in CHORD_INTERVALS.items():
        t_norm = sorted(set(i % 12 for i in template))
        if normalized == t_norm:
            return quality
    return None


def score_chord_identification(responses: List[ChordResponse]) -> Dict[str, float]:
    if not responses:
        return {"accuracy": 0.0, "avg_time_ms": 0.0}
    correct = sum(1 for r in responses if r.correct)
    avg_time = sum(r.response_time_ms for r in responses) / len(responses)
    per_type: Dict[str, List[bool]] = {}
    for r in responses:
        key = r.presented.value
        per_type.setdefault(key, []).append(r.correct)
    type_accuracy = {k: sum(v) / len(v) for k, v in per_type.items()}
    return {
        "accuracy": correct / len(responses),
        "avg_time_ms": avg_time,
        "per_type_accuracy": type_accuracy,
    }


@dataclass
class MelodicDictationResult:
    target_pitches: List[int]
    transcribed_pitches: List[int]
    target_rhythms: List[float]
    transcribed_rhythms: List[float]

    def pitch_accuracy(self) -> float:
        if not self.target_pitches:
            return 1.0
        n = min(len(self.target_pitches), len(self.transcribed_pitches))
        if n == 0:
            return 0.0
        correct = sum(1 for i in range(n)
                      if self.target_pitches[i] == self.transcribed_pitches[i])
        length_penalty = n / max(len(self.target_pitches), len(self.transcribed_pitches))
        return (correct / n) * length_penalty

    def rhythm_accuracy(self, tolerance: float = 0.1) -> float:
        if not self.target_rhythms:
            return 1.0
        n = min(len(self.target_rhythms), len(self.transcribed_rhythms))
        if n == 0:
            return 0.0
        correct = sum(1 for i in range(n)
                      if abs(self.target_rhythms[i] - self.transcribed_rhythms[i]) <= tolerance)
        length_penalty = n / max(len(self.target_rhythms), len(self.transcribed_rhythms))
        return (correct / n) * length_penalty

    def contour_accuracy(self) -> float:
        if len(self.target_pitches) < 2 or len(self.transcribed_pitches) < 2:
            return 0.0
        n = min(len(self.target_pitches), len(self.transcribed_pitches)) - 1
        target_contour = [np.sign(self.target_pitches[i + 1] - self.target_pitches[i])
                          for i in range(n)]
        trans_contour = [np.sign(self.transcribed_pitches[i + 1] - self.transcribed_pitches[i])
                         for i in range(n)]
        correct = sum(1 for a, b in zip(target_contour, trans_contour) if a == b)
        return correct / n if n > 0 else 0.0


@dataclass
class HarmonicDictationResult:
    target_chords: List[str]
    transcribed_chords: List[str]

    def chord_accuracy(self) -> float:
        if not self.target_chords:
            return 1.0
        n = min(len(self.target_chords), len(self.transcribed_chords))
        if n == 0:
            return 0.0
        correct = sum(1 for i in range(n)
                      if self.target_chords[i] == self.transcribed_chords[i])
        return correct / len(self.target_chords)

    def root_accuracy(self) -> float:
        if not self.target_chords:
            return 1.0
        n = min(len(self.target_chords), len(self.transcribed_chords))
        if n == 0:
            return 0.0
        correct = 0
        for i in range(n):
            t_root = self.target_chords[i].rstrip("mMdimaugmaj7975+o")
            a_root = self.transcribed_chords[i].rstrip("mMdimaugmaj7975+o")
            if t_root == a_root:
                correct += 1
        return correct / len(self.target_chords)


@dataclass
class EloSkill:
    rating: float = 1000.0
    k_factor: float = 32.0
    history: List[Tuple[float, bool]] = field(default_factory=list)

    def expected_score(self, difficulty: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((difficulty - self.rating) / 400.0))

    def update(self, difficulty: float, correct: bool) -> float:
        expected = self.expected_score(difficulty)
        actual = 1.0 if correct else 0.0
        delta = self.k_factor * (actual - expected)
        self.rating += delta
        self.history.append((self.rating, correct))
        return delta

    def recent_accuracy(self, n: int = 20) -> float:
        recent = self.history[-n:]
        if not recent:
            return 0.5
        return sum(1 for _, c in recent if c) / len(recent)


INTERVAL_DIFFICULTY = {
    "P1": 200, "P8": 400, "P5": 600, "P4": 700,
    "M3": 800, "m3": 850, "M6": 900, "m6": 950,
    "M2": 1000, "m2": 1050, "M7": 1100, "m7": 1150, "TT": 1200,
}

CHORD_DIFFICULTY = {
    ChordQuality.MAJOR: 500, ChordQuality.MINOR: 600,
    ChordQuality.DIMINISHED: 800, ChordQuality.AUGMENTED: 850,
    ChordQuality.DOMINANT_7: 900, ChordQuality.MAJOR_7: 950,
    ChordQuality.MINOR_7: 1000, ChordQuality.HALF_DIM_7: 1100,
    ChordQuality.DIM_7: 1150, ChordQuality.MINOR_MAJOR_7: 1200,
}


def select_exercise_difficulty(skill: EloSkill, target_p: float = 0.7) -> float:
    return skill.rating - 400.0 * math.log10(1.0 / target_p - 1.0)


def generate_interval_exercise(skill: EloSkill,
                               rng: Optional[np.random.Generator] = None) -> Tuple[str, int]:
    rng = rng or np.random.default_rng()
    target_diff = select_exercise_difficulty(skill)
    candidates = sorted(INTERVAL_DIFFICULTY.items(),
                        key=lambda x: abs(x[1] - target_diff))
    weights = np.array([1.0 / (1.0 + abs(d - target_diff)) for _, d in candidates])
    weights /= weights.sum()
    idx = rng.choice(len(candidates), p=weights)
    name, _ = candidates[idx]
    semitones = INTERVAL_SEMITONES[name]
    base = rng.integers(48, 72)
    return name, int(base)


def generate_chord_exercise(skill: EloSkill,
                            rng: Optional[np.random.Generator] = None
                            ) -> Tuple[ChordQuality, int]:
    rng = rng or np.random.default_rng()
    target_diff = select_exercise_difficulty(skill)
    candidates = sorted(CHORD_DIFFICULTY.items(),
                        key=lambda x: abs(x[1] - target_diff))
    weights = np.array([1.0 / (1.0 + abs(d - target_diff)) for _, d in candidates])
    weights /= weights.sum()
    idx = rng.choice(len(candidates), p=weights)
    quality, _ = candidates[idx]
    root = rng.integers(48, 68)
    return quality, int(root)


def generate_rhythm_exercise(n_beats: int = 4, subdivisions: int = 4,
                             density: float = 0.5,
                             rng: Optional[np.random.Generator] = None) -> List[float]:
    rng = rng or np.random.default_rng()
    total_slots = n_beats * subdivisions
    pattern = rng.random(total_slots) < density
    pattern[0] = True
    durations = []
    i = 0
    while i < total_slots:
        if pattern[i]:
            dur = 1
            while i + dur < total_slots and not pattern[i + dur]:
                dur += 1
            durations.append(dur / subdivisions)
            i += dur
        else:
            i += 1
    return durations
