"""Ear training: interval recognition, chord identification, rhythm dictation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

INTERVAL_NAMES = [
    "unison", "minor 2nd", "major 2nd", "minor 3rd", "major 3rd",
    "perfect 4th", "tritone", "perfect 5th", "minor 6th", "major 6th",
    "minor 7th", "major 7th", "octave",
]

CHORD_TYPES = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "dominant 7th": [0, 4, 7, 10],
    "major 7th": [0, 4, 7, 11],
    "minor 7th": [0, 3, 7, 10],
    "half-diminished 7th": [0, 3, 6, 10],
    "diminished 7th": [0, 3, 6, 9],
}


@dataclass
class ExerciseResult:
    correct: bool
    expected: str
    given: str
    response_time_ms: float = 0.0


class IntervalTrainer:
    def __init__(self, difficulty: int = 1):
        self._difficulty = min(max(difficulty, 1), 5)
        self._history: list[ExerciseResult] = []

    @property
    def available_intervals(self) -> list[int]:
        if self._difficulty <= 1:
            return [0, 5, 7, 12]
        elif self._difficulty <= 2:
            return [0, 3, 4, 5, 7, 12]
        elif self._difficulty <= 3:
            return list(range(0, 13))
        else:
            return list(range(0, 25))

    def generate_exercise(self, root: int = 60) -> dict:
        intervals = self.available_intervals
        semitones = random.choice(intervals)
        return {
            "root_midi": root,
            "target_midi": root + semitones,
            "semitones": semitones,
            "interval_name": INTERVAL_NAMES[semitones % 13],
        }

    def check_answer(self, exercise: dict, answer: str) -> ExerciseResult:
        expected = exercise["interval_name"]
        correct = answer.lower().strip() == expected.lower()
        result = ExerciseResult(correct=correct, expected=expected, given=answer)
        self._history.append(result)
        return result

    def accuracy(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if r.correct) / len(self._history)


class ChordTrainer:
    def __init__(self, chord_types: list[str] | None = None):
        self._types = chord_types or ["major", "minor", "diminished", "augmented"]
        self._history: list[ExerciseResult] = []

    def generate_exercise(self, root: int = 60) -> dict:
        chord_type = random.choice(self._types)
        intervals = CHORD_TYPES.get(chord_type, [0, 4, 7])
        return {
            "root_midi": root,
            "chord_type": chord_type,
            "midi_notes": [root + i for i in intervals],
        }

    def check_answer(self, exercise: dict, answer: str) -> ExerciseResult:
        expected = exercise["chord_type"]
        correct = answer.lower().strip() == expected.lower()
        result = ExerciseResult(correct=correct, expected=expected, given=answer)
        self._history.append(result)
        return result

    def accuracy(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for r in self._history if r.correct) / len(self._history)


class RhythmDictation:
    PATTERNS = {
        1: [[1, 1, 1, 1], [2, 2], [2, 1, 1], [1, 1, 2]],
        2: [[1, 0.5, 0.5, 1, 1], [0.5, 0.5, 0.5, 0.5, 2], [1.5, 0.5, 1, 1]],
        3: [[0.5, 0.5, 0.25, 0.25, 0.5, 2], [1, 0.75, 0.25, 1, 1]],
    }

    def __init__(self, difficulty: int = 1):
        self._difficulty = min(max(difficulty, 1), 3)

    def generate_pattern(self) -> list[float]:
        patterns = self.PATTERNS.get(self._difficulty, self.PATTERNS[1])
        return list(random.choice(patterns))

    @staticmethod
    def check_pattern(expected: list[float], given: list[float], tolerance: float = 0.1) -> float:
        if len(expected) != len(given):
            return 0.0
        matches = sum(
            1 for e, g in zip(expected, given) if abs(e - g) <= tolerance
        )
        return matches / len(expected)
