"""Adaptive music exercise generation: intervals, chords, rhythms, dictation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class DifficultyLevel(Enum):
    BEGINNER = 1
    ELEMENTARY = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


class ExerciseType(Enum):
    INTERVAL_RECOGNITION = auto()
    CHORD_IDENTIFICATION = auto()
    RHYTHM_READING = auto()
    MELODIC_DICTATION = auto()
    SCALE_PRACTICE = auto()
    TRANSPOSITION = auto()


_INTERVAL_NAMES = [
    "P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7", "P8"
]

_INTERVALS_BY_LEVEL: dict[DifficultyLevel, list[int]] = {
    DifficultyLevel.BEGINNER: [0, 3, 4, 5, 7, 12],
    DifficultyLevel.ELEMENTARY: [0, 2, 3, 4, 5, 7, 8, 9, 12],
    DifficultyLevel.INTERMEDIATE: list(range(13)),
    DifficultyLevel.ADVANCED: list(range(13)),
    DifficultyLevel.EXPERT: list(range(13)),
}

_CHORD_TYPES_BY_LEVEL: dict[DifficultyLevel, list[tuple[str, tuple[int, ...]]]] = {
    DifficultyLevel.BEGINNER: [("major", (0, 4, 7)), ("minor", (0, 3, 7))],
    DifficultyLevel.ELEMENTARY: [
        ("major", (0, 4, 7)), ("minor", (0, 3, 7)),
        ("diminished", (0, 3, 6)), ("augmented", (0, 4, 8)),
    ],
    DifficultyLevel.INTERMEDIATE: [
        ("major", (0, 4, 7)), ("minor", (0, 3, 7)),
        ("dim", (0, 3, 6)), ("aug", (0, 4, 8)),
        ("dom7", (0, 4, 7, 10)), ("maj7", (0, 4, 7, 11)), ("min7", (0, 3, 7, 10)),
    ],
    DifficultyLevel.ADVANCED: [
        ("maj7", (0, 4, 7, 11)), ("min7", (0, 3, 7, 10)),
        ("dom7", (0, 4, 7, 10)), ("dim7", (0, 3, 6, 9)),
        ("m7b5", (0, 3, 6, 10)), ("aug7", (0, 4, 8, 10)),
        ("min_maj7", (0, 3, 7, 11)),
    ],
    DifficultyLevel.EXPERT: [
        ("dom9", (0, 4, 7, 10, 14)), ("maj9", (0, 4, 7, 11, 14)),
        ("min9", (0, 3, 7, 10, 14)), ("dom13", (0, 4, 7, 10, 14, 21)),
        ("alt", (0, 4, 8, 10, 13)),
    ],
}


@dataclass
class Exercise:
    exercise_type: ExerciseType
    difficulty: DifficultyLevel
    content: dict[str, Any] = field(default_factory=dict)
    answer: Any = None
    hints: list[str] = field(default_factory=list)
    time_limit_seconds: float = 30.0


class ExerciseGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def interval_exercise(self, difficulty: DifficultyLevel = DifficultyLevel.BEGINNER) -> Exercise:
        pool = _INTERVALS_BY_LEVEL.get(difficulty, list(range(13)))
        semitones = int(self.rng.choice(pool))
        root = int(self.rng.randint(48, 73))
        return Exercise(
            exercise_type=ExerciseType.INTERVAL_RECOGNITION,
            difficulty=difficulty,
            content={"root_midi": root, "semitones": semitones, "direction": "ascending"},
            answer=_INTERVAL_NAMES[semitones] if semitones < len(_INTERVAL_NAMES) else f"{semitones}st",
            hints=["Listen for the quality (major/minor/perfect)", "Sing the interval"],
            time_limit_seconds=15.0 + (5 - difficulty.value) * 5,
        )

    def chord_exercise(self, difficulty: DifficultyLevel = DifficultyLevel.BEGINNER) -> Exercise:
        pool = _CHORD_TYPES_BY_LEVEL.get(difficulty, _CHORD_TYPES_BY_LEVEL[DifficultyLevel.BEGINNER])
        idx = int(self.rng.randint(len(pool)))
        name, intervals = pool[idx]
        root = int(self.rng.randint(48, 73))
        midi_notes = tuple(root + i for i in intervals)
        return Exercise(
            exercise_type=ExerciseType.CHORD_IDENTIFICATION,
            difficulty=difficulty,
            content={"root_midi": root, "midi_notes": midi_notes},
            answer=name,
            hints=["Focus on the third (major vs minor)", "Count the notes"],
            time_limit_seconds=20.0 + (5 - difficulty.value) * 5,
        )

    def rhythm_exercise(self, difficulty: DifficultyLevel = DifficultyLevel.BEGINNER) -> Exercise:
        if difficulty.value <= 2:
            beats = 4
            subdivisions = [1, 1, 2, 1]
        elif difficulty.value == 3:
            beats = int(self.rng.choice([3, 4, 6]))
            subdivisions = [int(self.rng.choice([1, 2, 3])) for _ in range(beats)]
        else:
            beats = int(self.rng.choice([5, 7, 9, 11]))
            subdivisions = [int(self.rng.choice([1, 2, 3, 4])) for _ in range(beats)]
        return Exercise(
            exercise_type=ExerciseType.RHYTHM_READING,
            difficulty=difficulty,
            content={"beats": beats, "subdivisions": subdivisions},
            answer=subdivisions,
            time_limit_seconds=10.0 * beats,
        )

    def melodic_dictation(
        self, difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
        n_notes: int = 4,
    ) -> Exercise:
        if difficulty.value <= 2:
            scale = [0, 2, 4, 5, 7]
        elif difficulty.value == 3:
            scale = [0, 2, 3, 5, 7, 8, 10]
        else:
            scale = list(range(12))
        root = int(self.rng.randint(60, 73))
        melody = [root]
        for _ in range(n_notes - 1):
            step = int(self.rng.choice(scale))
            melody.append(root + step)
        return Exercise(
            exercise_type=ExerciseType.MELODIC_DICTATION,
            difficulty=difficulty,
            content={"root_midi": root, "n_notes": n_notes},
            answer=melody,
            time_limit_seconds=5.0 * n_notes + 10,
        )

    def generate_batch(
        self,
        exercise_type: ExerciseType,
        difficulty: DifficultyLevel,
        count: int = 10,
    ) -> list[Exercise]:
        generators = {
            ExerciseType.INTERVAL_RECOGNITION: self.interval_exercise,
            ExerciseType.CHORD_IDENTIFICATION: self.chord_exercise,
            ExerciseType.RHYTHM_READING: self.rhythm_exercise,
            ExerciseType.MELODIC_DICTATION: self.melodic_dictation,
        }
        gen = generators.get(exercise_type, self.interval_exercise)
        return [gen(difficulty) for _ in range(count)]


def grade_answer(exercise: Exercise, student_answer: Any) -> tuple[bool, float, str]:
    if exercise.answer is None:
        return True, 1.0, "No answer required"
    if isinstance(exercise.answer, str):
        correct = str(student_answer).strip().lower() == exercise.answer.strip().lower()
        return correct, 1.0 if correct else 0.0, "Correct!" if correct else f"Expected: {exercise.answer}"
    if isinstance(exercise.answer, list):
        if not isinstance(student_answer, list):
            return False, 0.0, "Expected a list answer"
        matches = sum(1 for a, b in zip(exercise.answer, student_answer) if a == b)
        total = max(len(exercise.answer), 1)
        score = matches / total
        return score >= 0.8, score, f"{matches}/{total} correct"
    correct = student_answer == exercise.answer
    return correct, 1.0 if correct else 0.0, "Correct!" if correct else f"Expected: {exercise.answer}"
