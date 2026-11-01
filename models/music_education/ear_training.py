"""Ear training: interval recognition, chord identification, melodic dictation scoring."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class IntervalQuality(Enum):
    PERFECT = "perfect"
    MAJOR = "major"
    MINOR = "minor"
    AUGMENTED = "augmented"
    DIMINISHED = "diminished"


INTERVAL_NAMES = {
    0: ("P", 1), 1: ("m", 2), 2: ("M", 2), 3: ("m", 3), 4: ("M", 3),
    5: ("P", 4), 6: ("A", 4), 7: ("P", 5), 8: ("m", 6), 9: ("M", 6),
    10: ("m", 7), 11: ("M", 7), 12: ("P", 8),
}

INTERVAL_DIFFICULTY = {
    0: 1, 7: 1, 12: 1,
    5: 2, 4: 2, 3: 3, 9: 3,
    2: 3, 10: 4, 8: 4, 1: 5, 11: 5, 6: 5,
}


@dataclass(frozen=True)
class IntervalQuestion:
    root_midi: int
    interval_semitones: int
    ascending: bool = True
    harmonic: bool = False


@dataclass(frozen=True)
class ChordQuestion:
    root_midi: int
    chord_type: str
    inversion: int = 0


@dataclass(frozen=True)
class DictationExercise:
    notes_midi: List[int]
    rhythm_beats: List[float]
    key_center: int = 60
    tempo_bpm: float = 80.0


@dataclass(frozen=True)
class EarTrainingScore:
    correct: int
    total: int
    accuracy: float
    avg_response_time_ms: float
    difficulty_breakdown: Dict[int, float]


def interval_name(semitones: int) -> str:
    s = abs(semitones) % 12
    if s in INTERVAL_NAMES:
        quality, degree = INTERVAL_NAMES[s]
        return f"{quality}{degree}"
    return f"?{s}"


def interval_difficulty(semitones: int) -> int:
    return INTERVAL_DIFFICULTY.get(abs(semitones) % 12, 3)


def generate_interval_question(difficulty: int = 3,
                                rng: Optional[np.random.RandomState] = None) -> IntervalQuestion:
    if rng is None:
        rng = np.random.RandomState()
    candidates = [s for s, d in INTERVAL_DIFFICULTY.items() if d <= difficulty]
    if not candidates:
        candidates = [0, 7, 12]
    semitones = int(rng.choice(candidates))
    root = int(rng.randint(48, 72))
    ascending = bool(rng.choice([True, False]))
    harmonic = bool(rng.choice([True, False]))
    return IntervalQuestion(root, semitones, ascending, harmonic)


def score_interval_answer(question: IntervalQuestion, answer_semitones: int) -> bool:
    return abs(question.interval_semitones) % 12 == abs(answer_semitones) % 12


CHORD_TEMPLATES = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dim7": (0, 3, 6, 9),
}

CHORD_DIFFICULTY = {
    "major": 1, "minor": 1,
    "diminished": 2, "augmented": 2,
    "dom7": 3, "maj7": 3, "min7": 3,
    "dim7": 4,
}


def chord_pitches(root_midi: int, chord_type: str,
                  inversion: int = 0) -> Tuple[int, ...]:
    template = CHORD_TEMPLATES.get(chord_type, (0,))
    pitches = [root_midi + t for t in template]
    for _ in range(inversion % len(pitches)):
        pitches.append(pitches.pop(0) + 12)
    return tuple(pitches)


def generate_chord_question(difficulty: int = 2,
                             rng: Optional[np.random.RandomState] = None) -> ChordQuestion:
    if rng is None:
        rng = np.random.RandomState()
    candidates = [c for c, d in CHORD_DIFFICULTY.items() if d <= difficulty]
    if not candidates:
        candidates = ["major", "minor"]
    chord_type = str(rng.choice(candidates))
    root = int(rng.randint(48, 72))
    inversion = int(rng.randint(0, len(CHORD_TEMPLATES.get(chord_type, (0,)))))
    return ChordQuestion(root, chord_type, inversion)


def score_chord_answer(question: ChordQuestion, answer_type: str) -> bool:
    return question.chord_type == answer_type


def generate_dictation(length: int = 8, key: int = 60,
                       scale_type: str = "major",
                       rng: Optional[np.random.RandomState] = None) -> DictationExercise:
    if rng is None:
        rng = np.random.RandomState()
    if scale_type == "major":
        scale = [0, 2, 4, 5, 7, 9, 11]
    elif scale_type == "minor":
        scale = [0, 2, 3, 5, 7, 8, 10]
    else:
        scale = [0, 2, 4, 5, 7, 9, 11]
    notes = []
    current_idx = 3
    for _ in range(length):
        step = int(rng.choice([-2, -1, 0, 1, 2]))
        current_idx = max(0, min(len(scale) - 1, current_idx + step))
        octave_offset = 0
        if current_idx >= len(scale):
            octave_offset = 12
            current_idx = current_idx % len(scale)
        notes.append(key + scale[current_idx] + octave_offset)
    rhythm_options = [0.5, 1.0, 1.0, 1.5, 2.0]
    rhythm = [float(rng.choice(rhythm_options)) for _ in range(length)]
    return DictationExercise(notes, rhythm, key)


def score_dictation(exercise: DictationExercise,
                    answer_midi: List[int]) -> float:
    if not exercise.notes_midi or not answer_midi:
        return 0.0
    n = min(len(exercise.notes_midi), len(answer_midi))
    correct = sum(1 for i in range(n)
                  if exercise.notes_midi[i] == answer_midi[i])
    length_penalty = abs(len(exercise.notes_midi) - len(answer_midi)) * 0.1
    return max(0.0, correct / len(exercise.notes_midi) - length_penalty)


def contour_score(exercise: DictationExercise,
                  answer_midi: List[int]) -> float:
    if len(exercise.notes_midi) < 2 or len(answer_midi) < 2:
        return 0.0
    n = min(len(exercise.notes_midi), len(answer_midi)) - 1
    correct_contour = 0
    for i in range(n):
        expected = np.sign(exercise.notes_midi[i + 1] - exercise.notes_midi[i])
        actual = np.sign(answer_midi[i + 1] - answer_midi[i])
        if expected == actual:
            correct_contour += 1
    return correct_contour / max(n, 1)


def adaptive_difficulty(scores: List[float], current: int,
                        threshold_up: float = 0.8,
                        threshold_down: float = 0.4) -> int:
    if len(scores) < 3:
        return current
    recent = np.mean(scores[-5:])
    if recent >= threshold_up:
        return min(current + 1, 8)
    elif recent <= threshold_down:
        return max(current - 1, 1)
    return current


def session_summary(scores: List[float],
                    response_times_ms: List[float],
                    difficulties: List[int]) -> EarTrainingScore:
    correct = sum(1 for s in scores if s >= 0.99)
    total = len(scores)
    accuracy = correct / max(total, 1)
    avg_rt = float(np.mean(response_times_ms)) if response_times_ms else 0.0
    breakdown: Dict[int, float] = {}
    for d, s in zip(difficulties, scores):
        if d not in breakdown:
            breakdown[d] = []
        breakdown[d].append(s)
    breakdown_avg = {d: float(np.mean(v)) for d, v in breakdown.items()}
    return EarTrainingScore(correct, total, accuracy, avg_rt, breakdown_avg)
