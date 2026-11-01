"""Sight-reading difficulty estimation and exercise generation.

Models sight-reading complexity per Lehmann & McArthur (2002) predictors:
rhythmic complexity, intervallic difficulty, key familiarity, texture density.
Generates graded exercises via constrained random walk. Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence
from enum import Enum


class DifficultyLevel(Enum):
    BEGINNER = 1
    ELEMENTARY = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


@dataclass
class SightReadingMetrics:
    rhythmic_complexity: float
    intervallic_difficulty: float
    key_difficulty: float
    texture_density: float
    accidental_density: float
    range_span: int
    tempo_factor: float
    overall_difficulty: float
    level: DifficultyLevel


@dataclass
class Exercise:
    pitches: List[int]
    durations: List[float]
    key_signature: int
    time_signature: Tuple[int, int]
    tempo: float
    difficulty: float


KEY_DIFFICULTY = {
    0: 0.0, 1: 0.15, 2: 0.25, -1: 0.15, -2: 0.25,
    3: 0.4, -3: 0.4, 4: 0.55, -4: 0.55,
    5: 0.7, -5: 0.7, 6: 0.85, -6: 0.85,
    7: 1.0, -7: 1.0,
}


def rhythmic_complexity(durations: Sequence[float],
                         time_sig: Tuple[int, int] = (4, 4)) -> float:
    if len(durations) < 2:
        return 0.0
    beat_dur = 4.0 / time_sig[1]
    syncopation = 0
    pos = 0.0
    for d in durations:
        on_beat = abs(pos % beat_dur) < 0.01
        if not on_beat and d >= beat_dur:
            syncopation += 1
        pos += d

    dur_set = set(round(d * 16) for d in durations)
    variety = len(dur_set) / max(len(durations), 1)

    dotted = sum(1 for d in durations if abs(d * 16 % 3) < 0.5 and d > 0.125)
    dotted_ratio = dotted / max(len(durations), 1)

    triplet = sum(1 for d in durations
                  if abs(d - round(d * 12) / 12.0) < 0.01 and d > 0)
    triplet_ratio = triplet / max(len(durations), 1)

    sync_ratio = syncopation / max(len(durations), 1)
    score = 0.35 * sync_ratio + 0.25 * variety + 0.2 * dotted_ratio + 0.2 * triplet_ratio
    return float(np.clip(score, 0, 1))


def intervallic_difficulty(pitches: Sequence[int]) -> float:
    if len(pitches) < 2:
        return 0.0
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    weights = {0: 0, 1: 0.1, 2: 0.1, 3: 0.2, 4: 0.2, 5: 0.3,
               6: 0.7, 7: 0.3, 8: 0.5, 9: 0.5, 10: 0.6, 11: 0.7, 12: 0.4}
    total = sum(weights.get(min(iv, 12), 0.8) for iv in intervals)
    return float(np.clip(total / max(len(intervals), 1), 0, 1))


def accidental_density(pitches: Sequence[int], key_sig: int = 0) -> float:
    if not pitches:
        return 0.0
    scale_pcs = set()
    root = (key_sig * 7) % 12
    major_steps = [0, 2, 4, 5, 7, 9, 11]
    for s in major_steps:
        scale_pcs.add((root + s) % 12)
    out = sum(1 for p in pitches if p % 12 not in scale_pcs)
    return float(out / len(pitches))


def texture_density(pitches: Sequence[int], durations: Sequence[float],
                     voices: int = 1) -> float:
    if not pitches or not durations:
        return 0.0
    total_time = sum(durations)
    if total_time < 0.01:
        return 0.0
    notes_per_sec = len(pitches) / total_time
    density = notes_per_sec / 8.0
    voice_factor = (voices - 1) * 0.3
    return float(np.clip(density + voice_factor, 0, 1))


def range_difficulty(pitches: Sequence[int]) -> float:
    if len(pitches) < 2:
        return 0.0
    span = max(pitches) - min(pitches)
    return float(np.clip(span / 36.0, 0, 1))


def tempo_difficulty(bpm: float) -> float:
    if bpm < 60:
        return 0.1
    if bpm < 80:
        return 0.2
    if bpm < 100:
        return 0.3
    if bpm < 120:
        return 0.5
    if bpm < 144:
        return 0.7
    if bpm < 176:
        return 0.85
    return 1.0


def estimate_difficulty(
    pitches: Sequence[int],
    durations: Sequence[float],
    key_sig: int = 0,
    time_sig: Tuple[int, int] = (4, 4),
    bpm: float = 100.0,
    voices: int = 1,
) -> SightReadingMetrics:
    rc = rhythmic_complexity(durations, time_sig)
    iv = intervallic_difficulty(pitches)
    kd = KEY_DIFFICULTY.get(key_sig, 0.5)
    td = texture_density(pitches, durations, voices)
    ad = accidental_density(pitches, key_sig)
    rng = range_difficulty(pitches)
    tf = tempo_difficulty(bpm)

    overall = (0.25 * rc + 0.20 * iv + 0.10 * kd + 0.15 * td +
               0.10 * ad + 0.05 * rng + 0.15 * tf)

    if overall < 0.15:
        level = DifficultyLevel.BEGINNER
    elif overall < 0.30:
        level = DifficultyLevel.ELEMENTARY
    elif overall < 0.50:
        level = DifficultyLevel.INTERMEDIATE
    elif overall < 0.70:
        level = DifficultyLevel.ADVANCED
    else:
        level = DifficultyLevel.EXPERT

    span = max(pitches) - min(pitches) if pitches else 0
    return SightReadingMetrics(rc, iv, kd, td, ad, span, tf, float(overall), level)


def generate_exercise(
    target_difficulty: float = 0.3,
    length: int = 16,
    key_sig: int = 0,
    time_sig: Tuple[int, int] = (4, 4),
    rng_seed: Optional[int] = None,
) -> Exercise:
    rng = np.random.RandomState(rng_seed)
    root = 60 + (key_sig * 7) % 12
    major_scale = [0, 2, 4, 5, 7, 9, 11]

    if target_difficulty < 0.2:
        intervals = [-2, -1, 0, 1, 2]
        dur_pool = [0.5, 1.0]
        tempo = 80.0
    elif target_difficulty < 0.4:
        intervals = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        dur_pool = [0.25, 0.5, 0.75, 1.0]
        tempo = 100.0
    elif target_difficulty < 0.6:
        intervals = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        dur_pool = [0.25, 0.5, 0.75, 1.0, 1.5]
        tempo = 120.0
    else:
        intervals = list(range(-7, 8))
        dur_pool = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0]
        tempo = 140.0

    pitches = [root]
    for _ in range(length - 1):
        step = rng.choice(intervals)
        p = pitches[-1] + step
        p = np.clip(p, root - 12, root + 24)
        nearest = min(major_scale, key=lambda s: abs((p % 12) - (root + s) % 12))
        p = (p // 12) * 12 + (root + nearest) % 12
        pitches.append(int(p))

    durations = [float(rng.choice(dur_pool)) for _ in range(length)]
    difficulty = estimate_difficulty(pitches, durations, key_sig, time_sig, tempo).overall_difficulty

    return Exercise(pitches, durations, key_sig, time_sig, tempo, difficulty)


def adaptive_exercise_sequence(
    current_score: float,
    n_exercises: int = 5,
    step_size: float = 0.05,
    rng_seed: Optional[int] = None,
) -> List[Exercise]:
    exercises = []
    target = current_score
    for i in range(n_exercises):
        target = np.clip(target + step_size, 0.05, 0.95)
        ex = generate_exercise(target_difficulty=target, rng_seed=(rng_seed or 0) + i)
        exercises.append(ex)
    return exercises


def reading_speed_estimate(
    pitches: Sequence[int],
    durations: Sequence[float],
    accuracy: float,
    time_taken: float,
) -> float:
    if time_taken <= 0:
        return 0.0
    expected_time = sum(durations)
    speed_ratio = expected_time / time_taken
    return float(speed_ratio * accuracy)


def weakness_profile(
    results: Sequence[Tuple[SightReadingMetrics, float]],
) -> dict:
    if not results:
        return {}
    dims = ["rhythmic", "intervallic", "key", "texture", "accidentals"]
    weakness = {d: 0.0 for d in dims}
    for metrics, score in results:
        inv_score = 1.0 - score
        weakness["rhythmic"] += metrics.rhythmic_complexity * inv_score
        weakness["intervallic"] += metrics.intervallic_difficulty * inv_score
        weakness["key"] += metrics.key_difficulty * inv_score
        weakness["texture"] += metrics.texture_density * inv_score
        weakness["accidentals"] += metrics.accidental_density * inv_score
    n = len(results)
    return {k: v / n for k, v in weakness.items()}
