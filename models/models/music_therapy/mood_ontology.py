"""Mood ontology: mood taxonomy, mood-music mapping, valence-arousal projection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class MoodCategory(Enum):
    HAPPY = "happy"
    SAD = "sad"
    CALM = "calm"
    ANXIOUS = "anxious"
    ENERGETIC = "energetic"
    MELANCHOLIC = "melancholic"
    ANGRY = "angry"
    PEACEFUL = "peaceful"
    NOSTALGIC = "nostalgic"
    HOPEFUL = "hopeful"


MOOD_VA_COORDINATES: dict[MoodCategory, tuple[float, float]] = {
    MoodCategory.HAPPY: (0.8, 0.7),
    MoodCategory.SAD: (-0.6, -0.4),
    MoodCategory.CALM: (0.3, -0.6),
    MoodCategory.ANXIOUS: (-0.4, 0.7),
    MoodCategory.ENERGETIC: (0.5, 0.9),
    MoodCategory.MELANCHOLIC: (-0.5, -0.3),
    MoodCategory.ANGRY: (-0.7, 0.8),
    MoodCategory.PEACEFUL: (0.4, -0.7),
    MoodCategory.NOSTALGIC: (-0.2, -0.2),
    MoodCategory.HOPEFUL: (0.6, 0.3),
}


@dataclass(frozen=True)
class MoodState:
    valence: float
    arousal: float

    @property
    def quadrant(self) -> str:
        if self.valence >= 0 and self.arousal >= 0:
            return "Q1_happy_energetic"
        if self.valence < 0 and self.arousal >= 0:
            return "Q2_tense_angry"
        if self.valence < 0 and self.arousal < 0:
            return "Q3_sad_depressed"
        return "Q4_calm_relaxed"


@dataclass(frozen=True)
class MoodTransition:
    source: MoodCategory
    target: MoodCategory
    distance: float
    recommended_steps: int


def mood_distance(a: MoodCategory, b: MoodCategory) -> float:
    va, aa = MOOD_VA_COORDINATES[a]
    vb, ab = MOOD_VA_COORDINATES[b]
    return float(np.sqrt((va - vb) ** 2 + (aa - ab) ** 2))


def classify_mood(valence: float, arousal: float) -> MoodCategory:
    best: MoodCategory = MoodCategory.CALM
    best_dist = float("inf")
    for mood, (v, a) in MOOD_VA_COORDINATES.items():
        d = (valence - v) ** 2 + (arousal - a) ** 2
        if d < best_dist:
            best_dist = d
            best = mood
    return best


def mood_trajectory(
    start: MoodCategory,
    end: MoodCategory,
    n_steps: int = 5,
) -> list[MoodState]:
    v_start, a_start = MOOD_VA_COORDINATES[start]
    v_end, a_end = MOOD_VA_COORDINATES[end]
    vs = np.linspace(v_start, v_end, n_steps)
    ars = np.linspace(a_start, a_end, n_steps)
    return [MoodState(valence=float(v), arousal=float(a)) for v, a in zip(vs, ars)]


def plan_mood_transition(
    source: MoodCategory,
    target: MoodCategory,
) -> MoodTransition:
    dist = mood_distance(source, target)
    steps = max(2, int(np.ceil(dist * 4)))
    return MoodTransition(
        source=source,
        target=target,
        distance=dist,
        recommended_steps=steps,
    )


def mood_music_features(mood: MoodCategory) -> dict[str, float]:
    v, a = MOOD_VA_COORDINATES[mood]
    tempo = 80 + 60 * (a + 1) / 2
    energy = (a + 1) / 2
    brightness = (v + 1) / 2
    return {
        "target_tempo": float(np.clip(tempo, 40, 200)),
        "target_energy": float(np.clip(energy, 0, 1)),
        "target_brightness": float(np.clip(brightness, 0, 1)),
        "target_valence": float(np.clip((v + 1) / 2, 0, 1)),
    }


def mood_consistency(
    mood_sequence: Sequence[MoodCategory],
) -> float:
    if len(mood_sequence) < 2:
        return 1.0
    distances = [
        mood_distance(mood_sequence[i], mood_sequence[i + 1])
        for i in range(len(mood_sequence) - 1)
    ]
    max_dist = np.sqrt(2) * 2
    mean_dist = float(np.mean(distances))
    return max(0.0, 1.0 - mean_dist / max_dist)
