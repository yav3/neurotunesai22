"""Mood and tag taxonomies: hierarchical mood space, tag prediction, cross-modal mapping."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class MoodCategory(Enum):
    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"
    AGGRESSIVE = "aggressive"
    ROMANTIC = "romantic"
    MELANCHOLIC = "melancholic"
    TRIUMPHANT = "triumphant"
    MYSTERIOUS = "mysterious"
    PLAYFUL = "playful"
    DARK = "dark"
    ETHEREAL = "ethereal"
    NOSTALGIC = "nostalgic"
    ANXIOUS = "anxious"
    PEACEFUL = "peaceful"
    EPIC = "epic"


MOOD_VAD_MAP: dict[MoodCategory, tuple[float, float, float]] = {
    MoodCategory.HAPPY: (0.8, 0.6, 0.7),
    MoodCategory.SAD: (0.2, 0.2, 0.3),
    MoodCategory.ENERGETIC: (0.7, 0.9, 0.7),
    MoodCategory.CALM: (0.6, 0.1, 0.5),
    MoodCategory.AGGRESSIVE: (0.2, 0.9, 0.8),
    MoodCategory.ROMANTIC: (0.7, 0.3, 0.4),
    MoodCategory.MELANCHOLIC: (0.3, 0.3, 0.3),
    MoodCategory.TRIUMPHANT: (0.9, 0.8, 0.9),
    MoodCategory.MYSTERIOUS: (0.4, 0.4, 0.3),
    MoodCategory.PLAYFUL: (0.8, 0.7, 0.5),
    MoodCategory.DARK: (0.2, 0.5, 0.6),
    MoodCategory.ETHEREAL: (0.6, 0.2, 0.2),
    MoodCategory.NOSTALGIC: (0.4, 0.3, 0.3),
    MoodCategory.ANXIOUS: (0.2, 0.7, 0.4),
    MoodCategory.PEACEFUL: (0.7, 0.1, 0.4),
    MoodCategory.EPIC: (0.7, 0.8, 0.8),
}


@dataclass(frozen=True)
class MoodTag:
    category: MoodCategory
    confidence: float = 1.0
    intensity: float = 0.5

    @property
    def vad(self) -> tuple[float, float, float]:
        base = MOOD_VAD_MAP[self.category]
        return (
            base[0] * self.intensity + 0.5 * (1 - self.intensity),
            base[1] * self.intensity + 0.5 * (1 - self.intensity),
            base[2] * self.intensity + 0.5 * (1 - self.intensity),
        )


def vad_to_mood(valence: float, arousal: float, dominance: float,
                top_k: int = 3) -> list[tuple[MoodCategory, float]]:
    query = np.array([valence, arousal, dominance])
    scores: list[tuple[MoodCategory, float]] = []
    for mood, vad in MOOD_VAD_MAP.items():
        ref = np.array(vad)
        dist = float(np.linalg.norm(query - ref))
        similarity = max(0.0, 1.0 - dist / np.sqrt(3))
        scores.append((mood, similarity))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


MUSIC_TAG_TAXONOMY: dict[str, list[str]] = {
    "genre": [
        "rock", "pop", "jazz", "classical", "electronic", "hip-hop", "r&b",
        "country", "folk", "blues", "metal", "punk", "reggae", "latin",
        "world", "ambient", "indie", "soul", "funk", "disco",
    ],
    "era": [
        "baroque", "classical_era", "romantic", "20th_century", "contemporary",
        "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s",
    ],
    "instrument": [
        "guitar", "piano", "drums", "bass", "violin", "saxophone", "trumpet",
        "synthesizer", "vocals", "flute", "cello", "organ", "harmonica",
    ],
    "tempo": ["very_slow", "slow", "moderate", "fast", "very_fast"],
    "mood": [m.value for m in MoodCategory],
    "use_case": [
        "study", "workout", "relaxation", "party", "driving", "sleep",
        "meditation", "focus", "romance", "celebration",
    ],
}


@dataclass
class TagProfile:
    tags: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def add_tag(self, category: str, tag: str, confidence: float = 1.0) -> None:
        if category not in self.tags:
            self.tags[category] = []
        self.tags[category].append((tag, confidence))

    def top_tags(self, category: str, k: int = 3) -> list[tuple[str, float]]:
        if category not in self.tags:
            return []
        return sorted(self.tags[category], key=lambda x: -x[1])[:k]

    def all_tags(self, min_confidence: float = 0.5) -> list[tuple[str, str, float]]:
        result: list[tuple[str, str, float]] = []
        for cat, tags in self.tags.items():
            for tag, conf in tags:
                if conf >= min_confidence:
                    result.append((cat, tag, conf))
        result.sort(key=lambda x: -x[2])
        return result


def tag_similarity(profile_a: TagProfile, profile_b: TagProfile) -> float:
    tags_a = {(cat, tag) for cat, tag, _ in profile_a.all_tags(0.0)}
    tags_b = {(cat, tag) for cat, tag, _ in profile_b.all_tags(0.0)}
    if not tags_a and not tags_b:
        return 1.0
    if not tags_a or not tags_b:
        return 0.0
    return len(tags_a & tags_b) / len(tags_a | tags_b)
