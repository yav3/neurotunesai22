from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class TransitionType(Enum):
    SAME_KEY = "same_key"
    ADJACENT_KEY = "adjacent_key"
    PARALLEL_KEY = "parallel_key"
    ENERGY_BOOST = "energy_boost"
    ENERGY_DROP = "energy_drop"
    TEMPO_MATCH = "tempo_match"
    MOOD_SHIFT = "mood_shift"
    INCOMPATIBLE = "incompatible"


@dataclass
class TransitionScore:
    overall: float
    harmonic: float
    tempo: float
    energy: float
    mood: float
    transition_type: TransitionType
    bpm_ratio: float
    key_distance: int


CAMELOT_NUMBERS = {
    "Ab minor": (1, "A"), "B major": (1, "B"),
    "Eb minor": (2, "A"), "Gb major": (2, "B"),
    "Bb minor": (3, "A"), "Db major": (3, "B"),
    "F minor": (4, "A"), "Ab major": (4, "B"),
    "C minor": (5, "A"), "Eb major": (5, "B"),
    "G minor": (6, "A"), "Bb major": (6, "B"),
    "D minor": (7, "A"), "F major": (7, "B"),
    "A minor": (8, "A"), "C major": (8, "B"),
    "E minor": (9, "A"), "G major": (9, "B"),
    "B minor": (10, "A"), "D major": (10, "B"),
    "F# minor": (11, "A"), "A major": (11, "B"),
    "Db minor": (12, "A"), "E major": (12, "B"),
    "C# minor": (12, "A"),
}


def _camelot_distance(key1: str, key2: str) -> int:
    c1 = CAMELOT_NUMBERS.get(key1)
    c2 = CAMELOT_NUMBERS.get(key2)
    if c1 is None or c2 is None:
        return 7
    num1, let1 = c1
    num2, let2 = c2
    num_dist = min(abs(num1 - num2), 12 - abs(num1 - num2))
    if let1 == let2:
        return num_dist
    if num_dist == 0:
        return 1
    return num_dist + 1


class TransitionScorer:
    def __init__(
        self,
        w_harmonic: float = 0.35,
        w_tempo: float = 0.25,
        w_energy: float = 0.25,
        w_mood: float = 0.15,
    ):
        self.w_harmonic = w_harmonic
        self.w_tempo = w_tempo
        self.w_energy = w_energy
        self.w_mood = w_mood

    def score(
        self,
        track_a: dict[str, Any],
        track_b: dict[str, Any],
    ) -> TransitionScore:
        key_a = track_a.get("key", "C major")
        key_b = track_b.get("key", "C major")
        bpm_a = float(track_a.get("bpm", 120))
        bpm_b = float(track_b.get("bpm", 120))
        energy_a = float(track_a.get("energy", 0.5))
        energy_b = float(track_b.get("energy", 0.5))
        valence_a = float(track_a.get("valence", 0.5))
        valence_b = float(track_b.get("valence", 0.5))
        arousal_a = float(track_a.get("arousal", 0.5))
        arousal_b = float(track_b.get("arousal", 0.5))

        key_dist = _camelot_distance(key_a, key_b)
        harmonic = max(0.0, 1.0 - key_dist / 6.0)

        bpm_ratio = min(bpm_a, bpm_b) / max(bpm_a, bpm_b) if max(bpm_a, bpm_b) > 0 else 1.0
        double_time = min(bpm_a, bpm_b) * 2
        half_time = max(bpm_a, bpm_b) / 2
        if abs(double_time - max(bpm_a, bpm_b)) < 5:
            bpm_ratio = max(bpm_ratio, 0.85)
        if abs(half_time - min(bpm_a, bpm_b)) < 5:
            bpm_ratio = max(bpm_ratio, 0.85)
        tempo = bpm_ratio

        energy_diff = abs(energy_a - energy_b)
        energy_score = 1.0 - energy_diff

        mood_dist = np.sqrt((valence_a - valence_b) ** 2 + (arousal_a - arousal_b) ** 2)
        mood = max(0.0, 1.0 - mood_dist / np.sqrt(2))

        overall = (
            self.w_harmonic * harmonic
            + self.w_tempo * tempo
            + self.w_energy * energy_score
            + self.w_mood * mood
        )

        transition_type = self._classify_transition(
            key_dist, bpm_ratio, energy_a, energy_b, mood_dist,
        )

        return TransitionScore(
            overall=float(np.clip(overall, 0, 1)),
            harmonic=harmonic,
            tempo=tempo,
            energy=energy_score,
            mood=mood,
            transition_type=transition_type,
            bpm_ratio=bpm_ratio,
            key_distance=key_dist,
        )

    def _classify_transition(
        self,
        key_dist: int,
        bpm_ratio: float,
        energy_a: float,
        energy_b: float,
        mood_dist: float,
    ) -> TransitionType:
        if key_dist == 0:
            return TransitionType.SAME_KEY
        if key_dist == 1:
            return TransitionType.ADJACENT_KEY
        energy_diff = energy_b - energy_a
        if energy_diff > 0.3:
            return TransitionType.ENERGY_BOOST
        if energy_diff < -0.3:
            return TransitionType.ENERGY_DROP
        if bpm_ratio > 0.95:
            return TransitionType.TEMPO_MATCH
        if mood_dist > 0.7:
            return TransitionType.MOOD_SHIFT
        if key_dist <= 2:
            return TransitionType.PARALLEL_KEY
        return TransitionType.INCOMPATIBLE

    def rank_transitions(
        self,
        current_track: dict[str, Any],
        candidates: list[dict[str, Any]],
        top_n: int = 10,
    ) -> list[tuple[int, TransitionScore]]:
        scored = []
        for i, candidate in enumerate(candidates):
            score = self.score(current_track, candidate)
            scored.append((i, score))
        scored.sort(key=lambda x: x[1].overall, reverse=True)
        return scored[:top_n]
