"""Playlist transition quality analysis between tracks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TransitionFeature(Enum):
    KEY = "key"
    TEMPO = "tempo"
    ENERGY = "energy"
    TIMBRE = "timbre"
    GENRE = "genre"
    MOOD = "mood"


@dataclass
class TransitionScore:
    feature: TransitionFeature
    similarity: float
    penalty: float


def key_compatibility(key1: int, mode1: int, key2: int, mode2: int) -> float:
    """Circle-of-fifths distance normalised to [0, 1]."""
    fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    d = abs(fifths[key1 % 12] - fifths[key2 % 12])
    d = min(d, 12 - d)
    key_sim = 1.0 - d / 6.0
    mode_bonus = 0.1 if mode1 == mode2 else -0.05
    return max(0.0, min(1.0, key_sim + mode_bonus))


def tempo_compatibility(bpm1: float, bpm2: float) -> float:
    """Ratio-based tempo compatibility; halving/doubling acceptable."""
    if bpm1 <= 0 or bpm2 <= 0:
        return 0.0
    ratio = max(bpm1, bpm2) / min(bpm1, bpm2)
    if ratio <= 1.05:
        return 1.0
    if abs(ratio - 2.0) < 0.1:
        return 0.8
    if abs(ratio - 0.5) < 0.05:
        return 0.8
    return max(0.0, 1.0 - (ratio - 1.0) / 2.0)


def energy_flow(e1: float, e2: float, target_direction: str = "maintain") -> float:
    """Penalise abrupt energy shifts unless intentional."""
    diff = e2 - e1
    if target_direction == "maintain":
        return max(0.0, 1.0 - abs(diff) * 2)
    if target_direction == "increase":
        return max(0.0, min(1.0, diff * 3 + 0.5))
    if target_direction == "decrease":
        return max(0.0, min(1.0, -diff * 3 + 0.5))
    return 1.0 - abs(diff)


def compute_transition_score(
    track1_features: dict,
    track2_features: dict,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted transition score between two tracks."""
    if weights is None:
        weights = {"key": 0.3, "tempo": 0.3, "energy": 0.25, "timbre": 0.15}
    score = 0.0
    total_w = 0.0
    if "key" in weights:
        ks = key_compatibility(
            track1_features.get("key", 0), track1_features.get("mode", 1),
            track2_features.get("key", 0), track2_features.get("mode", 1),
        )
        score += weights["key"] * ks
        total_w += weights["key"]
    if "tempo" in weights:
        ts = tempo_compatibility(track1_features.get("tempo", 120), track2_features.get("tempo", 120))
        score += weights["tempo"] * ts
        total_w += weights["tempo"]
    if "energy" in weights:
        es = energy_flow(track1_features.get("energy", 0.5), track2_features.get("energy", 0.5))
        score += weights["energy"] * es
        total_w += weights["energy"]
    return score / total_w if total_w > 0 else 0.0


def optimal_crossfade_point(
    audio1_envelope: np.ndarray,
    audio2_envelope: np.ndarray,
    sr: int = 44100,
) -> float:
    """Find energy-matched crossfade point (seconds from end of track 1)."""
    window = min(len(audio1_envelope), sr * 30)
    tail = audio1_envelope[-window:]
    head = audio2_envelope[:window] if len(audio2_envelope) >= window else audio2_envelope
    n = min(len(tail), len(head))
    if n == 0:
        return 5.0
    diff = np.abs(tail[:n] - head[:n])
    best_idx = int(np.argmin(diff))
    return (window - best_idx) / sr
