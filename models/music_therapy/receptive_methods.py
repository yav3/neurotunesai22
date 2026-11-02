"""Receptive listening technique analysis."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ListeningSession:
    genre: str = "classical"
    tempo_bpm: float = 72.0
    mode: str = "ionian"
    duration_sec: float = 1200.0
    valence_trajectory: list[float] = field(default_factory=list)


def relaxation_playlist_score(
    tempos: list[float], valences: list[float]
) -> float:
    if len(tempos) < 2 or len(valences) < 2:
        return 0.5
    tempo_arr = np.array(tempos)
    val_arr = np.array(valences)
    tempo_descent = float(np.mean(np.diff(tempo_arr) <= 0))
    val_smoothness = 1.0 - float(np.std(np.diff(val_arr)))
    final_calm = 1.0 - min(1.0, val_arr[-1])
    return float(np.clip(0.4 * tempo_descent + 0.3 * val_smoothness + 0.3 * final_calm, 0, 1))


def guided_imagery_tempo_profile(duration_sec: float, n_points: int = 20) -> list[tuple[float, float]]:
    t = np.linspace(0, 1, n_points)
    # GIM sessions: induction (slow) -> working (moderate) -> return (slow)
    bpm = 60 + 20 * np.sin(np.pi * t) * np.exp(-0.5 * (t - 0.5) ** 2 / 0.1)
    times = (t * duration_sec).tolist()
    return list(zip(times, bpm.tolist()))


def song_discussion_features(lyrics_text: str) -> dict[str, float]:
    words = lyrics_text.lower().split()
    total = max(len(words), 1)
    emotion_words = {"love", "hate", "fear", "joy", "sad", "happy", "angry", "hope",
                     "pain", "loss", "grief", "peace", "calm", "rage", "lonely"}
    identity_words = {"i", "me", "my", "myself", "am", "self", "soul", "heart", "mind"}
    relation_words = {"you", "we", "us", "together", "apart", "friend", "family", "home"}
    return {
        "emotional_density": sum(1 for w in words if w in emotion_words) / total,
        "self_reference": sum(1 for w in words if w in identity_words) / total,
        "relational_content": sum(1 for w in words if w in relation_words) / total,
        "word_count": float(total),
    }


def session_arc_quality(valence_trajectory: list[float]) -> float:
    if len(valence_trajectory) < 3:
        return 0.5
    arr = np.array(valence_trajectory)
    # Good arc: starts matched, explores, resolves upward
    resolution = max(0.0, float(arr[-1] - arr[0]))
    smoothness = 1.0 / (1.0 + float(np.std(np.diff(arr))))
    return float(np.clip(0.5 * resolution + 0.5 * smoothness, 0, 1))
