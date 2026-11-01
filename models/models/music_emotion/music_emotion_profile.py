from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from models.music_emotion.plutchik import (
    EmotionIntensity,
    PlutchikEmotion,
    PlutchikWheel,
    primary_base,
)


@dataclass
class MusicFeatureVector:
    bpm: float = 120.0
    loudness_db: float = -8.0
    danceability: float = 0.5
    acousticness: float = 0.3
    energy: float = 0.5
    mode: float = 1.0  # 0 = minor, 1 = major
    melodic_dissonance: float = 0.2
    percussiveness: float = 0.5
    isochrony_regularity: float = 0.7
    key_stability: float = 0.8
    brightness: float = 0.5
    melodic_direction: float = 0.0  # -1 descending, +1 ascending
    dynamic_range: float = 10.0
    harmonic_complexity: float = 0.3


@dataclass
class EmotionProfile:
    valence: float
    arousal: float
    dominance: float
    plutchik_scores: dict[PlutchikEmotion, float]
    primary_emotion: PlutchikEmotion
    secondary_emotion: Optional[PlutchikEmotion]
    intensity: EmotionIntensity
    quadrant: str


# Centroids for each primary Plutchik emotion in (valence, arousal, dominance) space.
_EMOTION_CENTROIDS: dict[PlutchikEmotion, tuple[float, float, float]] = {
    PlutchikEmotion.JOY:          (0.90, 0.70, 0.70),
    PlutchikEmotion.TRUST:        (0.70, 0.40, 0.55),
    PlutchikEmotion.FEAR:         (0.15, 0.80, 0.20),
    PlutchikEmotion.SURPRISE:     (0.55, 0.85, 0.45),
    PlutchikEmotion.SADNESS:      (0.15, 0.20, 0.25),
    PlutchikEmotion.DISGUST:      (0.20, 0.35, 0.45),
    PlutchikEmotion.ANGER:        (0.15, 0.85, 0.75),
    PlutchikEmotion.ANTICIPATION: (0.65, 0.60, 0.60),
}

# BPM normalisation bounds
_BPM_MIN = 40.0
_BPM_MAX = 200.0


class MusicEmotionMapper:
    QUADRANT_RANGES: dict[str, dict[str, tuple[float, float]]] = {
        "high_arousal_high_valence": {
            "bpm": (120.0, 180.0),
            "loudness_db": (-5.0, 0.0),
            "danceability": (0.7, 1.0),
            "acousticness": (0.0, 0.3),
        },
        "high_arousal_low_valence": {
            "bpm": (130.0, 200.0),
            "loudness_db": (-3.0, 0.0),
            "danceability": (0.3, 0.8),
            "acousticness": (0.0, 0.25),
        },
        "low_arousal_low_valence": {
            "bpm": (40.0, 80.0),
            "loudness_db": (-20.0, -10.0),
            "danceability": (0.1, 0.4),
            "acousticness": (0.6, 1.0),
        },
        "low_arousal_high_valence": {
            "bpm": (60.0, 100.0),
            "loudness_db": (-15.0, -8.0),
            "danceability": (0.3, 0.6),
            "acousticness": (0.5, 1.0),
        },
    }

    @staticmethod
    def _norm_bpm(bpm: float) -> float:
        return float(np.clip((bpm - _BPM_MIN) / (_BPM_MAX - _BPM_MIN), 0.0, 1.0))

    @classmethod
    def features_to_vad(
        cls, features: MusicFeatureVector,
    ) -> tuple[float, float, float]:
        t_norm = cls._norm_bpm(features.bpm)
        valence = (
            0.45 * features.mode
            + 0.25 * (1.0 - abs(t_norm - 0.5) * 2.0)
            + 0.20 * features.brightness
            + 0.10 * features.danceability
        )
        arousal = (
            0.50 * t_norm
            + 0.30 * features.energy
            + 0.20 * features.brightness
        )
        dominance = (
            0.60 * (1.0 - abs(features.danceability - 0.6))
            + 0.20 * features.energy
            + 0.20 * (1.0 - abs(t_norm - 0.5))
        )
        return (
            float(np.clip(valence, 0.0, 1.0)),
            float(np.clip(arousal, 0.0, 1.0)),
            float(np.clip(dominance, 0.0, 1.0)),
        )

    @staticmethod
    def vad_to_plutchik(
        v: float, a: float, d: float,
    ) -> dict[PlutchikEmotion, float]:
        scores: dict[PlutchikEmotion, float] = {}
        for emotion, (cv, ca, cd) in _EMOTION_CENTROIDS.items():
            dist = math.sqrt((v - cv) ** 2 + (a - ca) ** 2 + (d - cd) ** 2)
            scores[emotion] = 1.0 / (1.0 + dist * 4.0)
        total = sum(scores.values()) or 1.0
        return {e: s / total for e, s in scores.items()}

    @classmethod
    def features_to_emotion_profile(
        cls, features: MusicFeatureVector,
    ) -> EmotionProfile:
        v, a, d = cls.features_to_vad(features)
        scores = cls.vad_to_plutchik(v, a, d)
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_emotions[0][0]
        secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
        top_score = sorted_emotions[0][1]
        if top_score > 0.18:
            intensity = EmotionIntensity.HIGH
        elif top_score > 0.14:
            intensity = EmotionIntensity.MEDIUM
        else:
            intensity = EmotionIntensity.LOW
        quadrant = cls.classify_quadrant(v, a)
        return EmotionProfile(
            valence=v,
            arousal=a,
            dominance=d,
            plutchik_scores=scores,
            primary_emotion=primary,
            secondary_emotion=secondary,
            intensity=intensity,
            quadrant=quadrant,
        )

    @staticmethod
    def classify_quadrant(valence: float, arousal: float) -> str:
        a_label = "high_arousal" if arousal >= 0.5 else "low_arousal"
        v_label = "high_valence" if valence >= 0.5 else "low_valence"
        return f"{a_label}_{v_label}"
