"""Music emotion recognition: audio-feature-to-emotion mapping, classifier framework."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class EmotionCategory(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    RELAXED = "relaxed"
    EXCITED = "excited"
    TENDER = "tender"
    FEARFUL = "fearful"
    NOSTALGIC = "nostalgic"


@dataclass
class AudioFeatures:
    tempo: float = 120.0
    mode_major: float = 1.0
    spectral_centroid: float = 2000.0
    loudness_db: float = -14.0
    harmonic_complexity: float = 0.5
    rhythmic_regularity: float = 0.8
    pitch_range: float = 12.0
    articulation_legato: float = 0.5


@dataclass
class EmotionPrediction:
    category: EmotionCategory
    valence: float
    arousal: float
    dominance: float
    confidence: float
    feature_contributions: Dict[str, float] = field(default_factory=dict)


FEATURE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "valence": {"tempo": 0.15, "mode_major": 0.35, "spectral_centroid": 0.10,
                "loudness_db": 0.05, "harmonic_complexity": -0.10,
                "pitch_range": 0.10, "articulation_legato": 0.15},
    "arousal": {"tempo": 0.35, "spectral_centroid": 0.20, "loudness_db": 0.20,
                "rhythmic_regularity": -0.05, "pitch_range": 0.10,
                "articulation_legato": -0.10},
    "dominance": {"loudness_db": 0.30, "tempo": 0.15, "spectral_centroid": 0.15,
                  "pitch_range": 0.15, "harmonic_complexity": 0.10,
                  "articulation_legato": -0.15},
}


def _normalize_feature(name: str, value: float) -> float:
    ranges: Dict[str, Tuple[float, float]] = {
        "tempo": (40.0, 220.0),
        "mode_major": (0.0, 1.0),
        "spectral_centroid": (200.0, 8000.0),
        "loudness_db": (-40.0, 0.0),
        "harmonic_complexity": (0.0, 1.0),
        "rhythmic_regularity": (0.0, 1.0),
        "pitch_range": (0.0, 48.0),
        "articulation_legato": (0.0, 1.0),
    }
    lo, hi = ranges.get(name, (0.0, 1.0))
    return max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_vad(features: AudioFeatures) -> Tuple[float, float, float]:
    feat_dict = {
        "tempo": features.tempo, "mode_major": features.mode_major,
        "spectral_centroid": features.spectral_centroid,
        "loudness_db": features.loudness_db,
        "harmonic_complexity": features.harmonic_complexity,
        "rhythmic_regularity": features.rhythmic_regularity,
        "pitch_range": features.pitch_range,
        "articulation_legato": features.articulation_legato,
    }
    vad: List[float] = []
    for dim in ["valence", "arousal", "dominance"]:
        s = 0.0
        weights = FEATURE_WEIGHTS[dim]
        for fname, w in weights.items():
            s += w * _normalize_feature(fname, feat_dict.get(fname, 0.5))
        vad.append(max(-1.0, min(1.0, s * 2 - 0.5)))
    return (vad[0], vad[1], vad[2])


def vad_to_category(valence: float, arousal: float) -> EmotionCategory:
    if valence >= 0 and arousal >= 0:
        return EmotionCategory.HAPPY if valence > arousal else EmotionCategory.EXCITED
    elif valence >= 0 and arousal < 0:
        return EmotionCategory.RELAXED if valence > -arousal else EmotionCategory.TENDER
    elif valence < 0 and arousal >= 0:
        return EmotionCategory.ANGRY if arousal > -valence else EmotionCategory.FEARFUL
    else:
        return EmotionCategory.SAD if -valence > -arousal else EmotionCategory.NOSTALGIC


def predict_emotion(features: AudioFeatures) -> EmotionPrediction:
    v, a, d = predict_vad(features)
    cat = vad_to_category(v, a)
    confidence = min(1.0, (abs(v) + abs(a)) / 2.0 + 0.3)
    feat_dict = {
        "tempo": features.tempo, "mode_major": features.mode_major,
        "spectral_centroid": features.spectral_centroid,
        "loudness_db": features.loudness_db,
    }
    contributions: Dict[str, float] = {}
    for fname, fval in feat_dict.items():
        n = _normalize_feature(fname, fval)
        contributions[fname] = n * FEATURE_WEIGHTS["valence"].get(fname, 0.0)
    return EmotionPrediction(category=cat, valence=v, arousal=a, dominance=d,
                             confidence=confidence, feature_contributions=contributions)


def emotion_trajectory(feature_sequence: List[AudioFeatures]) -> List[Tuple[float, float, float]]:
    return [predict_vad(f) for f in feature_sequence]


def emotion_transition_matrix(predictions: List[EmotionCategory]) -> Dict[Tuple[str, str], int]:
    matrix: Dict[Tuple[str, str], int] = {}
    for i in range(len(predictions) - 1):
        key = (predictions[i].value, predictions[i + 1].value)
        matrix[key] = matrix.get(key, 0) + 1
    return matrix
