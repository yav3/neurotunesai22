"""Track annotation pipeline: combines feature extraction with metadata tagging."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .feature_pipeline import FeaturePipeline, PipelineConfig, FeatureResult


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MOOD_QUADRANTS = {
    "happy": (0.5, 1.0, 0.5, 1.0),
    "angry": (0.0, 0.5, 0.5, 1.0),
    "sad": (0.0, 0.5, 0.0, 0.5),
    "relaxed": (0.5, 1.0, 0.0, 0.5),
}


@dataclass
class TrackAnnotation:
    tempo_bpm: float = 0.0
    key: str = ""
    key_confidence: float = 0.0
    mode: str = ""
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    mood_quadrant: str = ""
    brightness: float = 0.0
    energy: float = 0.0
    danceability: float = 0.0
    spectral_centroid: float = 0.0
    tags: list[str] = field(default_factory=list)
    features: FeatureResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "tempo_bpm": self.tempo_bpm,
            "key": self.key,
            "key_confidence": self.key_confidence,
            "mode": self.mode,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "mood_quadrant": self.mood_quadrant,
            "brightness": self.brightness,
            "energy": self.energy,
            "danceability": self.danceability,
            "spectral_centroid": self.spectral_centroid,
            "tags": self.tags,
        }
        return d


def classify_mood_quadrant(valence: float, arousal: float) -> str:
    best = "neutral"
    best_dist = float("inf")
    for name, (v_lo, v_hi, a_lo, a_hi) in MOOD_QUADRANTS.items():
        v_center = (v_lo + v_hi) / 2
        a_center = (a_lo + a_hi) / 2
        dist = (valence - v_center) ** 2 + (arousal - a_center) ** 2
        if dist < best_dist:
            best_dist = dist
            best = name
    return best


def derive_tags(annotation: TrackAnnotation) -> list[str]:
    tags = []
    if annotation.tempo_bpm > 0:
        if annotation.tempo_bpm < 80:
            tags.append("slow")
        elif annotation.tempo_bpm < 120:
            tags.append("moderate")
        elif annotation.tempo_bpm < 160:
            tags.append("fast")
        else:
            tags.append("very_fast")

    if annotation.energy > 0.7:
        tags.append("high_energy")
    elif annotation.energy < 0.3:
        tags.append("low_energy")

    if annotation.brightness > 0.5:
        tags.append("bright")
    else:
        tags.append("dark")

    if annotation.danceability > 0.6:
        tags.append("danceable")

    if annotation.valence > 0.6:
        tags.append("positive")
    elif annotation.valence < 0.4:
        tags.append("negative")

    tags.append(annotation.mood_quadrant)
    return tags


class AnnotationPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.feature_pipeline = FeaturePipeline(config)

    def annotate(self, signal: np.ndarray, sr: int | None = None) -> TrackAnnotation:
        features = self.feature_pipeline.extract(signal, sr)
        ann = TrackAnnotation(features=features)

        ann.tempo_bpm = features.rhythm.get("tempo_estimate", 0.0)
        ann.spectral_centroid = features.spectral.get("centroid", 0.0)
        ann.brightness = features.timbral.get("brightness", 0.0)
        ann.energy = features.rhythm.get("beat_strength", 0.0)
        ann.danceability = features.rhythm.get("pulse_clarity", 0.0)

        key_idx = int(features.harmonic.get("estimated_key_idx", 0))
        ann.key = NOTE_NAMES[key_idx % 12]
        ann.key_confidence = features.harmonic.get("key_confidence", 0.0)

        ann.valence = features.emotion.get("valence", 0.5)
        ann.arousal = features.emotion.get("arousal", 0.5)
        ann.dominance = features.emotion.get("dominance", 0.5)
        ann.mood_quadrant = classify_mood_quadrant(ann.valence, ann.arousal)

        ann.tags = derive_tags(ann)
        return ann

    def annotate_batch(self, signals: list[np.ndarray], sr: int | None = None) -> list[TrackAnnotation]:
        return [self.annotate(s, sr) for s in signals]
