"""Auto-tagging: genre classification, mood tagging, instrument detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class AutoTagger:
    """Rule-based auto-tagger using audio features."""

    FEATURE_RULES: list[tuple[str, dict[str, tuple[float, float]]]] = [
        ("acoustic", {"acousticness": (0.7, 1.0)}),
        ("electronic", {"acousticness": (0.0, 0.3), "energy": (0.5, 1.0)}),
        ("danceable", {"danceability": (0.7, 1.0)}),
        ("chill", {"energy": (0.0, 0.35), "valence": (0.3, 0.7)}),
        ("intense", {"energy": (0.8, 1.0), "arousal": (0.7, 1.0)}),
        ("upbeat", {"valence": (0.7, 1.0), "energy": (0.6, 1.0)}),
        ("dark", {"valence": (0.0, 0.3), "dominance": (0.0, 0.4)}),
        ("bright", {"brightness": (0.6, 1.0), "treble_energy": (0.5, 1.0)}),
        ("bass-heavy", {"bass_energy": (0.6, 1.0)}),
        ("lo-fi", {"spectral_rolloff": (0.0, 0.4), "energy": (0.2, 0.5)}),
    ]

    def tag(self, features: dict[str, float]) -> list[str]:
        tags: list[str] = []
        for tag_name, conditions in self.FEATURE_RULES:
            match = True
            for feat, (lo, hi) in conditions.items():
                val = features.get(feat, 0.5)
                if not (lo <= val <= hi):
                    match = False
                    break
            if match:
                tags.append(tag_name)
        return tags

    def tag_confidence(self, features: dict[str, float]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for tag_name, conditions in self.FEATURE_RULES:
            confidences: list[float] = []
            for feat, (lo, hi) in conditions.items():
                val = features.get(feat, 0.5)
                if lo <= val <= hi:
                    mid = (lo + hi) / 2.0
                    half_range = (hi - lo) / 2.0
                    dist = abs(val - mid) / half_range if half_range > 0 else 0.0
                    confidences.append(1.0 - dist * 0.5)
                else:
                    confidences.append(0.0)
            avg = float(np.mean(confidences)) if confidences else 0.0
            if avg > 0:
                results.append((tag_name, avg))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class GenreClassifier:
    """Feature-centroid genre classifier."""

    GENRE_CENTROIDS: dict[str, dict[str, float]] = {
        "Pop": {"bpm": 120, "valence": 0.65, "energy": 0.65, "danceability": 0.7, "acousticness": 0.3},
        "Rock": {"bpm": 130, "valence": 0.5, "energy": 0.8, "danceability": 0.45, "acousticness": 0.35},
        "Electronic": {"bpm": 128, "valence": 0.5, "energy": 0.75, "danceability": 0.8, "acousticness": 0.1},
        "Jazz": {"bpm": 110, "valence": 0.55, "energy": 0.4, "danceability": 0.5, "acousticness": 0.7},
        "Classical": {"bpm": 90, "valence": 0.45, "energy": 0.35, "danceability": 0.2, "acousticness": 0.9},
        "Hip-Hop": {"bpm": 95, "valence": 0.55, "energy": 0.7, "danceability": 0.75, "acousticness": 0.2},
        "R&B": {"bpm": 100, "valence": 0.6, "energy": 0.5, "danceability": 0.65, "acousticness": 0.4},
        "Country": {"bpm": 115, "valence": 0.6, "energy": 0.55, "danceability": 0.55, "acousticness": 0.65},
        "Metal": {"bpm": 140, "valence": 0.35, "energy": 0.95, "danceability": 0.3, "acousticness": 0.15},
        "Folk": {"bpm": 105, "valence": 0.55, "energy": 0.35, "danceability": 0.4, "acousticness": 0.85},
    }

    def classify(self, features: dict[str, float], top_k: int = 3) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for genre, centroid in self.GENRE_CENTROIDS.items():
            dist = 0.0
            count = 0
            for feat, center in centroid.items():
                val = features.get(feat, center)
                scale = 200.0 if feat == "bpm" else 1.0
                dist += ((val - center) / scale) ** 2
                count += 1
            if count > 0:
                dist = np.sqrt(dist / count)
            similarity = float(np.exp(-dist * 5.0))
            results.append((genre, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class MoodTagger:
    """VAD-based mood tagging using quadrant mapping."""

    MOOD_MAP: dict[str, tuple[float, float, float]] = {
        "Happy": (0.8, 0.7, 0.6),
        "Excited": (0.7, 0.9, 0.7),
        "Energetic": (0.6, 0.9, 0.8),
        "Angry": (0.2, 0.9, 0.8),
        "Tense": (0.3, 0.7, 0.4),
        "Sad": (0.2, 0.2, 0.3),
        "Melancholic": (0.3, 0.3, 0.3),
        "Peaceful": (0.6, 0.2, 0.5),
        "Calm": (0.5, 0.2, 0.5),
        "Serene": (0.7, 0.15, 0.5),
        "Triumphant": (0.8, 0.8, 0.9),
        "Nostalgic": (0.4, 0.3, 0.3),
        "Dreamy": (0.55, 0.25, 0.4),
        "Dark": (0.15, 0.5, 0.6),
        "Euphoric": (0.9, 0.85, 0.7),
    }

    def tag(self, valence: float, arousal: float, dominance: float = 0.5, top_k: int = 3) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for mood, (mv, ma, md) in self.MOOD_MAP.items():
            dist = np.sqrt((valence - mv) ** 2 + (arousal - ma) ** 2 + (dominance - md) ** 2)
            similarity = float(np.exp(-dist * 3.0))
            results.append((mood, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class InstrumentDetector:
    """Spectral-feature instrument likelihood estimation."""

    INSTRUMENT_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
        "vocals": {"spectral_centroid": (0.2, 0.7), "spectral_flatness": (0.1, 0.4)},
        "guitar": {"spectral_centroid": (0.2, 0.5), "spectral_flatness": (0.15, 0.45)},
        "piano": {"spectral_centroid": (0.15, 0.6), "spectral_flatness": (0.05, 0.3)},
        "drums": {"spectral_centroid": (0.3, 0.9), "spectral_flatness": (0.5, 0.95)},
        "bass": {"spectral_centroid": (0.02, 0.15), "spectral_flatness": (0.1, 0.4)},
        "strings": {"spectral_centroid": (0.1, 0.4), "spectral_flatness": (0.05, 0.25)},
        "synth": {"spectral_centroid": (0.2, 0.8), "spectral_flatness": (0.3, 0.8)},
        "brass": {"spectral_centroid": (0.3, 0.6), "spectral_flatness": (0.1, 0.35)},
    }

    def detect(self, features: dict[str, float], threshold: float = 0.3) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for instrument, profile in self.INSTRUMENT_PROFILES.items():
            likelihoods: list[float] = []
            for feat, (lo, hi) in profile.items():
                val = features.get(feat, 0.5)
                if lo <= val <= hi:
                    mid = (lo + hi) / 2
                    half = (hi - lo) / 2
                    dist = abs(val - mid) / half if half > 0 else 0.0
                    likelihoods.append(1.0 - dist * 0.4)
                else:
                    if val < lo:
                        likelihoods.append(max(0, 1.0 - (lo - val) * 5))
                    else:
                        likelihoods.append(max(0, 1.0 - (val - hi) * 5))
            avg = float(np.mean(likelihoods)) if likelihoods else 0.0
            if avg >= threshold:
                results.append((instrument, avg))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
