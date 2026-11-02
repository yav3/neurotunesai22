"""Metadata enrichment: infer missing tags from audio features, cross-reference, confidence scoring."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class TagSource(str, Enum):
    ACOUSTIC = "acoustic"
    INFERRED = "inferred"
    USER = "user"
    EXTERNAL = "external"


@dataclass
class EnrichedTag:
    key: str
    value: str
    confidence: float
    source: TagSource


@dataclass
class EnrichmentResult:
    tags: List[EnrichedTag] = field(default_factory=list)
    coverage: float = 0.0
    digest: str = ""


GENRE_CENTROIDS: Dict[str, np.ndarray] = {
    "classical": np.array([0.3, 0.1, 0.7, 0.2, 0.8, 0.1]),
    "rock": np.array([0.7, 0.8, 0.4, 0.6, 0.3, 0.5]),
    "electronic": np.array([0.5, 0.9, 0.2, 0.8, 0.2, 0.7]),
    "jazz": np.array([0.5, 0.3, 0.6, 0.4, 0.7, 0.3]),
    "folk": np.array([0.4, 0.2, 0.5, 0.3, 0.6, 0.2]),
    "hip_hop": np.array([0.6, 0.7, 0.3, 0.7, 0.2, 0.6]),
    "ambient": np.array([0.2, 0.1, 0.3, 0.1, 0.5, 0.8]),
    "metal": np.array([0.9, 0.9, 0.3, 0.7, 0.2, 0.4]),
}

MOOD_FEATURE_MAP: Dict[str, Tuple[float, float, float]] = {
    "happy": (0.7, 0.8, 0.6),
    "sad": (0.3, 0.2, 0.4),
    "energetic": (0.8, 0.9, 0.5),
    "calm": (0.3, 0.1, 0.7),
    "aggressive": (0.9, 0.8, 0.3),
    "melancholic": (0.3, 0.3, 0.5),
    "uplifting": (0.7, 0.6, 0.7),
    "dark": (0.4, 0.5, 0.2),
}

TEMPO_DESCRIPTORS = [
    (0, 60, "very_slow"),
    (60, 90, "slow"),
    (90, 120, "moderate"),
    (120, 140, "fast"),
    (140, 180, "very_fast"),
    (180, 999, "extreme"),
]


def infer_genre(features: np.ndarray, top_k: int = 3) -> List[EnrichedTag]:
    if features.ndim != 1 or len(features) < 6:
        return []
    feat = features[:6]
    distances = {}
    for genre, centroid in GENRE_CENTROIDS.items():
        distances[genre] = float(np.linalg.norm(feat - centroid))
    sorted_genres = sorted(distances.items(), key=lambda x: x[1])
    max_dist = max(d for _, d in sorted_genres) if sorted_genres else 1.0
    tags = []
    for genre, dist in sorted_genres[:top_k]:
        conf = max(0.0, 1.0 - dist / max(max_dist, 1e-12))
        tags.append(EnrichedTag(key="genre", value=genre, confidence=conf, source=TagSource.ACOUSTIC))
    return tags


def infer_mood(valence: float, arousal: float, dominance: float = 0.5) -> List[EnrichedTag]:
    vad = np.array([valence, arousal, dominance])
    scores = {}
    for mood, (v, a, d) in MOOD_FEATURE_MAP.items():
        ref = np.array([v, a, d])
        scores[mood] = float(np.linalg.norm(vad - ref))
    sorted_moods = sorted(scores.items(), key=lambda x: x[1])
    max_d = max(d for _, d in sorted_moods) if sorted_moods else 1.0
    return [
        EnrichedTag(
            key="mood", value=m, confidence=max(0.0, 1.0 - d / max(max_d, 1e-12)),
            source=TagSource.INFERRED,
        )
        for m, d in sorted_moods[:3]
    ]


def infer_tempo_descriptor(bpm: float) -> EnrichedTag:
    for lo, hi, label in TEMPO_DESCRIPTORS:
        if lo <= bpm < hi:
            return EnrichedTag(key="tempo_class", value=label, confidence=0.9, source=TagSource.ACOUSTIC)
    return EnrichedTag(key="tempo_class", value="unknown", confidence=0.0, source=TagSource.ACOUSTIC)


def infer_key_tag(chroma: np.ndarray) -> Optional[EnrichedTag]:
    if chroma.ndim != 1 or len(chroma) != 12:
        return None
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    idx = int(np.argmax(chroma))
    total = float(np.sum(chroma))
    if total < 1e-12:
        return None
    conf = float(chroma[idx]) / total
    return EnrichedTag(key="key", value=NOTE_NAMES[idx], confidence=conf, source=TagSource.ACOUSTIC)


def infer_instrumentation(spectral_centroid_hz: float, spectral_flatness: float) -> EnrichedTag:
    if spectral_flatness > 0.5:
        value = "percussion_dominant"
    elif spectral_centroid_hz < 300:
        value = "bass_dominant"
    elif spectral_centroid_hz < 1500:
        value = "midrange_dominant"
    else:
        value = "treble_dominant"
    return EnrichedTag(
        key="instrumentation", value=value,
        confidence=min(1.0, 0.5 + abs(spectral_flatness - 0.5)),
        source=TagSource.ACOUSTIC,
    )


def cross_reference_tags(
    tags_a: List[EnrichedTag], tags_b: List[EnrichedTag],
) -> List[EnrichedTag]:
    merged: Dict[Tuple[str, str], float] = {}
    for t in tags_a + tags_b:
        k = (t.key, t.value)
        merged[k] = max(merged.get(k, 0.0), t.confidence)
    return [
        EnrichedTag(key=k, value=v, confidence=c, source=TagSource.INFERRED)
        for (k, v), c in sorted(merged.items(), key=lambda x: -x[1])
    ]


def tag_coverage(tags: List[EnrichedTag], required_keys: List[str]) -> float:
    present = {t.key for t in tags if t.confidence > 0.3}
    if not required_keys:
        return 1.0
    return len(present & set(required_keys)) / len(required_keys)


def enrichment_digest(tags: List[EnrichedTag]) -> str:
    data = "|".join(f"{t.key}={t.value}:{t.confidence:.3f}" for t in sorted(tags, key=lambda t: t.key))
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def enrich(
    features: np.ndarray,
    bpm: float = 0.0,
    chroma: Optional[np.ndarray] = None,
    valence: float = 0.5,
    arousal: float = 0.5,
    spectral_centroid_hz: float = 1000.0,
    spectral_flatness: float = 0.3,
) -> EnrichmentResult:
    tags: List[EnrichedTag] = []
    tags.extend(infer_genre(features))
    tags.extend(infer_mood(valence, arousal))
    if bpm > 0:
        tags.append(infer_tempo_descriptor(bpm))
    if chroma is not None:
        kt = infer_key_tag(chroma)
        if kt:
            tags.append(kt)
    tags.append(infer_instrumentation(spectral_centroid_hz, spectral_flatness))
    required = ["genre", "mood", "tempo_class", "key", "instrumentation"]
    cov = tag_coverage(tags, required)
    return EnrichmentResult(tags=tags, coverage=cov, digest=enrichment_digest(tags))
