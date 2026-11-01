"""Cantometrics-inspired parametric song description.

Based on Alan Lomax's Cantometrics coding system — a multi-dimensional
parametric description of singing style across cultures.  37 coding
parameters covering vocal quality, social organization of performance,
rhythmic features, melodic contour, and text handling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class VocalWidth(Enum):
    NARROW = 1
    MEDIUM = 2
    WIDE = 3
    VERY_WIDE = 4


class NasalQuality(Enum):
    NONE = 0
    SLIGHT = 1
    MODERATE = 2
    HEAVY = 3


class RegisterType(Enum):
    CHEST = auto()
    HEAD = auto()
    FALSETTO = auto()
    MIXED = auto()
    THROAT = auto()


class SocialOrg(Enum):
    SOLO = auto()
    UNISON = auto()
    HETEROPHONY = auto()
    POLYPHONY = auto()
    CALL_RESPONSE = auto()
    INTERLOCK = auto()


class RhythmicOrg(Enum):
    FREE = auto()
    SIMPLE = auto()
    COMPLEX = auto()
    POLYRHYTHMIC = auto()


class TextRelation(Enum):
    SYLLABIC = auto()
    MELISMATIC = auto()
    MIXED = auto()


@dataclass
class CantometricsProfile:
    """Full cantometric coding for a single performance or style."""
    vocal_width: VocalWidth = VocalWidth.MEDIUM
    nasal_quality: NasalQuality = NasalQuality.NONE
    register: RegisterType = RegisterType.CHEST
    social_org: SocialOrg = SocialOrg.SOLO
    rhythmic_org: RhythmicOrg = RhythmicOrg.SIMPLE
    text_relation: TextRelation = TextRelation.SYLLABIC
    # Continuous parameters (0-1 scales)
    vocal_tension: float = 0.5
    ornamentation_density: float = 0.2
    melodic_range: float = 0.5     # normalized semitones / 36
    melodic_contour_slope: float = 0.0  # positive=ascending tendency
    rhythmic_regularity: float = 0.7
    accent_strength: float = 0.5
    tempo_variability: float = 0.3
    group_cohesion: float = 0.5  # 0=loose, 1=tight unison
    tonal_blend: float = 0.5
    wordiness: float = 0.5  # syllables per beat
    repetition_level: float = 0.5
    glissando_frequency: float = 0.1
    vibrato_extent: float = 0.3
    rasp: float = 0.0
    yodel: float = 0.0
    volume_variation: float = 0.3
    # Metadata
    tradition: str = ""
    region: str = ""
    notes: str = ""


def profile_to_vector(profile: CantometricsProfile) -> np.ndarray:
    """Convert a profile to a numeric feature vector for comparison."""
    return np.array([
        profile.vocal_width.value / 4.0,
        profile.nasal_quality.value / 3.0,
        float(hash(profile.register.name) % 5) / 5.0,
        float(hash(profile.social_org.name) % 6) / 6.0,
        float(hash(profile.rhythmic_org.name) % 4) / 4.0,
        float(hash(profile.text_relation.name) % 3) / 3.0,
        profile.vocal_tension,
        profile.ornamentation_density,
        profile.melodic_range,
        profile.melodic_contour_slope,
        profile.rhythmic_regularity,
        profile.accent_strength,
        profile.tempo_variability,
        profile.group_cohesion,
        profile.tonal_blend,
        profile.wordiness,
        profile.repetition_level,
        profile.glissando_frequency,
        profile.vibrato_extent,
        profile.rasp,
        profile.yodel,
        profile.volume_variation,
    ], dtype=float)


def profile_distance(a: CantometricsProfile, b: CantometricsProfile) -> float:
    """Euclidean distance between two cantometric profiles."""
    va, vb = profile_to_vector(a), profile_to_vector(b)
    return float(np.linalg.norm(va - vb))


def profile_similarity(a: CantometricsProfile, b: CantometricsProfile) -> float:
    """Cosine similarity between two cantometric profiles."""
    va, vb = profile_to_vector(a), profile_to_vector(b)
    dot = float(np.dot(va, vb))
    norms = np.linalg.norm(va) * np.linalg.norm(vb)
    if norms < 1e-10:
        return 0.0
    return dot / norms


def cluster_profiles(
    profiles: List[CantometricsProfile],
    n_clusters: int = 5,
    n_iterations: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering of cantometric profiles.

    Returns (labels, centroids).
    """
    rng = np.random.RandomState(seed)
    vectors = np.array([profile_to_vector(p) for p in profiles])
    n = len(vectors)
    if n == 0:
        return np.array([]), np.array([])
    k = min(n_clusters, n)

    # Random initialization
    idx = rng.choice(n, k, replace=False)
    centroids = vectors[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iterations):
        # assign
        for i in range(n):
            dists = np.linalg.norm(vectors[i] - centroids, axis=1)
            labels[i] = int(np.argmin(dists))
        # update
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centroids[c] = vectors[mask].mean(axis=0)
            else:
                new_centroids[c] = centroids[c]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


def tradition_fingerprint(
    profiles: List[CantometricsProfile],
) -> np.ndarray:
    """Average cantometric vector for a tradition (its "fingerprint")."""
    if not profiles:
        return np.zeros(22)
    vectors = np.array([profile_to_vector(p) for p in profiles])
    return vectors.mean(axis=0)


def most_distinctive_features(
    tradition_a: List[CantometricsProfile],
    tradition_b: List[CantometricsProfile],
    feature_names: Optional[List[str]] = None,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Find the features that most distinguish two traditions."""
    if feature_names is None:
        feature_names = [
            "vocal_width", "nasal_quality", "register", "social_org",
            "rhythmic_org", "text_relation", "vocal_tension",
            "ornamentation_density", "melodic_range", "contour_slope",
            "rhythmic_regularity", "accent_strength", "tempo_variability",
            "group_cohesion", "tonal_blend", "wordiness", "repetition_level",
            "glissando_frequency", "vibrato_extent", "rasp", "yodel",
            "volume_variation",
        ]
    fp_a = tradition_fingerprint(tradition_a)
    fp_b = tradition_fingerprint(tradition_b)
    diffs = np.abs(fp_a - fp_b)
    top_idx = np.argsort(diffs)[::-1][:top_n]
    result = []
    for i in top_idx:
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        result.append((name, float(diffs[i])))
    return result


# Pre-configured tradition prototypes
WEST_AFRICAN_GRIOT = CantometricsProfile(
    vocal_width=VocalWidth.WIDE, nasal_quality=NasalQuality.MODERATE,
    register=RegisterType.CHEST, social_org=SocialOrg.CALL_RESPONSE,
    rhythmic_org=RhythmicOrg.POLYRHYTHMIC, text_relation=TextRelation.MELISMATIC,
    vocal_tension=0.7, ornamentation_density=0.6, melodic_range=0.6,
    tradition="West African Griot", region="West Africa",
)

APPALACHIAN_BALLAD = CantometricsProfile(
    vocal_width=VocalWidth.NARROW, nasal_quality=NasalQuality.SLIGHT,
    register=RegisterType.CHEST, social_org=SocialOrg.SOLO,
    rhythmic_org=RhythmicOrg.SIMPLE, text_relation=TextRelation.SYLLABIC,
    vocal_tension=0.4, ornamentation_density=0.15, melodic_range=0.4,
    tradition="Appalachian Ballad", region="Appalachian USA",
)

PYGMY_POLYPHONY = CantometricsProfile(
    vocal_width=VocalWidth.MEDIUM, nasal_quality=NasalQuality.NONE,
    register=RegisterType.MIXED, social_org=SocialOrg.INTERLOCK,
    rhythmic_org=RhythmicOrg.POLYRHYTHMIC, text_relation=TextRelation.SYLLABIC,
    vocal_tension=0.3, ornamentation_density=0.4, melodic_range=0.5,
    group_cohesion=0.9, yodel=0.6,
    tradition="Pygmy Polyphony", region="Central Africa",
)

HINDUSTANI_CLASSICAL = CantometricsProfile(
    vocal_width=VocalWidth.WIDE, nasal_quality=NasalQuality.SLIGHT,
    register=RegisterType.CHEST, social_org=SocialOrg.SOLO,
    rhythmic_org=RhythmicOrg.COMPLEX, text_relation=TextRelation.MELISMATIC,
    vocal_tension=0.5, ornamentation_density=0.8, melodic_range=0.8,
    glissando_frequency=0.7, vibrato_extent=0.5,
    tradition="Hindustani Classical", region="North India",
)

TRADITION_PROTOTYPES = [
    WEST_AFRICAN_GRIOT, APPALACHIAN_BALLAD, PYGMY_POLYPHONY, HINDUSTANI_CLASSICAL,
]
