"""Performance practice conventions across musical traditions.

Models tradition-specific performance parameters: tempo conventions,
dynamic ranges, vibrato characteristics, articulation norms, ensemble
coordination patterns, and improvisatory freedom levels.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class PerformanceTradition(Enum):
    WESTERN_CLASSICAL = "western_classical"
    JAZZ = "jazz"
    NORTH_INDIAN = "north_indian"
    SOUTH_INDIAN = "south_indian"
    ARABIC = "arabic"
    JAVANESE_GAMELAN = "javanese_gamelan"
    WEST_AFRICAN = "west_african"
    FLAMENCO = "flamenco"
    CELTIC = "celtic"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    KOREAN = "korean"


class VibratoType(Enum):
    NONE = "none"
    CONTINUOUS = "continuous"
    LATE_ONSET = "late_onset"
    TERMINAL = "terminal"
    WIDE_SLOW = "wide_slow"
    NARROW_FAST = "narrow_fast"
    PITCH_INFLECTED = "pitch_inflected"


@dataclass
class VibratoProfile:
    type: VibratoType
    rate_hz: float          # cycles per second
    extent_cents: float     # peak-to-peak in cents
    onset_delay: float      # fraction of note before vibrato starts
    dynamic_coupling: float # 0 = independent, 1 = fully coupled to dynamics


@dataclass
class TempoConvention:
    typical_bpm_range: Tuple[float, float]
    rubato_freedom: float           # 0 = strict, 1 = maximal
    acceleration_convention: str    # "none", "slight", "dramatic"
    section_tempo_contrast: float   # ratio between slow/fast sections


@dataclass
class DynamicConvention:
    range_db: Tuple[float, float]   # typical dynamic range
    crescendo_preference: float     # 0 = terraced, 1 = continuous
    accent_frequency: float         # accents per beat
    subito_usage: float             # 0 = never, 1 = frequent


@dataclass
class ArticulationProfile:
    legato_ratio: float             # fraction of notes connected
    staccato_ratio: float
    portamento_frequency: float     # per phrase
    attack_sharpness: float         # 0 = soft, 1 = sharp


@dataclass
class ImprovisationProfile:
    freedom_level: float            # 0 = fully composed, 1 = fully improvised
    structure_type: str             # "none", "modal", "harmonic", "rhythmic"
    ornamentation_density: float    # ornaments per beat
    call_response_ratio: float      # 0 = solo, 1 = constant exchange


@dataclass
class PerformancePractice:
    tradition: PerformanceTradition
    vibrato: VibratoProfile
    tempo: TempoConvention
    dynamics: DynamicConvention
    articulation: ArticulationProfile
    improvisation: ImprovisationProfile


PRACTICE_CATALOG: Dict[PerformanceTradition, PerformancePractice] = {
    PerformanceTradition.WESTERN_CLASSICAL: PerformancePractice(
        tradition=PerformanceTradition.WESTERN_CLASSICAL,
        vibrato=VibratoProfile(VibratoType.CONTINUOUS, 5.5, 30, 0.1, 0.3),
        tempo=TempoConvention((40, 200), 0.3, "slight", 1.8),
        dynamics=DynamicConvention((-60, -10), 0.8, 0.2, 0.3),
        articulation=ArticulationProfile(0.6, 0.15, 0.05, 0.5),
        improvisation=ImprovisationProfile(0.05, "none", 0.3, 0.0),
    ),
    PerformanceTradition.JAZZ: PerformancePractice(
        tradition=PerformanceTradition.JAZZ,
        vibrato=VibratoProfile(VibratoType.TERMINAL, 5.0, 40, 0.4, 0.5),
        tempo=TempoConvention((60, 300), 0.2, "none", 1.3),
        dynamics=DynamicConvention((-50, -15), 0.6, 0.4, 0.1),
        articulation=ArticulationProfile(0.4, 0.2, 0.1, 0.6),
        improvisation=ImprovisationProfile(0.8, "harmonic", 0.5, 0.5),
    ),
    PerformanceTradition.NORTH_INDIAN: PerformancePractice(
        tradition=PerformanceTradition.NORTH_INDIAN,
        vibrato=VibratoProfile(VibratoType.PITCH_INFLECTED, 4.0, 60, 0.05, 0.2),
        tempo=TempoConvention((30, 240), 0.5, "dramatic", 4.0),
        dynamics=DynamicConvention((-45, -15), 0.5, 0.3, 0.1),
        articulation=ArticulationProfile(0.7, 0.05, 0.3, 0.3),
        improvisation=ImprovisationProfile(0.85, "modal", 2.5, 0.3),
    ),
    PerformanceTradition.ARABIC: PerformancePractice(
        tradition=PerformanceTradition.ARABIC,
        vibrato=VibratoProfile(VibratoType.WIDE_SLOW, 3.5, 50, 0.15, 0.4),
        tempo=TempoConvention((50, 180), 0.6, "slight", 2.0),
        dynamics=DynamicConvention((-40, -15), 0.6, 0.3, 0.2),
        articulation=ArticulationProfile(0.65, 0.1, 0.2, 0.35),
        improvisation=ImprovisationProfile(0.7, "modal", 1.8, 0.4),
    ),
    PerformanceTradition.JAVANESE_GAMELAN: PerformancePractice(
        tradition=PerformanceTradition.JAVANESE_GAMELAN,
        vibrato=VibratoProfile(VibratoType.NONE, 0, 0, 0, 0),
        tempo=TempoConvention((30, 120), 0.15, "none", 3.0),
        dynamics=DynamicConvention((-30, -10), 0.3, 0.1, 0.0),
        articulation=ArticulationProfile(0.3, 0.4, 0.0, 0.7),
        improvisation=ImprovisationProfile(0.3, "rhythmic", 1.0, 0.7),
    ),
    PerformanceTradition.WEST_AFRICAN: PerformancePractice(
        tradition=PerformanceTradition.WEST_AFRICAN,
        vibrato=VibratoProfile(VibratoType.NONE, 0, 0, 0, 0),
        tempo=TempoConvention((80, 200), 0.1, "none", 1.2),
        dynamics=DynamicConvention((-35, -10), 0.2, 0.6, 0.0),
        articulation=ArticulationProfile(0.2, 0.5, 0.0, 0.8),
        improvisation=ImprovisationProfile(0.6, "rhythmic", 0.5, 0.8),
    ),
    PerformanceTradition.FLAMENCO: PerformancePractice(
        tradition=PerformanceTradition.FLAMENCO,
        vibrato=VibratoProfile(VibratoType.PITCH_INFLECTED, 4.5, 45, 0.1, 0.6),
        tempo=TempoConvention((60, 240), 0.7, "dramatic", 3.0),
        dynamics=DynamicConvention((-50, -5), 0.7, 0.5, 0.5),
        articulation=ArticulationProfile(0.4, 0.3, 0.15, 0.7),
        improvisation=ImprovisationProfile(0.6, "modal", 2.2, 0.5),
    ),
    PerformanceTradition.CELTIC: PerformancePractice(
        tradition=PerformanceTradition.CELTIC,
        vibrato=VibratoProfile(VibratoType.NONE, 0, 0, 0, 0),
        tempo=TempoConvention((60, 180), 0.2, "none", 1.5),
        dynamics=DynamicConvention((-35, -15), 0.4, 0.3, 0.1),
        articulation=ArticulationProfile(0.5, 0.2, 0.05, 0.5),
        improvisation=ImprovisationProfile(0.4, "none", 2.0, 0.0),
    ),
}


def practice_vector(practice: PerformancePractice) -> np.ndarray:
    return np.array([
        practice.vibrato.rate_hz / 10.0,
        practice.vibrato.extent_cents / 100.0,
        practice.vibrato.onset_delay,
        practice.tempo.typical_bpm_range[0] / 300.0,
        practice.tempo.typical_bpm_range[1] / 300.0,
        practice.tempo.rubato_freedom,
        practice.tempo.section_tempo_contrast / 5.0,
        practice.dynamics.crescendo_preference,
        practice.dynamics.accent_frequency,
        practice.articulation.legato_ratio,
        practice.articulation.staccato_ratio,
        practice.articulation.attack_sharpness,
        practice.improvisation.freedom_level,
        practice.improvisation.ornamentation_density / 3.0,
        practice.improvisation.call_response_ratio,
    ])


def practice_distance(a: PerformancePractice, b: PerformancePractice) -> float:
    va = practice_vector(a)
    vb = practice_vector(b)
    return float(np.linalg.norm(va - vb))


def practice_similarity(a: PerformancePractice, b: PerformancePractice) -> float:
    d = practice_distance(a, b)
    return float(np.exp(-d * d / 2.0))


def pairwise_practice_matrix(practices: List[PerformancePractice]) -> np.ndarray:
    n = len(practices)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = practice_distance(practices[i], practices[j])
            mat[i, j] = mat[j, i] = d
    return mat


def classify_tradition(practice: PerformancePractice) -> Tuple[PerformanceTradition, float]:
    best_trad = practice.tradition
    best_sim = 0.0
    for trad, catalog_practice in PRACTICE_CATALOG.items():
        sim = practice_similarity(practice, catalog_practice)
        if sim > best_sim:
            best_sim = sim
            best_trad = trad
    return best_trad, best_sim


def vibrato_similarity(a: VibratoProfile, b: VibratoProfile) -> float:
    if a.type == VibratoType.NONE and b.type == VibratoType.NONE:
        return 1.0
    if a.type == VibratoType.NONE or b.type == VibratoType.NONE:
        return 0.0
    rate_sim = 1.0 - abs(a.rate_hz - b.rate_hz) / 10.0
    extent_sim = 1.0 - abs(a.extent_cents - b.extent_cents) / 100.0
    type_match = 1.0 if a.type == b.type else 0.5
    return max(0.0, (rate_sim + extent_sim + type_match) / 3.0)


def tempo_flexibility_score(practice: PerformancePractice) -> float:
    bpm_range = practice.tempo.typical_bpm_range[1] - practice.tempo.typical_bpm_range[0]
    normalised_range = bpm_range / 300.0
    return (normalised_range + practice.tempo.rubato_freedom + practice.tempo.section_tempo_contrast / 5.0) / 3.0


def expressive_freedom_score(practice: PerformancePractice) -> float:
    return (
        practice.improvisation.freedom_level * 0.3 +
        practice.tempo.rubato_freedom * 0.2 +
        practice.dynamics.crescendo_preference * 0.15 +
        practice.improvisation.ornamentation_density / 3.0 * 0.2 +
        practice.articulation.portamento_frequency * 0.15
    )
