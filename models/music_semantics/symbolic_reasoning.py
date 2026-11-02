"""Rule-based musical inference and symbolic reasoning.

Feature-to-style classification, genre inference from acoustic
descriptors, rule-based harmonic analysis, and compositional
heuristic evaluation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum


class StyleCategory(Enum):
    CLASSICAL = "classical"
    JAZZ = "jazz"
    ROCK = "rock"
    ELECTRONIC = "electronic"
    FOLK = "folk"
    BLUES = "blues"
    METAL = "metal"
    AMBIENT = "ambient"
    HIP_HOP = "hip_hop"
    LATIN = "latin"
    UNKNOWN = "unknown"


class HarmonicFunction(Enum):
    TONIC = "tonic"
    PREDOMINANT = "predominant"
    DOMINANT = "dominant"
    CHROMATIC = "chromatic"


@dataclass
class FeatureProfile:
    tempo_bpm: float = 120.0
    spectral_centroid_hz: float = 2000.0
    spectral_rolloff_hz: float = 4000.0
    zero_crossing_rate: float = 0.05
    rms_energy: float = 0.1
    harmonic_ratio: float = 0.5
    percussive_ratio: float = 0.5
    chroma_std: float = 0.1
    onset_rate: float = 2.0
    dynamic_range_db: float = 20.0
    pitch_range_semitones: int = 24


@dataclass
class StyleInference:
    category: StyleCategory
    confidence: float
    supporting_rules: List[str]
    feature_matches: Dict[str, float]


@dataclass
class HarmonicRule:
    name: str
    antecedent: str
    consequent: str
    strength: float = 1.0


@dataclass
class CompositionEvaluation:
    score: float
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]


_STYLE_RULES: Dict[StyleCategory, List[Tuple[str, callable]]] = {}


def _tempo_in(lo: float, hi: float):
    return lambda p: lo <= p.tempo_bpm <= hi


def _centroid_above(hz: float):
    return lambda p: p.spectral_centroid_hz > hz


def _centroid_below(hz: float):
    return lambda p: p.spectral_centroid_hz < hz


def _harmonic_above(t: float):
    return lambda p: p.harmonic_ratio > t


def _percussive_above(t: float):
    return lambda p: p.percussive_ratio > t


def _zcr_above(t: float):
    return lambda p: p.zero_crossing_rate > t


def _onset_rate_above(t: float):
    return lambda p: p.onset_rate > t


def _dynamic_range_above(t: float):
    return lambda p: p.dynamic_range_db > t


STYLE_RULE_DEFS: Dict[StyleCategory, List[Tuple[str, callable]]] = {
    StyleCategory.CLASSICAL: [
        ("tempo 60-140", _tempo_in(60, 140)),
        ("high harmonic ratio", _harmonic_above(0.7)),
        ("wide dynamic range", _dynamic_range_above(30)),
        ("low ZCR", lambda p: p.zero_crossing_rate < 0.04),
        ("wide pitch range", lambda p: p.pitch_range_semitones > 36),
    ],
    StyleCategory.JAZZ: [
        ("tempo 80-200", _tempo_in(80, 200)),
        ("moderate harmonic ratio", _harmonic_above(0.5)),
        ("high chroma variance", lambda p: p.chroma_std > 0.15),
        ("moderate onset rate", lambda p: 2 <= p.onset_rate <= 6),
    ],
    StyleCategory.ROCK: [
        ("tempo 100-160", _tempo_in(100, 160)),
        ("high energy", lambda p: p.rms_energy > 0.15),
        ("moderate centroid", lambda p: 1500 < p.spectral_centroid_hz < 4000),
        ("high percussive", _percussive_above(0.4)),
    ],
    StyleCategory.ELECTRONIC: [
        ("tempo 118-150", _tempo_in(118, 150)),
        ("high centroid", _centroid_above(3000)),
        ("high onset rate", _onset_rate_above(4)),
        ("low harmonic ratio", lambda p: p.harmonic_ratio < 0.4),
    ],
    StyleCategory.FOLK: [
        ("tempo 80-140", _tempo_in(80, 140)),
        ("high harmonic ratio", _harmonic_above(0.6)),
        ("low centroid", _centroid_below(2500)),
        ("narrow pitch range", lambda p: p.pitch_range_semitones < 24),
    ],
    StyleCategory.BLUES: [
        ("tempo 60-130", _tempo_in(60, 130)),
        ("moderate harmonic", lambda p: 0.4 < p.harmonic_ratio < 0.7),
        ("moderate energy", lambda p: 0.05 < p.rms_energy < 0.2),
        ("moderate chroma std", lambda p: 0.08 < p.chroma_std < 0.18),
    ],
    StyleCategory.METAL: [
        ("tempo 100-220", _tempo_in(100, 220)),
        ("very high energy", lambda p: p.rms_energy > 0.25),
        ("high centroid", _centroid_above(3500)),
        ("high ZCR", _zcr_above(0.08)),
        ("high percussive", _percussive_above(0.5)),
    ],
    StyleCategory.AMBIENT: [
        ("tempo 60-100", _tempo_in(60, 100)),
        ("low energy", lambda p: p.rms_energy < 0.05),
        ("low onset rate", lambda p: p.onset_rate < 1.5),
        ("low centroid", _centroid_below(2000)),
        ("narrow dynamic range", lambda p: p.dynamic_range_db < 15),
    ],
}


def classify_style(profile: FeatureProfile) -> StyleInference:
    """Classify style from an acoustic feature profile using rule matching."""
    best_cat = StyleCategory.UNKNOWN
    best_score = 0.0
    best_rules: List[str] = []
    best_matches: Dict[str, float] = {}

    for cat, rules in STYLE_RULE_DEFS.items():
        matched = []
        for name, pred in rules:
            if pred(profile):
                matched.append(name)
        score = len(matched) / len(rules) if rules else 0.0
        if score > best_score:
            best_score = score
            best_cat = cat
            best_rules = matched
            best_matches = {name: 1.0 for name in matched}

    return StyleInference(
        category=best_cat,
        confidence=best_score,
        supporting_rules=best_rules,
        feature_matches=best_matches,
    )


def classify_all_styles(profile: FeatureProfile) -> List[StyleInference]:
    """Score all style categories and return sorted by confidence."""
    results = []
    for cat, rules in STYLE_RULE_DEFS.items():
        matched = [name for name, pred in rules if pred(profile)]
        score = len(matched) / len(rules) if rules else 0.0
        results.append(StyleInference(
            category=cat,
            confidence=score,
            supporting_rules=matched,
            feature_matches={n: 1.0 for n in matched},
        ))
    results.sort(key=lambda s: s.confidence, reverse=True)
    return results


# --- Harmonic reasoning ---

_DEGREE_FUNCTION: Dict[int, HarmonicFunction] = {
    0: HarmonicFunction.TONIC,
    1: HarmonicFunction.CHROMATIC,
    2: HarmonicFunction.PREDOMINANT,
    3: HarmonicFunction.TONIC,
    4: HarmonicFunction.PREDOMINANT,
    5: HarmonicFunction.PREDOMINANT,
    6: HarmonicFunction.PREDOMINANT,
    7: HarmonicFunction.DOMINANT,
    8: HarmonicFunction.CHROMATIC,
    9: HarmonicFunction.TONIC,
    10: HarmonicFunction.DOMINANT,
    11: HarmonicFunction.DOMINANT,
}


def harmonic_function(root_pc: int, key_pc: int) -> HarmonicFunction:
    """Classify the harmonic function of a chord root relative to a key."""
    degree = (root_pc - key_pc) % 12
    return _DEGREE_FUNCTION.get(degree, HarmonicFunction.CHROMATIC)


def is_authentic_cadence(penultimate_root: int, final_root: int, key: int) -> bool:
    """V -> I cadence check."""
    return (penultimate_root - key) % 12 == 7 and (final_root - key) % 12 == 0


def is_plagal_cadence(penultimate_root: int, final_root: int, key: int) -> bool:
    """IV -> I cadence check."""
    return (penultimate_root - key) % 12 == 5 and (final_root - key) % 12 == 0


def is_deceptive_cadence(penultimate_root: int, final_root: int, key: int) -> bool:
    """V -> vi cadence check."""
    return (penultimate_root - key) % 12 == 7 and (final_root - key) % 12 == 9


def is_half_cadence(final_root: int, key: int) -> bool:
    """Phrase ending on V."""
    return (final_root - key) % 12 == 7


def functional_progression_score(roots: List[int], key: int) -> float:
    """Score how well a root progression follows T-PD-D-T functional logic.

    Returns value in [0, 1] where 1 = perfectly functional.
    """
    if len(roots) < 2:
        return 1.0

    order = {
        HarmonicFunction.TONIC: 0,
        HarmonicFunction.PREDOMINANT: 1,
        HarmonicFunction.DOMINANT: 2,
    }

    good_transitions = 0
    total = len(roots) - 1
    for i in range(total):
        f_a = harmonic_function(roots[i], key)
        f_b = harmonic_function(roots[i + 1], key)
        if f_a == HarmonicFunction.CHROMATIC or f_b == HarmonicFunction.CHROMATIC:
            good_transitions += 0.5
            continue
        o_a = order.get(f_a, -1)
        o_b = order.get(f_b, -1)
        if o_b > o_a or (o_b == 0 and o_a == 2):
            good_transitions += 1.0
        elif o_b == o_a:
            good_transitions += 0.5

    return good_transitions / total


# --- Composition evaluation ---

def evaluate_melody(
    pitches: np.ndarray,
    durations: np.ndarray,
    key: int = 0,
) -> CompositionEvaluation:
    """Heuristic evaluation of a monophonic melody."""
    strengths: List[str] = []
    weaknesses: List[str] = []
    suggestions: List[str] = []
    score = 50.0

    if len(pitches) < 4:
        return CompositionEvaluation(score=0, strengths=[], weaknesses=["too short"], suggestions=[])

    pitch_range = int(np.ptp(pitches))
    if 8 <= pitch_range <= 16:
        strengths.append("good melodic range")
        score += 10
    elif pitch_range > 24:
        weaknesses.append("excessive range")
        score -= 10
    elif pitch_range < 5:
        weaknesses.append("narrow range")
        suggestions.append("expand melodic range beyond a 4th")
        score -= 5

    intervals = np.diff(pitches)
    steps = np.sum(np.abs(intervals) <= 2)
    step_ratio = steps / len(intervals)
    if 0.5 <= step_ratio <= 0.8:
        strengths.append("good step/skip balance")
        score += 10
    elif step_ratio > 0.9:
        weaknesses.append("too many stepwise motions")
        suggestions.append("add more skips for interest")
        score -= 5

    large_leaps = np.sum(np.abs(intervals) > 7)
    if large_leaps > len(intervals) * 0.15:
        weaknesses.append("too many large leaps")
        suggestions.append("resolve large leaps by step in opposite direction")
        score -= 10

    direction_changes = 0
    for i in range(1, len(intervals)):
        if intervals[i] * intervals[i - 1] < 0:
            direction_changes += 1
    contour_variety = direction_changes / max(len(intervals) - 1, 1)
    if 0.3 <= contour_variety <= 0.6:
        strengths.append("varied contour")
        score += 5

    climax_pos = np.argmax(pitches) / len(pitches)
    if 0.5 <= climax_pos <= 0.8:
        strengths.append("well-placed climax")
        score += 5
    elif climax_pos < 0.2:
        suggestions.append("place climax later in the phrase")

    score = float(np.clip(score, 0, 100))
    return CompositionEvaluation(
        score=score,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
    )


def evaluate_harmonic_progression(
    roots: List[int],
    qualities: List[str],
    key: int = 0,
) -> CompositionEvaluation:
    """Heuristic evaluation of a chord progression."""
    strengths: List[str] = []
    weaknesses: List[str] = []
    suggestions: List[str] = []
    score = 50.0

    if len(roots) < 2:
        return CompositionEvaluation(score=0, strengths=[], weaknesses=["too short"], suggestions=[])

    func_score = functional_progression_score(roots, key)
    if func_score > 0.7:
        strengths.append("strong functional logic")
        score += 15
    elif func_score < 0.3:
        weaknesses.append("weak functional progression")
        suggestions.append("use more T→PD→D→T motion")
        score -= 10

    if is_authentic_cadence(roots[-2], roots[-1], key):
        strengths.append("authentic cadence at end")
        score += 10

    unique_roots = len(set(r % 12 for r in roots))
    variety = unique_roots / min(len(roots), 12)
    if variety > 0.5:
        strengths.append("harmonic variety")
        score += 5
    elif variety < 0.25:
        weaknesses.append("repetitive harmony")
        suggestions.append("introduce more chord variety")
        score -= 5

    score = float(np.clip(score, 0, 100))
    return CompositionEvaluation(
        score=score,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
    )
