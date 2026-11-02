"""Symbolic grounding: map musical features to semantic/conceptual meaning.

Grounding theory for music: maps low-level musical features (pitch, rhythm,
dynamics, timbre) to cross-modal semantic domains (motion, space, emotion,
body, nature, narrative) using established psychoacoustic and embodied-cognition
correspondences.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SemanticDomain(str, Enum):
    MOTION = "motion"
    SPACE = "space"
    EMOTION = "emotion"
    BODY = "body"
    NATURE = "nature"
    SOCIAL = "social"
    TEMPORAL = "temporal"
    NARRATIVE = "narrative"


@dataclass
class GroundingLink:
    musical_feature: str
    semantic_concept: str
    domain: SemanticDomain
    strength: float


KINESTHETIC_MAP: Dict[str, Tuple[str, float]] = {
    "ascending_step": ("climbing", 0.8),
    "descending_step": ("descending", 0.8),
    "ascending_leap": ("jumping", 0.9),
    "descending_leap": ("falling", 0.9),
    "repeated": ("standing", 0.6),
    "trill": ("shaking", 0.7),
    "arpeggio_up": ("rising", 0.85),
    "arpeggio_down": ("sinking", 0.85),
    "chromatic_ascent": ("straining", 0.7),
    "chromatic_descent": ("sliding", 0.7),
}

SPATIAL_MAP: Dict[str, Tuple[str, float]] = {
    "high_register": ("bright", 0.9),
    "low_register": ("dark", 0.9),
    "wide_range": ("expansive", 0.8),
    "narrow_range": ("confined", 0.7),
    "ascending": ("upward", 0.85),
    "descending": ("downward", 0.85),
    "mid_register": ("grounded", 0.5),
}

TEMPO_ACTIVITY: Dict[str, Tuple[float, float, str]] = {
    "static": (0, 40, "stillness"),
    "contemplative": (40, 66, "meditation"),
    "walking": (66, 100, "walking"),
    "moderate": (100, 120, "purposeful"),
    "energetic": (120, 150, "running"),
    "frantic": (150, 220, "urgency"),
    "ecstatic": (220, 400, "frenzy"),
}


def pitch_to_spatial(pitches: np.ndarray) -> Dict[str, float]:
    if len(pitches) == 0:
        return {}
    mean_p = float(np.mean(pitches))
    range_p = float(np.ptp(pitches))
    result: Dict[str, float] = {}
    if mean_p > 72:
        result["bright"] = min(1.0, (mean_p - 72) / 24.0)
        result["light"] = result["bright"] * 0.8
    elif mean_p < 48:
        result["dark"] = min(1.0, (48 - mean_p) / 24.0)
        result["heavy"] = result["dark"] * 0.7
    else:
        result["grounded"] = 1.0 - abs(mean_p - 60) / 12.0
    if range_p > 24:
        result["expansive"] = min(1.0, range_p / 48.0)
    elif range_p < 7:
        result["confined"] = 1.0 - range_p / 7.0
    return result


def rhythm_to_motion(onsets_ms: np.ndarray,
                     bpm: float) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if len(onsets_ms) < 2:
        result["stillness"] = 1.0
        return result
    ioi = np.diff(onsets_ms)
    mean_ioi = float(np.mean(ioi))
    cv = float(np.std(ioi) / max(mean_ioi, 1.0))
    local_bpm = 60_000.0 / max(mean_ioi, 1.0)
    if local_bpm < 66:
        result["swaying"] = 0.8
    elif local_bpm < 100:
        result["walking"] = 0.8
    elif local_bpm < 140:
        result["striding"] = 0.7
    else:
        result["running"] = 0.9
    if cv > 0.5:
        result["stumbling"] = min(1.0, cv)
    elif cv < 0.1:
        result["mechanical"] = 1.0 - cv * 10
    else:
        result["flowing"] = 1.0 - abs(cv - 0.2) * 3
    return result


def dynamics_to_energy(amplitudes: np.ndarray) -> Dict[str, float]:
    if len(amplitudes) == 0:
        return {"silence": 1.0}
    mean_amp = float(np.mean(amplitudes))
    result: Dict[str, float] = {}
    if mean_amp > 0.7:
        result["powerful"] = min(1.0, mean_amp)
        result["intense"] = mean_amp * 0.8
    elif mean_amp > 0.3:
        result["moderate_energy"] = 1.0 - abs(mean_amp - 0.5) * 4
    else:
        result["delicate"] = 1.0 - mean_amp * 3
        result["intimate"] = result["delicate"] * 0.7
    if len(amplitudes) > 2:
        gradient = np.diff(amplitudes)
        if np.mean(gradient) > 0.01:
            result["building"] = min(1.0, float(np.mean(gradient)) * 10)
        elif np.mean(gradient) < -0.01:
            result["fading"] = min(1.0, float(-np.mean(gradient)) * 10)
    return result


def timbre_brightness_to_semantic(spectral_centroid_hz: float) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if spectral_centroid_hz > 4000:
        result["brilliant"] = min(1.0, (spectral_centroid_hz - 4000) / 4000)
        result["sharp"] = result["brilliant"] * 0.6
    elif spectral_centroid_hz > 2000:
        result["warm"] = 1.0 - (spectral_centroid_hz - 2000) / 2000
        result["clear"] = (spectral_centroid_hz - 2000) / 2000
    elif spectral_centroid_hz > 500:
        result["warm"] = min(1.0, (spectral_centroid_hz - 500) / 1500)
        result["mellow"] = 1.0 - result["warm"]
    else:
        result["dark"] = min(1.0, (500 - spectral_centroid_hz) / 500)
        result["heavy"] = result["dark"] * 0.6
    return result


def harmonic_tension_to_narrative(
        tension_curve: np.ndarray) -> List[Tuple[float, float, str]]:
    if len(tension_curve) == 0:
        return []
    n = len(tension_curve)
    segments: List[Tuple[float, float, str]] = []
    positions = np.linspace(0, 1, n)
    max_idx = int(np.argmax(tension_curve))
    climax_pos = positions[max_idx]
    for i in range(n):
        t = positions[i]
        if t < climax_pos * 0.3:
            label = "exposition"
        elif t < climax_pos * 0.7:
            label = "rising_action"
        elif t < climax_pos * 1.1:
            label = "climax"
        elif t < climax_pos + (1 - climax_pos) * 0.5:
            label = "falling_action"
        else:
            label = "resolution"
        segments.append((float(t), float(tension_curve[i]), label))
    return segments


def tempo_to_activity(bpm: float) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name, (lo, hi, descriptor) in TEMPO_ACTIVITY.items():
        if lo <= bpm < hi:
            position = (bpm - lo) / max(hi - lo, 1)
            result[descriptor] = 0.5 + 0.5 * position
    if not result:
        result["extreme"] = 1.0
    return result


def mode_to_affect(is_major: bool) -> Dict[str, float]:
    if is_major:
        return {"happy": 0.6, "bright": 0.7, "confident": 0.5, "open": 0.5}
    return {"sad": 0.5, "dark": 0.6, "introspective": 0.6, "tender": 0.4}


def aggregate_semantics(feature_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    count: Dict[str, int] = {}
    for domain_results in feature_dict.values():
        for concept, weight in domain_results.items():
            combined[concept] = combined.get(concept, 0.0) + weight
            count[concept] = count.get(concept, 0) + 1
    for concept in combined:
        combined[concept] /= count[concept]
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}
    return dict(sorted(combined.items(), key=lambda x: -x[1]))


def semantic_similarity(profile_a: Dict[str, float],
                        profile_b: Dict[str, float]) -> float:
    all_keys = set(profile_a) | set(profile_b)
    if not all_keys:
        return 0.0
    a = np.array([profile_a.get(k, 0.0) for k in all_keys])
    b = np.array([profile_b.get(k, 0.0) for k in all_keys])
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def describe_musical_moment(pitches: np.ndarray, onsets_ms: np.ndarray,
                            amplitudes: np.ndarray,
                            bpm: float) -> List[Tuple[str, float]]:
    features: Dict[str, Dict[str, float]] = {}
    features["pitch"] = pitch_to_spatial(pitches)
    features["rhythm"] = rhythm_to_motion(onsets_ms, bpm)
    features["dynamics"] = dynamics_to_energy(amplitudes)
    features["tempo"] = tempo_to_activity(bpm)
    profile = aggregate_semantics(features)
    return sorted(profile.items(), key=lambda x: -x[1])
