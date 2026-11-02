"""Cognitive simplification: reduce musical complexity for cognitive accessibility.

Algorithms for simplifying musical content for listeners with cognitive
disabilities or processing difficulties: rhythmic simplification,
melodic reduction, harmonic thinning, texture reduction, structural
regularisation, and cognitive load estimation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SimplificationLevel(str, Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    MAXIMAL = "maximal"


class CognitiveMetric(str, Enum):
    INFORMATION_RATE = "information_rate"
    RHYTHMIC_COMPLEXITY = "rhythmic_complexity"
    HARMONIC_COMPLEXITY = "harmonic_complexity"
    MELODIC_COMPLEXITY = "melodic_complexity"
    TEXTURAL_DENSITY = "textural_density"
    STRUCTURAL_REGULARITY = "structural_regularity"


@dataclass
class CognitiveProfile:
    processing_speed: float = 1.0
    working_memory_capacity: int = 7
    attention_span_s: float = 30.0
    preferred_tempo_range: Tuple[float, float] = (60.0, 120.0)
    max_simultaneous_voices: int = 4


@dataclass
class SimplificationResult:
    original_complexity: float
    simplified_complexity: float
    reduction_pct: float
    operations_applied: List[str]


def estimate_information_rate(pitches: np.ndarray,
                              onsets_ms: np.ndarray) -> float:
    if len(pitches) < 2 or len(onsets_ms) < 2:
        return 0.0
    duration_s = (onsets_ms[-1] - onsets_ms[0]) / 1000.0
    if duration_s <= 0:
        return 0.0
    unique_pitches = len(np.unique(pitches))
    n_events = len(pitches)
    if unique_pitches <= 1:
        return n_events / duration_s * 0.5
    entropy = 0.0
    for p in np.unique(pitches):
        prob = np.sum(pitches == p) / len(pitches)
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy * n_events / duration_s


def rhythmic_complexity(onsets_ms: np.ndarray, bpm: float) -> float:
    if len(onsets_ms) < 3:
        return 0.0
    ioi = np.diff(onsets_ms)
    beat_ms = 60_000.0 / bpm
    ratios = ioi / beat_ms
    quantised = np.round(ratios * 4) / 4
    deviations = np.abs(ratios - quantised)
    syncopation = np.sum(np.abs(quantised - np.round(quantised))) / len(quantised)
    irregularity = float(np.std(ioi) / max(np.mean(ioi), 1e-6))
    return float(np.clip(syncopation * 0.4 + irregularity * 0.3 +
                         np.mean(deviations) * 0.3, 0, 1))


def melodic_complexity(pitches: np.ndarray) -> float:
    if len(pitches) < 3:
        return 0.0
    intervals = np.abs(np.diff(pitches.astype(float)))
    mean_interval = float(np.mean(intervals))
    range_pct = float(np.ptp(pitches)) / 88.0
    direction_changes = 0
    signs = np.sign(np.diff(pitches.astype(float)))
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0:
            direction_changes += 1
    contour_complexity = direction_changes / max(len(signs) - 1, 1)
    return float(np.clip(mean_interval / 12.0 * 0.3 +
                         range_pct * 0.3 + contour_complexity * 0.4, 0, 1))


def harmonic_complexity_score(chord_roots: np.ndarray,
                              key_pc: int = 0) -> float:
    if len(chord_roots) < 2:
        return 0.0
    intervals = np.abs(np.diff(chord_roots.astype(float))) % 12
    diatonic = {0, 2, 4, 5, 7, 9, 11}
    chromatic_count = sum(1 for r in chord_roots if int(r) % 12 not in diatonic)
    chromatic_ratio = chromatic_count / len(chord_roots)
    motion_distance = float(np.mean(intervals)) / 6.0
    return float(np.clip(chromatic_ratio * 0.5 + motion_distance * 0.5, 0, 1))


def cognitive_load_estimate(pitches: np.ndarray, onsets_ms: np.ndarray,
                            bpm: float, n_voices: int = 1) -> Dict[str, float]:
    info = estimate_information_rate(pitches, onsets_ms)
    rhyth = rhythmic_complexity(onsets_ms, bpm)
    melod = melodic_complexity(pitches)
    texture = min(1.0, n_voices / 8.0)
    overall = (info / 10.0) * 0.3 + rhyth * 0.25 + melod * 0.25 + texture * 0.2
    return {
        "information_rate": info,
        "rhythmic": rhyth,
        "melodic": melod,
        "textural": texture,
        "overall": float(np.clip(overall, 0, 1)),
    }


def simplify_rhythm(onsets_ms: np.ndarray, bpm: float,
                    level: SimplificationLevel = SimplificationLevel.MODERATE) -> np.ndarray:
    beat_ms = 60_000.0 / bpm
    if level == SimplificationLevel.MINIMAL:
        grid = beat_ms / 2
    elif level == SimplificationLevel.MODERATE:
        grid = beat_ms
    elif level == SimplificationLevel.SIGNIFICANT:
        grid = beat_ms * 2
    else:
        grid = beat_ms * 4
    quantised = np.round(onsets_ms / grid) * grid
    _, unique_idx = np.unique(quantised, return_index=True)
    return quantised[np.sort(unique_idx)]


def simplify_melody(pitches: np.ndarray,
                    level: SimplificationLevel = SimplificationLevel.MODERATE) -> np.ndarray:
    if len(pitches) < 3:
        return pitches.copy()
    if level == SimplificationLevel.MINIMAL:
        max_interval = 7
    elif level == SimplificationLevel.MODERATE:
        max_interval = 5
    elif level == SimplificationLevel.SIGNIFICANT:
        max_interval = 3
    else:
        max_interval = 2
    result = pitches.copy().astype(float)
    for i in range(1, len(result)):
        interval = result[i] - result[i - 1]
        if abs(interval) > max_interval:
            direction = 1 if interval > 0 else -1
            result[i] = result[i - 1] + direction * max_interval
    return result.astype(int)


def reduce_texture(voices: List[np.ndarray],
                   max_voices: int = 2) -> List[np.ndarray]:
    if len(voices) <= max_voices:
        return voices
    ranges = [(float(np.ptp(v)), i) for i, v in enumerate(voices)]
    ranges.sort(key=lambda x: -x[0])
    keep_indices = sorted([idx for _, idx in ranges[:max_voices]])
    return [voices[i] for i in keep_indices]


def regularise_structure(section_durations_s: np.ndarray,
                         target_regularity: float = 0.8) -> np.ndarray:
    if len(section_durations_s) < 2:
        return section_durations_s.copy()
    mean_dur = float(np.mean(section_durations_s))
    result = section_durations_s.copy()
    for i in range(len(result)):
        result[i] = result[i] * (1 - target_regularity) + mean_dur * target_regularity
    return result


def adapt_tempo_to_profile(current_bpm: float,
                           profile: CognitiveProfile) -> float:
    lo, hi = profile.preferred_tempo_range
    adapted = current_bpm * profile.processing_speed
    return float(np.clip(adapted, lo, hi))


def create_simplified_version(pitches: np.ndarray, onsets_ms: np.ndarray,
                              bpm: float, n_voices: int,
                              profile: CognitiveProfile,
                              level: SimplificationLevel = SimplificationLevel.MODERATE
                              ) -> Tuple[np.ndarray, np.ndarray, SimplificationResult]:
    load_before = cognitive_load_estimate(pitches, onsets_ms, bpm, n_voices)
    new_onsets = simplify_rhythm(onsets_ms, bpm, level)
    min_len = min(len(pitches), len(new_onsets))
    new_pitches = simplify_melody(pitches[:min_len], level)
    new_onsets = new_onsets[:min_len]
    load_after = cognitive_load_estimate(new_pitches, new_onsets, bpm,
                                        min(n_voices, profile.max_simultaneous_voices))
    ops = ["rhythm_quantise", "melody_simplify"]
    if n_voices > profile.max_simultaneous_voices:
        ops.append("texture_reduce")
    reduction = 1.0 - load_after["overall"] / max(load_before["overall"], 1e-6)
    return new_pitches, new_onsets, SimplificationResult(
        original_complexity=load_before["overall"],
        simplified_complexity=load_after["overall"],
        reduction_pct=float(np.clip(reduction * 100, 0, 100)),
        operations_applied=ops,
    )
