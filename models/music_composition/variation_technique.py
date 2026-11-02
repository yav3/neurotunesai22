"""Variation technique: theme-and-variations procedures, melodic
embellishment, reharmonization, rhythmic transformation, invertible
counterpoint, ground bass / passacaglia / chaconne patterns.

Based on Nelson (1948) variation form analysis, Schoenberg (1967)
developing variation, Caplin (1998) formal functions. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class VariationType(Enum):
    MELODIC = "melodic"
    RHYTHMIC = "rhythmic"
    HARMONIC = "harmonic"
    TEXTURAL = "textural"
    ORNAMENTAL = "ornamental"
    COMBINATORIAL = "combinatorial"


class EmbellishmentType(Enum):
    PASSING_TONE = "passing_tone"
    NEIGHBOR_TONE = "neighbor_tone"
    APPOGGIATURA = "appoggiatura"
    ESCAPE_TONE = "escape_tone"
    SUSPENSION = "suspension"
    ANTICIPATION = "anticipation"


@dataclass
class Theme:
    pitches: np.ndarray
    durations: np.ndarray
    chord_roots: Optional[np.ndarray] = None


@dataclass
class Variation:
    pitches: np.ndarray
    durations: np.ndarray
    variation_type: VariationType = VariationType.MELODIC
    similarity_to_theme: float = 1.0


@dataclass
class VariationSet:
    theme: Theme
    variations: List[Variation] = field(default_factory=list)


def melodic_embellishment(pitches: np.ndarray,
                          density: float = 0.3,
                          seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    result = []
    for i, p in enumerate(pitches):
        result.append(int(p))
        if i < len(pitches) - 1 and rng.random() < density:
            target = int(pitches[i + 1])
            diff = target - int(p)
            if abs(diff) > 2:
                result.append(int(p) + np.sign(diff))
            elif abs(diff) == 2:
                result.append(int(p) + np.sign(diff))
            else:
                result.append(int(p) + (1 if rng.random() > 0.5 else -1))
    return np.array(result)


def rhythmic_augmentation(durations: np.ndarray,
                          factor: float = 2.0) -> np.ndarray:
    return durations * factor


def rhythmic_diminution(durations: np.ndarray,
                        factor: float = 2.0) -> np.ndarray:
    return durations / factor


def rhythmic_displacement(durations: np.ndarray,
                          shift: float = 0.5) -> np.ndarray:
    result = durations.copy()
    result[0] += shift
    if len(result) > 1:
        result[-1] -= shift
    return np.maximum(result, 0.01)


def dotted_rhythm(durations: np.ndarray) -> np.ndarray:
    result = durations.copy()
    for i in range(0, len(result) - 1, 2):
        total = result[i] + result[i + 1]
        result[i] = total * 0.75
        result[i + 1] = total * 0.25
    return result


def lombardic_rhythm(durations: np.ndarray) -> np.ndarray:
    result = durations.copy()
    for i in range(0, len(result) - 1, 2):
        total = result[i] + result[i + 1]
        result[i] = total * 0.25
        result[i + 1] = total * 0.75
    return result


def reharmonize(chord_roots: np.ndarray,
                mode: str = "substitute") -> np.ndarray:
    result = chord_roots.copy()
    tritone_sub = {0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11,
                   6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}
    relative = {0: 9, 2: 11, 4: 0, 5: 2, 7: 4, 9: 5, 11: 7}
    for i in range(len(result)):
        pc = int(result[i]) % 12
        if mode == "tritone" and pc in tritone_sub:
            result[i] = int(result[i]) - pc + tritone_sub[pc]
        elif mode == "substitute" and pc in relative:
            result[i] = int(result[i]) - pc + relative[pc]
    return result


def invertible_counterpoint(melody: np.ndarray,
                            interval: int = 12) -> np.ndarray:
    pivot = np.mean(melody)
    return 2 * pivot - melody + interval


def ground_bass_variations(bass: np.ndarray, n_variations: int = 4,
                           seed: int = 42) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    variations = [bass.copy()]
    for v in range(n_variations):
        var = bass.copy()
        for i in range(len(var)):
            offset = rng.choice([-2, -1, 0, 0, 0, 1, 2])
            var[i] += offset
        variations.append(var)
    return variations


def theme_similarity(theme: np.ndarray, variation: np.ndarray) -> float:
    n = min(len(theme), len(variation))
    if n == 0:
        return 0.0
    intervals_t = np.diff(theme[:n].astype(float))
    intervals_v = np.diff(variation[:n].astype(float))
    m = min(len(intervals_t), len(intervals_v))
    if m == 0:
        return 1.0
    contour_t = np.sign(intervals_t[:m])
    contour_v = np.sign(intervals_v[:m])
    contour_match = float(np.mean(contour_t == contour_v))
    interval_diff = np.mean(np.abs(intervals_t[:m] - intervals_v[:m]))
    interval_sim = 1.0 / (1.0 + interval_diff)
    return 0.5 * contour_match + 0.5 * interval_sim


def developing_variation(pitches: np.ndarray,
                         n_steps: int = 4,
                         mutation_rate: float = 0.2,
                         seed: int = 42) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    chain = [pitches.copy()]
    current = pitches.copy()
    for _ in range(n_steps):
        new = current.copy()
        for i in range(len(new)):
            if rng.random() < mutation_rate:
                new[i] += rng.choice([-2, -1, 1, 2])
        chain.append(new)
        current = new
    return chain


def sequence_variation(pitches: np.ndarray,
                       interval: int = 2,
                       n_repeats: int = 3) -> np.ndarray:
    parts = [pitches + interval * i for i in range(n_repeats)]
    return np.concatenate(parts)


def ornamental_variation(pitches: np.ndarray,
                         durations: np.ndarray,
                         ornament_prob: float = 0.4,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    new_pitches = []
    new_durations = []
    for i in range(len(pitches)):
        p = int(pitches[i])
        d = float(durations[i])
        if rng.random() < ornament_prob and d > 0.1:
            choice = rng.choice(["trill", "mordent", "turn"])
            if choice == "trill":
                rep = max(2, int(d / 0.05))
                for r in range(rep):
                    new_pitches.append(p if r % 2 == 0 else p + 2)
                    new_durations.append(d / rep)
            elif choice == "mordent":
                new_pitches.extend([p, p + 1, p])
                new_durations.extend([d * 0.5, d * 0.25, d * 0.25])
            else:
                new_pitches.extend([p, p + 2, p, p - 1, p])
                new_durations.extend([d * 0.2] * 5)
        else:
            new_pitches.append(p)
            new_durations.append(d)
    return np.array(new_pitches), np.array(new_durations)


def textural_thickening(pitches: np.ndarray,
                        intervals: List[int] = None) -> np.ndarray:
    if intervals is None:
        intervals = [0, 4, 7]
    voices = []
    for iv in intervals:
        voices.append(pitches + iv)
    return np.column_stack(voices)


def variation_distance(v1: Variation, v2: Variation) -> float:
    n = min(len(v1.pitches), len(v2.pitches))
    if n == 0:
        return 1.0
    pitch_diff = np.mean(np.abs(v1.pitches[:n].astype(float) - v2.pitches[:n].astype(float)))
    dur_n = min(len(v1.durations), len(v2.durations))
    if dur_n > 0:
        dur_diff = np.mean(np.abs(v1.durations[:dur_n] - v2.durations[:dur_n]))
    else:
        dur_diff = 0.0
    return float(pitch_diff / 12.0 + dur_diff)


def cumulative_variation_arc(variations: List[Variation]) -> np.ndarray:
    if len(variations) < 2:
        return np.array([0.0])
    distances = []
    for i in range(1, len(variations)):
        d = variation_distance(variations[i - 1], variations[i])
        distances.append(d)
    return np.cumsum(distances)
