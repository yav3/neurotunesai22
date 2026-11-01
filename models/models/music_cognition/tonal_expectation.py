"""Tonal expectation models: Krumhansl key profiles, tonal tension, pitch-class distributions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class Mode(str, Enum):
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class KeyEstimate:
    tonic: int
    mode: Mode
    correlation: float
    name: str


@dataclass
class TonalTensionProfile:
    cloud_diameter: float
    cloud_momentum: float
    tensile_strain: float
    combined: float


def _rotate(profile: List[float], shift: int) -> List[float]:
    return profile[-shift:] + profile[:-shift]


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x)) or 1e-12
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y)) or 1e-12
    return num / (dx * dy)


def pitch_class_distribution(midi_notes: List[int], durations: Optional[List[float]] = None) -> List[float]:
    dist = [0.0] * 12
    for i, note in enumerate(midi_notes):
        w = durations[i] if durations and i < len(durations) else 1.0
        dist[note % 12] += w
    total = sum(dist) or 1.0
    return [d / total for d in dist]


def estimate_key(midi_notes: List[int], durations: Optional[List[float]] = None) -> KeyEstimate:
    dist = pitch_class_distribution(midi_notes, durations)
    best: Optional[KeyEstimate] = None
    for tonic in range(12):
        for mode, profile in [(Mode.MAJOR, KRUMHANSL_MAJOR), (Mode.MINOR, KRUMHANSL_MINOR)]:
            r = _pearson(dist, _rotate(profile, tonic))
            name = f"{PITCH_NAMES[tonic]} {mode.value}"
            ke = KeyEstimate(tonic=tonic, mode=mode, correlation=r, name=name)
            if best is None or r > best.correlation:
                best = ke
    assert best is not None
    return best


def key_profile_correlation(dist: List[float], tonic: int, mode: Mode) -> float:
    profile = KRUMHANSL_MAJOR if mode == Mode.MAJOR else KRUMHANSL_MINOR
    return _pearson(dist, _rotate(profile, tonic))


def tonal_tension(dist_prev: List[float], dist_curr: List[float], key_tonic: int, key_mode: Mode) -> TonalTensionProfile:
    profile = KRUMHANSL_MAJOR if key_mode == Mode.MAJOR else KRUMHANSL_MINOR
    rotated = _rotate(profile, key_tonic)
    cloud_diam = math.sqrt(sum((a - b) ** 2 for a, b in zip(dist_curr, rotated)))
    momentum = math.sqrt(sum((a - b) ** 2 for a, b in zip(dist_curr, dist_prev)))
    strain = 1.0 - _pearson(dist_curr, rotated)
    combined = 0.4 * cloud_diam + 0.3 * momentum + 0.3 * strain
    return TonalTensionProfile(cloud_diameter=cloud_diam, cloud_momentum=momentum,
                                tensile_strain=strain, combined=combined)


def modulation_distance(key1_tonic: int, key1_mode: Mode,
                        key2_tonic: int, key2_mode: Mode) -> float:
    p1 = _rotate(KRUMHANSL_MAJOR if key1_mode == Mode.MAJOR else KRUMHANSL_MINOR, key1_tonic)
    p2 = _rotate(KRUMHANSL_MAJOR if key2_mode == Mode.MAJOR else KRUMHANSL_MINOR, key2_tonic)
    return 1.0 - _pearson(p1, p2)


def surprisal(pitch_class: int, distribution: List[float]) -> float:
    p = distribution[pitch_class % 12]
    if p <= 0:
        return 20.0
    return -math.log2(p)


def entropy(distribution: List[float]) -> float:
    return -sum(p * math.log2(p) for p in distribution if p > 0)


def key_clarity(distribution: List[float]) -> float:
    correlations: List[float] = []
    for tonic in range(12):
        for profile in [KRUMHANSL_MAJOR, KRUMHANSL_MINOR]:
            correlations.append(_pearson(distribution, _rotate(profile, tonic)))
    correlations.sort(reverse=True)
    return correlations[0] - correlations[1] if len(correlations) >= 2 else correlations[0]
