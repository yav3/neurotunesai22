"""Tonal cognition: Krumhansl key profiles, key finding, tonal hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "KeyProfile", "KeyFindingResult", "TonalDistance",
    "krumhansl_major_profile", "krumhansl_minor_profile",
    "key_finding_correlation", "find_key",
    "tonal_hierarchy_distance", "inter_key_distance",
    "toroidal_key_coordinates", "detect_modulations",
    "tonal_tension_curve", "pitch_class_distribution",
]

KRUMHANSL_MAJOR = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
])

KRUMHANSL_MINOR = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
])

TEMPERLEY_MAJOR = np.array([
    5.0, 2.0, 3.5, 2.0, 4.5, 4.0,
    2.0, 4.5, 2.0, 3.5, 1.5, 4.0,
])

TEMPERLEY_MINOR = np.array([
    5.0, 2.0, 3.5, 4.5, 2.0, 3.5,
    2.0, 4.5, 3.5, 2.0, 1.5, 4.0,
])

KEY_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass
class KeyProfile:
    weights: np.ndarray
    mode: str
    tonic: int

    @property
    def key_name(self) -> str:
        mode_suffix = "" if self.mode == "major" else "m"
        return f"{KEY_NAMES[self.tonic % 12]}{mode_suffix}"


def krumhansl_major_profile(tonic: int = 0) -> KeyProfile:
    weights = np.roll(KRUMHANSL_MAJOR, tonic)
    return KeyProfile(weights=weights, mode="major", tonic=tonic)


def krumhansl_minor_profile(tonic: int = 0) -> KeyProfile:
    weights = np.roll(KRUMHANSL_MINOR, tonic)
    return KeyProfile(weights=weights, mode="minor", tonic=tonic)


def pitch_class_distribution(pitches: np.ndarray, durations: Optional[np.ndarray] = None) -> np.ndarray:
    pcs = np.asarray(pitches, dtype=int) % 12
    dist = np.zeros(12)
    if durations is not None:
        for pc, dur in zip(pcs, durations):
            dist[pc] += dur
    else:
        for pc in pcs:
            dist[pc] += 1
    total = dist.sum()
    if total > 0:
        dist /= total
    return dist


@dataclass
class KeyFindingResult:
    key: str
    tonic: int
    mode: str
    correlation: float
    all_correlations: np.ndarray
    confidence: float


def key_finding_correlation(
    pc_distribution: np.ndarray,
    profile: np.ndarray,
) -> float:
    d = pc_distribution - pc_distribution.mean()
    p = profile - profile.mean()
    num = np.dot(d, p)
    den = np.sqrt(np.dot(d, d) * np.dot(p, p))
    return float(num / max(den, 1e-15))


def find_key(
    pitches: np.ndarray,
    durations: Optional[np.ndarray] = None,
    profile_type: str = "krumhansl",
) -> KeyFindingResult:
    dist = pitch_class_distribution(pitches, durations)
    if profile_type == "temperley":
        major_base, minor_base = TEMPERLEY_MAJOR, TEMPERLEY_MINOR
    else:
        major_base, minor_base = KRUMHANSL_MAJOR, KRUMHANSL_MINOR
    corrs = np.zeros(24)
    for t in range(12):
        maj_profile = np.roll(major_base, t)
        min_profile = np.roll(minor_base, t)
        corrs[t] = key_finding_correlation(dist, maj_profile)
        corrs[t + 12] = key_finding_correlation(dist, min_profile)
    best = int(np.argmax(corrs))
    if best < 12:
        tonic, mode = best, "major"
    else:
        tonic, mode = best - 12, "minor"
    sorted_corrs = np.sort(corrs)[::-1]
    confidence = float(sorted_corrs[0] - sorted_corrs[1])
    mode_suffix = "" if mode == "major" else "m"
    return KeyFindingResult(
        key=f"{KEY_NAMES[tonic]}{mode_suffix}",
        tonic=tonic, mode=mode,
        correlation=float(corrs[best]),
        all_correlations=corrs,
        confidence=confidence,
    )


@dataclass
class TonalDistance:
    key1: str
    key2: str
    distance: float
    relationship: str


def inter_key_distance(
    tonic1: int, mode1: str,
    tonic2: int, mode2: str,
) -> float:
    if mode1 == "major":
        p1 = KRUMHANSL_MAJOR
    else:
        p1 = KRUMHANSL_MINOR
    if mode2 == "major":
        p2 = KRUMHANSL_MAJOR
    else:
        p2 = KRUMHANSL_MINOR
    profile1 = np.roll(p1, tonic1)
    profile2 = np.roll(p2, tonic2)
    corr = key_finding_correlation(profile1, profile2)
    return float(1.0 - corr)


def tonal_hierarchy_distance(
    tonic1: int, mode1: str,
    tonic2: int, mode2: str,
) -> TonalDistance:
    dist = inter_key_distance(tonic1, mode1, tonic2, mode2)
    interval = (tonic2 - tonic1) % 12
    if interval == 0 and mode1 == mode2:
        rel = "same"
    elif interval == 0:
        rel = "parallel"
    elif (mode1 == "major" and interval == 9) or (mode1 == "minor" and interval == 3):
        rel = "relative"
    elif interval == 7:
        rel = "dominant"
    elif interval == 5:
        rel = "subdominant"
    elif interval in (1, 11):
        rel = "chromatic_neighbor"
    else:
        rel = "distant"
    m1 = "" if mode1 == "major" else "m"
    m2 = "" if mode2 == "major" else "m"
    return TonalDistance(
        key1=f"{KEY_NAMES[tonic1]}{m1}",
        key2=f"{KEY_NAMES[tonic2]}{m2}",
        distance=dist, relationship=rel,
    )


def toroidal_key_coordinates(tonic: int, mode: str) -> Tuple[float, float]:
    fifths_pos = (tonic * 7) % 12
    angle_fifths = 2 * np.pi * fifths_pos / 12
    angle_thirds = 0.0 if mode == "major" else np.pi / 3
    x = np.cos(angle_fifths) + 0.3 * np.cos(angle_thirds)
    y = np.sin(angle_fifths) + 0.3 * np.sin(angle_thirds)
    return (float(x), float(y))


def detect_modulations(
    pitches: np.ndarray,
    window_size: int = 20,
    hop_size: int = 5,
    threshold: float = 0.3,
) -> List[Dict]:
    p = np.asarray(pitches, dtype=float)
    n = len(p)
    modulations = []
    prev_key = None
    for start in range(0, n - window_size + 1, hop_size):
        window = p[start:start + window_size]
        result = find_key(window)
        if prev_key is not None and result.key != prev_key:
            if result.confidence > threshold:
                modulations.append({
                    "position": start,
                    "from_key": prev_key,
                    "to_key": result.key,
                    "confidence": result.confidence,
                })
        prev_key = result.key
    return modulations


def tonal_tension_curve(
    pitches: np.ndarray,
    key_tonic: int = 0,
    key_mode: str = "major",
    window: int = 8,
) -> np.ndarray:
    p = np.asarray(pitches, dtype=float)
    n = len(p)
    if n < window:
        return np.zeros(max(n, 1))
    if key_mode == "major":
        profile = np.roll(KRUMHANSL_MAJOR, key_tonic)
    else:
        profile = np.roll(KRUMHANSL_MINOR, key_tonic)
    profile = profile / profile.max()
    tension = np.zeros(n)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        local_dist = pitch_class_distribution(p[start:end])
        stability = np.sum(local_dist * profile)
        tension[i] = 1.0 - stability
    return tension
