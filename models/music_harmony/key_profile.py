"""Key detection via Krumhansl-Schmuckler probe-tone profiles and pitch-class analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.music_harmony.modulation import Key


KRUMHANSL_MAJOR = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
], dtype=np.float64)

KRUMHANSL_MINOR = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
], dtype=np.float64)

TEMPERLEY_MAJOR = np.array([
    5.0, 2.0, 3.5, 2.0, 4.5, 4.0,
    2.0, 4.5, 2.0, 3.5, 1.5, 4.0,
], dtype=np.float64)

TEMPERLEY_MINOR = np.array([
    5.0, 2.0, 3.5, 4.5, 2.0, 3.5,
    2.0, 4.5, 3.5, 2.0, 1.5, 4.0,
], dtype=np.float64)

AARDEN_MAJOR = np.array([
    17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587,
    0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122,
], dtype=np.float64)

AARDEN_MINOR = np.array([
    18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362,
    0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623,
], dtype=np.float64)


PROFILE_SETS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "krumhansl": (KRUMHANSL_MAJOR, KRUMHANSL_MINOR),
    "temperley": (TEMPERLEY_MAJOR, TEMPERLEY_MINOR),
    "aarden": (AARDEN_MAJOR, AARDEN_MINOR),
}


@dataclass
class KeyEstimate:
    key: Key
    correlation: float
    all_correlations: dict[str, float]


def pitch_class_histogram(pitches: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    pcs = np.mod(pitches.astype(int), 12)
    hist = np.zeros(12, dtype=np.float64)
    if weights is not None and len(weights) == len(pcs):
        for pc, w in zip(pcs, weights):
            hist[pc] += w
    else:
        for pc in pcs:
            hist[pc] += 1.0
    return hist


def _correlate_profile(histogram: np.ndarray, profile: np.ndarray) -> float:
    if np.std(histogram) < 1e-10 or np.std(profile) < 1e-10:
        return 0.0
    return float(np.corrcoef(histogram, profile)[0, 1])


def detect_key(
    pitches: np.ndarray,
    weights: np.ndarray | None = None,
    profile_set: str = "krumhansl",
) -> KeyEstimate:
    hist = pitch_class_histogram(pitches, weights)
    major_profile, minor_profile = PROFILE_SETS.get(
        profile_set, (KRUMHANSL_MAJOR, KRUMHANSL_MINOR)
    )

    best_key: Key | None = None
    best_corr = -2.0
    all_corrs: dict[str, float] = {}

    for root in range(12):
        rotated = np.roll(hist, -root)

        c_maj = _correlate_profile(rotated, major_profile)
        key_maj = Key(root=root, mode="major")
        all_corrs[key_maj.name] = c_maj
        if c_maj > best_corr:
            best_corr = c_maj
            best_key = key_maj

        c_min = _correlate_profile(rotated, minor_profile)
        key_min = Key(root=root, mode="minor")
        all_corrs[key_min.name] = c_min
        if c_min > best_corr:
            best_corr = c_min
            best_key = key_min

    if best_key is None:
        best_key = Key(root=0, mode="major")
    return KeyEstimate(key=best_key, correlation=best_corr, all_correlations=all_corrs)


def key_clarity(estimate: KeyEstimate) -> float:
    corrs = sorted(estimate.all_correlations.values(), reverse=True)
    if len(corrs) < 2:
        return 1.0
    return corrs[0] - corrs[1]


def windowed_key_detection(
    pitches: np.ndarray,
    window_size: int = 16,
    hop_size: int = 4,
    profile_set: str = "krumhansl",
) -> list[KeyEstimate]:
    results: list[KeyEstimate] = []
    n = len(pitches)
    if n < window_size:
        return [detect_key(pitches, profile_set=profile_set)]
    for start in range(0, n - window_size + 1, hop_size):
        segment = pitches[start:start + window_size]
        results.append(detect_key(segment, profile_set=profile_set))
    return results


def key_stability(estimates: list[KeyEstimate]) -> float:
    if len(estimates) <= 1:
        return 1.0
    keys = [e.key for e in estimates]
    same = sum(1 for a, b in zip(keys[:-1], keys[1:]) if a == b)
    return same / (len(keys) - 1)


def pitch_class_entropy(histogram: np.ndarray) -> float:
    total = histogram.sum()
    if total < 1e-10:
        return 0.0
    p = histogram / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))
