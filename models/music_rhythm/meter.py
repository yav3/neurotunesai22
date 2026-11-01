"""Meter analysis: time signature, downbeat detection, metric hierarchy."""
from __future__ import annotations
import numpy as np


def metric_hierarchy(numerator: int) -> list[float]:
    """Return metric weight profile for a given numerator.

    Weights follow the standard metric hierarchy:
    4/4 → [1.0, 0.25, 0.5, 0.25]  (strong, weak, medium, weak)
    3/4 → [1.0, 0.33, 0.33]        (strong, weak, weak)
    """
    if numerator <= 0:
        return []
    weights = [0.0] * numerator
    weights[0] = 1.0  # downbeat is always strongest

    if numerator == 1:
        return weights

    # binary subdivision
    level = numerator
    strength = 0.5
    while level > 1:
        half = level // 2
        for i in range(half, numerator, level):
            if weights[i] == 0.0:
                weights[i] = strength
        level = half
        strength *= 0.5

    # fill remaining zeros with minimum weight
    min_w = strength
    for i in range(numerator):
        if weights[i] == 0.0:
            weights[i] = min_w

    return weights


def metric_weight_profile(
    beat_times: np.ndarray,
    onset_strengths: np.ndarray,
    candidates: list[int] | None = None,
) -> dict[int, float]:
    """Score candidate meters by correlation with onset strengths.

    For each candidate numerator, tile the metric hierarchy and
    correlate with observed onset strengths.
    """
    if candidates is None:
        candidates = [2, 3, 4, 6]

    n = len(beat_times)
    if n < 2 or len(onset_strengths) != n:
        return {c: 0.0 for c in candidates}

    scores = {}
    for num in candidates:
        template = metric_hierarchy(num)
        if not template:
            scores[num] = 0.0
            continue
        # tile template to match length
        tiled = np.array([template[i % num] for i in range(n)])
        # normalize both
        t_centered = tiled - np.mean(tiled)
        o_centered = onset_strengths - np.mean(onset_strengths)
        denom = np.linalg.norm(t_centered) * np.linalg.norm(o_centered)
        if denom < 1e-12:
            scores[num] = 0.0
        else:
            scores[num] = float(np.dot(t_centered, o_centered) / denom)
    return scores


def downbeat_detection(
    beat_times: np.ndarray,
    onset_strengths: np.ndarray,
    meter: int = 4,
) -> np.ndarray:
    """Identify downbeats from beat positions.

    Uses onset strength peaks with metric-period spacing.
    Returns indices into beat_times that are downbeats.
    """
    n = len(beat_times)
    if n == 0:
        return np.array([], dtype=int)
    if meter <= 0:
        meter = 4

    # find the offset (phase) that maximizes total strength at downbeat positions
    best_offset = 0
    best_score = -np.inf
    for offset in range(min(meter, n)):
        indices = np.arange(offset, n, meter)
        score = float(np.sum(onset_strengths[indices]))
        if score > best_score:
            best_score = score
            best_offset = offset

    return np.arange(best_offset, n, meter)


def time_signature_from_beats(
    beat_times: np.ndarray,
    onset_strengths: np.ndarray,
) -> tuple[int, int]:
    """Estimate time signature from beat positions and strengths.

    Returns (numerator, denominator). Denominator is always 4 (quarter note).
    """
    if len(beat_times) < 4:
        return (4, 4)

    scores = metric_weight_profile(beat_times, onset_strengths, [2, 3, 4, 6])
    best = max(scores, key=lambda k: scores[k])
    return (best, 4)


class MeterAnalyzer:
    """Full meter analysis: time signature, downbeats, metric profile."""

    def __init__(self, candidates: list[int] | None = None) -> None:
        self.candidates = candidates or [2, 3, 4, 6]

    def analyze(
        self,
        beat_times: np.ndarray,
        onset_strengths: np.ndarray,
    ) -> dict:
        if len(beat_times) < 2:
            return {
                "time_signature": (4, 4),
                "downbeat_indices": np.array([], dtype=int),
                "metric_scores": {},
            }
        ts = time_signature_from_beats(beat_times, onset_strengths)
        db = downbeat_detection(beat_times, onset_strengths, ts[0])
        scores = metric_weight_profile(beat_times, onset_strengths, self.candidates)
        return {
            "time_signature": ts,
            "downbeat_indices": db,
            "metric_scores": scores,
        }
