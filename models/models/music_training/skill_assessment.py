"""Musical skill assessment and practice optimization.

Quantitative evaluation of musical performance: intonation accuracy,
rhythmic precision, sight-reading difficulty, and spaced-repetition
scheduling for practice sessions.  Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Intonation assessment ────────────────────────────────────────────

def intonation_error_cents(played_hz: np.ndarray, target_hz: np.ndarray) -> np.ndarray:
    """Cent deviation of each played pitch from its target.

    cents = 1200 · log₂(played / target).
    """
    played = np.asarray(played_hz, dtype=float)
    target = np.asarray(target_hz, dtype=float)
    mask = (played > 0) & (target > 0)
    out = np.zeros_like(played)
    out[mask] = 1200.0 * np.log2(played[mask] / target[mask])
    return out


def intonation_score(cents_errors: np.ndarray, tolerance: float = 50.0) -> float:
    """Score ∈ [0,1]: fraction of notes within ±tolerance cents."""
    errs = np.abs(np.asarray(cents_errors, dtype=float))
    if len(errs) == 0:
        return 1.0
    return float(np.mean(errs <= tolerance))


def intonation_bias(cents_errors: np.ndarray) -> float:
    """Signed mean cent error — positive = sharp, negative = flat."""
    errs = np.asarray(cents_errors, dtype=float)
    return float(np.mean(errs)) if len(errs) > 0 else 0.0


# ── Rhythm assessment ────────────────────────────────────────────────

def onset_timing_error(played_onsets: np.ndarray, target_onsets: np.ndarray) -> np.ndarray:
    """Absolute timing offset (seconds) for each played onset vs its target.

    Matches each played onset to the nearest target onset.
    """
    played = np.asarray(played_onsets, dtype=float)
    target = np.asarray(target_onsets, dtype=float)
    if len(played) == 0 or len(target) == 0:
        return np.array([])
    diffs = np.abs(played[:, None] - target[None, :])
    nearest = diffs.min(axis=1)
    return nearest


def rhythm_precision_score(
    timing_errors: np.ndarray, tolerance_sec: float = 0.05
) -> float:
    """Fraction of onsets within ±tolerance of grid."""
    errs = np.asarray(timing_errors, dtype=float)
    if len(errs) == 0:
        return 1.0
    return float(np.mean(errs <= tolerance_sec))


def rhythm_consistency(timing_errors: np.ndarray) -> float:
    """Low std of timing errors → high consistency. Returns 1/(1+σ)."""
    errs = np.asarray(timing_errors, dtype=float)
    if len(errs) < 2:
        return 1.0
    return float(1.0 / (1.0 + np.std(errs)))


# ── Sight-reading difficulty ─────────────────────────────────────────

@dataclass
class SightReadingFeatures:
    """Features extracted from a musical score for difficulty estimation."""
    note_density: float = 0.0
    pitch_range_semitones: int = 0
    max_interval_semitones: int = 0
    accidental_count: int = 0
    syncopation_count: int = 0
    tempo_bpm: float = 120.0
    key_signature_sharps_flats: int = 0
    time_signature_complexity: float = 1.0


def sight_reading_difficulty(features: SightReadingFeatures) -> float:
    """Heuristic difficulty score ∈ [0, 10].

    Weighted combination of score features:
    - note_density (notes/beat): high density → harder
    - pitch_range: wide range → harder
    - max_interval: large leaps → harder
    - accidentals: more → harder
    - syncopation: more → harder
    - tempo: faster → harder
    - key_signature: more sharps/flats → harder
    - time_signature_complexity: irregular meters → harder
    """
    d = 0.0
    d += min(features.note_density / 4.0, 1.0) * 2.0
    d += min(features.pitch_range_semitones / 48.0, 1.0) * 1.5
    d += min(features.max_interval_semitones / 24.0, 1.0) * 1.0
    d += min(features.accidental_count / 20.0, 1.0) * 1.0
    d += min(features.syncopation_count / 10.0, 1.0) * 1.0
    d += min(features.tempo_bpm / 200.0, 1.0) * 1.5
    d += min(features.key_signature_sharps_flats / 7.0, 1.0) * 0.5
    d += min(features.time_signature_complexity / 3.0, 1.0) * 0.5
    return min(d, 10.0)


# ── Spaced repetition for practice ───────────────────────────────────

@dataclass
class PracticeItem:
    """A piece or exercise tracked by the spaced-repetition scheduler."""
    item_id: str = ""
    ease_factor: float = 2.5
    interval_days: float = 1.0
    repetitions: int = 0
    last_review: float = 0.0
    due: float = 0.0


def sm2_update(item: PracticeItem, quality: int, current_time: float = 0.0) -> PracticeItem:
    """SM-2 spaced repetition update (Wozniak 1990).

    quality: 0–5 self-rating (0=blackout, 5=perfect).
    Returns an updated copy of the item.
    """
    q = max(0, min(5, quality))
    ef = item.ease_factor + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
    ef = max(ef, 1.3)

    if q < 3:
        reps = 0
        interval = 1.0
    else:
        reps = item.repetitions + 1
        if reps == 1:
            interval = 1.0
        elif reps == 2:
            interval = 6.0
        else:
            interval = item.interval_days * ef

    return PracticeItem(
        item_id=item.item_id,
        ease_factor=ef,
        interval_days=interval,
        repetitions=reps,
        last_review=current_time,
        due=current_time + interval,
    )


def schedule_practice_session(
    items: list[PracticeItem], current_time: float, max_items: int = 20
) -> list[PracticeItem]:
    """Select the most overdue items for today's practice session."""
    overdue = [(i, current_time - i.due) for i in items if i.due <= current_time]
    overdue.sort(key=lambda x: -x[1])
    return [i for i, _ in overdue[:max_items]]


# ── Performance comparison ───────────────────────────────────────────

@dataclass
class PerformanceSnapshot:
    """A timestamped record of performance metrics."""
    timestamp: float = 0.0
    intonation_score: float = 0.0
    rhythm_score: float = 0.0
    note_accuracy: float = 0.0
    tempo_stability: float = 0.0
    dynamics_range: float = 0.0


def performance_trend(snapshots: list[PerformanceSnapshot], metric: str = "intonation_score") -> np.ndarray:
    """Extract a time series of a given metric from snapshots."""
    return np.array([getattr(s, metric, 0.0) for s in snapshots])


def improvement_rate(scores: np.ndarray) -> float:
    """Linear regression slope of scores over indices — positive = improving."""
    n = len(scores)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = scores.mean()
    num = np.sum((x - x_mean) * (scores - y_mean))
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


def weakness_detection(
    snapshots: list[PerformanceSnapshot],
    threshold: float = 0.6,
) -> list[str]:
    """Identify metrics consistently below threshold."""
    if not snapshots:
        return []
    metrics = ["intonation_score", "rhythm_score", "note_accuracy",
               "tempo_stability", "dynamics_range"]
    weak = []
    for m in metrics:
        vals = performance_trend(snapshots, m)
        if len(vals) > 0 and np.mean(vals) < threshold:
            weak.append(m)
    return weak
