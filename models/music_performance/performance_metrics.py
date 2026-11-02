"""Performance metrics: timing accuracy, pitch accuracy, dynamic fidelity, overall scoring.

Quantitative evaluation of musical performance against a reference score/MIDI.
All metrics return values in [0, 1] where 1 is perfect adherence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PerformanceScore:
    timing_accuracy: float
    pitch_accuracy: float
    dynamic_fidelity: float
    articulation_score: float
    overall: float

    @staticmethod
    def compute(timing: float, pitch: float, dynamics: float,
                articulation: float,
                weights: tuple[float, ...] = (0.3, 0.3, 0.2, 0.2)) -> "PerformanceScore":
        w = np.array(weights)
        w = w / w.sum()
        vals = np.array([timing, pitch, dynamics, articulation])
        overall = float(np.dot(w, vals))
        return PerformanceScore(timing, pitch, dynamics, articulation, overall)


def timing_accuracy(performed_onsets: np.ndarray,
                    reference_onsets: np.ndarray,
                    tolerance_ms: float = 50.0) -> float:
    if len(performed_onsets) == 0 or len(reference_onsets) == 0:
        return 0.0
    tol = tolerance_ms / 1000.0
    matched = 0
    used = set()
    for p in performed_onsets:
        diffs = np.abs(reference_onsets - p)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= tol and best_idx not in used:
            matched += 1
            used.add(best_idx)
    precision = matched / len(performed_onsets)
    recall = matched / len(reference_onsets)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def pitch_accuracy(performed_pitches: np.ndarray,
                   reference_pitches: np.ndarray,
                   tolerance_cents: float = 50.0) -> float:
    n = min(len(performed_pitches), len(reference_pitches))
    if n == 0:
        return 0.0
    perf = performed_pitches[:n]
    ref = reference_pitches[:n]
    cents_diff = np.abs(perf - ref) * 100
    within = np.sum(cents_diff <= tolerance_cents)
    return float(within / n)


def dynamic_fidelity(performed_velocities: np.ndarray,
                     reference_velocities: np.ndarray) -> float:
    n = min(len(performed_velocities), len(reference_velocities))
    if n == 0:
        return 0.0
    perf = performed_velocities[:n].astype(float)
    ref = reference_velocities[:n].astype(float)
    if np.std(ref) < 1e-6:
        return 1.0 if np.std(perf) < 1e-6 else 0.5
    corr = np.corrcoef(perf, ref)[0, 1]
    if np.isnan(corr):
        return 0.5
    return float(np.clip((corr + 1) / 2, 0, 1))


def articulation_score(performed_durations: np.ndarray,
                       reference_durations: np.ndarray,
                       tolerance_ratio: float = 0.2) -> float:
    n = min(len(performed_durations), len(reference_durations))
    if n == 0:
        return 0.0
    perf = performed_durations[:n]
    ref = reference_durations[:n]
    ratios = perf / np.maximum(ref, 1e-6)
    within = np.sum(np.abs(ratios - 1.0) <= tolerance_ratio)
    return float(within / n)


def tempo_stability(beat_intervals: np.ndarray) -> float:
    if len(beat_intervals) < 2:
        return 1.0
    cv = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)
    return float(np.clip(1.0 - cv, 0, 1))


def rhythmic_evenness(note_onsets: np.ndarray, target_ioi: float) -> float:
    if len(note_onsets) < 2:
        return 1.0
    ioi = np.diff(note_onsets)
    deviations = np.abs(ioi - target_ioi) / target_ioi
    return float(np.clip(1.0 - np.mean(deviations), 0, 1))


@dataclass
class ProgressTracker:
    piece_id: str
    scores: list[PerformanceScore]

    def trend(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        vals = np.array([s.overall for s in self.scores])
        x = np.arange(len(vals))
        coeffs = np.polyfit(x, vals, 1)
        return float(coeffs[0])

    def best(self) -> float:
        if not self.scores:
            return 0.0
        return max(s.overall for s in self.scores)

    def latest(self) -> float:
        if not self.scores:
            return 0.0
        return self.scores[-1].overall
