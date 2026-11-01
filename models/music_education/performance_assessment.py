"""Performance assessment: rubric scoring, error classification, DTW alignment, improvement tracking."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ErrorType(Enum):
    WRONG_NOTE = "wrong_note"
    WRONG_RHYTHM = "wrong_rhythm"
    MISSED_NOTE = "missed_note"
    ADDED_NOTE = "added_note"
    TIMING_DRIFT = "timing_drift"
    DYNAMICS_ERROR = "dynamics_error"
    ARTICULATION_ERROR = "articulation_error"


@dataclass
class PerformanceError:
    error_type: ErrorType
    position: float
    severity: float = 1.0
    expected: float = 0.0
    actual: float = 0.0


@dataclass
class RubricScore:
    pitch_accuracy: float = 0.0
    rhythm_accuracy: float = 0.0
    dynamics: float = 0.0
    articulation: float = 0.0
    expression: float = 0.0

    @property
    def overall(self) -> float:
        weights = [0.3, 0.3, 0.15, 0.1, 0.15]
        scores = [self.pitch_accuracy, self.rhythm_accuracy,
                  self.dynamics, self.articulation, self.expression]
        return float(np.dot(weights, scores))


def dtw_alignment(reference: np.ndarray, performance: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    n, m = len(reference), len(performance)
    if n == 0 or m == 0:
        return np.array([]), []
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(float(reference[i - 1]) - float(performance[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [(cost[i - 1, j - 1], i - 1, j - 1),
                      (cost[i - 1, j], i - 1, j),
                      (cost[i, j - 1], i, j - 1)]
        _, i, j = min(candidates)
    path.reverse()
    return cost[:n, :m], path


def classify_errors(reference_pitches: np.ndarray,
                    performance_pitches: np.ndarray,
                    reference_onsets: np.ndarray,
                    performance_onsets: np.ndarray,
                    pitch_tolerance: float = 0.5,
                    timing_tolerance: float = 0.1) -> list[PerformanceError]:
    errors = []
    _, path = dtw_alignment(reference_pitches, performance_pitches)
    ref_matched = set()
    perf_matched = set()
    for ri, pi in path:
        ref_matched.add(ri)
        perf_matched.add(pi)
        pitch_diff = abs(float(reference_pitches[ri]) - float(performance_pitches[pi]))
        if pitch_diff > pitch_tolerance:
            errors.append(PerformanceError(
                ErrorType.WRONG_NOTE, float(reference_onsets[ri]),
                severity=min(pitch_diff / 12.0, 1.0),
                expected=float(reference_pitches[ri]),
                actual=float(performance_pitches[pi]),
            ))
        if ri < len(reference_onsets) and pi < len(performance_onsets):
            time_diff = abs(float(reference_onsets[ri]) - float(performance_onsets[pi]))
            if time_diff > timing_tolerance:
                errors.append(PerformanceError(
                    ErrorType.TIMING_DRIFT, float(reference_onsets[ri]),
                    severity=min(time_diff / 0.5, 1.0),
                ))
    for i in range(len(reference_pitches)):
        if i not in ref_matched:
            pos = float(reference_onsets[i]) if i < len(reference_onsets) else 0.0
            errors.append(PerformanceError(ErrorType.MISSED_NOTE, pos))
    for i in range(len(performance_pitches)):
        if i not in perf_matched:
            pos = float(performance_onsets[i]) if i < len(performance_onsets) else 0.0
            errors.append(PerformanceError(ErrorType.ADDED_NOTE, pos))
    return errors


def score_performance(reference_pitches: np.ndarray,
                      performance_pitches: np.ndarray,
                      reference_onsets: np.ndarray,
                      performance_onsets: np.ndarray) -> RubricScore:
    errors = classify_errors(reference_pitches, performance_pitches,
                             reference_onsets, performance_onsets)
    n_ref = max(len(reference_pitches), 1)
    pitch_errors = sum(1 for e in errors if e.error_type == ErrorType.WRONG_NOTE)
    rhythm_errors = sum(1 for e in errors if e.error_type in (ErrorType.WRONG_RHYTHM, ErrorType.TIMING_DRIFT))
    missed = sum(1 for e in errors if e.error_type == ErrorType.MISSED_NOTE)
    pitch_acc = max(0.0, 1.0 - (pitch_errors + missed) / n_ref)
    rhythm_acc = max(0.0, 1.0 - rhythm_errors / n_ref)
    return RubricScore(
        pitch_accuracy=pitch_acc * 100,
        rhythm_accuracy=rhythm_acc * 100,
        dynamics=75.0,
        articulation=75.0,
        expression=70.0,
    )


def difficulty_adjustment(score: RubricScore, base_difficulty: float) -> float:
    if score.overall > 90:
        return min(base_difficulty + 0.1, 1.0)
    if score.overall < 60:
        return max(base_difficulty - 0.1, 0.0)
    return base_difficulty


@dataclass
class ProgressRecord:
    scores: list[float] = field(default_factory=list)

    def add(self, score: float) -> None:
        self.scores.append(score)

    @property
    def trend(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        x = np.arange(len(self.scores), dtype=float)
        y = np.array(self.scores)
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    @property
    def best(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def recent_average(self) -> float:
        if not self.scores:
            return 0.0
        window = self.scores[-5:]
        return float(np.mean(window))
