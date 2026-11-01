"""Practice analytics: session tracking, learning curves, spaced repetition."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class PracticeSession:
    duration_sec: float
    tempo_sequence: np.ndarray
    error_positions: List[int]
    repetition_count: int = 1
    piece_id: str = ""


@dataclass
class PracticeReport:
    accuracy: float
    tempo_stability: float
    improvement_rate: float
    problem_spots: List[Tuple[int, int]]
    estimated_mastery: float


def tempo_stability(tempo_seq: np.ndarray) -> float:
    if len(tempo_seq) < 2:
        return 1.0
    mean_t = float(np.mean(tempo_seq))
    if mean_t < 1e-12:
        return 0.0
    cv = float(np.std(tempo_seq) / mean_t)
    return max(1.0 - cv, 0.0)


def error_density(
    error_positions: List[int],
    total_notes: int,
    window: int = 10,
) -> np.ndarray:
    if total_notes <= 0:
        return np.array([])
    indicator = np.zeros(total_notes)
    for p in error_positions:
        if 0 <= p < total_notes:
            indicator[p] = 1.0
    if total_notes < window:
        return np.array([float(np.mean(indicator))])
    kernel = np.ones(window) / window
    return np.convolve(indicator, kernel, mode="valid")


def identify_problem_spots(
    error_positions: List[int],
    total_notes: int,
    threshold: float = 0.3,
    window: int = 10,
) -> List[Tuple[int, int]]:
    density = error_density(error_positions, total_notes, window)
    if len(density) == 0:
        return []
    above = density >= threshold
    spots = []
    in_spot = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_spot:
            start = i
            in_spot = True
        elif not above[i] and in_spot:
            spots.append((start, i + window - 1))
            in_spot = False
    if in_spot:
        spots.append((start, len(above) - 1 + window - 1))
    return spots


def learning_curve(session_accuracies: List[float]) -> Tuple[float, float]:
    if len(session_accuracies) < 2:
        return (0.0, session_accuracies[0] if session_accuracies else 0.0)
    y = np.array(session_accuracies)
    n = len(y)
    x = np.arange(n, dtype=float)
    asymptote = float(np.max(y))
    gap = asymptote - y[0]
    if gap < 0.01:
        return (0.0, asymptote)
    progress = (y - y[0]) / max(gap, 1e-12)
    progress = np.clip(progress, 1e-6, 1.0 - 1e-6)
    log_compl = np.log(1.0 - progress + 1e-12)
    valid = np.isfinite(log_compl) & (log_compl < 0)
    if np.sum(valid) < 2:
        return (0.1, asymptote)
    slope = float(-np.mean(log_compl[valid]) / max(np.mean(x[valid]), 1.0))
    return (max(slope, 0.0), asymptote)


def spaced_repetition_schedule(
    mastery_level: float,
    last_practice_days_ago: int = 0,
) -> int:
    base_interval = 1.0
    if mastery_level >= 0.95:
        base_interval = 14.0
    elif mastery_level >= 0.85:
        base_interval = 7.0
    elif mastery_level >= 0.7:
        base_interval = 3.0
    elif mastery_level >= 0.5:
        base_interval = 2.0
    else:
        base_interval = 1.0
    decay = np.exp(-0.1 * last_practice_days_ago)
    effective_mastery = mastery_level * decay
    if effective_mastery < 0.5:
        return 1
    return max(int(base_interval), 1)


def analyze_practice(
    sessions: List[PracticeSession],
    total_notes: int = 100,
) -> PracticeReport:
    if not sessions:
        return PracticeReport(0.0, 0.0, 0.0, [], 0.0)
    accuracies = []
    all_errors: List[int] = []
    stabilities = []
    for s in sessions:
        acc = 1.0 - len(s.error_positions) / max(total_notes, 1)
        accuracies.append(max(acc, 0.0))
        all_errors.extend(s.error_positions)
        stabilities.append(tempo_stability(s.tempo_sequence))

    rate, asymptote = learning_curve(accuracies)
    spots = identify_problem_spots(all_errors, total_notes)
    latest_acc = accuracies[-1]
    avg_stability = float(np.mean(stabilities))
    mastery = 0.6 * latest_acc + 0.4 * avg_stability

    return PracticeReport(
        accuracy=latest_acc,
        tempo_stability=avg_stability,
        improvement_rate=rate,
        problem_spots=spots,
        estimated_mastery=mastery,
    )
