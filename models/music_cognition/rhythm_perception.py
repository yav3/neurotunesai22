"""Rhythm perception, entrainment, and beat induction."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class AccentType(Enum):
    METRIC = "metric"
    PHENOMENAL = "phenomenal"
    GROUPING = "grouping"


@dataclass(frozen=True)
class PerceptualAccent:
    time: float
    accent_type: AccentType
    strength: float


class MetricalGrid:
    def __init__(self, accents: list[PerceptualAccent]):
        self.accents = sorted(accents, key=lambda a: a.time)

    @classmethod
    def from_tempo(cls, bpm: float, meter: tuple[int, int] = (4, 4),
                   duration_s: float = 10.0) -> MetricalGrid:
        beat_dur = 60.0 / bpm
        beats_per_bar = meter[0]
        accents: list[PerceptualAccent] = []
        t = 0.0
        beat_idx = 0
        while t < duration_s:
            pos_in_bar = beat_idx % beats_per_bar
            if pos_in_bar == 0:
                strength = 1.0
            elif beats_per_bar == 4 and pos_in_bar == 2:
                strength = 0.6
            else:
                strength = 0.3
            accents.append(PerceptualAccent(t, AccentType.METRIC, strength))
            for sub in range(1, meter[1] // meter[1]):
                st = t + beat_dur * sub / (meter[1] // meter[1])
                if st < duration_s:
                    accents.append(PerceptualAccent(st, AccentType.METRIC, 0.1))
            t += beat_dur
            beat_idx += 1
        return cls(accents)

    def strength_at(self, time: float, tolerance: float = 0.05) -> float:
        for a in self.accents:
            if abs(a.time - time) <= tolerance:
                return a.strength
        return 0.0


def compute_syncopation(onsets: Sequence[float], grid: MetricalGrid,
                        tolerance: float = 0.05) -> float:
    if not onsets or not grid.accents:
        return 0.0
    score = 0.0
    sorted_onsets = sorted(onsets)
    grid_times = [a.time for a in grid.accents]
    for onset in sorted_onsets:
        onset_strength = grid.strength_at(onset, tolerance)
        next_strengths = [a.strength for a in grid.accents if a.time > onset + tolerance]
        if next_strengths:
            next_strength = next_strengths[0]
            if onset_strength < next_strength:
                score += next_strength - onset_strength
    return score


def compute_groove_complexity(onsets: Sequence[float], grid: MetricalGrid,
                              tolerance: float = 0.05) -> float:
    if not onsets:
        return 0.0
    syncopated = 0.0
    total_weight = 0.0
    for onset in onsets:
        s = grid.strength_at(onset, tolerance)
        total_weight += 1.0
        if s < 0.3:
            syncopated += 1.0 - s
    return syncopated / total_weight if total_weight > 0 else 0.0


def beat_induction(onsets: Sequence[float], min_bpm: float = 40.0,
                   max_bpm: float = 240.0) -> float:
    if len(onsets) < 3:
        return 120.0
    onsets_arr = np.array(sorted(onsets))
    iois = np.diff(onsets_arr)
    iois = iois[iois > 0]
    if len(iois) == 0:
        return 120.0
    min_ioi = 60.0 / max_bpm
    max_ioi = 60.0 / min_bpm
    candidates = np.linspace(min_ioi, max_ioi, 500)
    scores = np.zeros(len(candidates))
    for i, c in enumerate(candidates):
        residuals = np.mod(iois, c)
        residuals = np.minimum(residuals, c - residuals)
        scores[i] = np.sum(np.exp(-0.5 * (residuals / (c * 0.1)) ** 2))
    best = candidates[np.argmax(scores)]
    return 60.0 / best


def phase_coherence(onsets: Sequence[float], tempo_bpm: float) -> float:
    if len(onsets) < 2 or tempo_bpm <= 0:
        return 0.0
    period = 60.0 / tempo_bpm
    phases = np.array(onsets) % period
    angles = 2.0 * np.pi * phases / period
    r = np.abs(np.mean(np.exp(1j * angles)))
    return float(r)


@dataclass
class PerceptualGrouping:
    @staticmethod
    def group_by_proximity(onsets: Sequence[float],
                           threshold_ms: float = 300.0) -> list[list[float]]:
        if not onsets:
            return []
        sorted_o = sorted(onsets)
        groups: list[list[float]] = [[sorted_o[0]]]
        for i in range(1, len(sorted_o)):
            if (sorted_o[i] - sorted_o[i - 1]) * 1000.0 > threshold_ms:
                groups.append([])
            groups[-1].append(sorted_o[i])
        return groups

    @staticmethod
    def group_by_similarity(onsets: Sequence[float],
                            features: np.ndarray,
                            threshold: float = 1.0) -> list[list[int]]:
        if len(onsets) == 0 or features.size == 0:
            return []
        n = len(onsets)
        labels = np.full(n, -1, dtype=int)
        current = 0
        labels[0] = current
        for i in range(1, n):
            if np.linalg.norm(features[i] - features[labels == labels[i - 1]].mean(axis=0)) < threshold:
                labels[i] = labels[i - 1]
            else:
                current += 1
                labels[i] = current
        groups: list[list[int]] = []
        for g in range(current + 1):
            groups.append([int(x) for x in np.where(labels == g)[0]])
        return groups
