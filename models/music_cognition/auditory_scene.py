"""Auditory scene analysis: stream segregation, grouping, attention modeling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AuditoryStream:
    stream_id: int
    pitch_center: float
    timbre_descriptor: np.ndarray
    onset_times: tuple[float, ...]
    loudness: float = 0.5

    def duration(self) -> float:
        if len(self.onset_times) < 2:
            return 0.0
        return self.onset_times[-1] - self.onset_times[0]


@dataclass
class SceneAnalyzer:
    proximity_threshold: float = 6.0
    continuity_weight: float = 0.7
    similarity_weight: float = 0.3
    max_streams: int = 8

    def segregate_by_pitch(self, pitches: Sequence[float],
                           onsets: Sequence[float]) -> list[list[int]]:
        if not pitches:
            return []
        streams: list[list[int]] = [[0]]
        stream_last_pitch: list[float] = [pitches[0]]
        for i in range(1, len(pitches)):
            best_stream = -1
            best_dist = float('inf')
            for s_idx, last_p in enumerate(stream_last_pitch):
                dist = abs(pitches[i] - last_p)
                if dist < best_dist and dist < self.proximity_threshold:
                    best_dist = dist
                    best_stream = s_idx
            if best_stream >= 0:
                streams[best_stream].append(i)
                stream_last_pitch[best_stream] = pitches[i]
            elif len(streams) < self.max_streams:
                streams.append([i])
                stream_last_pitch.append(pitches[i])
            else:
                closest = min(range(len(stream_last_pitch)),
                              key=lambda s: abs(pitches[i] - stream_last_pitch[s]))
                streams[closest].append(i)
                stream_last_pitch[closest] = pitches[i]
        return streams

    def stream_coherence(self, stream_pitches: Sequence[float]) -> float:
        if len(stream_pitches) < 2:
            return 1.0
        intervals = [abs(stream_pitches[i + 1] - stream_pitches[i])
                     for i in range(len(stream_pitches) - 1)]
        mean_interval = np.mean(intervals)
        if mean_interval < 1e-10:
            return 1.0
        var_interval = np.var(intervals)
        return 1.0 / (1.0 + var_interval / (mean_interval + 1e-10))


def gestalt_proximity(onsets: Sequence[float], threshold: float = 0.3) -> list[list[int]]:
    if not onsets:
        return []
    groups: list[list[int]] = [[0]]
    for i in range(1, len(onsets)):
        if onsets[i] - onsets[i - 1] <= threshold:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def gestalt_similarity(features: np.ndarray, threshold: float = 0.5) -> list[list[int]]:
    n = len(features)
    if n == 0:
        return []
    labels = [-1] * n
    current_label = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = current_label
        for j in range(i + 1, n):
            if labels[j] >= 0:
                continue
            dist = float(np.linalg.norm(features[i] - features[j]))
            if dist <= threshold:
                labels[j] = current_label
        current_label += 1
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab not in groups:
            groups[lab] = []
        groups[lab].append(i)
    return list(groups.values())


def attention_salience(loudness: Sequence[float], novelty: Sequence[float],
                       loudness_weight: float = 0.6) -> np.ndarray:
    l = np.array(loudness, dtype=np.float64)
    n = np.array(novelty, dtype=np.float64)
    l_max = np.max(np.abs(l)) + 1e-10
    n_max = np.max(np.abs(n)) + 1e-10
    return loudness_weight * (l / l_max) + (1 - loudness_weight) * (n / n_max)


@dataclass
class RhythmicExpectation:
    beat_period: float = 0.5
    phase: float = 0.0
    tolerance: float = 0.1
    coupling_strength: float = 0.5

    def predict_next_beat(self, current_time: float) -> float:
        cycles_elapsed = (current_time - self.phase) / self.beat_period
        next_cycle = np.ceil(cycles_elapsed)
        return self.phase + next_cycle * self.beat_period

    def surprise(self, event_time: float) -> float:
        predicted = self.predict_next_beat(event_time - self.beat_period)
        deviation = abs(event_time - predicted)
        return deviation / (self.beat_period + 1e-10)

    def update(self, observed_beat: float) -> None:
        predicted = self.predict_next_beat(observed_beat - self.beat_period)
        error = observed_beat - predicted
        self.phase += self.coupling_strength * error
        period_error = abs(observed_beat - self.phase) - self.beat_period
        self.beat_period += 0.1 * self.coupling_strength * period_error


def rhythmic_complexity(onsets: Sequence[float], beat_period: float = 0.5) -> float:
    if len(onsets) < 2:
        return 0.0
    quantized = [round(t / beat_period) * beat_period for t in onsets]
    deviations = [abs(o - q) / beat_period for o, q in zip(onsets, quantized)]
    ioi = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    if not ioi:
        return 0.0
    ioi_ratios = [ioi[i + 1] / (ioi[i] + 1e-10) for i in range(len(ioi) - 1)]
    non_simple = sum(1 for r in ioi_ratios if abs(r - round(r)) > 0.15)
    ratio_complexity = non_simple / max(len(ioi_ratios), 1)
    mean_deviation = np.mean(deviations)
    return float(np.clip(0.5 * ratio_complexity + 0.5 * mean_deviation, 0, 1))
