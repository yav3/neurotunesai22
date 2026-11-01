"""Cognitive load estimation for music listening and performance.

Models working memory demands, information density, and processing
complexity based on Sweller's cognitive load theory adapted to music.
Covers intrinsic load (harmonic/rhythmic complexity), extraneous load
(timbral density, textural clutter), and germane load (schema formation,
pattern recognition). Uses Miller's 7+-2 chunk capacity and Cowan's
embedded-processes model for short-term musical memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


MILLERS_LIMIT = 7
COWAN_LIMIT = 4


@dataclass
class CognitiveLoadEstimate:
    intrinsic: float
    extraneous: float
    germane: float

    @property
    def total(self) -> float:
        return self.intrinsic + self.extraneous + self.germane

    @property
    def overload_risk(self) -> float:
        return float(np.clip(self.total / 3.0, 0.0, 1.0))


@dataclass
class InformationDensity:
    events_per_second: float
    unique_pitch_classes: int
    harmonic_rhythm: float
    avg_ioi: float

    @property
    def density_score(self) -> float:
        pc_factor = self.unique_pitch_classes / 12.0
        rate_factor = min(self.events_per_second / 10.0, 1.0)
        harm_factor = min(self.harmonic_rhythm / 2.0, 1.0)
        return float(np.clip(0.4 * rate_factor + 0.35 * pc_factor + 0.25 * harm_factor, 0, 1))


@dataclass
class WorkingMemoryModel:
    capacity: int = MILLERS_LIMIT
    decay_rate: float = 0.15
    rehearsal_boost: float = 0.3

    def chunk_count(self, items: Sequence[float], similarity_threshold: float = 1.5) -> int:
        if len(items) == 0:
            return 0
        chunks = 1
        current_chunk_start = items[0]
        for i in range(1, len(items)):
            if abs(items[i] - items[i - 1]) > similarity_threshold:
                chunks += 1
                current_chunk_start = items[i]
        return chunks

    def memory_load(self, n_chunks: int) -> float:
        return float(np.clip(n_chunks / self.capacity, 0.0, 1.5))

    def retention_probability(self, n_items: int, elapsed_seconds: float,
                               rehearsals: int = 0) -> float:
        base = np.exp(-self.decay_rate * elapsed_seconds)
        boost = 1.0 + self.rehearsal_boost * rehearsals
        capacity_penalty = max(0.0, 1.0 - max(0, n_items - self.capacity) * 0.15)
        return float(np.clip(base * boost * capacity_penalty, 0.0, 1.0))


def harmonic_information_rate(chord_sequence: Sequence[int],
                               durations: Sequence[float]) -> float:
    if not chord_sequence or not durations:
        return 0.0
    n = len(chord_sequence)
    total_dur = sum(durations[:n])
    if total_dur < 1e-10:
        return 0.0
    changes = sum(1 for i in range(1, n) if chord_sequence[i] != chord_sequence[i - 1])
    return changes / total_dur


def pitch_entropy(pitch_classes: Sequence[int]) -> float:
    if not pitch_classes:
        return 0.0
    counts = np.zeros(12, dtype=np.float64)
    for pc in pitch_classes:
        counts[pc % 12] += 1.0
    total = counts.sum()
    if total < 1:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def rhythmic_entropy(ioi_sequence: Sequence[float], n_bins: int = 16) -> float:
    if len(ioi_sequence) < 2:
        return 0.0
    arr = np.array(ioi_sequence, dtype=np.float64)
    arr = arr[arr > 0]
    if len(arr) < 2:
        return 0.0
    hist, _ = np.histogram(arr, bins=n_bins, density=True)
    bin_width = (arr.max() - arr.min()) / n_bins
    if bin_width < 1e-10:
        return 0.0
    probs = hist * bin_width
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-15)))


def textural_density(n_voices: int, frequency_spread_hz: float,
                     dynamic_range_db: float) -> float:
    voice_factor = min(n_voices / 8.0, 1.0)
    spread_factor = float(np.clip(frequency_spread_hz / 4000.0, 0.0, 1.0))
    dynamic_factor = float(np.clip(dynamic_range_db / 60.0, 0.0, 1.0))
    return float(np.clip(0.5 * voice_factor + 0.3 * (1.0 - spread_factor) + 0.2 * (1.0 - dynamic_factor), 0, 1))


def estimate_cognitive_load(
    pitch_classes: Sequence[int],
    ioi_sequence: Sequence[float],
    n_voices: int = 1,
    chord_changes_per_second: float = 0.5,
    familiarity: float = 0.5,
) -> CognitiveLoadEstimate:
    p_ent = pitch_entropy(pitch_classes)
    r_ent = rhythmic_entropy(ioi_sequence)
    max_pitch_ent = np.log2(12)
    intrinsic = float(np.clip(
        0.5 * (p_ent / max_pitch_ent) + 0.3 * min(r_ent / 3.0, 1.0) +
        0.2 * min(chord_changes_per_second / 2.0, 1.0),
        0, 1
    ))
    extraneous = textural_density(n_voices, 2000.0, 30.0)
    germane = float(np.clip(1.0 - familiarity, 0, 1))
    return CognitiveLoadEstimate(
        intrinsic=intrinsic,
        extraneous=extraneous,
        germane=germane,
    )


def attention_resource_allocation(
    salience_values: np.ndarray,
    total_capacity: float = 1.0,
) -> np.ndarray:
    if len(salience_values) == 0:
        return np.array([], dtype=np.float64)
    s = np.asarray(salience_values, dtype=np.float64)
    s = np.maximum(s, 0.0)
    total = s.sum()
    if total < 1e-10:
        return np.full_like(s, total_capacity / max(len(s), 1))
    return s / total * total_capacity


def processing_bottleneck(event_rate: float, complexity: float,
                           skill_level: float = 0.5) -> float:
    raw_demand = event_rate * complexity
    effective_capacity = 5.0 + 10.0 * skill_level
    return float(np.clip(raw_demand / effective_capacity, 0.0, 1.0))


def musical_syntax_complexity(intervals: Sequence[int]) -> float:
    if len(intervals) < 2:
        return 0.0
    arr = np.array(intervals, dtype=np.float64)
    abs_intervals = np.abs(arr)
    large_leaps = np.sum(abs_intervals > 7)
    direction_changes = sum(
        1 for i in range(1, len(arr)) if arr[i] * arr[i - 1] < 0
    )
    n = len(arr)
    leap_ratio = large_leaps / n
    change_ratio = direction_changes / max(n - 1, 1)
    interval_variety = len(set(int(x) for x in arr)) / max(n, 1)
    return float(np.clip(
        0.35 * leap_ratio + 0.35 * change_ratio + 0.3 * interval_variety,
        0, 1
    ))


@dataclass
class PerformanceCognitiveLoad:
    reading_load: float
    motor_load: float
    memory_load: float
    coordination_load: float

    @property
    def total(self) -> float:
        return (0.3 * self.reading_load + 0.3 * self.motor_load +
                0.2 * self.memory_load + 0.2 * self.coordination_load)


def estimate_performance_load(
    notes_per_second: float,
    key_signature_accidentals: int,
    simultaneous_voices: int,
    sight_reading: bool = False,
) -> PerformanceCognitiveLoad:
    reading = float(np.clip(
        0.4 * min(notes_per_second / 8.0, 1.0) +
        0.3 * min(key_signature_accidentals / 7.0, 1.0) +
        0.3 * (1.0 if sight_reading else 0.3),
        0, 1
    ))
    motor = float(np.clip(min(notes_per_second / 12.0, 1.0), 0, 1))
    memory = float(np.clip(
        0.6 * (0.2 if not sight_reading else 0.8) +
        0.4 * min(simultaneous_voices / 4.0, 1.0),
        0, 1
    ))
    coordination = float(np.clip(min(simultaneous_voices / 4.0, 1.0), 0, 1))
    return PerformanceCognitiveLoad(
        reading_load=reading,
        motor_load=motor,
        memory_load=memory,
        coordination_load=coordination,
    )
