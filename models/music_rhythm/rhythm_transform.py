"""Rhythm transforms: augmentation, diminution, displacement, rotation, density operations."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RhythmEvent:
    onset: float
    duration: float
    velocity: float = 0.8

    @property
    def offset(self) -> float:
        return self.onset + self.duration


@dataclass
class RhythmSequence:
    events: List[RhythmEvent] = field(default_factory=list)
    time_signature: Tuple[int, int] = (4, 4)
    tempo_bpm: float = 120.0

    @property
    def total_duration(self) -> float:
        if not self.events:
            return 0.0
        return max(e.offset for e in self.events)

    @property
    def density(self) -> float:
        if self.total_duration <= 0:
            return 0.0
        return len(self.events) / self.total_duration

    def onsets(self) -> np.ndarray:
        return np.array([e.onset for e in self.events])

    def ioi(self) -> np.ndarray:
        ons = self.onsets()
        if len(ons) < 2:
            return np.array([])
        return np.diff(ons)


def augment(seq: RhythmSequence, factor: float = 2.0) -> RhythmSequence:
    events = [RhythmEvent(e.onset * factor, e.duration * factor, e.velocity)
              for e in seq.events]
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def diminish(seq: RhythmSequence, factor: float = 2.0) -> RhythmSequence:
    return augment(seq, 1.0 / factor)


def displace(seq: RhythmSequence, offset: float) -> RhythmSequence:
    events = [RhythmEvent(e.onset + offset, e.duration, e.velocity)
              for e in seq.events]
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def rotate(seq: RhythmSequence, steps: int) -> RhythmSequence:
    if not seq.events or steps == 0:
        return RhythmSequence(list(seq.events), seq.time_signature, seq.tempo_bpm)
    n = len(seq.events)
    steps = steps % n
    ioi_vals = seq.ioi()
    if len(ioi_vals) == 0:
        return RhythmSequence(list(seq.events), seq.time_signature, seq.tempo_bpm)
    ioi_list = list(ioi_vals)
    ioi_list.append(seq.events[0].duration)
    rotated_ioi = ioi_list[steps:] + ioi_list[:steps]
    events = []
    t = 0.0
    durations = [seq.events[(i + steps) % n].duration for i in range(n)]
    velocities = [seq.events[(i + steps) % n].velocity for i in range(n)]
    for i in range(n):
        events.append(RhythmEvent(t, durations[i], velocities[i]))
        if i < len(rotated_ioi):
            t += rotated_ioi[i]
        else:
            t += durations[i]
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def retrograde(seq: RhythmSequence) -> RhythmSequence:
    if not seq.events:
        return RhythmSequence([], seq.time_signature, seq.tempo_bpm)
    total = seq.total_duration
    events = []
    for e in reversed(seq.events):
        new_onset = total - e.offset
        events.append(RhythmEvent(max(0.0, new_onset), e.duration, e.velocity))
    events.sort(key=lambda ev: ev.onset)
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def interleave(a: RhythmSequence, b: RhythmSequence) -> RhythmSequence:
    combined = list(a.events) + list(b.events)
    combined.sort(key=lambda e: e.onset)
    return RhythmSequence(combined, a.time_signature, a.tempo_bpm)


def thin(seq: RhythmSequence, keep_ratio: float = 0.5,
         seed: Optional[int] = None) -> RhythmSequence:
    if not seq.events or keep_ratio >= 1.0:
        return RhythmSequence(list(seq.events), seq.time_signature, seq.tempo_bpm)
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(len(seq.events) * keep_ratio))
    indices = sorted(rng.choice(len(seq.events), size=n_keep, replace=False))
    events = [seq.events[i] for i in indices]
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def densify(seq: RhythmSequence, subdivisions: int = 2) -> RhythmSequence:
    if not seq.events or subdivisions <= 1:
        return RhythmSequence(list(seq.events), seq.time_signature, seq.tempo_bpm)
    events = []
    for e in seq.events:
        sub_dur = e.duration / subdivisions
        for k in range(subdivisions):
            events.append(RhythmEvent(
                e.onset + k * sub_dur,
                sub_dur * 0.9,
                e.velocity * (1.0 if k == 0 else 0.6),
            ))
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def accent_pattern(seq: RhythmSequence, pattern: List[float]) -> RhythmSequence:
    if not pattern:
        return RhythmSequence(list(seq.events), seq.time_signature, seq.tempo_bpm)
    events = []
    for i, e in enumerate(seq.events):
        accent = pattern[i % len(pattern)]
        events.append(RhythmEvent(e.onset, e.duration, min(1.0, e.velocity * accent)))
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def quantize(seq: RhythmSequence, grid: float = 0.25) -> RhythmSequence:
    if grid <= 0:
        return RhythmSequence(list(seq.events), seq.time_signature, seq.tempo_bpm)
    events = []
    for e in seq.events:
        q_onset = round(e.onset / grid) * grid
        q_dur = max(grid, round(e.duration / grid) * grid)
        events.append(RhythmEvent(q_onset, q_dur, e.velocity))
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def swing_transform(seq: RhythmSequence, ratio: float = 0.67,
                     grid: float = 0.5) -> RhythmSequence:
    if grid <= 0:
        return RhythmSequence(list(seq.events), seq.time_signature, seq.tempo_bpm)
    events = []
    for e in seq.events:
        beat_pos = e.onset / grid
        beat_idx = int(beat_pos)
        frac = beat_pos - beat_idx
        if frac > 0.4 and frac < 0.6:
            new_frac = ratio
            new_onset = (beat_idx + new_frac) * grid
            events.append(RhythmEvent(new_onset, e.duration, e.velocity))
        else:
            events.append(RhythmEvent(e.onset, e.duration, e.velocity))
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def humanize(seq: RhythmSequence, timing_std: float = 0.01,
             velocity_std: float = 0.05,
             seed: Optional[int] = None) -> RhythmSequence:
    rng = np.random.default_rng(seed)
    events = []
    for e in seq.events:
        t_jitter = rng.normal(0, timing_std)
        v_jitter = rng.normal(0, velocity_std)
        events.append(RhythmEvent(
            max(0.0, e.onset + t_jitter),
            e.duration,
            max(0.01, min(1.0, e.velocity + v_jitter)),
        ))
    events.sort(key=lambda ev: ev.onset)
    return RhythmSequence(events, seq.time_signature, seq.tempo_bpm)


def necklace_class(binary: List[int]) -> Tuple[int, ...]:
    n = len(binary)
    if n == 0:
        return ()
    best = tuple(binary)
    for r in range(1, n):
        rotated = tuple(binary[r:] + binary[:r])
        if rotated < best:
            best = rotated
    return best


def rhythm_similarity(a: RhythmSequence, b: RhythmSequence,
                      grid: float = 0.125) -> float:
    if not a.events and not b.events:
        return 1.0
    max_t = max(a.total_duration, b.total_duration, grid)
    n_bins = int(math.ceil(max_t / grid))
    vec_a = np.zeros(n_bins)
    vec_b = np.zeros(n_bins)
    for e in a.events:
        idx = min(int(e.onset / grid), n_bins - 1)
        vec_a[idx] = 1.0
    for e in b.events:
        idx = min(int(e.onset / grid), n_bins - 1)
        vec_b[idx] = 1.0
    dot = float(np.dot(vec_a, vec_b))
    norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return dot / norm if norm > 1e-12 else 0.0


def syncopation_strength(seq: RhythmSequence, beats_per_bar: int = 4,
                          beat_duration: float = 1.0) -> float:
    if not seq.events:
        return 0.0
    weights = np.zeros(beats_per_bar)
    for i in range(beats_per_bar):
        if i == 0:
            weights[i] = 1.0
        elif beats_per_bar % 2 == 0 and i == beats_per_bar // 2:
            weights[i] = 0.7
        elif i % 2 == 0:
            weights[i] = 0.4
        else:
            weights[i] = 0.2
    syncopation = 0.0
    for e in seq.events:
        beat = int(e.onset / beat_duration) % beats_per_bar
        syncopation += (1.0 - weights[beat]) * e.velocity
    return syncopation / len(seq.events)
