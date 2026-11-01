"""Auditory stream segregation — Bregman (1990) primitive grouping principles.

Models sequential and simultaneous integration/segregation of auditory
objects, the perceptual basis for polyphonic listening and the cocktail
party effect in music.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────

FISSION_BOUNDARY_SEMITONES = 3.0
TEMPORAL_COHERENCE_WINDOW_MS = 150.0
MAX_STREAMS = 8
ONSET_GROUPING_THRESHOLD_MS = 30.0


# ── Data types ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditoryEvent:
    onset_ms: float
    duration_ms: float
    frequency_hz: float
    amplitude: float = 1.0
    timbre_centroid: float = 0.0


@dataclass
class Stream:
    stream_id: int
    events: List[AuditoryEvent] = field(default_factory=list)

    @property
    def mean_frequency(self) -> float:
        if not self.events:
            return 0.0
        return float(np.mean([e.frequency_hz for e in self.events]))

    @property
    def duration_ms(self) -> float:
        if not self.events:
            return 0.0
        first = min(e.onset_ms for e in self.events)
        last = max(e.onset_ms + e.duration_ms for e in self.events)
        return last - first


@dataclass(frozen=True)
class SegregationResult:
    streams: List[Stream]
    n_streams: int
    fission_probability: float
    coherence_matrix: np.ndarray


# ── Frequency proximity ──────────────────────────────────────────────────

def frequency_to_semitones(f1: float, f2: float) -> float:
    if f1 <= 0 or f2 <= 0:
        return float("inf")
    return 12.0 * math.log2(f2 / f1)


def frequency_proximity(f1: float, f2: float) -> float:
    """Proximity in [0, 1]; 1 = identical, decays Gaussian with semitone distance."""
    st = abs(frequency_to_semitones(f1, f2))
    sigma = FISSION_BOUNDARY_SEMITONES
    return float(np.exp(-0.5 * (st / sigma) ** 2))


def temporal_proximity(onset1_ms: float, onset2_ms: float) -> float:
    """Temporal proximity in [0, 1]; Gaussian decay with TEMPORAL_COHERENCE_WINDOW_MS."""
    dt = abs(onset2_ms - onset1_ms)
    sigma = TEMPORAL_COHERENCE_WINDOW_MS
    return float(np.exp(-0.5 * (dt / sigma) ** 2))


def timbre_proximity(centroid1: float, centroid2: float, sigma: float = 500.0) -> float:
    """Spectral centroid proximity in [0, 1]."""
    return float(np.exp(-0.5 * ((centroid2 - centroid1) / max(sigma, 1e-12)) ** 2))


# ── Coherence matrix ────────────────────────────────────────────────────

def coherence_matrix(
    events: Sequence[AuditoryEvent],
    w_freq: float = 0.5,
    w_time: float = 0.3,
    w_timbre: float = 0.2,
) -> np.ndarray:
    """Pairwise coherence for sequential grouping (van Noorden 1975 model)."""
    n = len(events)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            fp = frequency_proximity(events[i].frequency_hz, events[j].frequency_hz)
            tp = temporal_proximity(events[i].onset_ms, events[j].onset_ms)
            tb = timbre_proximity(events[i].timbre_centroid, events[j].timbre_centroid)
            c = w_freq * fp + w_time * tp + w_timbre * tb
            mat[i, j] = c
            mat[j, i] = c
        mat[i, i] = 1.0
    return mat


# ── Fission boundary ────────────────────────────────────────────────────

def fission_probability(
    interval_semitones: float,
    tempo_ioi_ms: float,
    boundary_st: float = FISSION_BOUNDARY_SEMITONES,
) -> float:
    """Van Noorden fission probability: large intervals + fast tempo → segregation."""
    freq_factor = 1.0 / (1.0 + np.exp(-(abs(interval_semitones) - boundary_st)))
    tempo_factor = 1.0 / (1.0 + np.exp((tempo_ioi_ms - TEMPORAL_COHERENCE_WINDOW_MS) / 30.0))
    return float(freq_factor * 0.7 + tempo_factor * 0.3)


# ── Stream segregation (greedy nearest-stream) ──────────────────────────

def segregate_streams(
    events: Sequence[AuditoryEvent],
    threshold: float = 0.4,
    max_streams: int = MAX_STREAMS,
) -> SegregationResult:
    """Greedy sequential stream assignment based on coherence threshold."""
    sorted_events = sorted(events, key=lambda e: e.onset_ms)
    streams: List[Stream] = []
    cmat = coherence_matrix(sorted_events)

    for idx, event in enumerate(sorted_events):
        best_stream: Optional[int] = None
        best_score = -1.0

        for si, stream in enumerate(streams):
            last_event_idx = -1
            for j, ev in enumerate(sorted_events):
                if ev is stream.events[-1]:
                    last_event_idx = j
                    break
            if last_event_idx >= 0:
                score = cmat[last_event_idx, idx]
                if score > threshold and score > best_score:
                    best_score = score
                    best_stream = si

        if best_stream is not None:
            streams[best_stream].events.append(event)
        elif len(streams) < max_streams:
            streams.append(Stream(stream_id=len(streams), events=[event]))
        else:
            worst_idx = min(range(len(streams)), key=lambda s: len(streams[s].events))
            streams[worst_idx].events.append(event)

    if len(sorted_events) >= 2:
        intervals = [
            abs(frequency_to_semitones(
                sorted_events[i].frequency_hz,
                sorted_events[i + 1].frequency_hz,
            ))
            for i in range(len(sorted_events) - 1)
        ]
        iois = [
            sorted_events[i + 1].onset_ms - sorted_events[i].onset_ms
            for i in range(len(sorted_events) - 1)
        ]
        mean_interval = float(np.mean(intervals))
        mean_ioi = float(np.mean(iois)) if iois else 200.0
        fission_prob = fission_probability(mean_interval, mean_ioi)
    else:
        fission_prob = 0.0

    return SegregationResult(
        streams=streams,
        n_streams=len(streams),
        fission_probability=fission_prob,
        coherence_matrix=cmat,
    )


# ── Simultaneous grouping ───────────────────────────────────────────────

def onset_synchrony_groups(
    events: Sequence[AuditoryEvent],
    threshold_ms: float = ONSET_GROUPING_THRESHOLD_MS,
) -> List[List[int]]:
    """Group events by onset synchrony — simultaneous onsets fuse into chords."""
    sorted_indices = sorted(range(len(events)), key=lambda i: events[i].onset_ms)
    groups: List[List[int]] = []
    current_group: List[int] = []

    for idx in sorted_indices:
        if not current_group:
            current_group.append(idx)
        else:
            last_onset = events[current_group[-1]].onset_ms
            if events[idx].onset_ms - last_onset <= threshold_ms:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
    if current_group:
        groups.append(current_group)
    return groups


def harmonic_fusion_score(frequencies: Sequence[float], tolerance_cents: float = 50.0) -> float:
    """How well a set of simultaneous frequencies fuse into a harmonic series."""
    if len(frequencies) < 2:
        return 1.0
    f0_candidates = np.array(frequencies)
    best_score = 0.0
    for f0 in f0_candidates:
        if f0 <= 0:
            continue
        ratios = np.array(frequencies) / f0
        nearest_int = np.round(ratios)
        nearest_int = np.clip(nearest_int, 1, None)
        deviations_cents = 1200.0 * np.abs(np.log2(np.clip(ratios / nearest_int, 1e-12, None)))
        within = np.sum(deviations_cents < tolerance_cents)
        score = within / len(frequencies)
        if score > best_score:
            best_score = score
    return float(best_score)


# ── Stream salience ──────────────────────────────────────────────────────

def stream_salience(stream: Stream) -> float:
    """Perceptual salience: function of density, amplitude, and register."""
    if not stream.events:
        return 0.0
    dur = stream.duration_ms
    density = len(stream.events) / max(dur, 1.0) * 1000.0
    mean_amp = float(np.mean([e.amplitude for e in stream.events]))
    mean_freq = stream.mean_frequency
    register_weight = 1.0 / (1.0 + np.exp(-(math.log2(max(mean_freq, 20)) - 9.0)))
    return float(np.clip(density * 0.3 + mean_amp * 0.5 + register_weight * 0.2, 0, 1))


def stream_continuity(stream: Stream) -> float:
    """Temporal continuity: 1 = evenly spaced, 0 = large gaps."""
    if len(stream.events) < 2:
        return 1.0
    onsets = sorted(e.onset_ms for e in stream.events)
    iois = np.diff(onsets)
    if len(iois) == 0 or np.mean(iois) == 0:
        return 1.0
    cv = float(np.std(iois) / np.mean(iois))
    return float(np.exp(-cv))
