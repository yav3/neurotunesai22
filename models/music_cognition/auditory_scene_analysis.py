"""Auditory scene analysis (Bregman 1990).

Models how the auditory system groups acoustic elements into
coherent streams via proximity, harmonicity, onset synchrony,
common fate, and continuity.  Pure numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np


class StreamingCue(Enum):
    PROXIMITY = "proximity"
    HARMONICITY = "harmonicity"
    ONSET_SYNCHRONY = "onset_synchrony"
    COMMON_FATE = "common_fate"
    TIMBRE = "timbre"
    SPATIAL = "spatial"


@dataclass
class AuditoryObject:
    frequency_band: Tuple[float, float]
    onset: float
    duration: float
    amplitude: float = 1.0


@dataclass
class StreamState:
    objects: List[AuditoryObject] = field(default_factory=list)
    coherence: float = 0.0


def frequency_proximity(f1: float, f2: float) -> float:
    if f1 <= 0 or f2 <= 0:
        return 0.0
    semitones = 12.0 * math.log2(max(f1, f2) / min(f1, f2))
    fission_boundary = 3.0
    coherence_boundary = 15.0
    if semitones <= fission_boundary:
        return 1.0
    if semitones >= coherence_boundary:
        return 0.0
    return 1.0 - (semitones - fission_boundary) / (coherence_boundary - fission_boundary)


def onset_synchrony(onset_a: float, onset_b: float, tolerance: float = 0.03) -> float:
    diff = abs(onset_a - onset_b)
    if diff <= tolerance:
        return 1.0
    return float(np.exp(-(diff - tolerance) ** 2 / (2 * 0.02**2)))


def harmonic_template_match(partials: np.ndarray, f0_candidate: float) -> float:
    if f0_candidate <= 0 or len(partials) == 0:
        return 0.0
    ratios = partials / f0_candidate
    nearest_int = np.round(ratios)
    deviations = np.abs(ratios - nearest_int)
    max_harmonic = 12
    valid = (nearest_int >= 1) & (nearest_int <= max_harmonic)
    if not np.any(valid):
        return 0.0
    match_score = np.mean(np.exp(-50.0 * deviations[valid] ** 2))
    return float(match_score)


def common_fate_score(am_a: np.ndarray, am_b: np.ndarray) -> float:
    n = min(len(am_a), len(am_b))
    if n < 2:
        return 0.0
    a = am_a[:n]
    b = am_b[:n]
    std_a, std_b = np.std(a), np.std(b)
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return max(0.0, r)


def old_plus_new_heuristic(
    existing_spectrum: np.ndarray,
    new_spectrum: np.ndarray,
) -> np.ndarray:
    residual = new_spectrum - existing_spectrum
    return np.maximum(residual, 0.0)


def stream_segregation(
    objects: List[AuditoryObject],
    max_streams: int = 4,
) -> List[StreamState]:
    if len(objects) == 0:
        return []
    sorted_objs = sorted(objects, key=lambda o: o.onset)
    streams: List[StreamState] = [StreamState(objects=[sorted_objs[0]], coherence=1.0)]
    for obj in sorted_objs[1:]:
        best_stream = -1
        best_score = -1.0
        obj_center = (obj.frequency_band[0] + obj.frequency_band[1]) / 2.0
        for si, stream in enumerate(streams):
            last = stream.objects[-1]
            last_center = (last.frequency_band[0] + last.frequency_band[1]) / 2.0
            prox = frequency_proximity(obj_center, last_center)
            sync = onset_synchrony(obj.onset, last.onset + last.duration)
            score = 0.6 * prox + 0.4 * sync
            if score > best_score:
                best_score = score
                best_stream = si
        if best_score > 0.4 and best_stream >= 0:
            streams[best_stream].objects.append(obj)
            streams[best_stream].coherence = (streams[best_stream].coherence + best_score) / 2.0
        elif len(streams) < max_streams:
            streams.append(StreamState(objects=[obj], coherence=0.5))
        else:
            worst = min(range(len(streams)), key=lambda i: streams[i].coherence)
            if best_score > streams[worst].coherence:
                streams[worst].objects.append(obj)
    return streams


def streaming_boundary(tempo_hz: float) -> Tuple[float, float]:
    if tempo_hz <= 0:
        return (3.0, 15.0)
    fission = max(1.0, 3.0 + 2.0 * math.log2(max(tempo_hz, 0.5)))
    coherence = max(fission + 2.0, 15.0 - 3.0 * math.log2(max(tempo_hz, 0.5)))
    return (fission, coherence)


def continuity_illusion_check(
    gap_duration: float,
    masker_level: float,
    signal_level: float,
) -> bool:
    if gap_duration > 0.3:
        return False
    if masker_level < signal_level:
        return False
    margin = masker_level - signal_level
    max_gap = 0.05 + 0.01 * margin
    return gap_duration <= max_gap


def melodic_fission_probability(
    interval_semitones: float,
    rate_hz: float,
) -> float:
    fission_thresh, coherence_thresh = streaming_boundary(rate_hz)
    st = abs(interval_semitones)
    if st <= fission_thresh:
        return 0.0
    if st >= coherence_thresh:
        return 1.0
    return (st - fission_thresh) / (coherence_thresh - fission_thresh)


def primitive_grouping_score(objects: List[AuditoryObject]) -> float:
    if len(objects) <= 1:
        return 0.0
    streams = stream_segregation(objects)
    return len(streams) / len(objects)
