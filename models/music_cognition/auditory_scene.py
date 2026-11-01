"""Auditory scene analysis: stream segregation, Gestalt grouping, attention modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "AuditoryStream", "StreamingEvent", "SceneAnalysis",
    "segregate_streams", "gestalt_proximity_score",
    "gestalt_similarity_score", "gestalt_common_fate",
    "gestalt_good_continuation", "detect_stream_boundaries",
    "perceptual_voice_separation", "attention_salience",
    "stream_coherence",
]


@dataclass
class AuditoryStream:
    stream_id: int
    note_indices: List[int]
    mean_pitch: float
    coherence: float


@dataclass
class StreamingEvent:
    time: float
    event_type: str
    stream_ids: List[int]


@dataclass
class SceneAnalysis:
    streams: List[AuditoryStream]
    boundaries: List[StreamingEvent]
    attention_weights: np.ndarray


def gestalt_proximity_score(
    pitches: np.ndarray, times: np.ndarray,
    pitch_weight: float = 1.0, time_weight: float = 0.5,
) -> np.ndarray:
    n = len(pitches)
    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dp = abs(pitches[i] - pitches[j]) * pitch_weight
            dt = abs(times[i] - times[j]) * time_weight
            dist = np.sqrt(dp ** 2 + dt ** 2)
            scores[i, j] = np.exp(-dist / max(np.mean(np.abs(np.diff(pitches))), 1e-6))
    return scores


def gestalt_similarity_score(features: np.ndarray) -> np.ndarray:
    n = features.shape[0]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = features / norms
    return (normalized @ normalized.T + 1.0) / 2.0


def gestalt_common_fate(velocities: np.ndarray) -> np.ndarray:
    n = len(velocities)
    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if velocities[i] == 0 and velocities[j] == 0:
                scores[i, j] = 1.0
            else:
                ratio = min(abs(velocities[i]), abs(velocities[j])) / max(abs(velocities[i]), abs(velocities[j]), 1e-9)
                same_dir = 1.0 if np.sign(velocities[i]) == np.sign(velocities[j]) else 0.0
                scores[i, j] = ratio * same_dir
    return scores


def gestalt_good_continuation(pitch_sequence: np.ndarray) -> float:
    if len(pitch_sequence) < 3:
        return 1.0
    intervals = np.diff(pitch_sequence)
    if len(intervals) < 2:
        return 1.0
    stepwise = np.abs(intervals) <= 2.0
    continuity = float(np.mean(stepwise))
    direction_changes = np.sum(np.diff(np.sign(intervals)) != 0)
    smoothness = 1.0 - direction_changes / max(len(intervals) - 1, 1)
    return float(0.5 * continuity + 0.5 * smoothness)


def segregate_streams(
    pitches: np.ndarray, times: np.ndarray,
    max_pitch_gap: float = 7.0, max_time_gap: float = 0.5,
) -> List[AuditoryStream]:
    n = len(pitches)
    if n == 0:
        return []
    order = np.argsort(times)
    assigned = [-1] * n
    streams: List[List[int]] = []
    for idx in order:
        best_stream = -1
        best_dist = float("inf")
        for s_id, s_notes in enumerate(streams):
            last_idx = s_notes[-1]
            dp = abs(pitches[idx] - pitches[last_idx])
            dt = times[idx] - times[last_idx]
            if dp <= max_pitch_gap and 0 < dt <= max_time_gap:
                dist = dp + dt
                if dist < best_dist:
                    best_dist = dist
                    best_stream = s_id
        if best_stream >= 0:
            streams[best_stream].append(idx)
            assigned[idx] = best_stream
        else:
            assigned[idx] = len(streams)
            streams.append([idx])
    result = []
    for s_id, notes in enumerate(streams):
        p = pitches[notes]
        mean_p = float(np.mean(p))
        gc = gestalt_good_continuation(p) if len(p) >= 3 else 1.0
        result.append(AuditoryStream(s_id, notes, mean_p, gc))
    return result


def detect_stream_boundaries(
    pitches: np.ndarray, times: np.ndarray, threshold: float = 10.0
) -> List[StreamingEvent]:
    if len(pitches) < 2:
        return []
    boundaries = []
    intervals = np.abs(np.diff(pitches))
    for i, iv in enumerate(intervals):
        if iv > threshold:
            boundaries.append(StreamingEvent(
                time=float(times[i + 1]),
                event_type="split",
                stream_ids=[i, i + 1],
            ))
    return boundaries


def perceptual_voice_separation(
    pitches: np.ndarray, times: np.ndarray, max_voices: int = 4
) -> List[List[int]]:
    streams = segregate_streams(pitches, times)
    streams.sort(key=lambda s: len(s.note_indices), reverse=True)
    voices = [s.note_indices for s in streams[:max_voices]]
    if len(streams) > max_voices:
        for s in streams[max_voices:]:
            best_v = 0
            best_diff = float("inf")
            for v_id, v_notes in enumerate(voices):
                v_mean = float(np.mean(pitches[v_notes]))
                diff = abs(s.mean_pitch - v_mean)
                if diff < best_diff:
                    best_diff = diff
                    best_v = v_id
            voices[best_v].extend(s.note_indices)
    return voices


def attention_salience(
    pitches: np.ndarray, velocities: np.ndarray, novelty: Optional[np.ndarray] = None
) -> np.ndarray:
    n = len(pitches)
    if n == 0:
        return np.array([])
    p_norm = (pitches - pitches.min()) / max(pitches.max() - pitches.min(), 1e-9)
    v_norm = velocities / max(velocities.max(), 1e-9)
    salience = 0.3 * p_norm + 0.4 * v_norm
    if novelty is not None and len(novelty) == n:
        salience += 0.3 * (novelty / max(novelty.max(), 1e-9))
    else:
        salience += 0.3 * 0.5
    return salience / max(salience.max(), 1e-9)


def stream_coherence(stream: AuditoryStream, pitches: np.ndarray) -> float:
    if len(stream.note_indices) < 2:
        return 1.0
    p = pitches[stream.note_indices]
    return gestalt_good_continuation(p)
