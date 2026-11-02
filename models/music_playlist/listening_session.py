"""Listening session modeling: attention curves, fatigue estimation,
flow state detection, session segmentation and pacing.

Pure numpy — no external frameworks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ListenerState(str, Enum):
    ENGAGED = "engaged"
    FLOW = "flow"
    PASSIVE = "passive"
    FATIGUED = "fatigued"
    DISTRACTED = "distracted"


@dataclass
class TrackEvent:
    track_index: int
    start_time: float
    duration: float
    energy: float
    valence: float
    arousal: float
    tempo: float
    skipped: bool = False
    skip_time: Optional[float] = None
    replay_count: int = 0


@dataclass
class SessionState:
    time: float
    attention: float
    fatigue: float
    arousal_level: float
    listener_state: ListenerState
    cumulative_energy: float


@dataclass
class SessionAnalysis:
    total_duration: float
    active_listening_time: float
    skip_rate: float
    mean_attention: float
    fatigue_onset_time: Optional[float]
    flow_segments: List[Tuple[float, float]]
    energy_trajectory: np.ndarray
    state_timeline: List[SessionState]


def attention_decay(t: float, half_life: float = 1200.0,
                    baseline: float = 0.3) -> float:
    decay = math.exp(-0.693 * t / half_life)
    return baseline + (1.0 - baseline) * decay


def attention_boost(current: float, novelty: float,
                    max_boost: float = 0.3) -> float:
    boost = max_boost * novelty
    return min(1.0, current + boost)


def fatigue_accumulation(listening_time: float,
                         mean_loudness: float,
                         complexity: float) -> float:
    time_factor = 1.0 - math.exp(-listening_time / 3600.0)
    loudness_factor = max(0.0, (mean_loudness - 0.5) * 2.0)
    complexity_factor = max(0.0, complexity - 0.3)
    fatigue = 0.5 * time_factor + 0.3 * loudness_factor + 0.2 * complexity_factor
    return min(1.0, max(0.0, fatigue))


def novelty_score(current_features: np.ndarray,
                  history_features: np.ndarray) -> float:
    if history_features.shape[0] == 0:
        return 1.0
    diffs = np.linalg.norm(history_features - current_features, axis=1)
    return float(np.mean(diffs))


def classify_listener_state(attention: float, fatigue: float,
                            arousal: float) -> ListenerState:
    if fatigue > 0.7:
        return ListenerState.FATIGUED
    if attention < 0.3:
        return ListenerState.DISTRACTED
    if attention > 0.8 and 0.4 < arousal < 0.8 and fatigue < 0.3:
        return ListenerState.FLOW
    if attention > 0.5:
        return ListenerState.ENGAGED
    return ListenerState.PASSIVE


def energy_trajectory(events: List[TrackEvent]) -> np.ndarray:
    if not events:
        return np.array([])
    energies = np.array([e.energy for e in events])
    return energies


def smooth_energy_curve(energies: np.ndarray,
                        window: int = 3) -> np.ndarray:
    if len(energies) < window:
        return energies.copy()
    kernel = np.ones(window) / window
    padded = np.pad(energies, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(energies)]


def skip_rate(events: List[TrackEvent]) -> float:
    if not events:
        return 0.0
    skipped = sum(1 for e in events if e.skipped)
    return skipped / len(events)


def mean_skip_position(events: List[TrackEvent]) -> float:
    skip_positions = []
    for e in events:
        if e.skipped and e.skip_time is not None and e.duration > 0:
            skip_positions.append(e.skip_time / e.duration)
    if not skip_positions:
        return 1.0
    return float(np.mean(skip_positions))


def replay_affinity(events: List[TrackEvent]) -> np.ndarray:
    return np.array([e.replay_count for e in events], dtype=np.float64)


def detect_flow_segments(states: List[SessionState],
                         min_duration: float = 120.0) -> List[Tuple[float, float]]:
    segments = []
    in_flow = False
    flow_start = 0.0

    for s in states:
        if s.listener_state == ListenerState.FLOW:
            if not in_flow:
                flow_start = s.time
                in_flow = True
        else:
            if in_flow:
                duration = s.time - flow_start
                if duration >= min_duration:
                    segments.append((flow_start, s.time))
                in_flow = False

    if in_flow and states:
        last_time = states[-1].time
        if last_time - flow_start >= min_duration:
            segments.append((flow_start, last_time))

    return segments


def fatigue_onset(states: List[SessionState],
                  threshold: float = 0.5) -> Optional[float]:
    for s in states:
        if s.fatigue >= threshold:
            return s.time
    return None


def optimal_session_length(fatigue_rate: float,
                           target_engagement: float = 0.6) -> float:
    if fatigue_rate <= 0:
        return 7200.0
    t = -3600.0 * math.log(1.0 - target_engagement) / (fatigue_rate + 1e-12)
    return min(max(t, 300.0), 7200.0)


def pacing_recommendation(states: List[SessionState],
                          upcoming_energy: float) -> Dict[str, float]:
    if not states:
        return {"recommended_energy": 0.5, "tempo_adjustment": 1.0,
                "novelty_need": 0.5}

    last = states[-1]
    recent = states[-min(5, len(states)):]
    recent_energy = np.mean([s.cumulative_energy / max(s.time, 1.0)
                             for s in recent])

    if last.listener_state == ListenerState.FATIGUED:
        rec_energy = max(0.2, upcoming_energy * 0.6)
        tempo_adj = 0.85
        novelty = 0.8
    elif last.listener_state == ListenerState.FLOW:
        rec_energy = upcoming_energy
        tempo_adj = 1.0
        novelty = 0.3
    elif last.listener_state == ListenerState.DISTRACTED:
        rec_energy = min(1.0, upcoming_energy * 1.3)
        tempo_adj = 1.1
        novelty = 0.9
    else:
        rec_energy = upcoming_energy
        tempo_adj = 1.0
        novelty = 0.5

    return {
        "recommended_energy": float(np.clip(rec_energy, 0.0, 1.0)),
        "tempo_adjustment": float(np.clip(tempo_adj, 0.7, 1.3)),
        "novelty_need": float(np.clip(novelty, 0.0, 1.0)),
    }


def simulate_session(events: List[TrackEvent],
                     attention_half_life: float = 1200.0) -> SessionAnalysis:
    states = []
    cumulative_time = 0.0
    cumulative_energy = 0.0
    feature_history = []
    active_time = 0.0

    for event in events:
        features = np.array([event.energy, event.valence,
                             event.arousal, event.tempo / 200.0])

        base_attention = attention_decay(cumulative_time, attention_half_life)
        nov = novelty_score(features, np.array(feature_history) if feature_history else np.empty((0, 4)))
        att = attention_boost(base_attention, min(nov, 1.0))

        if event.skipped:
            att *= 0.7

        effective_dur = event.skip_time if event.skipped and event.skip_time else event.duration
        fat = fatigue_accumulation(cumulative_time, event.energy, nov)
        arousal_level = float(np.clip(event.arousal * att, 0, 1))

        state = classify_listener_state(att, fat, arousal_level)

        cumulative_energy += event.energy * effective_dur
        cumulative_time += effective_dur
        if not event.skipped:
            active_time += effective_dur

        states.append(SessionState(
            time=cumulative_time, attention=att, fatigue=fat,
            arousal_level=arousal_level, listener_state=state,
            cumulative_energy=cumulative_energy,
        ))
        feature_history.append(features)

    flow_segs = detect_flow_segments(states)
    fat_onset = fatigue_onset(states)
    energies = energy_trajectory(events)
    sr = skip_rate(events)
    mean_att = float(np.mean([s.attention for s in states])) if states else 0.0

    return SessionAnalysis(
        total_duration=cumulative_time,
        active_listening_time=active_time,
        skip_rate=sr,
        mean_attention=mean_att,
        fatigue_onset_time=fat_onset,
        flow_segments=flow_segs,
        energy_trajectory=energies,
        state_timeline=states,
    )
