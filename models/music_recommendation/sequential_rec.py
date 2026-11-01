"""Sequential recommendation: session-aware next-track prediction.

Models the temporal dynamics of listening sessions: transition probability
matrices, skip prediction, session segmentation by mood/energy arcs,
and playlist continuation via feature-space trajectory extrapolation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ListeningEvent:
    track_id: str
    features: np.ndarray
    timestamp_s: float
    duration_listened_s: float
    total_duration_s: float
    skipped: bool = False


@dataclass
class SessionSegment:
    events: List[ListeningEvent]
    centroid: np.ndarray
    energy_trend: float
    valence_trend: float


def skip_probability(duration_listened: float, total_duration: float,
                     position_in_session: int, session_length: int) -> float:
    listen_ratio = duration_listened / max(total_duration, 1.0)
    fatigue = min(1.0, position_in_session / max(session_length, 1))
    skip_base = 1.0 / (1.0 + np.exp(5.0 * (listen_ratio - 0.4)))
    return float(np.clip(skip_base * (1.0 + 0.3 * fatigue), 0, 1))


def transition_matrix(sessions: List[List[str]],
                      n_tracks: int) -> np.ndarray:
    trans = np.zeros((n_tracks, n_tracks))
    for session in sessions:
        for i in range(len(session) - 1):
            a = hash(session[i]) % n_tracks
            b = hash(session[i + 1]) % n_tracks
            trans[a, b] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return trans / row_sums


def session_energy_arc(events: List[ListeningEvent],
                       energy_idx: int = 0) -> np.ndarray:
    if not events:
        return np.array([])
    energies = np.array([e.features[energy_idx] if len(e.features) > energy_idx else 0.0
                         for e in events])
    return energies


def segment_session(events: List[ListeningEvent],
                    max_gap_s: float = 1800.0,
                    similarity_threshold: float = 0.5) -> List[SessionSegment]:
    if not events:
        return []
    segments: List[SessionSegment] = []
    current: List[ListeningEvent] = [events[0]]
    for i in range(1, len(events)):
        gap = events[i].timestamp_s - events[i - 1].timestamp_s
        if gap > max_gap_s:
            segments.append(_make_segment(current))
            current = [events[i]]
            continue
        feat_sim = _cosine(events[i].features, current[-1].features)
        if feat_sim < similarity_threshold and len(current) >= 3:
            segments.append(_make_segment(current))
            current = [events[i]]
        else:
            current.append(events[i])
    if current:
        segments.append(_make_segment(current))
    return segments


def _make_segment(events: List[ListeningEvent]) -> SessionSegment:
    features = np.array([e.features for e in events])
    centroid = features.mean(axis=0)
    if len(events) >= 2 and features.shape[1] > 0:
        energy_trend = float(np.polyfit(range(len(events)), features[:, 0], 1)[0])
        val_idx = min(1, features.shape[1] - 1)
        valence_trend = float(np.polyfit(range(len(events)), features[:, val_idx], 1)[0])
    else:
        energy_trend = 0.0
        valence_trend = 0.0
    return SessionSegment(events=events, centroid=centroid,
                          energy_trend=energy_trend, valence_trend=valence_trend)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def trajectory_extrapolation(recent_features: np.ndarray,
                             n_steps: int = 3,
                             momentum: float = 0.7) -> np.ndarray:
    if len(recent_features) < 2:
        return np.tile(recent_features[-1] if len(recent_features) > 0
                       else np.zeros(1), (n_steps, 1))
    velocity = np.diff(recent_features, axis=0)
    avg_vel = np.mean(velocity[-3:], axis=0) if len(velocity) >= 3 else np.mean(velocity, axis=0)
    predictions = np.zeros((n_steps, recent_features.shape[1]))
    current = recent_features[-1].copy()
    vel = avg_vel.copy()
    for i in range(n_steps):
        current = current + vel
        predictions[i] = current
        vel *= momentum
    return predictions


def next_track_scores(candidate_features: np.ndarray,
                      predicted_point: np.ndarray,
                      diversity_weight: float = 0.2,
                      recent_features: Optional[np.ndarray] = None) -> np.ndarray:
    dists = np.linalg.norm(candidate_features - predicted_point, axis=1)
    relevance = 1.0 / (1.0 + dists)
    if recent_features is not None and len(recent_features) > 0:
        recent_center = recent_features.mean(axis=0)
        diversity = np.linalg.norm(candidate_features - recent_center, axis=1)
        diversity = diversity / max(diversity.max(), 1e-10)
    else:
        diversity = np.zeros(len(candidate_features))
    scores = (1.0 - diversity_weight) * relevance + diversity_weight * diversity
    return scores


def listening_fatigue_model(session_duration_min: float,
                            n_tracks: int,
                            skip_rate: float) -> Dict[str, float]:
    base_fatigue = 1.0 - np.exp(-session_duration_min / 60.0)
    skip_boost = skip_rate * 0.5
    fatigue = float(np.clip(base_fatigue + skip_boost, 0, 1))
    engagement = 1.0 - fatigue
    tracks_per_hour = n_tracks / max(session_duration_min / 60, 0.01)
    return {
        "fatigue": fatigue,
        "engagement": engagement,
        "tracks_per_hour": tracks_per_hour,
        "recommend_break": fatigue > 0.7,
    }


def session_coherence(events: List[ListeningEvent]) -> float:
    if len(events) < 2:
        return 1.0
    sims = []
    for i in range(len(events) - 1):
        sims.append(_cosine(events[i].features, events[i + 1].features))
    return float(np.mean(sims))
