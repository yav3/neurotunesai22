"""Playlist transition quality analysis between tracks.

Transition scoring, classification, playlist flow optimization (greedy NN,
2-opt TSP), DJ set segmentation, and energy/mood arc analysis.
Pure numpy — no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class TransitionFeature(Enum):
    KEY = "key"
    TEMPO = "tempo"
    ENERGY = "energy"
    TIMBRE = "timbre"
    GENRE = "genre"
    MOOD = "mood"


class TransitionMethod(Enum):
    CROSSFADE = "crossfade"
    BEATMATCH = "beatmatch"
    KEY_COMPATIBLE = "key_compatible"
    ENERGY_MATCH = "energy_match"
    MOOD_BRIDGE = "mood_bridge"
    CUT = "cut"


@dataclass
class TransitionDetail:
    feature: TransitionFeature
    similarity: float
    penalty: float


@dataclass
class FullTransitionScore:
    tempo_compatibility: float
    key_compatibility: float
    energy_flow: float
    timbre_similarity: float
    overall: float
    method: TransitionMethod = TransitionMethod.CROSSFADE


def key_compatibility(key1: int, mode1: int, key2: int, mode2: int) -> float:
    """Circle-of-fifths distance normalised to [0, 1]."""
    fifths = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    d = abs(fifths[key1 % 12] - fifths[key2 % 12])
    d = min(d, 12 - d)
    key_sim = 1.0 - d / 6.0
    mode_bonus = 0.1 if mode1 == mode2 else -0.05
    return max(0.0, min(1.0, key_sim + mode_bonus))


def tempo_compatibility(bpm1: float, bpm2: float) -> float:
    """Ratio-based tempo compatibility; halving/doubling acceptable."""
    if bpm1 <= 0 or bpm2 <= 0:
        return 0.0
    ratio = max(bpm1, bpm2) / min(bpm1, bpm2)
    if ratio <= 1.05:
        return 1.0
    if abs(ratio - 2.0) < 0.1:
        return 0.8
    if abs(ratio - 0.5) < 0.05:
        return 0.8
    return max(0.0, 1.0 - (ratio - 1.0) / 2.0)


def energy_flow(e1: float, e2: float, target_direction: str = "maintain") -> float:
    """Penalise abrupt energy shifts unless intentional."""
    diff = e2 - e1
    if target_direction == "maintain":
        return max(0.0, 1.0 - abs(diff) * 2)
    if target_direction == "increase":
        return max(0.0, min(1.0, diff * 3 + 0.5))
    if target_direction == "decrease":
        return max(0.0, min(1.0, -diff * 3 + 0.5))
    return 1.0 - abs(diff)


def compute_transition_score(
    track1_features: dict,
    track2_features: dict,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted transition score between two tracks."""
    if weights is None:
        weights = {"key": 0.3, "tempo": 0.3, "energy": 0.25, "timbre": 0.15}
    score = 0.0
    total_w = 0.0
    if "key" in weights:
        ks = key_compatibility(
            track1_features.get("key", 0), track1_features.get("mode", 1),
            track2_features.get("key", 0), track2_features.get("mode", 1),
        )
        score += weights["key"] * ks
        total_w += weights["key"]
    if "tempo" in weights:
        ts = tempo_compatibility(track1_features.get("tempo", 120), track2_features.get("tempo", 120))
        score += weights["tempo"] * ts
        total_w += weights["tempo"]
    if "energy" in weights:
        es = energy_flow(track1_features.get("energy", 0.5), track2_features.get("energy", 0.5))
        score += weights["energy"] * es
        total_w += weights["energy"]
    return score / total_w if total_w > 0 else 0.0


def optimal_crossfade_point(
    audio1_envelope: np.ndarray,
    audio2_envelope: np.ndarray,
    sr: int = 44100,
) -> float:
    """Find energy-matched crossfade point (seconds from end of track 1)."""
    window = min(len(audio1_envelope), sr * 30)
    tail = audio1_envelope[-window:]
    head = audio2_envelope[:window] if len(audio2_envelope) >= window else audio2_envelope
    n = min(len(tail), len(head))
    if n == 0:
        return 5.0
    diff = np.abs(tail[:n] - head[:n])
    best_idx = int(np.argmin(diff))
    return (window - best_idx) / sr


def timbre_similarity(mfcc1: np.ndarray, mfcc2: np.ndarray) -> float:
    """Cosine similarity between mean MFCC vectors."""
    if mfcc1.ndim == 2:
        mfcc1 = mfcc1.mean(axis=1)
    if mfcc2.ndim == 2:
        mfcc2 = mfcc2.mean(axis=1)
    n1, n2 = np.linalg.norm(mfcc1), np.linalg.norm(mfcc2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(mfcc1, mfcc2) / (n1 * n2))


def full_transition_score(
    track1: dict, track2: dict,
    weights: Optional[Dict[str, float]] = None,
) -> FullTransitionScore:
    """Compute a detailed multi-dimensional transition score."""
    if weights is None:
        weights = {"key": 0.3, "tempo": 0.25, "energy": 0.25, "timbre": 0.2}
    kc = key_compatibility(
        track1.get("key", 0), track1.get("mode", 1),
        track2.get("key", 0), track2.get("mode", 1),
    )
    tc = tempo_compatibility(track1.get("tempo", 120), track2.get("tempo", 120))
    ef = energy_flow(track1.get("energy", 0.5), track2.get("energy", 0.5))
    ts = 0.5
    if "mfcc" in track1 and "mfcc" in track2:
        ts = timbre_similarity(
            np.asarray(track1["mfcc"]), np.asarray(track2["mfcc"]),
        )
    w = weights
    total_w = w.get("key", 0) + w.get("tempo", 0) + w.get("energy", 0) + w.get("timbre", 0)
    overall = (
        w.get("key", 0) * kc + w.get("tempo", 0) * tc +
        w.get("energy", 0) * ef + w.get("timbre", 0) * ts
    ) / max(total_w, 1e-10)
    method = classify_transition(track1, track2)
    return FullTransitionScore(tc, kc, ef, ts, overall, method)


def optimal_crossfade_duration(bpm1: float, bpm2: float, energy_diff: float) -> float:
    """Heuristic crossfade duration (seconds) from tempo and energy gap."""
    avg_bpm = (bpm1 + bpm2) / 2.0
    beat_dur = 60.0 / max(avg_bpm, 30.0)
    base_beats = 8.0
    if abs(energy_diff) > 0.4:
        base_beats = 4.0
    elif abs(energy_diff) < 0.1:
        base_beats = 16.0
    bpm_ratio = max(bpm1, bpm2) / max(min(bpm1, bpm2), 1.0)
    if bpm_ratio > 1.15:
        base_beats = max(base_beats / 2, 2.0)
    return beat_dur * base_beats


def classify_transition(track1: dict, track2: dict) -> TransitionMethod:
    """Determine the best transition method between two tracks."""
    tc = tempo_compatibility(track1.get("tempo", 120), track2.get("tempo", 120))
    kc = key_compatibility(
        track1.get("key", 0), track1.get("mode", 1),
        track2.get("key", 0), track2.get("mode", 1),
    )
    e_diff = abs(track1.get("energy", 0.5) - track2.get("energy", 0.5))
    if tc > 0.95 and kc > 0.7:
        return TransitionMethod.BEATMATCH
    if kc > 0.85:
        return TransitionMethod.KEY_COMPATIBLE
    if e_diff < 0.1 and tc > 0.8:
        return TransitionMethod.ENERGY_MATCH
    m1 = track1.get("mood_valence", 0.5)
    m2 = track2.get("mood_valence", 0.5)
    if abs(m1 - m2) > 0.3:
        return TransitionMethod.MOOD_BRIDGE
    if tc < 0.5 or kc < 0.3:
        return TransitionMethod.CUT
    return TransitionMethod.CROSSFADE


def transition_matrix(tracks: Sequence[dict]) -> np.ndarray:
    """Pairwise transition score matrix for a list of tracks."""
    n = len(tracks)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i, j] = compute_transition_score(tracks[i], tracks[j])
    return mat


def playlist_flow_curve(
    tracks: Sequence[dict], feature: str = "energy",
) -> np.ndarray:
    """Extract the trajectory of a feature across an ordered playlist."""
    return np.array([t.get(feature, 0.5) for t in tracks])


def detect_flow_breaks(
    tracks: Sequence[dict],
    threshold: float = 0.3,
) -> List[int]:
    """Find indices where consecutive transition score drops below threshold."""
    breaks = []
    for i in range(len(tracks) - 1):
        s = compute_transition_score(tracks[i], tracks[i + 1])
        if s < threshold:
            breaks.append(i)
    return breaks


def reorder_for_flow(
    tracks: List[dict],
    start_idx: int = 0,
) -> List[int]:
    """Greedy nearest-neighbour reordering maximising transition scores."""
    n = len(tracks)
    if n <= 1:
        return list(range(n))
    mat = transition_matrix(tracks)
    visited = [False] * n
    order = [start_idx]
    visited[start_idx] = True
    for _ in range(n - 1):
        cur = order[-1]
        scores = mat[cur].copy()
        scores[visited] = -1.0
        nxt = int(np.argmax(scores))
        order.append(nxt)
        visited[nxt] = True
    return order


def _route_cost(order: List[int], mat: np.ndarray) -> float:
    cost = 0.0
    for i in range(len(order) - 1):
        cost += mat[order[i], order[i + 1]]
    return cost


def traveling_salesman_playlist(
    tracks: List[dict],
    iterations: int = 500,
) -> List[int]:
    """2-opt TSP heuristic for globally optimal playlist ordering."""
    n = len(tracks)
    if n <= 3:
        return list(range(n))
    mat = transition_matrix(tracks)
    order = reorder_for_flow(tracks)
    best_cost = _route_cost(order, mat)
    improved = True
    it = 0
    while improved and it < iterations:
        improved = False
        it += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                new_cost = _route_cost(new_order, mat)
                if new_cost > best_cost:
                    order = new_order
                    best_cost = new_cost
                    improved = True
    return order


def dj_set_segmentation(
    tracks: Sequence[dict],
    min_set_size: int = 3,
    break_threshold: float = 0.25,
) -> List[List[int]]:
    """Partition playlist into DJ sets at natural transition break points."""
    n = len(tracks)
    if n == 0:
        return []
    breaks = detect_flow_breaks(tracks, threshold=break_threshold)
    segments: List[List[int]] = []
    start = 0
    for b in breaks:
        seg_end = b + 1
        if seg_end - start >= min_set_size:
            segments.append(list(range(start, seg_end)))
            start = seg_end
    if start < n:
        if segments and n - start < min_set_size:
            segments[-1].extend(range(start, n))
        else:
            segments.append(list(range(start, n)))
    return segments


def energy_arc_score(
    tracks: Sequence[dict],
    target_shape: str = "peak",
) -> float:
    """How well the playlist energy matches a target arc shape.

    Shapes: 'peak' (build-climax-resolve), 'ramp' (monotone increase),
    'wave' (sinusoidal oscillation), 'flat' (constant).
    """
    energies = playlist_flow_curve(tracks, "energy")
    n = len(energies)
    if n < 2:
        return 1.0
    t = np.linspace(0, 1, n)
    if target_shape == "peak":
        target = np.sin(np.pi * t)
    elif target_shape == "ramp":
        target = t
    elif target_shape == "wave":
        target = 0.5 + 0.5 * np.sin(2 * np.pi * t)
    else:
        target = np.ones(n) * np.mean(energies)
    e_norm = energies - energies.mean()
    t_norm = target - target.mean()
    ne, nt = np.linalg.norm(e_norm), np.linalg.norm(t_norm)
    if ne < 1e-12 or nt < 1e-12:
        return 0.5
    return float(np.dot(e_norm, t_norm) / (ne * nt) * 0.5 + 0.5)
