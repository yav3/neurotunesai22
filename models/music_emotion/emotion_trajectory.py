"""Emotion trajectory analysis over musical time.

Models emotional arcs through a piece: transition matrices between emotion
states, momentum/inertia computation, arc shape detection (ramp, peak, valley,
oscillation), and trajectory distance metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class ArcShape(Enum):
    ASCENDING = auto()
    DESCENDING = auto()
    PEAK = auto()
    VALLEY = auto()
    PLATEAU = auto()
    OSCILLATING = auto()
    FLAT = auto()


@dataclass
class EmotionPoint:
    valence: float     # [-1, 1]
    arousal: float     # [-1, 1]
    dominance: float   # [-1, 1]
    time: float        # in seconds or beats


@dataclass
class TrajectoryStats:
    mean_valence: float
    mean_arousal: float
    mean_dominance: float
    valence_range: float
    arousal_range: float
    momentum: float
    inertia: float
    arc_shape: ArcShape
    transitions: np.ndarray  # (n_states, n_states) transition matrix
    state_labels: List[str]


@dataclass
class TrajectoryDistance:
    dtw_distance: float
    frechet_distance: float
    area_between: float


EMOTION_QUADRANTS = {
    "excited": (0.5, 0.5),
    "happy": (0.8, 0.3),
    "calm": (0.3, -0.5),
    "relaxed": (0.5, -0.3),
    "sad": (-0.6, -0.4),
    "depressed": (-0.7, -0.6),
    "angry": (-0.4, 0.7),
    "tense": (-0.3, 0.5),
}


def vad_to_quadrant(valence: float, arousal: float) -> str:
    """Map VAD point to nearest named emotion quadrant."""
    best_name = "neutral"
    best_dist = float("inf")
    for name, (v, a) in EMOTION_QUADRANTS.items():
        d = (valence - v) ** 2 + (arousal - a) ** 2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def emotion_momentum(trajectory: List[EmotionPoint]) -> float:
    """Average magnitude of emotion change per time step."""
    if len(trajectory) < 2:
        return 0.0
    deltas = []
    for i in range(1, len(trajectory)):
        dv = trajectory[i].valence - trajectory[i - 1].valence
        da = trajectory[i].arousal - trajectory[i - 1].arousal
        dd = trajectory[i].dominance - trajectory[i - 1].dominance
        dt = max(trajectory[i].time - trajectory[i - 1].time, 1e-6)
        deltas.append(np.sqrt(dv**2 + da**2 + dd**2) / dt)
    return float(np.mean(deltas))


def emotion_inertia(trajectory: List[EmotionPoint]) -> float:
    """Autocorrelation of valence series at lag 1 — high = slow emotional change."""
    if len(trajectory) < 3:
        return 0.0
    v = np.array([p.valence for p in trajectory])
    v_centered = v - v.mean()
    var = float(np.sum(v_centered ** 2))
    if var < 1e-10:
        return 1.0
    ac = float(np.sum(v_centered[:-1] * v_centered[1:])) / var
    return ac


def transition_matrix(
    trajectory: List[EmotionPoint],
    n_bins: int = 4,
) -> Tuple[np.ndarray, List[str]]:
    """Build emotion state transition matrix by quantizing VA space."""
    labels = ["low-V/low-A", "low-V/high-A", "high-V/low-A", "high-V/high-A"]
    states = []
    for p in trajectory:
        vi = 1 if p.valence >= 0 else 0
        ai = 1 if p.arousal >= 0 else 0
        states.append(vi * 2 + ai)
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)
    for i in range(len(states) - 1):
        mat[states[i], states[i + 1]] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, row_sums, where=row_sums > 0, out=mat)
    return mat, labels


def detect_arc_shape(trajectory: List[EmotionPoint]) -> ArcShape:
    """Classify the overall emotional arc shape from valence contour."""
    if len(trajectory) < 3:
        return ArcShape.FLAT
    v = np.array([p.valence for p in trajectory])
    v_range = v.max() - v.min()
    if v_range < 0.1:
        return ArcShape.FLAT

    # Linear regression slope
    x = np.arange(len(v), dtype=float)
    slope = float(np.polyfit(x, v, 1)[0])

    # Peak/valley detection
    mid = len(v) // 2
    first_half_mean = v[:mid].mean()
    second_half_mean = v[mid:].mean()
    peak_idx = int(np.argmax(v))
    valley_idx = int(np.argmin(v))

    # Oscillation: count zero-crossings of detrended signal
    detrended = v - np.linspace(v[0], v[-1], len(v))
    zc = int(np.sum(np.abs(np.diff(np.sign(detrended))) > 0))
    if zc > len(v) * 0.3:
        return ArcShape.OSCILLATING

    if slope > 0.01 and abs(slope) > v_range * 0.1 / len(v):
        return ArcShape.ASCENDING
    if slope < -0.01 and abs(slope) > v_range * 0.1 / len(v):
        return ArcShape.DESCENDING

    # Peak in middle third
    if len(v) // 3 < peak_idx < 2 * len(v) // 3:
        if v[peak_idx] > first_half_mean and v[peak_idx] > second_half_mean:
            return ArcShape.PEAK

    # Valley in middle third
    if len(v) // 3 < valley_idx < 2 * len(v) // 3:
        if v[valley_idx] < first_half_mean and v[valley_idx] < second_half_mean:
            return ArcShape.VALLEY

    if abs(first_half_mean - second_half_mean) < 0.05:
        return ArcShape.PLATEAU

    return ArcShape.ASCENDING if slope > 0 else ArcShape.DESCENDING


def analyze_trajectory(trajectory: List[EmotionPoint]) -> TrajectoryStats:
    """Full trajectory analysis."""
    v = np.array([p.valence for p in trajectory])
    a = np.array([p.arousal for p in trajectory])
    d = np.array([p.dominance for p in trajectory])
    trans, labels = transition_matrix(trajectory)
    return TrajectoryStats(
        mean_valence=float(v.mean()) if len(v) else 0.0,
        mean_arousal=float(a.mean()) if len(a) else 0.0,
        mean_dominance=float(d.mean()) if len(d) else 0.0,
        valence_range=float(v.ptp()) if len(v) else 0.0,
        arousal_range=float(a.ptp()) if len(a) else 0.0,
        momentum=emotion_momentum(trajectory),
        inertia=emotion_inertia(trajectory),
        arc_shape=detect_arc_shape(trajectory),
        transitions=trans,
        state_labels=labels,
    )


def dtw_trajectory_distance(
    traj_a: List[EmotionPoint],
    traj_b: List[EmotionPoint],
) -> float:
    """DTW distance between two emotion trajectories in VA space."""
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        return float("inf")
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt(
                (traj_a[i-1].valence - traj_b[j-1].valence) ** 2
                + (traj_a[i-1].arousal - traj_b[j-1].arousal) ** 2
            )
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return float(dp[n, m]) / max(n, m)


def frechet_trajectory_distance(
    traj_a: List[EmotionPoint],
    traj_b: List[EmotionPoint],
) -> float:
    """Discrete Fréchet distance between two VA trajectories."""
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        return float("inf")
    dp = np.full((n, m), -1.0)

    def _d(i: int, j: int) -> float:
        return float(np.sqrt(
            (traj_a[i].valence - traj_b[j].valence) ** 2
            + (traj_a[i].arousal - traj_b[j].arousal) ** 2
        ))

    def _rec(i: int, j: int) -> float:
        if dp[i, j] >= 0:
            return dp[i, j]
        if i == 0 and j == 0:
            dp[i, j] = _d(0, 0)
        elif i == 0:
            dp[i, j] = max(_rec(0, j - 1), _d(0, j))
        elif j == 0:
            dp[i, j] = max(_rec(i - 1, 0), _d(i, 0))
        else:
            dp[i, j] = max(
                min(_rec(i - 1, j), _rec(i, j - 1), _rec(i - 1, j - 1)),
                _d(i, j),
            )
        return dp[i, j]

    return _rec(n - 1, m - 1)


def trajectory_distance(
    traj_a: List[EmotionPoint],
    traj_b: List[EmotionPoint],
) -> TrajectoryDistance:
    """Compute multiple trajectory distance metrics."""
    dtw = dtw_trajectory_distance(traj_a, traj_b)
    frechet = frechet_trajectory_distance(traj_a, traj_b)
    # Area between curves (simple trapezoid for aligned trajectories)
    n = min(len(traj_a), len(traj_b))
    if n < 2:
        area = 0.0
    else:
        va = np.array([traj_a[i].valence for i in range(n)])
        vb = np.array([traj_b[i].valence for i in range(n)])
        area = float(np.trapz(np.abs(va - vb)))
    return TrajectoryDistance(dtw_distance=dtw, frechet_distance=frechet, area_between=area)
