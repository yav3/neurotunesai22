"""Ensemble synchronization: alignment, leader-follower, phase coupling, balance."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PerformerTrack:
    name: str
    onsets: np.ndarray
    velocities: Optional[np.ndarray] = None

    @property
    def n_events(self) -> int:
        return len(self.onsets)

    @property
    def mean_ioi(self) -> float:
        if self.n_events < 2:
            return 0.0
        return float(np.mean(np.diff(self.onsets)))


@dataclass
class SyncMetrics:
    mean_asynchrony: float
    std_asynchrony: float
    max_asynchrony: float
    sync_strength: float


def inter_onset_alignment(track_a: PerformerTrack, track_b: PerformerTrack,
                          tolerance: float = 0.1) -> SyncMetrics:
    if track_a.n_events == 0 or track_b.n_events == 0:
        return SyncMetrics(0.0, 0.0, 0.0, 0.0)

    asyncs = []
    j = 0
    for onset_a in track_a.onsets:
        while j < track_b.n_events - 1 and abs(track_b.onsets[j + 1] - onset_a) < abs(track_b.onsets[j] - onset_a):
            j += 1
        diff = track_b.onsets[j] - onset_a
        if abs(diff) <= tolerance:
            asyncs.append(diff)

    if not asyncs:
        return SyncMetrics(0.0, 0.0, 0.0, 0.0)

    arr = np.array(asyncs)
    sync_strength = 1.0 - min(float(np.std(arr)) / tolerance, 1.0)

    return SyncMetrics(
        mean_asynchrony=float(arr.mean()),
        std_asynchrony=float(arr.std()),
        max_asynchrony=float(np.max(np.abs(arr))),
        sync_strength=sync_strength,
    )


def pairwise_sync_matrix(tracks: List[PerformerTrack],
                         tolerance: float = 0.1) -> np.ndarray:
    n = len(tracks)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                metrics = inter_onset_alignment(tracks[i], tracks[j], tolerance)
                matrix[i, j] = metrics.sync_strength
    return matrix


def granger_precedence(leader: PerformerTrack, follower: PerformerTrack,
                       max_lag_ms: float = 50.0) -> float:
    if leader.n_events < 5 or follower.n_events < 5:
        return 0.0

    leader_ioi = np.diff(leader.onsets)
    follower_ioi = np.diff(follower.onsets)
    n = min(len(leader_ioi), len(follower_ioi))
    if n < 4:
        return 0.0

    l_ioi = leader_ioi[:n]
    f_ioi = follower_ioi[:n]

    corr_lf = np.corrcoef(l_ioi[:-1], f_ioi[1:])[0, 1] if n > 2 else 0.0
    corr_fl = np.corrcoef(f_ioi[:-1], l_ioi[1:])[0, 1] if n > 2 else 0.0

    if np.isnan(corr_lf):
        corr_lf = 0.0
    if np.isnan(corr_fl):
        corr_fl = 0.0

    return float(corr_lf - corr_fl)


def detect_leader(tracks: List[PerformerTrack]) -> Tuple[int, Dict[int, float]]:
    n = len(tracks)
    if n < 2:
        return 0, {}

    leadership_scores: Dict[int, float] = {}
    for i in range(n):
        score = 0.0
        for j in range(n):
            if i != j:
                prec = granger_precedence(tracks[i], tracks[j])
                score += prec
        leadership_scores[i] = score / (n - 1)

    leader_idx = max(leadership_scores, key=lambda k: leadership_scores[k])
    return leader_idx, leadership_scores


def sync_error_distribution(track_a: PerformerTrack, track_b: PerformerTrack,
                            tolerance: float = 0.1) -> Dict[str, float]:
    if track_a.n_events == 0 or track_b.n_events == 0:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}

    asyncs = []
    j = 0
    for onset_a in track_a.onsets:
        while j < track_b.n_events - 1 and abs(track_b.onsets[j + 1] - onset_a) < abs(track_b.onsets[j] - onset_a):
            j += 1
        diff = track_b.onsets[j] - onset_a
        if abs(diff) <= tolerance:
            asyncs.append(diff)

    if len(asyncs) < 3:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}

    arr = np.array(asyncs)
    mean = float(arr.mean())
    std = float(arr.std())

    if std < 1e-12:
        return {"mean": mean, "std": std, "skew": 0.0, "kurtosis": 0.0}

    z = (arr - mean) / std
    skew = float(np.mean(z ** 3))
    kurtosis = float(np.mean(z ** 4) - 3.0)

    return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurtosis}


def phase_coupling(onsets_a: np.ndarray, onsets_b: np.ndarray,
                   period: float) -> float:
    if len(onsets_a) == 0 or len(onsets_b) == 0 or period <= 0:
        return 0.0

    phases_a = (onsets_a % period) / period * 2 * np.pi
    phases_b = (onsets_b % period) / period * 2 * np.pi

    n = min(len(phases_a), len(phases_b))
    phase_diff = phases_a[:n] - phases_b[:n]

    plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
    return plv


def tempo_negotiation(tracks: List[PerformerTrack],
                      window_size: int = 8) -> Dict[int, float]:
    if len(tracks) < 2:
        return {}

    adaptability: Dict[int, float] = {}

    for i, track in enumerate(tracks):
        if track.n_events < window_size + 2:
            adaptability[i] = 0.0
            continue

        ioi = np.diff(track.onsets)
        all_others_ioi = []
        for j, other in enumerate(tracks):
            if j != i and other.n_events >= window_size + 2:
                all_others_ioi.append(np.diff(other.onsets))

        if not all_others_ioi:
            adaptability[i] = 0.0
            continue

        min_len = min(len(ioi), min(len(o) for o in all_others_ioi))
        if min_len < window_size:
            adaptability[i] = 0.0
            continue

        mean_other = np.mean([o[:min_len] for o in all_others_ioi], axis=0)

        corrections = []
        for t in range(1, min_len):
            deviation_prev = ioi[t - 1] - mean_other[t - 1]
            correction = -(ioi[t] - ioi[t - 1])
            if abs(deviation_prev) > 1e-6:
                corrections.append(correction / deviation_prev)

        adaptability[i] = float(np.mean(corrections)) if corrections else 0.0

    return adaptability


def sectional_balance(tracks: List[PerformerTrack],
                      section_start: float, section_end: float) -> Dict[int, float]:
    levels: Dict[int, float] = {}

    for i, track in enumerate(tracks):
        if track.velocities is None:
            levels[i] = 0.5
            continue

        mask = (track.onsets >= section_start) & (track.onsets < section_end)
        if not np.any(mask):
            levels[i] = 0.0
            continue

        vels = track.velocities[mask]
        levels[i] = float(np.mean(vels) / 127.0)

    return levels


def balance_deviation(levels: Dict[int, float], target_balance: Optional[Dict[int, float]] = None) -> float:
    if not levels:
        return 0.0

    if target_balance is None:
        mean_level = np.mean(list(levels.values()))
        target_balance = {k: mean_level for k in levels}

    deviations = []
    for k in levels:
        if k in target_balance:
            deviations.append(abs(levels[k] - target_balance[k]))

    return float(np.mean(deviations)) if deviations else 0.0


def ensemble_cohesion(tracks: List[PerformerTrack], tolerance: float = 0.1) -> float:
    if len(tracks) < 2:
        return 1.0
    sync_mat = pairwise_sync_matrix(tracks, tolerance)
    n = len(tracks)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sync_mat[i, j]
            count += 1
    return total / count if count > 0 else 0.0


def synchronization_over_time(tracks: List[PerformerTrack],
                              window_duration: float = 4.0,
                              hop: float = 1.0) -> List[Tuple[float, float]]:
    if len(tracks) < 2:
        return []

    all_onsets = np.concatenate([t.onsets for t in tracks])
    t_start = float(all_onsets.min())
    t_end = float(all_onsets.max())

    results = []
    t = t_start
    while t + window_duration <= t_end:
        windowed = []
        for track in tracks:
            mask = (track.onsets >= t) & (track.onsets < t + window_duration)
            if np.any(mask):
                windowed.append(PerformerTrack(
                    name=track.name,
                    onsets=track.onsets[mask],
                    velocities=track.velocities[mask] if track.velocities is not None else None,
                ))
        if len(windowed) >= 2:
            coh = ensemble_cohesion(windowed)
            results.append((t + window_duration / 2, coh))
        t += hop

    return results
