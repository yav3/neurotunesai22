"""Models for ensemble synchronization: asynchrony, entrainment, phase correction."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SyncMetric:
    mean_async_ms: float
    std_async_ms: float
    pairwise_coupling: float
    leader_follower_ratio: float


def compute_async(onsets_a: np.ndarray, onsets_b: np.ndarray) -> float:
    if len(onsets_a) == 0 or len(onsets_b) == 0:
        return 0.0
    n = min(len(onsets_a), len(onsets_b))
    return float(np.mean(onsets_a[:n] - onsets_b[:n]) * 1000)


def mutual_adaptation(
    onsets_leader: np.ndarray, onsets_follower: np.ndarray, window_beats: int = 4,
) -> float:
    n = min(len(onsets_leader), len(onsets_follower))
    if n < window_beats + 1:
        return 0.0
    ioi_l = np.diff(onsets_leader[:n])
    ioi_f = np.diff(onsets_follower[:n])
    m = min(len(ioi_l), len(ioi_f))
    if m < window_beats:
        return 0.0
    corrs = []
    for i in range(m - window_beats):
        seg_l = ioi_l[i: i + window_beats]
        seg_f = ioi_f[i: i + window_beats]
        if np.std(seg_l) < 1e-10 or np.std(seg_f) < 1e-10:
            continue
        r = np.corrcoef(seg_l, seg_f)[0, 1]
        if not math.isnan(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0


def granger_influence(
    onsets_a: np.ndarray, onsets_b: np.ndarray, lag_max: int = 4,
) -> tuple[float, float]:
    n = min(len(onsets_a), len(onsets_b))
    if n < lag_max + 2:
        return 0.0, 0.0
    ioi_a = np.diff(onsets_a[:n])
    ioi_b = np.diff(onsets_b[:n])
    m = min(len(ioi_a), len(ioi_b))
    if m <= lag_max:
        return 0.0, 0.0

    def _var_reduction(target: np.ndarray, predictor: np.ndarray, lags: int) -> float:
        y = target[lags:]
        X_self = np.column_stack([target[lags - i - 1: m - i - 1] for i in range(lags)])
        X_other = np.column_stack([predictor[lags - i - 1: m - i - 1] for i in range(lags)])
        n_obs = len(y)
        if n_obs <= 2 * lags:
            return 0.0
        res_self = y - X_self @ np.linalg.lstsq(X_self, y, rcond=None)[0]
        var_self = np.var(res_self)
        X_full = np.column_stack([X_self, X_other])
        res_full = y - X_full @ np.linalg.lstsq(X_full, y, rcond=None)[0]
        var_full = np.var(res_full)
        if var_self < 1e-15:
            return 0.0
        return max(0.0, 1.0 - var_full / var_self)

    a_to_b = _var_reduction(ioi_b[:m], ioi_a[:m], lag_max)
    b_to_a = _var_reduction(ioi_a[:m], ioi_b[:m], lag_max)
    return a_to_b, b_to_a


def entrainment_score(tempos: list[float]) -> float:
    if len(tempos) < 2:
        return 1.0
    arr = np.array(tempos)
    mean = np.mean(arr)
    if mean < 1e-10:
        return 0.0
    cv = float(np.std(arr) / mean)
    return max(0.0, 1.0 - cv)


def phase_correction_gain(asynchronies: list[float]) -> float:
    if len(asynchronies) < 3:
        return 0.0
    a = np.array(asynchronies)
    corrections = np.diff(a)
    n = min(len(corrections), len(a) - 1)
    if n < 2:
        return 0.0
    x = a[:n]
    y = corrections[:n]
    var_x = np.var(x)
    if var_x < 1e-15:
        return 0.0
    alpha = -float(np.cov(x, y)[0, 1] / var_x)
    return max(0.0, min(1.0, alpha))


def ensemble_cohesion(all_onsets: list[list[float]]) -> SyncMetric:
    if len(all_onsets) < 2:
        return SyncMetric(0, 0, 1.0, 1.0)
    asyncs = []
    couplings = []
    for i in range(len(all_onsets)):
        for j in range(i + 1, len(all_onsets)):
            a = np.array(all_onsets[i])
            b = np.array(all_onsets[j])
            asyncs.append(abs(compute_async(a, b)))
            couplings.append(mutual_adaptation(a, b))
    mean_a = float(np.mean(asyncs)) if asyncs else 0.0
    std_a = float(np.std(asyncs)) if asyncs else 0.0
    mean_c = float(np.mean(couplings)) if couplings else 0.0
    a0 = np.array(all_onsets[0])
    influences = []
    for i in range(1, len(all_onsets)):
        ai = np.array(all_onsets[i])
        fwd, bck = granger_influence(a0, ai)
        if fwd + bck > 0:
            influences.append(fwd / (fwd + bck))
    lf = float(np.mean(influences)) if influences else 0.5
    return SyncMetric(mean_a, std_a, mean_c, lf)
