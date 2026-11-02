"""Rhythmic feature extraction from onset times and beat grids.

Implements onset density, tempo stability, beat histogram, pulse clarity,
syncopation index (Longuet-Higgins & Lee 1984), groove features,
and the normalized pairwise variability index (nPVI, Grabe & Low 2002).
"""
from __future__ import annotations

import numpy as np


def onset_density(onset_times: np.ndarray, duration: float) -> float:
    """Onsets per second."""
    if duration <= 0:
        return 0.0
    return float(len(onset_times) / duration)


def inter_onset_intervals(onset_times: np.ndarray) -> np.ndarray:
    """IOI sequence from sorted onset times (seconds)."""
    times = np.sort(onset_times)
    if len(times) < 2:
        return np.array([])
    return np.diff(times)


def tempo_stability(onset_times: np.ndarray) -> float:
    """Coefficient of variation of inter-onset intervals. 0 = perfectly regular."""
    ioi = inter_onset_intervals(onset_times)
    if len(ioi) < 2:
        return 0.0
    mu = np.mean(ioi)
    if mu < 1e-12:
        return 0.0
    return float(np.std(ioi) / mu)


def beat_histogram(onset_times: np.ndarray, bpm_min: float = 30.0,
                   bpm_max: float = 300.0, n_bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation-based beat histogram.

    Returns (bpm_axis, histogram) where histogram peaks indicate likely tempi.
    """
    ioi = inter_onset_intervals(onset_times)
    if len(ioi) < 2:
        bpm_axis = np.linspace(bpm_min, bpm_max, n_bins)
        return bpm_axis, np.zeros(n_bins)

    bpm_axis = np.linspace(bpm_min, bpm_max, n_bins)
    period_axis = 60.0 / bpm_axis

    hist = np.zeros(n_bins)
    for p_idx, period in enumerate(period_axis):
        residuals = np.mod(ioi, period)
        residuals = np.minimum(residuals, period - residuals)
        hist[p_idx] = np.exp(-np.mean(residuals ** 2) / (2 * (period * 0.1) ** 2))

    mx = np.max(hist)
    if mx > 0:
        hist /= mx
    return bpm_axis, hist


def pulse_clarity(onset_times: np.ndarray, max_lag_s: float = 2.0,
                  sr_grid: float = 200.0) -> float:
    """Autocorrelation peak prominence of the onset signal — higher = clearer pulse."""
    if len(onset_times) < 3:
        return 0.0
    duration = onset_times[-1] - onset_times[0]
    if duration <= 0:
        return 0.0
    n_samples = int(duration * sr_grid) + 1
    signal = np.zeros(n_samples)
    for t in onset_times:
        idx = int((t - onset_times[0]) * sr_grid)
        if 0 <= idx < n_samples:
            signal[idx] = 1.0

    max_lag = int(max_lag_s * sr_grid)
    if max_lag > n_samples:
        max_lag = n_samples
    ac = np.correlate(signal, signal, mode='full')
    ac = ac[len(signal) - 1:][:max_lag]
    if ac[0] < 1e-12:
        return 0.0
    ac_norm = ac / ac[0]

    peaks = []
    for i in range(1, len(ac_norm) - 1):
        if ac_norm[i] > ac_norm[i - 1] and ac_norm[i] > ac_norm[i + 1]:
            peaks.append(ac_norm[i])
    if not peaks:
        return 0.0
    return float(max(peaks))


def syncopation_index(onsets_binary: np.ndarray, meter: tuple[int, int] = (4, 4)) -> float:
    """Longuet-Higgins & Lee (1984) syncopation metric.

    *onsets_binary* is a binary grid (1=onset, 0=rest) of length = subdivisions per bar.
    Higher values = more syncopated.
    """
    n = len(onsets_binary)
    if n == 0:
        return 0.0
    weights = np.zeros(n)
    level = int(np.log2(max(n, 1)))
    for i in range(n):
        w = 0
        pos = i
        for lv in range(level, -1, -1):
            step = 2 ** lv
            if pos % step == 0:
                w = lv
                break
        weights[i] = w

    score = 0.0
    for i in range(n):
        if onsets_binary[i] == 1:
            j = (i + 1) % n
            if onsets_binary[j] == 0:
                if weights[j] > weights[i]:
                    score += weights[j] - weights[i]
    max_possible = np.sum(weights)
    if max_possible < 1e-12:
        return 0.0
    return float(score / max_possible)


def swing_ratio(onset_times: np.ndarray, beat_times: np.ndarray) -> float:
    """Ratio of long-to-short eighth notes within each beat span.

    1.0 = straight, ~1.5 = light swing, ~2.0 = triplet swing.
    """
    if len(beat_times) < 2 or len(onset_times) < 3:
        return 1.0
    ratios = []
    for i in range(len(beat_times) - 1):
        bt = beat_times[i]
        bt_next = beat_times[i + 1]
        mid = (bt + bt_next) / 2
        early = onset_times[(onset_times >= bt) & (onset_times < mid)]
        late = onset_times[(onset_times >= mid) & (onset_times < bt_next)]
        if len(early) > 0 and len(late) > 0:
            dur_long = late[0] - early[-1]
            dur_short = bt_next - late[0]
            if dur_short > 1e-6:
                ratios.append(dur_long / dur_short)
    if not ratios:
        return 1.0
    return float(np.median(ratios))


def microtiming_deviation(onset_times: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
    """Signed deviation of each onset from its nearest grid point (seconds)."""
    if len(onset_times) == 0 or len(grid_times) == 0:
        return np.array([])
    deviations = np.zeros(len(onset_times))
    for i, ot in enumerate(onset_times):
        diffs = ot - grid_times
        idx = np.argmin(np.abs(diffs))
        deviations[i] = diffs[idx]
    return deviations


def tempo_drift(onset_times: np.ndarray, window_beats: int = 8) -> np.ndarray:
    """Instantaneous tempo curve from windowed IOI (BPM)."""
    ioi = inter_onset_intervals(onset_times)
    if len(ioi) < window_beats:
        if len(ioi) > 0:
            return np.array([60.0 / np.mean(ioi)])
        return np.array([0.0])
    tempos = []
    for i in range(len(ioi) - window_beats + 1):
        mean_ioi = np.mean(ioi[i:i + window_beats])
        if mean_ioi > 1e-6:
            tempos.append(60.0 / mean_ioi)
        else:
            tempos.append(0.0)
    return np.array(tempos)


def npvi(durations: np.ndarray) -> float:
    """Normalized pairwise variability index (Grabe & Low 2002).

    Measures rhythmic variability independent of tempo.
    Range [0, 200]; higher = more variable.
    """
    n = len(durations)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        s = durations[i] + durations[i + 1]
        if abs(s) > 1e-12:
            total += abs(durations[i] - durations[i + 1]) / (s / 2)
    return float(100.0 * total / (n - 1))


def rhythmic_complexity(onset_binary: np.ndarray) -> float:
    """Weighted sum of off-beat onsets as a complexity measure (0-1)."""
    n = len(onset_binary)
    if n == 0 or np.sum(onset_binary) == 0:
        return 0.0
    total_onsets = int(np.sum(onset_binary))
    off_beat = 0
    for i in range(n):
        if onset_binary[i] == 1 and i % 2 != 0:
            off_beat += 1
    return float(off_beat / total_onsets)
