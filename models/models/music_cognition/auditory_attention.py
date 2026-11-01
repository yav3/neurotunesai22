"""Auditory attention: salience, streaming, selective attention modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SalienceMap:
    times: np.ndarray
    salience: np.ndarray
    peak_times: list[float]

    @property
    def mean_salience(self) -> float:
        return float(np.mean(self.salience))


def spectral_salience(
    magnitude_spec: np.ndarray,
    sr: float = 22050.0,
    n_fft: int = 2048,
) -> np.ndarray:
    n_frames = magnitude_spec.shape[0]
    salience = np.zeros(n_frames)
    for t in range(1, n_frames - 1):
        diff = np.abs(magnitude_spec[t] - 0.5 * (magnitude_spec[t - 1] + magnitude_spec[t + 1]))
        salience[t] = float(np.sum(diff))
    if np.max(salience) > 0:
        salience /= np.max(salience)
    return salience


def onset_salience(
    onset_strength: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    n = len(onset_strength)
    salience = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local_mean = np.mean(onset_strength[lo:hi])
        salience[i] = max(0.0, onset_strength[i] - local_mean)
    mx = np.max(salience)
    if mx > 0:
        salience /= mx
    return salience


def compute_salience_map(
    times: np.ndarray,
    spectral_sal: np.ndarray | None = None,
    onset_sal: np.ndarray | None = None,
    harmonic_sal: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> SalienceMap:
    n = len(times)
    if weights is None:
        weights = {"spectral": 0.4, "onset": 0.4, "harmonic": 0.2}

    combined = np.zeros(n)
    total_w = 0.0

    if spectral_sal is not None and len(spectral_sal) >= n:
        w = weights.get("spectral", 0.4)
        combined += w * spectral_sal[:n]
        total_w += w
    if onset_sal is not None and len(onset_sal) >= n:
        w = weights.get("onset", 0.4)
        combined += w * onset_sal[:n]
        total_w += w
    if harmonic_sal is not None and len(harmonic_sal) >= n:
        w = weights.get("harmonic", 0.2)
        combined += w * harmonic_sal[:n]
        total_w += w

    if total_w > 0:
        combined /= total_w

    peaks: list[float] = []
    for i in range(1, n - 1):
        if combined[i] > combined[i - 1] and combined[i] > combined[i + 1]:
            if combined[i] > 0.5:
                peaks.append(float(times[i]))

    return SalienceMap(times=times, salience=combined, peak_times=peaks)


@dataclass(frozen=True)
class AuditoryStream:
    stream_id: int
    frequency_center: float
    onset_times: list[float]
    coherence: float


def stream_segregation(
    frequencies: np.ndarray,
    onset_times: np.ndarray,
    freq_threshold: float = 200.0,
    time_threshold: float = 0.3,
) -> list[AuditoryStream]:
    n = len(frequencies)
    if n == 0:
        return []

    assigned = [-1] * n
    streams: list[list[int]] = []

    for i in range(n):
        best_stream = -1
        best_dist = float("inf")
        for s_idx, members in enumerate(streams):
            last = members[-1]
            freq_dist = abs(frequencies[i] - frequencies[last])
            time_dist = onset_times[i] - onset_times[last]
            if freq_dist < freq_threshold and 0 < time_dist < time_threshold:
                if freq_dist < best_dist:
                    best_dist = freq_dist
                    best_stream = s_idx
        if best_stream >= 0:
            streams[best_stream].append(i)
            assigned[i] = best_stream
        else:
            assigned[i] = len(streams)
            streams.append([i])

    result: list[AuditoryStream] = []
    for s_idx, members in enumerate(streams):
        freqs = frequencies[members]
        times = [float(onset_times[m]) for m in members]
        if len(freqs) > 1:
            coherence = 1.0 - float(np.std(freqs) / (np.mean(freqs) + 1e-12))
        else:
            coherence = 1.0
        result.append(AuditoryStream(
            stream_id=s_idx,
            frequency_center=float(np.mean(freqs)),
            onset_times=times,
            coherence=max(0.0, coherence),
        ))
    return result


def attention_decay(
    salience: np.ndarray,
    decay_rate: float = 0.95,
) -> np.ndarray:
    n = len(salience)
    attention = np.zeros(n)
    level = 0.0
    for i in range(n):
        level = level * decay_rate + salience[i] * (1.0 - decay_rate)
        attention[i] = level
    return attention


def surprise_score(
    expected: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    diff = np.abs(observed - expected)
    return diff / (np.abs(expected) + 1e-12)
