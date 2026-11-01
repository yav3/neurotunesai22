"""Arousal dynamics: time-varying arousal from audio features, peak/trough detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ArousalPoint:
    time: float
    arousal: float
    feature_contributions: dict[str, float]


@dataclass(frozen=True)
class ArousalCurve:
    times: np.ndarray
    arousal: np.ndarray
    mean_arousal: float
    std_arousal: float
    peak_times: list[float]
    trough_times: list[float]


def loudness_to_arousal(rms: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    normalized = np.clip((db - floor_db) / (-floor_db), 0.0, 1.0)
    return normalized


def tempo_to_arousal(tempo_bpm: float) -> float:
    return float(np.clip((tempo_bpm - 40) / (200 - 40), 0.0, 1.0))


def spectral_centroid_to_arousal(centroid_hz: np.ndarray, max_hz: float = 8000.0) -> np.ndarray:
    return np.clip(centroid_hz / max_hz, 0.0, 1.0)


def onset_density_to_arousal(onset_density: np.ndarray, max_density: float = 15.0) -> np.ndarray:
    return np.clip(onset_density / max_density, 0.0, 1.0)


def harmonic_tension_to_arousal(tension: np.ndarray) -> np.ndarray:
    return np.clip(tension, 0.0, 1.0)


def compute_arousal_curve(
    times: np.ndarray,
    rms: np.ndarray | None = None,
    spectral_centroid: np.ndarray | None = None,
    onset_density: np.ndarray | None = None,
    tension: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> ArousalCurve:
    n = len(times)
    if weights is None:
        weights = {"loudness": 0.35, "brightness": 0.25, "density": 0.2, "tension": 0.2}

    arousal = np.zeros(n)
    total_weight = 0.0

    if rms is not None and len(rms) >= n:
        a = loudness_to_arousal(rms[:n])
        arousal += weights.get("loudness", 0.35) * a
        total_weight += weights.get("loudness", 0.35)

    if spectral_centroid is not None and len(spectral_centroid) >= n:
        a = spectral_centroid_to_arousal(spectral_centroid[:n])
        arousal += weights.get("brightness", 0.25) * a
        total_weight += weights.get("brightness", 0.25)

    if onset_density is not None and len(onset_density) >= n:
        a = onset_density_to_arousal(onset_density[:n])
        arousal += weights.get("density", 0.2) * a
        total_weight += weights.get("density", 0.2)

    if tension is not None and len(tension) >= n:
        a = harmonic_tension_to_arousal(tension[:n])
        arousal += weights.get("tension", 0.2) * a
        total_weight += weights.get("tension", 0.2)

    if total_weight > 0:
        arousal /= total_weight

    peaks = _find_local_extrema(arousal, mode="max")
    troughs = _find_local_extrema(arousal, mode="min")
    peak_times = [float(times[i]) for i in peaks]
    trough_times = [float(times[i]) for i in troughs]

    return ArousalCurve(
        times=times,
        arousal=arousal,
        mean_arousal=float(np.mean(arousal)),
        std_arousal=float(np.std(arousal)),
        peak_times=peak_times,
        trough_times=trough_times,
    )


def _find_local_extrema(arr: np.ndarray, mode: str = "max", min_distance: int = 5) -> list[int]:
    extrema: list[int] = []
    for i in range(1, len(arr) - 1):
        if mode == "max":
            if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                if not extrema or (i - extrema[-1]) >= min_distance:
                    extrema.append(i)
        else:
            if arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
                if not extrema or (i - extrema[-1]) >= min_distance:
                    extrema.append(i)
    return extrema


@dataclass(frozen=True)
class ArousalSegment:
    start_time: float
    end_time: float
    mean_arousal: float
    trend: str


def segment_arousal(curve: ArousalCurve, n_segments: int = 4) -> list[ArousalSegment]:
    n = len(curve.times)
    if n < 2:
        return []
    seg_size = max(1, n // n_segments)
    segments: list[ArousalSegment] = []
    for s in range(n_segments):
        start = s * seg_size
        end = min((s + 1) * seg_size, n)
        if start >= n:
            break
        seg_arousal = curve.arousal[start:end]
        mean_a = float(np.mean(seg_arousal))
        if len(seg_arousal) > 1:
            slope = float(np.polyfit(np.arange(len(seg_arousal)), seg_arousal, 1)[0])
            trend = "rising" if slope > 0.01 else ("falling" if slope < -0.01 else "stable")
        else:
            trend = "stable"
        segments.append(ArousalSegment(
            start_time=float(curve.times[start]),
            end_time=float(curve.times[min(end - 1, n - 1)]),
            mean_arousal=mean_a,
            trend=trend,
        ))
    return segments


def arousal_contrast(curve: ArousalCurve, window: int = 10) -> np.ndarray:
    n = len(curve.arousal)
    if n < window * 2:
        return np.zeros(n)
    contrast = np.zeros(n)
    for i in range(window, n - window):
        left = np.mean(curve.arousal[i - window : i])
        right = np.mean(curve.arousal[i : i + window])
        contrast[i] = abs(right - left)
    return contrast


def smooth_arousal(arousal: np.ndarray, window: int = 5) -> np.ndarray:
    if len(arousal) < window:
        return arousal.copy()
    kernel = np.ones(window) / window
    padded = np.pad(arousal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arousal)]
