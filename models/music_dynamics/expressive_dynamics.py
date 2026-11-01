"""Expressive dynamics: crescendo/diminuendo detection, dynamic contour, loudness analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DynamicProfile:
    mean_level: float
    dynamic_range: float
    crescendo_count: int
    diminuendo_count: int
    subito_count: int
    dynamic_variance: float


def _amplitude_to_db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-10))


def dynamic_contour(envelope: np.ndarray, hop: int = 512) -> np.ndarray:
    if len(envelope) == 0:
        return np.array([])
    n_frames = max(len(envelope) // hop, 1)
    contour = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(envelope))
        contour[i] = float(np.sqrt(np.mean(envelope[start:end] ** 2)))
    return _amplitude_to_db(contour)


def detect_crescendo(
    envelope: np.ndarray,
    sr: int = 44100,
    min_duration: float = 0.5,
    min_db_change: float = 3.0,
) -> List[Tuple[int, int, float]]:
    hop = max(sr // 20, 1)
    contour = dynamic_contour(envelope, hop)
    if len(contour) < 2:
        return []
    min_frames = max(int(min_duration * sr / hop), 2)
    results = []
    i = 0
    while i < len(contour) - 1:
        if contour[i + 1] > contour[i]:
            start = i
            while i < len(contour) - 1 and contour[i + 1] >= contour[i] - 0.5:
                i += 1
            end = i
            if end - start >= min_frames:
                db_change = float(contour[end] - contour[start])
                if db_change >= min_db_change:
                    results.append((start * hop, end * hop, db_change))
        i += 1
    return results


def detect_diminuendo(
    envelope: np.ndarray,
    sr: int = 44100,
    min_duration: float = 0.5,
    min_db_change: float = 3.0,
) -> List[Tuple[int, int, float]]:
    hop = max(sr // 20, 1)
    contour = dynamic_contour(envelope, hop)
    if len(contour) < 2:
        return []
    min_frames = max(int(min_duration * sr / hop), 2)
    results = []
    i = 0
    while i < len(contour) - 1:
        if contour[i + 1] < contour[i]:
            start = i
            while i < len(contour) - 1 and contour[i + 1] <= contour[i] + 0.5:
                i += 1
            end = i
            if end - start >= min_frames:
                db_change = float(contour[start] - contour[end])
                if db_change >= min_db_change:
                    results.append((start * hop, end * hop, db_change))
        i += 1
    return results


def detect_subito(
    envelope: np.ndarray,
    sr: int = 44100,
    threshold_db: float = 6.0,
    max_transition_ms: float = 50.0,
) -> List[int]:
    hop = max(sr // 100, 1)
    contour = dynamic_contour(envelope, hop)
    if len(contour) < 2:
        return []
    max_frames = max(int(max_transition_ms * sr / (1000.0 * hop)), 1)
    results = []
    for i in range(max_frames, len(contour)):
        diff = abs(contour[i] - contour[i - max_frames])
        if diff >= threshold_db:
            results.append(i * hop)
    return results


def loudness_histogram(
    envelope: np.ndarray,
    n_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(envelope) == 0:
        return np.array([]), np.array([])
    db = _amplitude_to_db(envelope)
    counts, edges = np.histogram(db, bins=n_bins)
    return counts.astype(float), edges


def dynamic_range_score(envelope: np.ndarray) -> float:
    if len(envelope) < 2:
        return 0.0
    db = _amplitude_to_db(envelope)
    actual_range = float(np.percentile(db, 95) - np.percentile(db, 5))
    theoretical_max = 96.0
    return min(actual_range / theoretical_max, 1.0)


def analyze_dynamics(envelope: np.ndarray, sr: int = 44100) -> DynamicProfile:
    if len(envelope) == 0:
        return DynamicProfile(0.0, 0.0, 0, 0, 0, 0.0)
    db = _amplitude_to_db(envelope)
    mean_level = float(np.mean(db))
    dynamic_range = float(np.percentile(db, 95) - np.percentile(db, 5))
    cresc = detect_crescendo(envelope, sr)
    dim = detect_diminuendo(envelope, sr)
    sub = detect_subito(envelope, sr)
    return DynamicProfile(
        mean_level=mean_level,
        dynamic_range=dynamic_range,
        crescendo_count=len(cresc),
        diminuendo_count=len(dim),
        subito_count=len(sub),
        dynamic_variance=float(np.var(db)),
    )
