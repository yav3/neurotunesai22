"""Query by humming: pitch contour matching for melody retrieval.

Implements pitch contour extraction from monophonic audio, contour
simplification, and DTW-based matching against a melody database for
query-by-humming (QBH) retrieval systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _autocorrelation_pitch(frame: np.ndarray, sr: int,
                           fmin: float = 80.0, fmax: float = 800.0) -> float:
    if len(frame) < 2 or np.max(np.abs(frame)) < 1e-8:
        return 0.0
    lag_min = max(1, int(sr / fmax))
    lag_max = min(len(frame) - 1, int(sr / fmin))
    if lag_min >= lag_max:
        return 0.0
    frame = frame - np.mean(frame)
    norm = np.sum(frame ** 2)
    if norm < 1e-12:
        return 0.0
    acf = np.correlate(frame, frame, mode="full")
    acf = acf[len(frame) - 1 :]
    acf = acf / (norm + 1e-12)
    search = acf[lag_min : lag_max + 1]
    if len(search) == 0:
        return 0.0
    peak_idx = np.argmax(search)
    if search[peak_idx] < 0.3:
        return 0.0
    lag = lag_min + peak_idx
    return float(sr / lag)


def extract_pitch_contour(signal: np.ndarray, sr: int = 22050,
                          frame_size: int = 2048, hop: int = 512,
                          fmin: float = 80.0, fmax: float = 800.0) -> np.ndarray:
    n_frames = max(1, 1 + (len(signal) - frame_size) // hop)
    contour = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start : start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        contour[i] = _autocorrelation_pitch(frame, sr, fmin, fmax)
    return contour


def hz_to_cents(hz: np.ndarray, ref: float = 440.0) -> np.ndarray:
    safe = np.where(hz > 0, hz, 1e-12)
    cents = 1200 * np.log2(safe / ref)
    cents[hz <= 0] = np.nan
    return cents


def simplify_contour(contour_cents: np.ndarray, tolerance: float = 50.0) -> np.ndarray:
    valid = ~np.isnan(contour_cents)
    if np.sum(valid) < 2:
        return contour_cents[valid]
    values = contour_cents[valid]
    simplified = [values[0]]
    for v in values[1:]:
        if abs(v - simplified[-1]) >= tolerance:
            simplified.append(v)
    return np.array(simplified)


def contour_to_intervals(contour_cents: np.ndarray) -> np.ndarray:
    valid = contour_cents[~np.isnan(contour_cents)]
    if len(valid) < 2:
        return np.array([])
    return np.diff(valid)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / max(n + m, 1))


def dtw_distance_banded(a: np.ndarray, b: np.ndarray, band: int = 10) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            d = abs(a[i - 1] - b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / max(n + m, 1))


@dataclass
class MelodyEntry:
    name: str
    contour_cents: np.ndarray
    intervals: np.ndarray
    metadata: dict = field(default_factory=dict)


class MelodyDatabase:
    """In-memory melody database for QBH retrieval."""

    def __init__(self) -> None:
        self._entries: list[MelodyEntry] = []

    def add(self, name: str, pitches_hz: np.ndarray,
            metadata: dict | None = None) -> None:
        cents = hz_to_cents(pitches_hz)
        intervals = contour_to_intervals(cents)
        self._entries.append(MelodyEntry(
            name, cents, intervals, metadata or {}
        ))

    def add_midi(self, name: str, midi_pitches: list[int],
                 metadata: dict | None = None) -> None:
        hz = 440.0 * 2 ** ((np.array(midi_pitches, dtype=float) - 69) / 12)
        self.add(name, hz, metadata)

    @property
    def size(self) -> int:
        return len(self._entries)

    def query(self, query_hz: np.ndarray, n_results: int = 5,
              use_intervals: bool = True, band: int = 15) -> list[tuple[str, float]]:
        if use_intervals:
            q_cents = hz_to_cents(query_hz)
            q_intervals = contour_to_intervals(q_cents)
            if len(q_intervals) == 0:
                return []
            results = []
            for entry in self._entries:
                if len(entry.intervals) == 0:
                    continue
                dist = dtw_distance_banded(q_intervals, entry.intervals, band)
                results.append((entry.name, dist))
        else:
            q_cents = hz_to_cents(query_hz)
            q_simplified = simplify_contour(q_cents)
            results = []
            for entry in self._entries:
                ref = simplify_contour(entry.contour_cents)
                if len(ref) == 0:
                    continue
                dist = dtw_distance_banded(q_simplified, ref, band)
                results.append((entry.name, dist))
        results.sort(key=lambda x: x[1])
        return results[:n_results]


@dataclass
class QBHResult:
    query_length: int
    matches: list[tuple[str, float]]
    best_match: str | None
    confidence: float


def query_by_humming(signal: np.ndarray, sr: int,
                     database: MelodyDatabase,
                     n_results: int = 5) -> QBHResult:
    contour = extract_pitch_contour(signal, sr)
    voiced = contour[contour > 0]
    if len(voiced) < 3:
        return QBHResult(len(contour), [], None, 0.0)
    matches = database.query(voiced, n_results)
    if not matches:
        return QBHResult(len(contour), [], None, 0.0)
    best_name, best_dist = matches[0]
    confidence = max(0.0, 1.0 - best_dist / 500.0)
    return QBHResult(len(contour), matches, best_name, confidence)
