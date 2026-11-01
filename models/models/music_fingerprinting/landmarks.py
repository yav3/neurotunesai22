"""Shazam-style constellation map fingerprinting via spectral peak landmarks."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Peak:
    time: int
    freq: int
    magnitude: float


@dataclass
class Landmark:
    anchor_time: int
    anchor_freq: int
    paired_freq: int
    delta_time: int

    def hash_value(self) -> int:
        return (self.anchor_freq & 0x3FF) | ((self.paired_freq & 0x3FF) << 10) | ((self.delta_time & 0xFFF) << 20)


@dataclass
class Match:
    track_id: str
    score: float
    offset: int
    n_matches: int


def find_peaks(
    spectrogram: np.ndarray,
    threshold: float = 0.01,
    neighborhood_size: int = 5,
) -> list[Peak]:
    n_freq, n_time = spectrogram.shape
    half = neighborhood_size // 2
    peaks = []
    for t in range(half, n_time - half):
        for f in range(half, n_freq - half):
            val = spectrogram[f, t]
            if val < threshold:
                continue
            region = spectrogram[
                max(0, f - half) : f + half + 1,
                max(0, t - half) : t + half + 1,
            ]
            if val >= region.max():
                peaks.append(Peak(time=t, freq=f, magnitude=val))
    return peaks


def form_pairs(
    peaks: list[Peak],
    fan_out: int = 5,
    time_delta_max: int = 64,
) -> list[Landmark]:
    sorted_peaks = sorted(peaks, key=lambda p: (p.time, p.freq))
    landmarks = []
    for i, anchor in enumerate(sorted_peaks):
        paired = 0
        for j in range(i + 1, len(sorted_peaks)):
            if paired >= fan_out:
                break
            target = sorted_peaks[j]
            dt = target.time - anchor.time
            if dt <= 0:
                continue
            if dt > time_delta_max:
                break
            landmarks.append(
                Landmark(
                    anchor_time=anchor.time,
                    anchor_freq=anchor.freq,
                    paired_freq=target.freq,
                    delta_time=dt,
                )
            )
            paired += 1
    return landmarks


def hash_landmarks(landmarks: list[Landmark]) -> list[tuple[int, int]]:
    return [(lm.hash_value(), lm.anchor_time) for lm in landmarks]


class ConstellationMap:
    """Constellation map: find spectral peaks and form landmark pairs."""

    def __init__(
        self,
        threshold: float = 0.01,
        neighborhood_size: int = 5,
        fan_out: int = 5,
        time_delta_max: int = 64,
    ) -> None:
        self.threshold = threshold
        self.neighborhood_size = neighborhood_size
        self.fan_out = fan_out
        self.time_delta_max = time_delta_max

    def extract(self, spectrogram: np.ndarray) -> list[tuple[int, int]]:
        peaks = find_peaks(spectrogram, self.threshold, self.neighborhood_size)
        landmarks = form_pairs(peaks, self.fan_out, self.time_delta_max)
        return hash_landmarks(landmarks)


class LandmarkIndex:
    """Inverted index for landmark-hash based audio identification."""

    def __init__(self) -> None:
        self._index: dict[int, list[tuple[str, int]]] = {}
        self._track_count: int = 0

    def add_track(self, track_id: str, hashes: list[tuple[int, int]]) -> None:
        for h, offset in hashes:
            self._index.setdefault(h, []).append((track_id, offset))
        self._track_count += 1

    @property
    def n_tracks(self) -> int:
        return self._track_count

    @property
    def n_hashes(self) -> int:
        return sum(len(v) for v in self._index.values())

    def query(
        self,
        hashes: list[tuple[int, int]],
        min_matches: int = 3,
    ) -> list[Match]:
        candidates: dict[str, list[tuple[int, int]]] = {}
        for h, query_offset in hashes:
            if h not in self._index:
                continue
            for track_id, db_offset in self._index[h]:
                candidates.setdefault(track_id, []).append((query_offset, db_offset))

        results = []
        for track_id, offsets in candidates.items():
            if len(offsets) < min_matches:
                continue
            deltas = [db_off - q_off for q_off, db_off in offsets]
            if not deltas:
                continue
            from collections import Counter
            delta_counts = Counter(deltas)
            best_delta, best_count = delta_counts.most_common(1)[0]
            score = best_count / max(len(hashes), 1)
            results.append(
                Match(
                    track_id=track_id,
                    score=score,
                    offset=best_delta,
                    n_matches=best_count,
                )
            )
        return sorted(results, key=lambda m: -m.score)
