"""Landmark-based audio fingerprinting (Shazam-style).

Peak-pair landmarks from a spectrogram are hashed into compact integers.
A database maps hashes → (track_id, time_offset) for O(1) lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Landmark:
    """A time-frequency peak pair used for fingerprinting."""
    time1: int
    freq1: int
    time2: int
    freq2: int

    @property
    def dt(self) -> int:
        return self.time2 - self.time1


@dataclass
class MatchResult:
    track_id: str
    offset_frames: int
    n_matches: int
    confidence: float


def find_spectral_peaks(
    spectrogram: np.ndarray,
    threshold: float = 0.0,
    max_peaks_per_frame: int = 10,
) -> list[tuple[int, int]]:
    """Find (time, freq) peaks above *threshold* in a magnitude spectrogram.

    Parameters
    ----------
    spectrogram : (n_frames, n_freq_bins)
    threshold : absolute minimum magnitude
    max_peaks_per_frame : keep only the N strongest per frame
    """
    peaks: list[tuple[int, int]] = []
    n_frames, n_bins = spectrogram.shape
    for t in range(n_frames):
        frame = spectrogram[t]
        candidates: list[tuple[float, int]] = []
        for f in range(1, n_bins - 1):
            val = frame[f]
            if val > threshold and val >= frame[f - 1] and val >= frame[f + 1]:
                candidates.append((val, f))
        candidates.sort(reverse=True)
        for _, f in candidates[:max_peaks_per_frame]:
            peaks.append((t, f))
    return peaks


def generate_landmarks(
    peaks: Sequence[tuple[int, int]],
    fan_value: int = 15,
    max_dt: int = 50,
) -> list[Landmark]:
    """Create peak-pair landmarks by fanning out from each anchor peak.

    For each peak, pair it with up to *fan_value* later peaks within
    *max_dt* frames.
    """
    sorted_peaks = sorted(peaks, key=lambda p: (p[0], p[1]))
    landmarks: list[Landmark] = []
    n = len(sorted_peaks)
    for i in range(n):
        t1, f1 = sorted_peaks[i]
        paired = 0
        for j in range(i + 1, n):
            if paired >= fan_value:
                break
            t2, f2 = sorted_peaks[j]
            dt = t2 - t1
            if dt <= 0:
                continue
            if dt > max_dt:
                break
            landmarks.append(Landmark(time1=t1, freq1=f1, time2=t2, freq2=f2))
            paired += 1
    return landmarks


def hash_landmark(lm: Landmark, freq_bits: int = 10, dt_bits: int = 10) -> int:
    """Pack a landmark into a single integer hash.

    Layout: [freq1 : freq_bits][freq2 : freq_bits][dt : dt_bits].
    """
    freq_mask = (1 << freq_bits) - 1
    dt_mask = (1 << dt_bits) - 1
    return (
        ((lm.freq1 & freq_mask) << (freq_bits + dt_bits))
        | ((lm.freq2 & freq_mask) << dt_bits)
        | (lm.dt & dt_mask)
    )


def build_fingerprint_db(
    tracks: dict[str, list[Landmark]],
) -> dict[int, list[tuple[str, int]]]:
    """Build a hash → [(track_id, anchor_time)] lookup table."""
    db: dict[int, list[tuple[str, int]]] = {}
    for track_id, landmarks in tracks.items():
        for lm in landmarks:
            h = hash_landmark(lm)
            db.setdefault(h, []).append((track_id, lm.time1))
    return db


def query_fingerprint(
    query_landmarks: list[Landmark],
    db: dict[int, list[tuple[str, int]]],
    min_matches: int = 5,
) -> list[MatchResult]:
    """Query a landmark database and return ranked matches.

    Groups matching landmarks by (track_id, time_offset_delta) and ranks
    by match count.
    """
    vote: dict[tuple[str, int], int] = {}
    for lm in query_landmarks:
        h = hash_landmark(lm)
        if h not in db:
            continue
        for track_id, ref_time in db[h]:
            delta = ref_time - lm.time1
            key = (track_id, delta)
            vote[key] = vote.get(key, 0) + 1

    if not vote:
        return []

    track_best: dict[str, tuple[int, int]] = {}
    for (tid, offset), count in vote.items():
        if tid not in track_best or count > track_best[tid][1]:
            track_best[tid] = (offset, count)

    total_query = max(len(query_landmarks), 1)
    results: list[MatchResult] = []
    for tid, (offset, count) in track_best.items():
        if count < min_matches:
            continue
        results.append(MatchResult(
            track_id=tid,
            offset_frames=offset,
            n_matches=count,
            confidence=min(count / total_query, 1.0),
        ))
    results.sort(key=lambda r: -r.confidence)
    return results
