"""Audio fingerprinting: spectral peak constellation, hash generation, matching."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SpectralPeak:
    time_frame: int
    freq_bin: int
    magnitude: float


@dataclass(frozen=True)
class FingerprintHash:
    anchor_freq: int
    target_freq: int
    delta_time: int
    anchor_time: int

    @property
    def hash_value(self) -> str:
        raw = f"{self.anchor_freq}|{self.target_freq}|{self.delta_time}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class FingerprintDatabase:
    index: dict[str, list[tuple[str, int]]] = field(default_factory=dict)

    def insert(self, track_id: str, hashes: Sequence[FingerprintHash]) -> int:
        count = 0
        for h in hashes:
            key = h.hash_value
            if key not in self.index:
                self.index[key] = []
            self.index[key].append((track_id, h.anchor_time))
            count += 1
        return count

    def query(self, hashes: Sequence[FingerprintHash],
              min_matches: int = 5) -> list[tuple[str, int, float]]:
        votes: dict[str, dict[int, int]] = {}
        for h in hashes:
            key = h.hash_value
            if key not in self.index:
                continue
            for track_id, stored_time in self.index[key]:
                offset = stored_time - h.anchor_time
                if track_id not in votes:
                    votes[track_id] = {}
                votes[track_id][offset] = votes[track_id].get(offset, 0) + 1
        results: list[tuple[str, int, float]] = []
        for track_id, offsets in votes.items():
            best_offset = max(offsets, key=offsets.__getitem__)
            count = offsets[best_offset]
            if count >= min_matches:
                confidence = min(1.0, count / max(len(hashes), 1))
                results.append((track_id, best_offset, confidence))
        results.sort(key=lambda x: -x[2])
        return results


def find_spectral_peaks(spectrogram: np.ndarray,
                        n_peaks_per_frame: int = 5,
                        min_magnitude: float = 0.01) -> list[SpectralPeak]:
    peaks: list[SpectralPeak] = []
    n_freq, n_time = spectrogram.shape
    for t in range(n_time):
        frame = spectrogram[:, t]
        indices = np.argsort(frame)[::-1][:n_peaks_per_frame]
        for idx in indices:
            if frame[idx] >= min_magnitude:
                peaks.append(SpectralPeak(t, int(idx), float(frame[idx])))
    return peaks


def generate_fingerprints(peaks: list[SpectralPeak],
                          fan_out: int = 10,
                          max_delta: int = 200,
                          min_delta: int = 1) -> list[FingerprintHash]:
    sorted_peaks = sorted(peaks, key=lambda p: (p.time_frame, p.freq_bin))
    hashes: list[FingerprintHash] = []
    for i, anchor in enumerate(sorted_peaks):
        targets_added = 0
        for j in range(i + 1, len(sorted_peaks)):
            if targets_added >= fan_out:
                break
            target = sorted_peaks[j]
            dt = target.time_frame - anchor.time_frame
            if dt < min_delta:
                continue
            if dt > max_delta:
                break
            hashes.append(FingerprintHash(
                anchor_freq=anchor.freq_bin,
                target_freq=target.freq_bin,
                delta_time=dt,
                anchor_time=anchor.time_frame,
            ))
            targets_added += 1
    return hashes


def fingerprint_similarity(hashes_a: Sequence[FingerprintHash],
                           hashes_b: Sequence[FingerprintHash]) -> float:
    set_a = {h.hash_value for h in hashes_a}
    set_b = {h.hash_value for h in hashes_b}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
