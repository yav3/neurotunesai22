"""Audio fingerprinting: constellation maps, landmark hashing, query matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import stft as scipy_stft

__all__ = [
    "SpectralPeak", "Landmark", "FingerprintMatch", "FingerprintDB",
    "find_spectral_peaks", "form_landmarks", "hash_landmark",
    "fingerprint_audio", "query_fingerprints", "time_coherence_score",
]


@dataclass
class SpectralPeak:
    time_idx: int
    freq_bin: int
    magnitude: float


@dataclass
class Landmark:
    anchor_time: int
    anchor_freq: int
    target_freq: int
    delta_t: int


@dataclass
class FingerprintMatch:
    track_id: str
    score: int
    time_offset: float
    coherence: float


class FingerprintDB:
    def __init__(self):
        self._table: Dict[int, List[Tuple[str, int]]] = {}

    def insert(self, track_id: str, hashes: List[Tuple[int, int]]) -> None:
        for h, anchor_t in hashes:
            self._table.setdefault(h, []).append((track_id, anchor_t))

    def lookup(self, h: int) -> List[Tuple[str, int]]:
        return self._table.get(h, [])

    @property
    def n_hashes(self) -> int:
        return sum(len(v) for v in self._table.values())


def find_spectral_peaks(
    spectrogram: np.ndarray,
    neighborhood_time: int = 10,
    neighborhood_freq: int = 5,
    min_amplitude: float = 0.0,
    contrast_ratio: float = 1.5,
) -> List[SpectralPeak]:
    n_freq, n_time = spectrogram.shape
    size = (neighborhood_freq * 2 + 1, neighborhood_time * 2 + 1)
    local_max = maximum_filter(spectrogram, size=size)
    is_peak = (spectrogram == local_max) & (spectrogram > min_amplitude)
    if contrast_ratio > 1.0:
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(spectrogram, size=size)
        is_peak &= spectrogram > contrast_ratio * local_mean
    coords = np.argwhere(is_peak)
    peaks = []
    for freq_bin, time_idx in coords:
        peaks.append(SpectralPeak(int(time_idx), int(freq_bin), float(spectrogram[freq_bin, time_idx])))
    peaks.sort(key=lambda p: p.magnitude, reverse=True)
    return peaks


def form_landmarks(
    peaks: List[SpectralPeak],
    fan_out: int = 10,
    max_delta_t: int = 50,
    max_delta_f: int = 100,
) -> List[Landmark]:
    landmarks = []
    by_time = sorted(peaks, key=lambda p: p.time_idx)
    for i, anchor in enumerate(by_time):
        targets_found = 0
        for j in range(i + 1, len(by_time)):
            if targets_found >= fan_out:
                break
            target = by_time[j]
            dt = target.time_idx - anchor.time_idx
            if dt <= 0:
                continue
            if dt > max_delta_t:
                break
            df = abs(target.freq_bin - anchor.freq_bin)
            if df > max_delta_f:
                continue
            landmarks.append(Landmark(anchor.time_idx, anchor.freq_bin, target.freq_bin, dt))
            targets_found += 1
    return landmarks


def hash_landmark(lm: Landmark) -> int:
    f1 = lm.anchor_freq & 0x3FF
    f2 = lm.target_freq & 0x3FF
    dt = lm.delta_t & 0xFFF
    return (f1 << 22) | (f2 << 12) | dt


def fingerprint_audio(
    audio: np.ndarray, sr: int = 22050, n_fft: int = 2048, hop: int = 512, **peak_kwargs
) -> List[Tuple[int, int]]:
    _, _, Zxx = scipy_stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    spectrogram = np.abs(Zxx)
    peaks = find_spectral_peaks(spectrogram, **peak_kwargs)
    landmarks = form_landmarks(peaks)
    return [(hash_landmark(lm), lm.anchor_time) for lm in landmarks]


def time_coherence_score(offsets: List[float], bin_width: float = 1.0) -> float:
    if len(offsets) < 2:
        return 0.0
    arr = np.array(offsets)
    lo, hi = arr.min(), arr.max()
    if hi - lo < bin_width:
        return 1.0
    n_bins = max(int((hi - lo) / bin_width) + 1, 1)
    counts, _ = np.histogram(arr, bins=n_bins)
    peak = int(counts.max())
    return peak / len(offsets)


def query_fingerprints(
    query_hashes: List[Tuple[int, int]], db: FingerprintDB, top_k: int = 5
) -> List[FingerprintMatch]:
    track_offsets: Dict[str, List[float]] = {}
    for h, q_time in query_hashes:
        for track_id, db_time in db.lookup(h):
            offset = db_time - q_time
            track_offsets.setdefault(track_id, []).append(offset)
    results = []
    for track_id, offsets in track_offsets.items():
        score = len(offsets)
        coherence = time_coherence_score(offsets)
        median_offset = float(np.median(offsets))
        results.append(FingerprintMatch(track_id, score, median_offset, coherence))
    results.sort(key=lambda m: (m.coherence * m.score), reverse=True)
    return results[:top_k]
