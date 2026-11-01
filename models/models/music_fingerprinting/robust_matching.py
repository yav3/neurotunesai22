"""Robust audio fingerprinting: spectral peak extraction, combinatorial hashing, histogram-based matching."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SpectralPeak:
    time_bin: int
    freq_bin: int
    magnitude: float


def find_spectral_peaks(
    spectrogram: np.ndarray,
    threshold_db: float = -60.0,
    min_freq_distance: int = 3,
    min_time_distance: int = 3,
) -> list[SpectralPeak]:
    mag = np.abs(spectrogram)
    mag_db = 20 * np.log10(mag + 1e-10)
    n_freq, n_time = mag_db.shape
    peaks = []
    for t in range(n_time):
        for f in range(n_freq):
            if mag_db[f, t] < threshold_db:
                continue
            f_lo = max(0, f - min_freq_distance)
            f_hi = min(n_freq, f + min_freq_distance + 1)
            t_lo = max(0, t - min_time_distance)
            t_hi = min(n_time, t + min_time_distance + 1)
            local = mag_db[f_lo:f_hi, t_lo:t_hi]
            if mag_db[f, t] >= np.max(local):
                peaks.append(SpectralPeak(time_bin=t, freq_bin=f, magnitude=float(mag_db[f, t])))
    return peaks


def generate_fingerprints(
    peaks: list[SpectralPeak], fan_out: int = 5, max_delta_t: int = 50
) -> list[tuple[int, int, int, int]]:
    sorted_peaks = sorted(peaks, key=lambda p: (p.time_bin, p.freq_bin))
    fingerprints = []
    for i, anchor in enumerate(sorted_peaks):
        paired = 0
        for j in range(i + 1, len(sorted_peaks)):
            target = sorted_peaks[j]
            dt = target.time_bin - anchor.time_bin
            if dt <= 0:
                continue
            if dt > max_delta_t:
                break
            fingerprints.append((anchor.freq_bin, target.freq_bin, dt, anchor.time_bin))
            paired += 1
            if paired >= fan_out:
                break
    return fingerprints


def hash_fingerprint(f1: int, f2: int, dt: int) -> int:
    return ((f1 & 0x3FF) << 20) | ((f2 & 0x3FF) << 10) | (dt & 0x3FF)


class FingerprintDB:
    def __init__(self) -> None:
        self._db: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._track_count = 0

    def insert(self, track_id: int, fingerprints: list[tuple[int, int, int, int]]) -> None:
        for f1, f2, dt, t_anchor in fingerprints:
            h = hash_fingerprint(f1, f2, dt)
            self._db[h].append((track_id, t_anchor))
        self._track_count = max(self._track_count, track_id + 1)

    def query(
        self, fingerprints: list[tuple[int, int, int, int]], min_matches: int = 3
    ) -> list[tuple[int, float, int]]:
        offset_histograms: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for f1, f2, dt, t_query in fingerprints:
            h = hash_fingerprint(f1, f2, dt)
            for track_id, t_ref in self._db.get(h, []):
                offset = t_ref - t_query
                offset_histograms[track_id][offset] += 1
        results = []
        total_q = max(len(fingerprints), 1)
        for track_id, hist in offset_histograms.items():
            best_offset = max(hist, key=hist.get)
            count = hist[best_offset]
            if count >= min_matches:
                score = count / total_q
                results.append((track_id, float(score), best_offset))
        return sorted(results, key=lambda x: -x[1])


def time_offset_histogram(matches: list[tuple[int, int]]) -> dict[int, int]:
    hist: dict[int, int] = defaultdict(int)
    for t_ref, t_query in matches:
        hist[t_ref - t_query] += 1
    return dict(hist)


def match_score(histogram: dict[int, int], total_query_prints: int) -> float:
    if not histogram or total_query_prints <= 0:
        return 0.0
    peak = max(histogram.values())
    return peak / total_query_prints


def noise_robustness_test(
    fingerprints_clean: list[tuple[int, int, int, int]],
    fingerprints_noisy: list[tuple[int, int, int, int]],
    db: FingerprintDB,
    track_id: int = 0,
) -> float:
    db.insert(track_id, fingerprints_clean)
    results = db.query(fingerprints_noisy, min_matches=1)
    for tid, score, offset in results:
        if tid == track_id:
            return score
    return 0.0
