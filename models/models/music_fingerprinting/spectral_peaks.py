"""Spectral peak fingerprinting: constellation maps, combinatorial hashing (Wang 2003)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Peak:
    time_idx: int
    freq_idx: int


@dataclass
class FingerprintHash:
    hash_val: int
    time_offset: int


class SpectralPeakExtractor:
    """Shazam-style constellation map fingerprinting."""

    @staticmethod
    def find_peaks_2d(
        spectrogram: np.ndarray, neighborhood_size: int = 5
    ) -> list[Peak]:
        """Detect local maxima in 2D time-frequency representation."""
        from scipy.ndimage import maximum_filter

        n_freq, n_time = spectrogram.shape
        if n_freq < neighborhood_size or n_time < neighborhood_size:
            return []
        local_max = maximum_filter(spectrogram, size=neighborhood_size)
        is_peak = (spectrogram == local_max) & (spectrogram > 0)
        freq_ids, time_ids = np.where(is_peak)
        peaks = [Peak(int(t), int(f)) for f, t in zip(freq_ids, time_ids)]
        peaks.sort(key=lambda p: p.time_idx)
        return peaks

    @staticmethod
    def peak_pairs(
        peaks: list[Peak],
        target_zone_time: int = 50,
        target_zone_freq: int = 100,
        fan_out: int = 5,
    ) -> list[tuple[Peak, Peak]]:
        """Generate anchor-target pairs within a fan-out zone (Wang 2003)."""
        pairs = []
        for i, anchor in enumerate(peaks):
            count = 0
            for j in range(i + 1, len(peaks)):
                target = peaks[j]
                dt = target.time_idx - anchor.time_idx
                if dt > target_zone_time:
                    break
                if dt <= 0:
                    continue
                df = abs(target.freq_idx - anchor.freq_idx)
                if df <= target_zone_freq:
                    pairs.append((anchor, target))
                    count += 1
                    if count >= fan_out:
                        break
        return pairs

    @staticmethod
    def fingerprint_hash(f1: int, f2: int, dt: int) -> int:
        """Pack (f1, f2, dt) into a 32-bit hash."""
        return ((f1 & 0x3FF) << 20) | ((f2 & 0x3FF) << 10) | (dt & 0x3FF)

    @staticmethod
    def generate_fingerprints(
        signal: np.ndarray,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        neighborhood_size: int = 5,
    ) -> list[FingerprintHash]:
        """Full pipeline: STFT → peaks → pairs → hashes."""
        if len(signal) < n_fft:
            return []
        n_frames = 1 + (len(signal) - n_fft) // hop_length
        window = np.hanning(n_fft)
        mag = np.zeros((n_fft // 2 + 1, n_frames))
        for i in range(n_frames):
            frame = signal[i * hop_length : i * hop_length + n_fft] * window
            mag[:, i] = np.abs(np.fft.rfft(frame))
        peaks = SpectralPeakExtractor.find_peaks_2d(mag, neighborhood_size)
        pairs = SpectralPeakExtractor.peak_pairs(peaks)
        hashes = []
        for anchor, target in pairs:
            dt = target.time_idx - anchor.time_idx
            h = SpectralPeakExtractor.fingerprint_hash(
                anchor.freq_idx, target.freq_idx, dt
            )
            hashes.append(FingerprintHash(h, anchor.time_idx))
        return hashes

    @staticmethod
    def match_fingerprints(
        query_hashes: list[FingerprintHash],
        db_hashes: dict[int, list[tuple[int, int]]],
    ) -> list[tuple[int, int, float]]:
        """Histogram alignment matching.

        ``db_hashes`` maps hash_val → list of (song_id, time_offset).
        Returns list of (song_id, offset, confidence) sorted by confidence desc.
        """
        from collections import Counter

        offset_counts: dict[int, Counter] = {}
        for qh in query_hashes:
            matches = db_hashes.get(qh.hash_val, [])
            for song_id, db_offset in matches:
                delta = db_offset - qh.time_offset
                if song_id not in offset_counts:
                    offset_counts[song_id] = Counter()
                offset_counts[song_id][delta] += 1

        results = []
        for song_id, counter in offset_counts.items():
            best_offset, best_count = counter.most_common(1)[0]
            confidence = best_count / max(len(query_hashes), 1)
            results.append((song_id, best_offset, confidence))
        results.sort(key=lambda x: -x[2])
        return results

    @staticmethod
    def constellation_density(
        peaks: list[Peak],
        time_range: tuple[int, int] | None = None,
        freq_range: tuple[int, int] | None = None,
    ) -> float:
        """Peak density metric for signal quality estimation."""
        if not peaks:
            return 0.0
        filtered = peaks
        if time_range is not None:
            filtered = [p for p in filtered if time_range[0] <= p.time_idx <= time_range[1]]
        if freq_range is not None:
            filtered = [p for p in filtered if freq_range[0] <= p.freq_idx <= freq_range[1]]
        if not filtered:
            return 0.0
        t_vals = [p.time_idx for p in filtered]
        f_vals = [p.freq_idx for p in filtered]
        t_span = max(t_vals) - min(t_vals) + 1
        f_span = max(f_vals) - min(f_vals) + 1
        area = t_span * f_span
        return len(filtered) / max(area, 1)
