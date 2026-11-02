"""Audio fingerprinting via constellation maps (Shazam-like approach)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter


class SpectrogramPeaks:
    @staticmethod
    def find_peaks(
        spectrogram: np.ndarray,
        threshold: float = 0.0,
        neighborhood_size: int = 20,
    ) -> list[tuple[int, int]]:
        if spectrogram.size == 0:
            return []
        local_max = maximum_filter(spectrogram, size=neighborhood_size)
        mask = (spectrogram == local_max) & (spectrogram > threshold)
        coords = np.argwhere(mask)
        return [(int(r[1]), int(r[0])) for r in coords]


class FingerprintHash:
    @staticmethod
    def hash_pair(
        peak1: tuple[int, int],
        peak2: tuple[int, int],
        fan_out: int = 15,
    ) -> int:
        t1, f1 = peak1
        t2, f2 = peak2
        dt = t2 - t1
        return (f1 & 0x3FF) << 20 | (f2 & 0x3FF) << 10 | (dt & 0x3FF)


class ConstellationMap:
    def __init__(self) -> None:
        self._peaks: list[tuple[int, int]] = []
        self._shape: tuple[int, int] = (0, 0)

    def build(
        self,
        signal: np.ndarray,
        sr: int = 22050,
        n_peaks: int = 200,
        n_fft: int = 1024,
        hop: int = 512,
    ) -> list[tuple[int, int]]:
        spec = self._stft_mag(signal, n_fft, hop)
        self._shape = spec.shape
        all_peaks = SpectrogramPeaks.find_peaks(spec, threshold=np.median(spec))
        all_peaks.sort(key=lambda p: spec[p[1], p[0]], reverse=True)
        self._peaks = all_peaks[:n_peaks]
        return list(self._peaks)

    def visualize(self) -> np.ndarray:
        if not self._peaks or self._shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)
        vis = np.zeros(self._shape, dtype=np.float32)
        for t, f in self._peaks:
            if 0 <= f < vis.shape[0] and 0 <= t < vis.shape[1]:
                vis[f, t] = 1.0
        return vis

    @staticmethod
    def _stft_mag(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
        window = np.hanning(n_fft)
        n_frames = max(1, 1 + (len(signal) - n_fft) // hop)
        n_bins = n_fft // 2 + 1
        spec = np.zeros((n_bins, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            end = start + n_fft
            frame = signal[start:end]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            windowed = frame * window
            spec[:, i] = np.abs(np.fft.rfft(windowed)).astype(np.float32)
        return spec


class AudioFingerprinter:
    def __init__(self, fan_out: int = 15, n_fft: int = 1024, hop: int = 512) -> None:
        self._fan_out = fan_out
        self._n_fft = n_fft
        self._hop = hop

    def fingerprint(self, signal: np.ndarray, sr: int = 22050) -> list[int]:
        cmap = ConstellationMap()
        peaks = cmap.build(signal, sr, n_peaks=300, n_fft=self._n_fft, hop=self._hop)
        peaks.sort(key=lambda p: p[0])
        hashes: list[int] = []
        for i, p1 in enumerate(peaks):
            targets = peaks[i + 1 : i + 1 + self._fan_out]
            for p2 in targets:
                if p2[0] > p1[0]:
                    h = FingerprintHash.hash_pair(p1, p2, self._fan_out)
                    hashes.append(h)
        return hashes

    def match(
        self,
        fp_query: list[int],
        fp_database: dict[str, list[int]],
    ) -> list[tuple[str, float]]:
        if not fp_query or not fp_database:
            return []
        query_set = Counter(fp_query)
        results: list[tuple[str, float]] = []
        for track_id, fp_db in fp_database.items():
            db_set = Counter(fp_db)
            shared = sum((query_set & db_set).values())
            total = max(len(fp_query), 1)
            confidence = shared / total
            if confidence > 0:
                results.append((track_id, confidence))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
