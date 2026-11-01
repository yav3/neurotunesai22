"""Onset detection, beat tracking, segmentation, and tempo analysis."""
from __future__ import annotations

from typing import Any

import numpy as np


class OnsetDetector:
    def __init__(self, n_fft: int = 1024, hop: int = 512) -> None:
        self._n_fft = n_fft
        self._hop = hop

    def spectral_flux(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        spec = self._stft_mag(signal)
        if spec.shape[1] < 2:
            return np.zeros(0)
        diff = np.diff(spec, axis=1)
        diff[diff < 0] = 0
        flux = np.sum(diff, axis=0)
        return flux

    def energy(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        n_frames = max(1, 1 + (len(signal) - self._n_fft) // self._hop)
        energies = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * self._hop
            end = start + self._n_fft
            frame = signal[start:end]
            energies[i] = np.sum(frame ** 2)
        if len(energies) < 2:
            return np.zeros(0)
        diff = np.diff(energies)
        diff[diff < 0] = 0
        return diff

    def complex_domain(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        window = np.hanning(self._n_fft)
        n_frames = max(1, 1 + (len(signal) - self._n_fft) // self._hop)
        stft = np.zeros((self._n_fft // 2 + 1, n_frames), dtype=complex)
        for i in range(n_frames):
            start = i * self._hop
            end = start + self._n_fft
            frame = signal[start:end]
            if len(frame) < self._n_fft:
                frame = np.pad(frame, (0, self._n_fft - len(frame)))
            stft[:, i] = np.fft.rfft(frame * window)
        if n_frames < 3:
            return np.zeros(max(0, n_frames - 1))
        odf = np.zeros(n_frames - 1)
        for i in range(1, n_frames):
            predicted_mag = np.abs(stft[:, i - 1])
            predicted_phase = np.angle(stft[:, i - 1])
            if i >= 2:
                predicted_phase = 2 * np.angle(stft[:, i - 1]) - np.angle(stft[:, i - 2])
            predicted = predicted_mag * np.exp(1j * predicted_phase)
            odf[i - 1] = np.sum(np.abs(stft[:, i] - predicted))
        return odf

    def pick_onsets(
        self, odf: np.ndarray, threshold: float = 0.0, min_spacing: int = 5,
    ) -> list[int]:
        if odf.size == 0:
            return []
        if threshold <= 0:
            threshold = float(np.mean(odf) + 0.5 * np.std(odf))
        peaks: list[int] = []
        for i in range(1, len(odf) - 1):
            if odf[i] > odf[i - 1] and odf[i] > odf[i + 1] and odf[i] > threshold:
                if not peaks or (i - peaks[-1]) >= min_spacing:
                    peaks.append(i)
        return peaks

    def _stft_mag(self, signal: np.ndarray) -> np.ndarray:
        window = np.hanning(self._n_fft)
        n_frames = max(1, 1 + (len(signal) - self._n_fft) // self._hop)
        n_bins = self._n_fft // 2 + 1
        spec = np.zeros((n_bins, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self._hop
            end = start + self._n_fft
            frame = signal[start:end]
            if len(frame) < self._n_fft:
                frame = np.pad(frame, (0, self._n_fft - len(frame)))
            spec[:, i] = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        return spec


class BeatTracker:
    def __init__(self, n_fft: int = 1024, hop: int = 512) -> None:
        self._onset = OnsetDetector(n_fft, hop)
        self._hop = hop

    def track(self, signal: np.ndarray, sr: int = 22050) -> list[float]:
        odf = self._onset.spectral_flux(signal, sr)
        if odf.size < 4:
            return []
        onsets = self._onset.pick_onsets(odf)
        if len(onsets) < 2:
            return [float(o * self._hop / sr) for o in onsets]
        intervals = np.diff(onsets)
        median_interval = float(np.median(intervals))
        if median_interval < 1:
            return [float(o * self._hop / sr) for o in onsets]
        beats: list[float] = []
        for o in onsets:
            t = float(o * self._hop / sr)
            beats.append(t)
        return beats

    def estimate_tempo(self, signal: np.ndarray, sr: int = 22050) -> float:
        odf = self._onset.spectral_flux(signal, sr)
        if odf.size < 4:
            return 0.0
        onsets = self._onset.pick_onsets(odf)
        if len(onsets) < 2:
            return 0.0
        intervals = np.diff(onsets) * self._hop / sr
        intervals = intervals[intervals > 0.15]
        intervals = intervals[intervals < 2.0]
        if len(intervals) == 0:
            return 0.0
        median_ioi = float(np.median(intervals))
        if median_ioi < 1e-6:
            return 0.0
        return 60.0 / median_ioi


class SegmentBoundaryDetector:
    def __init__(self, n_fft: int = 4096, hop: int = 2048) -> None:
        self._n_fft = n_fft
        self._hop = hop

    def detect(
        self, signal: np.ndarray, sr: int = 22050, kernel_size: int = 16,
    ) -> list[float]:
        from .chromagram import ChromaExtractor
        extractor = ChromaExtractor(n_fft=self._n_fft, hop=self._hop)
        chroma = extractor.extract(signal, sr)
        if chroma.shape[1] < kernel_size * 2:
            return []
        ssm = self._self_similarity(chroma)
        novelty = self._checkerboard_kernel(ssm, kernel_size)
        peaks = self._find_peaks(novelty)
        return [float(p * self._hop / sr) for p in peaks]

    @staticmethod
    def _self_similarity(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=0, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normed = features / norms
        return normed.T @ normed

    @staticmethod
    def _checkerboard_kernel(ssm: np.ndarray, size: int) -> np.ndarray:
        n = ssm.shape[0]
        novelty = np.zeros(n)
        half = size // 2
        for i in range(half, n - half):
            tl = ssm[i - half : i, i - half : i].mean()
            br = ssm[i : i + half, i : i + half].mean()
            tr = ssm[i - half : i, i : i + half].mean()
            bl = ssm[i : i + half, i - half : i].mean()
            novelty[i] = (tl + br) - (tr + bl)
        return novelty

    @staticmethod
    def _find_peaks(curve: np.ndarray, threshold_factor: float = 0.5) -> list[int]:
        threshold = np.mean(curve) + threshold_factor * np.std(curve)
        peaks: list[int] = []
        for i in range(1, len(curve) - 1):
            if curve[i] > curve[i - 1] and curve[i] > curve[i + 1] and curve[i] > threshold:
                peaks.append(i)
        return peaks


class TempoHistogram:
    def __init__(self) -> None:
        self._bpm_range = (30.0, 300.0)
        self._n_bins = 270

    def build(self, onsets_seconds: list[float]) -> tuple[np.ndarray, np.ndarray]:
        if len(onsets_seconds) < 2:
            bins = np.linspace(self._bpm_range[0], self._bpm_range[1], self._n_bins + 1)
            return np.zeros(self._n_bins), bins
        iois = np.diff(onsets_seconds)
        iois = iois[iois > 0.1]
        iois = iois[iois < 4.0]
        if len(iois) == 0:
            bins = np.linspace(self._bpm_range[0], self._bpm_range[1], self._n_bins + 1)
            return np.zeros(self._n_bins), bins
        bpms = 60.0 / iois
        bpms = bpms[(bpms >= self._bpm_range[0]) & (bpms <= self._bpm_range[1])]
        bins = np.linspace(self._bpm_range[0], self._bpm_range[1], self._n_bins + 1)
        hist, _ = np.histogram(bpms, bins=bins)
        return hist.astype(np.float64), bins

    def dominant_tempo(self, onsets_seconds: list[float]) -> float:
        hist, bins = self.build(onsets_seconds)
        if hist.sum() < 1:
            return 0.0
        centers = (bins[:-1] + bins[1:]) / 2
        return float(centers[np.argmax(hist)])
