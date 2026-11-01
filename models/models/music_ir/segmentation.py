"""Structural segmentation and chroma analysis (complements onset.py)."""
from __future__ import annotations

from typing import Any

import numpy as np


class StructuralSegmenter:
    """Self-similarity matrix segmentation for section boundary detection."""

    def __init__(self, n_fft: int = 2048, hop: int = 1024):
        self.n_fft = n_fft
        self.hop = hop

    def segment(
        self,
        signal: np.ndarray,
        sr: int = 22050,
        n_segments: int = 4,
    ) -> list[dict[str, Any]]:
        features = self._extract_frame_features(signal, sr)
        if len(features) < 2:
            return [{"start": 0.0, "end": float(len(signal) / sr), "label": "A"}]

        ssm = self._self_similarity_matrix(features)
        boundaries = self._find_boundaries(ssm, n_segments)
        boundaries = [0] + boundaries + [len(features)]

        segments: list[dict[str, Any]] = []
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i] * self.hop / sr
            end_time = boundaries[i + 1] * self.hop / sr
            segments.append({
                "start": float(start_time),
                "end": float(end_time),
                "label": labels[i % len(labels)],
                "frame_start": boundaries[i],
                "frame_end": boundaries[i + 1],
            })
        return segments

    def self_similarity(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        features = self._extract_frame_features(signal, sr)
        return self._self_similarity_matrix(features)

    def _extract_frame_features(self, signal: np.ndarray, sr: int) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float64)
        window = np.hanning(self.n_fft)
        n_frames = max(1, 1 + (len(sig) - self.n_fft) // self.hop)
        n_mfcc = 13
        features = np.zeros((n_frames, n_mfcc), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop
            frame = sig[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            power = np.abs(np.fft.rfft(frame * window)) ** 2
            log_power = np.log(power + 1e-10)
            dct_out = np.fft.rfft(log_power).real
            features[i, :] = dct_out[:n_mfcc]
        return features

    @staticmethod
    def _self_similarity_matrix(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = features / norms
        return (normalized @ normalized.T).astype(np.float32)

    @staticmethod
    def _find_boundaries(ssm: np.ndarray, n_segments: int) -> list[int]:
        n = ssm.shape[0]
        if n < 2 or n_segments <= 1:
            return []
        novelty = np.zeros(n, dtype=np.float64)
        kernel_size = max(4, n // 20)
        for i in range(kernel_size, n - kernel_size):
            before = ssm[i - kernel_size:i, i - kernel_size:i]
            after = ssm[i:i + kernel_size, i:i + kernel_size]
            cross = ssm[i - kernel_size:i, i:i + kernel_size]
            novelty[i] = float(np.mean(before) + np.mean(after) - 2 * np.mean(cross))

        n_boundaries = min(n_segments - 1, max(1, n // 4))
        peaks: list[tuple[int, float]] = []
        for i in range(1, n - 1):
            if novelty[i] > novelty[i - 1] and novelty[i] >= novelty[i + 1]:
                peaks.append((i, novelty[i]))
        peaks.sort(key=lambda x: x[1], reverse=True)
        return sorted([p[0] for p in peaks[:n_boundaries]])


class ChromaAnalyzer:
    """Chroma (pitch class) feature extraction."""

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, n_fft: int = 4096, hop: int = 2048):
        self.n_fft = n_fft
        self.hop = hop

    def chromagram(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float64)
        window = np.hanning(self.n_fft)
        n_frames = max(1, 1 + (len(sig) - self.n_fft) // self.hop)
        chroma = np.zeros((12, n_frames), dtype=np.float32)
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sr)

        pitch_bins = np.zeros(len(freqs), dtype=int)
        for i, f in enumerate(freqs):
            if f > 20:
                midi = 12 * np.log2(f / 440.0) + 69
                pitch_bins[i] = int(round(midi)) % 12
            else:
                pitch_bins[i] = -1

        for i in range(n_frames):
            start = i * self.hop
            frame = sig[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            power = np.abs(np.fft.rfft(frame * window)) ** 2
            for j, pc in enumerate(pitch_bins):
                if pc >= 0:
                    chroma[pc, i] += power[j]

        col_norms = np.linalg.norm(chroma, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-12)
        chroma /= col_norms
        return chroma

    def key_profile(self, signal: np.ndarray, sr: int = 22050) -> dict[str, float]:
        cg = self.chromagram(signal, sr)
        avg = cg.mean(axis=1)
        total = avg.sum()
        if total < 1e-12:
            return {n: 0.0 for n in self.NOTE_NAMES}
        return {self.NOTE_NAMES[i]: float(avg[i] / total) for i in range(12)}

    def estimate_key(self, signal: np.ndarray, sr: int = 22050) -> tuple[str, str]:
        profile = self.key_profile(signal, sr)
        best_note = max(profile, key=profile.get)
        idx = self.NOTE_NAMES.index(best_note)
        minor_third = profile[self.NOTE_NAMES[(idx + 3) % 12]]
        major_third = profile[self.NOTE_NAMES[(idx + 4) % 12]]
        mode = "major" if major_third >= minor_third else "minor"
        return best_note, mode
