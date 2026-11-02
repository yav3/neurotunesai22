"""Chroma analysis: extraction, distance, key detection, chord recognition."""
from __future__ import annotations

from typing import Any

import numpy as np


KRUMHANSL_KESSLER_MAJOR = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
], dtype=np.float64)

KRUMHANSL_KESSLER_MINOR = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
], dtype=np.float64)

PITCH_CLASSES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]

CHORD_TEMPLATES: dict[str, np.ndarray] = {}
for _i, _name in enumerate(PITCH_CLASSES):
    _major = np.zeros(12)
    _major[_i] = 1.0
    _major[(_i + 4) % 12] = 1.0
    _major[(_i + 7) % 12] = 1.0
    CHORD_TEMPLATES[f"{_name} major"] = _major

    _minor = np.zeros(12)
    _minor[_i] = 1.0
    _minor[(_i + 3) % 12] = 1.0
    _minor[(_i + 7) % 12] = 1.0
    CHORD_TEMPLATES[f"{_name} minor"] = _minor


class ChromaExtractor:
    def __init__(self, n_chroma: int = 12, n_fft: int = 4096, hop: int = 512) -> None:
        self._n_chroma = n_chroma
        self._n_fft = n_fft
        self._hop = hop

    def extract(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        if signal.size == 0:
            return np.zeros((self._n_chroma, 0), dtype=np.float32)
        window = np.hanning(self._n_fft)
        n_frames = max(1, 1 + (len(signal) - self._n_fft) // self._hop)
        n_bins = self._n_fft // 2 + 1
        chroma = np.zeros((self._n_chroma, n_frames), dtype=np.float32)
        freqs = np.fft.rfftfreq(self._n_fft, 1.0 / sr)
        chroma_map = self._build_chroma_map(freqs, sr)
        for i in range(n_frames):
            start = i * self._hop
            end = start + self._n_fft
            frame = signal[start:end]
            if len(frame) < self._n_fft:
                frame = np.pad(frame, (0, self._n_fft - len(frame)))
            spec = np.abs(np.fft.rfft(frame * window)) ** 2
            for c in range(self._n_chroma):
                chroma[c, i] = np.sum(spec[chroma_map == c])
        col_max = chroma.max(axis=0, keepdims=True)
        col_max[col_max < 1e-12] = 1.0
        chroma /= col_max
        return chroma

    def _build_chroma_map(self, freqs: np.ndarray, sr: int) -> np.ndarray:
        chroma_map = np.zeros(len(freqs), dtype=int)
        for i, f in enumerate(freqs):
            if f < 20:
                chroma_map[i] = 0
                continue
            midi = 69 + 12 * np.log2(f / 440.0)
            chroma_map[i] = int(round(midi)) % self._n_chroma
        return chroma_map


class ChromaDistance:
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten()
        b_flat = b.flatten()
        na, nb = np.linalg.norm(a_flat), np.linalg.norm(b_flat)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a_flat, b_flat) / (na * nb))

    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape[1] != b.shape[1]:
            n = min(a.shape[1], b.shape[1])
            a, b = a[:, :n], b[:, :n]
        return float(np.linalg.norm(a - b))

    @staticmethod
    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        a_mean = a.mean(axis=1)
        b_mean = b.mean(axis=1)
        na, nb = np.linalg.norm(a_mean), np.linalg.norm(b_mean)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a_mean, b_mean) / (na * nb))


class KeyDetector:
    def __init__(self) -> None:
        self._major_profile = KRUMHANSL_KESSLER_MAJOR
        self._minor_profile = KRUMHANSL_KESSLER_MINOR

    def detect(self, chroma: np.ndarray) -> tuple[str, str, float]:
        if chroma.size == 0:
            return "C", "major", 0.0
        chroma_sum = chroma.mean(axis=1)
        best_key, best_mode, best_corr = "C", "major", -1.0
        for shift in range(12):
            rolled = np.roll(chroma_sum, -shift)
            corr_maj = float(np.corrcoef(rolled, self._major_profile)[0, 1])
            corr_min = float(np.corrcoef(rolled, self._minor_profile)[0, 1])
            if corr_maj > best_corr:
                best_corr = corr_maj
                best_key = PITCH_CLASSES[shift]
                best_mode = "major"
            if corr_min > best_corr:
                best_corr = corr_min
                best_key = PITCH_CLASSES[shift]
                best_mode = "minor"
        return best_key, best_mode, max(0.0, best_corr)

    def detect_local(
        self, chroma: np.ndarray, window: int = 8,
    ) -> list[tuple[str, str, float]]:
        results: list[tuple[str, str, float]] = []
        n_frames = chroma.shape[1] if chroma.ndim == 2 else 0
        for i in range(0, n_frames, window):
            segment = chroma[:, i : i + window]
            results.append(self.detect(segment))
        return results


class ChordRecognizer:
    def __init__(self) -> None:
        self._templates = CHORD_TEMPLATES
        self._names = list(CHORD_TEMPLATES.keys())
        self._matrix = np.array(list(CHORD_TEMPLATES.values()))

    def recognize_frame(self, chroma_frame: np.ndarray) -> tuple[str, float]:
        if chroma_frame.size != 12:
            return "N", 0.0
        norms = np.linalg.norm(self._matrix, axis=1)
        frame_norm = np.linalg.norm(chroma_frame)
        if frame_norm < 1e-12:
            return "N", 0.0
        sims = self._matrix @ chroma_frame / (norms * frame_norm + 1e-12)
        best_idx = int(np.argmax(sims))
        return self._names[best_idx], float(sims[best_idx])

    def recognize_sequence(self, chroma: np.ndarray) -> list[tuple[str, float]]:
        if chroma.ndim != 2 or chroma.shape[0] != 12:
            return []
        return [self.recognize_frame(chroma[:, i]) for i in range(chroma.shape[1])]
