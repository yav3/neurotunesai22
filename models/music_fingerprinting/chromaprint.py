"""Chromaprint-style spectral fingerprinting via chroma energy quantization."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np


def _gray_code(n: int) -> int:
    return n ^ (n >> 1)


def _chroma_energy(spectrogram: np.ndarray, n_chroma: int = 12) -> np.ndarray:
    n_bins, n_frames = spectrogram.shape
    chroma = np.zeros((n_chroma, n_frames))
    for i in range(n_bins):
        chroma[i % n_chroma] += spectrogram[i] ** 2
    norms = chroma.sum(axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return chroma / norms


def _quantize_chroma(chroma: np.ndarray) -> np.ndarray:
    n_chroma, n_frames = chroma.shape
    quantized = np.zeros(n_frames, dtype=np.uint32)
    for t in range(n_frames):
        val = 0
        for c in range(min(n_chroma, 16)):
            if c + 1 < n_chroma and chroma[c, t] > chroma[c + 1, t]:
                val |= (1 << (2 * c))
            if chroma[c, t] > np.mean(chroma[:, t]):
                val |= (1 << (2 * c + 1))
        quantized[t] = _gray_code(val)
    return quantized


@dataclass
class SubFingerprint:
    hash_value: int
    offset: int


class SpectralFingerprinter:
    """Chromaprint-style fingerprinter using chroma energy quantization + gray code."""

    def __init__(self, n_chroma: int = 12, window_size: int = 16, hop_size: int = 4) -> None:
        self.n_chroma = n_chroma
        self.window_size = window_size
        self.hop_size = hop_size

    def fingerprint(self, spectrogram: np.ndarray) -> bytes:
        if spectrogram.ndim != 2 or spectrogram.shape[1] == 0:
            return b""
        chroma = _chroma_energy(spectrogram, self.n_chroma)
        quantized = _quantize_chroma(chroma)
        raw = b"".join(struct.pack("<I", int(q)) for q in quantized)
        return hashlib.sha256(raw).digest()

    def fingerprint_segments(
        self, spectrogram: np.ndarray, segment_size: int = 128
    ) -> list[bytes]:
        if spectrogram.ndim != 2 or spectrogram.shape[1] < segment_size:
            return [self.fingerprint(spectrogram)]
        fps = []
        n_frames = spectrogram.shape[1]
        step = max(1, segment_size // 2)
        for start in range(0, n_frames - segment_size + 1, step):
            seg = spectrogram[:, start : start + segment_size]
            fps.append(self.fingerprint(seg))
        return fps

    def sub_fingerprints(self, spectrogram: np.ndarray) -> list[SubFingerprint]:
        if spectrogram.ndim != 2 or spectrogram.shape[1] == 0:
            return []
        chroma = _chroma_energy(spectrogram, self.n_chroma)
        quantized = _quantize_chroma(chroma)
        n = len(quantized)
        subs = []
        for start in range(0, n - self.window_size + 1, self.hop_size):
            window = quantized[start : start + self.window_size]
            combined = 0
            for i, v in enumerate(window):
                combined ^= int(v) << (i % 8)
            subs.append(SubFingerprint(hash_value=combined, offset=start))
        return subs
