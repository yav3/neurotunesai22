"""Perceptual audio hashing for near-duplicate detection and content identification."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np


@dataclass
class AudioHash:
    hash_bytes: bytes
    duration_s: float
    sample_rate: int

    @property
    def hex(self) -> str:
        return self.hash_bytes.hex()

    def hamming_distance(self, other: AudioHash) -> int:
        a = int.from_bytes(self.hash_bytes[:8], "big")
        b = int.from_bytes(other.hash_bytes[:8], "big")
        return bin(a ^ b).count("1")


def spectral_flatness(spectrum: np.ndarray) -> float:
    spectrum = np.maximum(spectrum, 1e-10)
    geo = np.exp(np.mean(np.log(spectrum)))
    arith = np.mean(spectrum)
    return geo / arith if arith > 0 else 0.0


def bark_bands(spectrum: np.ndarray, sr: int = 44100, n_bands: int = 24) -> np.ndarray:
    n = len(spectrum)
    freqs = np.linspace(0, sr / 2, n)
    bark = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500) ** 2)
    band_edges = np.linspace(0, bark[-1], n_bands + 1)
    bands = np.zeros(n_bands)
    for i in range(n_bands):
        mask = (bark >= band_edges[i]) & (bark < band_edges[i + 1])
        if mask.any():
            bands[i] = np.mean(spectrum[mask])
    return bands


def perceptual_hash(
    spectrogram: np.ndarray,
    sr: int = 44100,
    n_bands: int = 24,
    n_frames: int = 32,
) -> AudioHash:
    if spectrogram.ndim != 2 or spectrogram.shape[1] == 0:
        return AudioHash(b"\x00" * 32, 0.0, sr)

    n_bins, total_frames = spectrogram.shape
    duration = total_frames * 512 / sr

    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    features = []
    for fi in frame_indices:
        col = spectrogram[:, fi]
        bands = bark_bands(col, sr, n_bands)
        features.append(bands)

    feature_matrix = np.array(features)
    median_val = np.median(feature_matrix)
    bits = (feature_matrix.flatten() > median_val).astype(np.uint8)

    byte_count = (len(bits) + 7) // 8
    raw = bytearray(byte_count)
    for i, b in enumerate(bits):
        if b:
            raw[i // 8] |= 1 << (i % 8)

    hash_bytes = hashlib.sha256(bytes(raw)).digest()
    return AudioHash(hash_bytes, duration, sr)


def hash_similarity(h1: AudioHash, h2: AudioHash) -> float:
    dist = h1.hamming_distance(h2)
    max_bits = min(len(h1.hash_bytes), len(h2.hash_bytes)) * 8
    if max_bits == 0:
        return 0.0
    return 1.0 - dist / max_bits


def content_id(audio: np.ndarray, sr: int = 44100, hop: int = 512) -> str:
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    n_frames = len(audio) // hop
    energy = np.array([
        np.sum(audio[i * hop:(i + 1) * hop] ** 2)
        for i in range(n_frames)
    ])
    raw = struct.pack(f"<{len(energy)}f", *energy)
    return hashlib.sha256(raw).hexdigest()[:32]


@dataclass
class HashIndex:
    entries: dict[str, AudioHash]

    def __init__(self) -> None:
        self.entries = {}

    def add(self, track_id: str, h: AudioHash) -> None:
        self.entries[track_id] = h

    def query(self, h: AudioHash, threshold: float = 0.8) -> list[tuple[str, float]]:
        results = []
        for tid, stored in self.entries.items():
            sim = hash_similarity(h, stored)
            if sim >= threshold:
                results.append((tid, sim))
        results.sort(key=lambda x: -x[1])
        return results
