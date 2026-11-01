"""Perceptual hashing for audio: bark-scale energy hashing, tempo-invariant
fingerprints, sub-fingerprint matching, hash distance metrics.

Pure numpy — no external audio frameworks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PerceptualHash:
    hash_bits: np.ndarray
    frame_index: int
    hop_seconds: float

    @property
    def hex_str(self) -> str:
        bits = self.hash_bits.astype(np.uint8)
        n_bytes = (len(bits) + 7) // 8
        result = []
        for i in range(n_bytes):
            byte = 0
            for j in range(8):
                idx = i * 8 + j
                if idx < len(bits) and bits[idx]:
                    byte |= (1 << (7 - j))
            result.append(f"{byte:02x}")
        return "".join(result)


@dataclass
class HashMatch:
    query_frame: int
    reference_frame: int
    distance: int
    confidence: float


def bark_scale(f_hz: float) -> float:
    return 13.0 * math.atan(0.00076 * f_hz) + 3.5 * math.atan((f_hz / 7500.0) ** 2)


def bark_band_edges(n_bands: int = 24) -> np.ndarray:
    max_bark = bark_scale(22050.0)
    return np.linspace(0.0, max_bark, n_bands + 1)


def bark_band_energies(spectrum: np.ndarray, sr: float,
                       n_bands: int = 24) -> np.ndarray:
    n = len(spectrum)
    freqs = np.linspace(0, sr / 2, n)
    edges = bark_band_edges(n_bands)
    bark_freqs = np.array([bark_scale(f) for f in freqs])

    energies = np.zeros(n_bands)
    power = np.abs(spectrum) ** 2
    for b in range(n_bands):
        mask = (bark_freqs >= edges[b]) & (bark_freqs < edges[b + 1])
        if np.any(mask):
            energies[b] = np.sum(power[mask])
    return energies


def _energy_difference_hash(energies: np.ndarray) -> np.ndarray:
    n = len(energies)
    bits = np.zeros(n - 1, dtype=np.uint8)
    for i in range(n - 1):
        if energies[i + 1] > energies[i]:
            bits[i] = 1
    return bits


def compute_frame_hash(spectrum: np.ndarray, sr: float,
                       n_bands: int = 32) -> np.ndarray:
    energies = bark_band_energies(spectrum, sr, n_bands)
    log_e = np.log(energies + 1e-10)
    return _energy_difference_hash(log_e)


def compute_perceptual_hash(spectrogram: np.ndarray, sr: float,
                            hop_size: int = 512,
                            n_bands: int = 32) -> List[PerceptualHash]:
    n_frames = spectrogram.shape[1] if spectrogram.ndim == 2 else 1
    hop_sec = hop_size / sr
    hashes = []

    if spectrogram.ndim == 1:
        bits = compute_frame_hash(spectrogram, sr, n_bands)
        hashes.append(PerceptualHash(hash_bits=bits, frame_index=0, hop_seconds=hop_sec))
    else:
        for i in range(n_frames):
            bits = compute_frame_hash(spectrogram[:, i], sr, n_bands)
            hashes.append(PerceptualHash(hash_bits=bits, frame_index=i,
                                         hop_seconds=hop_sec))
    return hashes


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    return int(np.sum(a[:n] != b[:n]))


def normalized_hamming(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return hamming_distance(a, b) / n


def bit_error_rate(a: np.ndarray, b: np.ndarray) -> float:
    return normalized_hamming(a, b)


def match_hashes(query: List[PerceptualHash],
                 reference: List[PerceptualHash],
                 max_distance: int = 5) -> List[HashMatch]:
    matches = []
    for qh in query:
        for rh in reference:
            d = hamming_distance(qh.hash_bits, rh.hash_bits)
            if d <= max_distance:
                n_bits = len(qh.hash_bits)
                confidence = 1.0 - d / max(n_bits, 1)
                matches.append(HashMatch(
                    query_frame=qh.frame_index,
                    reference_frame=rh.frame_index,
                    distance=d, confidence=confidence,
                ))
    return matches


def find_offset(matches: List[HashMatch],
                hop_seconds: float) -> Tuple[float, float]:
    if not matches:
        return 0.0, 0.0

    offsets = np.array([
        (m.reference_frame - m.query_frame) * hop_seconds for m in matches
    ])
    confidences = np.array([m.confidence for m in matches])

    hist_bins = max(10, len(offsets) // 2)
    counts, edges = np.histogram(offsets, bins=hist_bins)
    best_bin = np.argmax(counts)
    center = 0.5 * (edges[best_bin] + edges[best_bin + 1])

    in_bin = np.abs(offsets - center) < (edges[1] - edges[0])
    if np.any(in_bin):
        avg_conf = float(np.mean(confidences[in_bin]))
    else:
        avg_conf = 0.0
    return float(center), avg_conf


def tempo_invariant_hash(chroma: np.ndarray, n_bits: int = 64) -> np.ndarray:
    if chroma.ndim == 1:
        chroma = chroma.reshape(-1, 1)
    mean_chroma = np.mean(chroma, axis=1)
    median_val = np.median(mean_chroma)
    base_bits = (mean_chroma > median_val).astype(np.uint8)

    if chroma.shape[1] > 1:
        diff = np.diff(chroma, axis=1)
        mean_diff = np.mean(diff, axis=1)
        diff_bits = (mean_diff > 0).astype(np.uint8)
    else:
        diff_bits = np.zeros(chroma.shape[0], dtype=np.uint8)

    combined = np.concatenate([base_bits, diff_bits])
    if len(combined) >= n_bits:
        return combined[:n_bits]
    padded = np.zeros(n_bits, dtype=np.uint8)
    padded[:len(combined)] = combined
    return padded


def hash_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - normalized_hamming(a, b)


def locality_sensitive_hash(features: np.ndarray, n_hashes: int = 128,
                            seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = len(features)
    planes = rng.standard_normal((n_hashes, dim))
    return (planes @ features > 0).astype(np.uint8)


def hash_database_search(query_hash: np.ndarray,
                         database: List[Tuple[str, np.ndarray]],
                         threshold: float = 0.8) -> List[Tuple[str, float]]:
    results = []
    for name, ref_hash in database:
        sim = hash_similarity(query_hash, ref_hash)
        if sim >= threshold:
            results.append((name, sim))
    results.sort(key=lambda x: -x[1])
    return results


def sub_fingerprint_sequence(hashes: List[PerceptualHash],
                             window: int = 4) -> List[np.ndarray]:
    sequences = []
    for i in range(len(hashes) - window + 1):
        combined = np.concatenate([h.hash_bits for h in hashes[i:i + window]])
        sequences.append(combined)
    return sequences
