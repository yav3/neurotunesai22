"""Robust audio hashing: noise-tolerant perceptual hashing with error correction."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RobustHash:
    hash_bits: np.ndarray
    sub_hashes: List[np.ndarray] = field(default_factory=list)
    n_frames: int = 0
    confidence: float = 0.0


@dataclass
class MatchResult:
    distance: int
    similarity: float
    offset_frames: int = 0
    matched: bool = False


BARK_BANDS = [
    (20, 100), (100, 200), (200, 300), (300, 400), (400, 510),
    (510, 630), (630, 770), (770, 920), (920, 1080), (1080, 1270),
    (1270, 1480), (1480, 1720), (1720, 2000), (2000, 2320),
    (2320, 2700), (2700, 3150), (3150, 3700), (3700, 4400),
    (4400, 5300), (5300, 6400), (6400, 7700), (7700, 9500),
    (9500, 12000), (12000, 15500),
]


def bark_band_energies(
    spectrum: np.ndarray, sr: int, n_fft: int,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    n_bands = len(BARK_BANDS)
    energies = np.zeros(n_bands)
    mag = np.abs(spectrum) ** 2
    for i, (lo, hi) in enumerate(BARK_BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            energies[i] = np.mean(mag[mask])
    return energies


def energy_difference_hash(
    spectrogram: np.ndarray, sr: int, n_fft: int, n_bits_per_frame: int = 24,
) -> np.ndarray:
    if spectrogram.ndim != 2:
        return np.zeros(0, dtype=np.uint8)
    n_frames = spectrogram.shape[1]
    all_bits = []
    for t in range(n_frames):
        energies = bark_band_energies(spectrogram[:, t], sr, n_fft)
        bits = np.zeros(min(n_bits_per_frame, len(energies) - 1), dtype=np.uint8)
        for i in range(len(bits)):
            bits[i] = 1 if energies[i + 1] > energies[i] else 0
        all_bits.append(bits)
    if not all_bits:
        return np.zeros(0, dtype=np.uint8)
    return np.concatenate(all_bits)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    min_len = min(len(a), len(b))
    if min_len == 0:
        return max(len(a), len(b))
    return int(np.sum(a[:min_len] != b[:min_len])) + abs(len(a) - len(b))


def bit_error_rate(a: np.ndarray, b: np.ndarray) -> float:
    total = max(len(a), len(b))
    if total == 0:
        return 0.0
    return hamming_distance(a, b) / total


def majority_vote_hash(sub_hashes: List[np.ndarray]) -> np.ndarray:
    if not sub_hashes:
        return np.zeros(0, dtype=np.uint8)
    max_len = max(len(h) for h in sub_hashes)
    votes = np.zeros(max_len, dtype=float)
    counts = np.zeros(max_len, dtype=float)
    for h in sub_hashes:
        votes[:len(h)] += h.astype(float)
        counts[:len(h)] += 1.0
    counts = np.maximum(counts, 1.0)
    return (votes / counts >= 0.5).astype(np.uint8)


def repetition_encode(bits: np.ndarray, repetitions: int = 3) -> np.ndarray:
    return np.repeat(bits, repetitions)


def repetition_decode(encoded: np.ndarray, repetitions: int = 3) -> np.ndarray:
    n_bits = len(encoded) // repetitions
    decoded = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        chunk = encoded[i * repetitions:(i + 1) * repetitions]
        decoded[i] = 1 if np.sum(chunk) > repetitions // 2 else 0
    return decoded


def compute_robust_hash(
    spectrogram: np.ndarray, sr: int = 44100, n_fft: int = 2048,
    n_sub_bands: int = 4,
) -> RobustHash:
    if spectrogram.ndim != 2 or spectrogram.shape[0] == 0 or spectrogram.shape[1] == 0:
        return RobustHash(hash_bits=np.zeros(0, dtype=np.uint8), confidence=0.0)
    n_freq = spectrogram.shape[0]
    band_size = max(1, n_freq // n_sub_bands)
    sub_hashes = []
    for b in range(n_sub_bands):
        start = b * band_size
        end = min(start + band_size, n_freq)
        sub_spec = spectrogram[start:end, :]
        if sub_spec.shape[0] == 0:
            continue
        padded = np.zeros((n_fft // 2 + 1, sub_spec.shape[1]))
        padded[:sub_spec.shape[0], :] = sub_spec
        bits = energy_difference_hash(padded, sr, n_fft)
        sub_hashes.append(bits)
    if not sub_hashes:
        return RobustHash(hash_bits=np.zeros(0, dtype=np.uint8), confidence=0.0)
    combined = majority_vote_hash(sub_hashes)
    max_len = max(len(h) for h in sub_hashes)
    agreement = np.zeros(max_len)
    for h in sub_hashes:
        agreement[:len(h)] += (h == combined[:len(h)]).astype(float)
    confidence = float(np.mean(agreement / len(sub_hashes)))
    return RobustHash(
        hash_bits=combined, sub_hashes=sub_hashes,
        n_frames=spectrogram.shape[1], confidence=confidence,
    )


def match_hashes(
    query: RobustHash, reference: RobustHash, threshold: float = 0.3,
) -> MatchResult:
    if len(query.hash_bits) == 0 or len(reference.hash_bits) == 0:
        return MatchResult(distance=max(len(query.hash_bits), len(reference.hash_bits)),
                          similarity=0.0, matched=False)
    dist = hamming_distance(query.hash_bits, reference.hash_bits)
    ber = bit_error_rate(query.hash_bits, reference.hash_bits)
    sim = 1.0 - ber
    return MatchResult(distance=dist, similarity=sim, matched=sim >= (1.0 - threshold))


def sliding_match(
    query: RobustHash, reference: RobustHash,
    bits_per_frame: int = 23, threshold: float = 0.3,
) -> MatchResult:
    q = query.hash_bits
    r = reference.hash_bits
    if len(q) == 0 or len(r) == 0 or len(q) > len(r):
        return MatchResult(distance=len(q), similarity=0.0, matched=False)
    best_sim = 0.0
    best_offset = 0
    step = max(1, bits_per_frame)
    for offset in range(0, len(r) - len(q) + 1, step):
        chunk = r[offset:offset + len(q)]
        ber = bit_error_rate(q, chunk)
        sim = 1.0 - ber
        if sim > best_sim:
            best_sim = sim
            best_offset = offset // step
    dist = int((1.0 - best_sim) * len(q))
    return MatchResult(
        distance=dist, similarity=best_sim,
        offset_frames=best_offset, matched=best_sim >= (1.0 - threshold),
    )


def hash_to_hex(bits: np.ndarray) -> str:
    byte_chunks = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        val = 0
        for j, b in enumerate(chunk):
            val |= int(b) << (7 - j)
        byte_chunks.append(val)
    return bytes(byte_chunks).hex()
