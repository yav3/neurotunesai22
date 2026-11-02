"""Spectral hashing for audio fingerprinting — locality-sensitive hashing in spectral space."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class HashConfig:
    """Configuration for spectral hash computation."""
    n_bits: int = 32
    frame_size: int = 2048
    hop_size: int = 512
    n_bands: int = 33
    sample_rate: float = 22050.0
    min_freq: float = 300.0
    max_freq: float = 3000.0


@dataclass
class FingerprintMatch:
    """Result of a fingerprint lookup."""
    query_offset: int
    reference_offset: int
    reference_id: str
    score: float
    n_matching_hashes: int


@dataclass
class HashDatabase:
    """In-memory hash lookup table."""
    table: Dict[int, List[Tuple[str, int]]] = field(default_factory=dict)
    n_entries: int = 0

    def insert(self, hash_val: int, ref_id: str, offset: int) -> None:
        if hash_val not in self.table:
            self.table[hash_val] = []
        self.table[hash_val].append((ref_id, offset))
        self.n_entries += 1

    def lookup(self, hash_val: int) -> List[Tuple[str, int]]:
        return self.table.get(hash_val, [])


def mel_frequencies(n_bands: int, f_min: float, f_max: float) -> np.ndarray:
    mel_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
    mels = np.linspace(mel_min, mel_max, n_bands + 2)
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def bark_scale_bands(n_bands: int, f_min: float, f_max: float) -> np.ndarray:
    bark_min = 13.0 * math.atan(0.00076 * f_min) + 3.5 * math.atan((f_min / 7500.0) ** 2)
    bark_max = 13.0 * math.atan(0.00076 * f_max) + 3.5 * math.atan((f_max / 7500.0) ** 2)
    barks = np.linspace(bark_min, bark_max, n_bands + 2)
    return (np.tan(barks / 13.0) / 0.00076 + np.tan(barks / 3.5) * 7500.0) / 2.0


def compute_band_energies(
    signal: np.ndarray,
    cfg: HashConfig,
) -> np.ndarray:
    n_frames = max(1, (len(signal) - cfg.frame_size) // cfg.hop_size + 1)
    freq_edges = mel_frequencies(cfg.n_bands, cfg.min_freq, cfg.max_freq)
    bin_edges = (freq_edges * cfg.frame_size / cfg.sample_rate).astype(int)
    bin_edges = np.clip(bin_edges, 0, cfg.frame_size // 2)

    energies = np.zeros((n_frames, cfg.n_bands))
    window = np.hanning(cfg.frame_size)

    for i in range(n_frames):
        start = i * cfg.hop_size
        frame = signal[start:start + cfg.frame_size]
        if len(frame) < cfg.frame_size:
            frame = np.pad(frame, (0, cfg.frame_size - len(frame)))
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        for b in range(cfg.n_bands):
            lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
            if hi > lo:
                energies[i, b] = np.sum(spectrum[lo:hi])
            else:
                energies[i, b] = spectrum[min(lo, len(spectrum) - 1)]

    return energies


def energy_difference_hash(energies: np.ndarray) -> np.ndarray:
    n_frames, n_bands = energies.shape
    if n_frames < 2:
        return np.zeros(max(n_frames, 1), dtype=np.uint32)

    hashes = np.zeros(n_frames - 1, dtype=np.uint32)
    n_bits = min(n_bands - 1, 32)
    for i in range(n_frames - 1):
        h = 0
        for b in range(n_bits):
            if energies[i + 1, b] - energies[i + 1, b + 1] > energies[i, b] - energies[i, b + 1]:
                h |= (1 << b)
        hashes[i] = h
    return hashes


def spectral_peak_hash(
    signal: np.ndarray,
    cfg: HashConfig,
    n_peaks: int = 5,
) -> np.ndarray:
    n_frames = max(1, (len(signal) - cfg.frame_size) // cfg.hop_size + 1)
    window = np.hanning(cfg.frame_size)
    hashes = np.zeros(n_frames, dtype=np.uint64)

    for i in range(n_frames):
        start = i * cfg.hop_size
        frame = signal[start:start + cfg.frame_size]
        if len(frame) < cfg.frame_size:
            frame = np.pad(frame, (0, cfg.frame_size - len(frame)))
        mag = np.abs(np.fft.rfft(frame * window))
        peak_bins = np.argsort(mag)[-n_peaks:]
        peak_bins_sorted = np.sort(peak_bins)
        h = 0
        for j, b in enumerate(peak_bins_sorted):
            h ^= int(b) << (j * 12)
        hashes[i] = h & 0xFFFFFFFFFFFFFFFF
    return hashes


def hamming_distance(a: int, b: int) -> int:
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def hash_similarity(a: int, b: int, n_bits: int = 32) -> float:
    return 1.0 - hamming_distance(a, b) / max(n_bits, 1)


def build_hash_database(
    hashes: np.ndarray,
    ref_id: str,
    db: Optional[HashDatabase] = None,
) -> HashDatabase:
    if db is None:
        db = HashDatabase()
    for offset, h in enumerate(hashes):
        db.insert(int(h), ref_id, offset)
    return db


def query_database(
    query_hashes: np.ndarray,
    db: HashDatabase,
    threshold: int = 3,
) -> List[FingerprintMatch]:
    offset_votes: Dict[Tuple[str, int], int] = {}
    for q_off, qh in enumerate(query_hashes):
        for ref_id, ref_off in db.lookup(int(qh)):
            delta = ref_off - q_off
            key = (ref_id, delta)
            offset_votes[key] = offset_votes.get(key, 0) + 1

    matches = []
    for (ref_id, delta), count in offset_votes.items():
        if count >= threshold:
            matches.append(FingerprintMatch(
                query_offset=0,
                reference_offset=max(delta, 0),
                reference_id=ref_id,
                score=count / max(len(query_hashes), 1),
                n_matching_hashes=count,
            ))
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def fingerprint_robustness(
    original_hashes: np.ndarray,
    degraded_hashes: np.ndarray,
    n_bits: int = 32,
) -> float:
    n = min(len(original_hashes), len(degraded_hashes))
    if n == 0:
        return 0.0
    sims = [hash_similarity(int(original_hashes[i]), int(degraded_hashes[i]), n_bits) for i in range(n)]
    return float(np.mean(sims))


def bit_error_rate(
    original_hashes: np.ndarray,
    degraded_hashes: np.ndarray,
    n_bits: int = 32,
) -> float:
    n = min(len(original_hashes), len(degraded_hashes))
    if n == 0:
        return 1.0
    total_errors = sum(hamming_distance(int(original_hashes[i]), int(degraded_hashes[i])) for i in range(n))
    return total_errors / (n * n_bits)


def hash_entropy(hashes: np.ndarray, n_bits: int = 32) -> float:
    if len(hashes) == 0:
        return 0.0
    unique, counts = np.unique(hashes, return_counts=True)
    probs = counts / counts.sum()
    ent = -np.sum(probs * np.log2(probs + 1e-12))
    return float(ent) / max(n_bits, 1)


def collision_rate(hashes: np.ndarray) -> float:
    if len(hashes) == 0:
        return 0.0
    unique = len(np.unique(hashes))
    return 1.0 - unique / len(hashes)
