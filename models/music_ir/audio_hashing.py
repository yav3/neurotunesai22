"""Perceptual audio hashing for duplicate detection and near-duplicate retrieval.

Implements multiple hash families:
- Spectral band energy hash (robust to compression)
- Chroma-based hash (pitch-invariant to timbre changes)
- Bark-scale hash (perceptually weighted)
- MinHash for set-similarity of spectral peaks
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class AudioHash:
    """A perceptual hash of an audio segment."""
    bits: np.ndarray  # bool array
    frame_rate: float
    n_frames: int

    def hex(self) -> str:
        packed = np.packbits(self.bits)
        return packed.tobytes().hex()

    def hamming_distance(self, other: "AudioHash") -> int:
        return int(np.sum(self.bits != other.bits))

    def similarity(self, other: "AudioHash") -> float:
        n = len(self.bits)
        if n == 0:
            return 1.0
        return 1.0 - self.hamming_distance(other) / n


def _stft_magnitude(signal: np.ndarray, frame_size: int = 2048,
                    hop: int = 512) -> np.ndarray:
    n_frames = max(1, (len(signal) - frame_size) // hop + 1)
    window = np.hanning(frame_size)
    n_bins = frame_size // 2 + 1
    mag = np.zeros((n_frames, n_bins))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        mag[i] = np.abs(np.fft.rfft(frame * window))
    return mag


def _bark_bands(n_bins: int, sr: float) -> List[Tuple[int, int]]:
    """Critical band boundaries on the Bark scale (Zwicker)."""
    bark_edges = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                  1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                  4400, 5300, 6400, 7700, 9500, 12000, 15500]
    freqs = np.linspace(0, sr / 2, n_bins)
    bands = []
    for j in range(len(bark_edges) - 1):
        lo = np.searchsorted(freqs, bark_edges[j])
        hi = np.searchsorted(freqs, bark_edges[j + 1])
        if lo < hi and lo < n_bins:
            bands.append((lo, min(hi, n_bins)))
    return bands


def spectral_band_hash(signal: np.ndarray, sr: float = 22050,
                       n_bands: int = 32, frame_size: int = 2048,
                       hop: int = 512) -> AudioHash:
    """Hash based on spectral band energy differences.

    For each frame, compute energy in n_bands frequency bands.
    Hash bit = 1 if band energy > median energy across bands.
    Robust to volume changes and mild compression artifacts.
    """
    mag = _stft_magnitude(signal, frame_size, hop)
    n_frames, n_bins = mag.shape
    band_width = max(1, n_bins // n_bands)
    bits = []
    for i in range(n_frames):
        energies = []
        for b in range(n_bands):
            lo = b * band_width
            hi = min((b + 1) * band_width, n_bins)
            energies.append(np.sum(mag[i, lo:hi] ** 2))
        energies = np.array(energies)
        median_e = np.median(energies)
        bits.extend(energies > median_e)
    return AudioHash(bits=np.array(bits, dtype=bool),
                     frame_rate=sr / hop, n_frames=n_frames)


def bark_scale_hash(signal: np.ndarray, sr: float = 22050,
                    frame_size: int = 2048, hop: int = 512) -> AudioHash:
    """Perceptually weighted hash using Bark critical bands.

    Each bit: 1 if energy in Bark band exceeds the band's temporal mean.
    More perceptually meaningful than uniform frequency splitting.
    """
    mag = _stft_magnitude(signal, frame_size, hop)
    n_frames, n_bins = mag.shape
    bands = _bark_bands(n_bins, sr)
    n_bands = len(bands)
    band_energy = np.zeros((n_frames, n_bands))
    for j, (lo, hi) in enumerate(bands):
        band_energy[:, j] = np.sum(mag[:, lo:hi] ** 2, axis=1)
    temporal_mean = np.mean(band_energy, axis=0, keepdims=True)
    bits = (band_energy > temporal_mean).ravel()
    return AudioHash(bits=bits, frame_rate=sr / hop, n_frames=n_frames)


def chroma_hash(signal: np.ndarray, sr: float = 22050,
                frame_size: int = 4096, hop: int = 2048) -> AudioHash:
    """Pitch-class profile hash (robust to timbre changes).

    Projects spectrum onto 12 chroma bins, hashes by comparing
    each chroma bin to the frame's mean chroma energy.
    """
    mag = _stft_magnitude(signal, frame_size, hop)
    n_frames, n_bins = mag.shape
    freqs = np.linspace(0, sr / 2, n_bins)
    chroma = np.zeros((n_frames, 12))
    for i in range(n_frames):
        for k in range(1, n_bins):
            if freqs[k] > 0:
                pitch_class = int(round(12 * np.log2(freqs[k] / 440.0))) % 12
                chroma[i, pitch_class] += mag[i, k] ** 2
    mean_chroma = np.mean(chroma, axis=1, keepdims=True)
    mean_chroma = np.maximum(mean_chroma, 1e-10)
    bits = (chroma > mean_chroma).ravel()
    return AudioHash(bits=bits, frame_rate=sr / hop, n_frames=n_frames)


@dataclass
class MinHashSignature:
    """MinHash signature for set-similarity estimation."""
    signature: np.ndarray
    n_hashes: int

    def jaccard_estimate(self, other: "MinHashSignature") -> float:
        return float(np.mean(self.signature == other.signature))


def _spectral_peak_set(mag_frame: np.ndarray, threshold_db: float = -40.0,
                       n_bins: int = 1025) -> set:
    """Extract prominent spectral peaks as a discrete set."""
    peak_set = set()
    mag_db = 20 * np.log10(np.maximum(mag_frame, 1e-10))
    max_db = np.max(mag_db)
    for k in range(1, len(mag_db) - 1):
        if (mag_db[k] > mag_db[k-1] and mag_db[k] > mag_db[k+1]
                and mag_db[k] > max_db + threshold_db):
            peak_set.add(k)
    return peak_set


def minhash_audio(signal: np.ndarray, sr: float = 22050,
                  n_hashes: int = 128, frame_size: int = 2048,
                  hop: int = 512) -> MinHashSignature:
    """MinHash signature over spectral peak sets.

    Treats each frame's spectral peaks as a set; computes
    MinHash for Jaccard similarity estimation between audio segments.
    Uses universal hash family h(x) = (ax + b) mod p mod m.
    """
    mag = _stft_magnitude(signal, frame_size, hop)
    n_bins = mag.shape[1]
    prime = 2147483647  # Mersenne prime 2^31 - 1
    rng = np.random.RandomState(42)
    a_vals = rng.randint(1, prime, size=n_hashes)
    b_vals = rng.randint(0, prime, size=n_hashes)
    all_peaks: set = set()
    for i in range(mag.shape[0]):
        all_peaks |= _spectral_peak_set(mag[i], n_bins=n_bins)
    if not all_peaks:
        return MinHashSignature(
            signature=np.full(n_hashes, prime, dtype=np.int64),
            n_hashes=n_hashes)
    peak_array = np.array(sorted(all_peaks), dtype=np.int64)
    sig = np.full(n_hashes, prime, dtype=np.int64)
    for h in range(n_hashes):
        hvals = (a_vals[h] * peak_array + b_vals[h]) % prime
        sig[h] = np.min(hvals)
    return MinHashSignature(signature=sig, n_hashes=n_hashes)


@dataclass
class HashIndex:
    """Index of audio hashes for fast retrieval."""
    entries: List[Tuple[str, AudioHash]] = field(default_factory=list)
    _table: dict = field(default_factory=dict, repr=False)

    def add(self, track_id: str, audio_hash: AudioHash) -> None:
        self.entries.append((track_id, audio_hash))
        hex_key = audio_hash.hex()[:16]
        self._table.setdefault(hex_key, []).append(len(self.entries) - 1)

    def query(self, audio_hash: AudioHash,
              threshold: float = 0.8) -> List[Tuple[str, float]]:
        results = []
        for track_id, stored_hash in self.entries:
            sim = audio_hash.similarity(stored_hash)
            if sim >= threshold:
                results.append((track_id, sim))
        results.sort(key=lambda x: -x[1])
        return results


def bit_error_rate(h1: AudioHash, h2: AudioHash) -> float:
    """Bit error rate between two hashes (0 = identical, 1 = opposite)."""
    return 1.0 - h1.similarity(h2)


def hash_robustness(original: np.ndarray, degraded: np.ndarray,
                    sr: float = 22050,
                    hash_fn=spectral_band_hash) -> dict:
    """Measure hash robustness: how similar are hashes after degradation."""
    h_orig = hash_fn(original, sr)
    h_deg = hash_fn(degraded, sr)
    min_len = min(len(h_orig.bits), len(h_deg.bits))
    h_orig_trimmed = AudioHash(h_orig.bits[:min_len], h_orig.frame_rate,
                               h_orig.n_frames)
    h_deg_trimmed = AudioHash(h_deg.bits[:min_len], h_deg.frame_rate,
                              h_deg.n_frames)
    return {
        "similarity": h_orig_trimmed.similarity(h_deg_trimmed),
        "bit_error_rate": bit_error_rate(h_orig_trimmed, h_deg_trimmed),
        "n_bits": min_len,
    }
