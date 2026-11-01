"""Audio fingerprinting metadata: chromaprint, landmarks, similarity."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import struct


class FingerprintAlgorithm(Enum):
    CHROMAPRINT = "chromaprint"
    LANDMARK = "landmark"
    SPECTRAL_HASH = "spectral_hash"
    WAVEPRINT = "waveprint"


@dataclass
class FingerprintMeta:
    algorithm: FingerprintAlgorithm
    fingerprint: bytes
    duration_s: float
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    source_codec: str = "pcm"

    def hex_digest(self) -> str:
        return hashlib.sha256(self.fingerprint).hexdigest()

    def short_id(self) -> str:
        return self.hex_digest()[:16]


@dataclass
class ChromaprintConfig:
    sample_rate: int = 11025
    frame_size: int = 4096
    overlap: int = 2048
    num_filter_banks: int = 5
    algorithm: int = 1

    @property
    def hop_size(self) -> int:
        return self.frame_size - self.overlap


@dataclass
class LandmarkConfig:
    max_peaks_per_frame: int = 5
    fan_out: int = 15
    min_time_delta: int = 0
    max_time_delta: int = 200
    freq_bins: int = 1024


@dataclass
class FingerprintMatch:
    query_offset_s: float
    reference_offset_s: float
    confidence: float
    reference_id: str


def fingerprint_similarity(fp1: bytes, fp2: bytes) -> float:
    """Hamming-distance-based similarity between two fingerprint bitstrings."""
    if not fp1 or not fp2:
        return 0.0
    min_len = min(len(fp1), len(fp2))
    if min_len == 0:
        return 0.0
    matching_bits = 0
    total_bits = 0
    for i in range(min_len):
        xor = fp1[i] ^ fp2[i]
        matching_bits += 8 - bin(xor).count('1')
        total_bits += 8
    return matching_bits / total_bits if total_bits > 0 else 0.0


def generate_spectral_hash(
    magnitudes: list[list[float]],
    num_bands: int = 32,
) -> bytes:
    """Produce a compact binary hash from spectral magnitude frames."""
    result = bytearray()
    for frame in magnitudes:
        band_size = max(1, len(frame) // num_bands)
        for b in range(num_bands):
            start = b * band_size
            end = min(start + band_size, len(frame))
            band_energy = sum(frame[start:end]) / max(1, end - start)
            result.append(int(min(band_energy * 255, 255)))
    return bytes(result)


def landmark_hash(
    freq1: int, freq2: int, time_delta: int,
) -> int:
    """Compute a Shazam-style landmark hash from peak pairs."""
    return ((freq1 & 0x3FF) << 20) | ((freq2 & 0x3FF) << 10) | (time_delta & 0x3FF)


def landmark_hash_to_bytes(h: int) -> bytes:
    return struct.pack(">I", h)


def decode_landmark_hash(h: int) -> tuple[int, int, int]:
    """Unpack a landmark hash into (freq1, freq2, time_delta)."""
    freq1 = (h >> 20) & 0x3FF
    freq2 = (h >> 10) & 0x3FF
    time_delta = h & 0x3FF
    return freq1, freq2, time_delta


def match_fingerprints(
    query_hashes: list[tuple[int, int]],
    reference_hashes: list[tuple[int, int]],
    reference_id: str = "unknown",
    min_matches: int = 5,
) -> list[FingerprintMatch]:
    """Find time-aligned matches between query and reference landmark hashes.

    Each hash is (landmark_hash_value, time_offset).
    """
    ref_lookup: dict[int, list[int]] = {}
    for h, t in reference_hashes:
        ref_lookup.setdefault(h, []).append(t)
    offset_counts: dict[int, int] = {}
    for h, qt in query_hashes:
        if h in ref_lookup:
            for rt in ref_lookup[h]:
                delta = rt - qt
                offset_counts[delta] = offset_counts.get(delta, 0) + 1
    if not offset_counts:
        return []
    best_offset = max(offset_counts, key=offset_counts.get)  # type: ignore[arg-type]
    count = offset_counts[best_offset]
    if count < min_matches:
        return []
    total_possible = min(len(query_hashes), len(reference_hashes))
    confidence = count / total_possible if total_possible > 0 else 0.0
    return [FingerprintMatch(
        query_offset_s=0.0,
        reference_offset_s=best_offset * 0.01,
        confidence=min(confidence, 1.0),
        reference_id=reference_id,
    )]
