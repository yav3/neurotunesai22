"""Audio content identification: spectral hashing, landmark matching, fingerprint DB."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AudioFingerprint:
    hash_value: str
    duration: float
    offset: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True)
class ContentIDResult:
    matched: bool
    source_id: str = ""
    similarity: float = 0.0
    offset: float = 0.0
    fingerprint: AudioFingerprint | None = None


def spectral_hash(spectrogram: np.ndarray, hop_length: int = 512) -> str:
    n_frames = spectrogram.shape[1] if spectrogram.ndim == 2 else spectrogram.shape[0]
    n_bins = spectrogram.shape[0] if spectrogram.ndim == 2 else 1
    band_size = max(n_bins // 8, 1)
    bits = []
    for t in range(min(n_frames, 256)):
        col = spectrogram[:, t] if spectrogram.ndim == 2 else spectrogram
        for b in range(0, n_bins - band_size, band_size):
            upper = np.mean(col[b: b + band_size])
            lower = np.mean(col[b + band_size: b + 2 * band_size]) if b + 2 * band_size <= n_bins else 0.0
            bits.append(1 if upper > lower else 0)
    byte_arr = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(min(8, len(bits) - i)):
            byte_val |= bits[i + j] << j
        byte_arr.append(byte_val)
    return hashlib.sha256(bytes(byte_arr)).hexdigest()[:32]


def extract_landmarks(
    spectrogram: np.ndarray,
    n_peaks: int = 20,
    neighborhood: int = 5,
) -> list[tuple[int, int, int]]:
    if spectrogram.ndim != 2 or spectrogram.size == 0:
        return []
    n_freq, n_time = spectrogram.shape
    peaks: list[tuple[int, int, float]] = []
    for t in range(neighborhood, n_time - neighborhood):
        for f in range(neighborhood, n_freq - neighborhood):
            val = spectrogram[f, t]
            patch = spectrogram[
                f - neighborhood: f + neighborhood + 1,
                t - neighborhood: t + neighborhood + 1,
            ]
            if val >= np.max(patch):
                peaks.append((t, f, float(val)))
    peaks.sort(key=lambda x: -x[2])
    peaks = peaks[:n_peaks * 4]
    landmarks: list[tuple[int, int, int]] = []
    for i, (t1, f1, _) in enumerate(peaks[:n_peaks]):
        for t2, f2, _ in peaks[i + 1: i + 6]:
            dt = t2 - t1
            if 1 <= dt <= 64:
                landmarks.append((f1, f2, dt))
    return landmarks


def match_landmarks(
    query: list[tuple[int, int, int]],
    database: dict[str, list[tuple[int, int, int]]],
    threshold: float = 0.3,
) -> ContentIDResult:
    if not query or not database:
        return ContentIDResult(matched=False)
    query_set = set(query)
    best_id = ""
    best_sim = 0.0
    for source_id, db_landmarks in database.items():
        db_set = set(db_landmarks)
        intersection = len(query_set & db_set)
        union = len(query_set | db_set)
        sim = intersection / max(union, 1)
        if sim > best_sim:
            best_sim = sim
            best_id = source_id
    return ContentIDResult(
        matched=best_sim >= threshold,
        source_id=best_id,
        similarity=best_sim,
    )


class FingerprintDatabase:
    def __init__(self) -> None:
        self._db: dict[str, list[tuple[int, int, int]]] = {}

    @property
    def size(self) -> int:
        return len(self._db)

    def add(self, source_id: str, landmarks: list[tuple[int, int, int]]) -> None:
        self._db[source_id] = landmarks

    def query(
        self, landmarks: list[tuple[int, int, int]], threshold: float = 0.3,
    ) -> ContentIDResult:
        return match_landmarks(landmarks, self._db, threshold)

    def remove(self, source_id: str) -> bool:
        return self._db.pop(source_id, None) is not None
