"""Cover song detection: chroma-based similarity, key-invariant matching.

Implements Optimal Transposition Index (OTI), cross-recurrence plots,
and dynamic time warping on chroma features for identifying cover versions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def chroma_from_signal(signal: np.ndarray, sr: float = 44100.0,
                       n_fft: int = 4096, hop: int = 2048) -> np.ndarray:
    n_frames = max(0, (len(signal) - n_fft) // hop + 1)
    if n_frames == 0:
        return np.zeros((0, 12))
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    chroma = np.zeros((n_frames, 12))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + n_fft] * window
        mag = np.abs(np.fft.rfft(frame))
        for j, f in enumerate(freqs):
            if f < 20 or f > 5000:
                continue
            midi = 69 + 12 * np.log2(f / 440.0)
            pitch_class = int(round(midi)) % 12
            chroma[i, pitch_class] += mag[j] ** 2
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    chroma /= norms
    return chroma


def optimal_transposition_index(chroma_a: np.ndarray,
                                chroma_b: np.ndarray) -> int:
    if len(chroma_a) == 0 or len(chroma_b) == 0:
        return 0
    mean_a = np.mean(chroma_a, axis=0)
    mean_b = np.mean(chroma_b, axis=0)
    best_shift = 0
    best_sim = -1.0
    for shift in range(12):
        shifted = np.roll(mean_b, shift)
        sim = np.dot(mean_a, shifted) / (np.linalg.norm(mean_a) * np.linalg.norm(shifted) + 1e-12)
        if sim > best_sim:
            best_sim = sim
            best_shift = shift
    return best_shift


def transpose_chroma(chroma: np.ndarray, semitones: int) -> np.ndarray:
    return np.roll(chroma, semitones % 12, axis=1)


def cross_recurrence_matrix(chroma_a: np.ndarray, chroma_b: np.ndarray,
                            threshold: float = 0.8) -> np.ndarray:
    n = len(chroma_a)
    m = len(chroma_b)
    if n == 0 or m == 0:
        return np.zeros((n, m))
    sim = chroma_a @ chroma_b.T
    crp = (sim >= threshold).astype(float)
    return crp


def qmax_score(crp: np.ndarray, min_run: int = 3) -> float:
    n, m = crp.shape
    if n == 0 or m == 0:
        return 0.0
    max_run = 0
    for offset in range(-n + 1, m):
        run = 0
        for i in range(n):
            j = i + offset
            if 0 <= j < m and crp[i, j] > 0:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
    if max_run < min_run:
        return 0.0
    return max_run / min(n, m)


def dtw_chroma(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    n = len(chroma_a)
    m = len(chroma_b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = 1.0 - np.dot(chroma_a[i - 1], chroma_b[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m]) / max(n, m)


@dataclass(frozen=True)
class CoverMatch:
    query_id: str
    reference_id: str
    score: float
    transposition: int
    method: str


@dataclass
class CoverDetector:
    oti_enabled: bool = True
    qmax_threshold: float = 0.1
    dtw_threshold: float = 0.6
    crp_similarity_threshold: float = 0.75

    def compare(self, chroma_a: np.ndarray, chroma_b: np.ndarray,
                query_id: str = "", ref_id: str = "") -> CoverMatch:
        if self.oti_enabled:
            oti = optimal_transposition_index(chroma_a, chroma_b)
            chroma_b_t = transpose_chroma(chroma_b, oti)
        else:
            oti = 0
            chroma_b_t = chroma_b
        crp = cross_recurrence_matrix(chroma_a, chroma_b_t,
                                      self.crp_similarity_threshold)
        q = qmax_score(crp)
        dtw_dist = dtw_chroma(chroma_a, chroma_b_t)
        dtw_sim = max(0.0, 1.0 - dtw_dist) if dtw_dist < float("inf") else 0.0
        score = 0.5 * q + 0.5 * dtw_sim
        method = "qmax+dtw"
        return CoverMatch(query_id, ref_id, score, oti, method)

    def is_cover(self, chroma_a: np.ndarray, chroma_b: np.ndarray) -> bool:
        match = self.compare(chroma_a, chroma_b)
        return match.score >= self.qmax_threshold


def hpcp(signal: np.ndarray, sr: float = 44100.0, n_fft: int = 4096,
         hop: int = 2048, harmonics: int = 3) -> np.ndarray:
    n_frames = max(0, (len(signal) - n_fft) // hop + 1)
    if n_frames == 0:
        return np.zeros((0, 12))
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    result = np.zeros((n_frames, 12))
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + n_fft] * window
        mag = np.abs(np.fft.rfft(frame))
        for j, f in enumerate(freqs):
            if f < 50 or f > 5000:
                continue
            for h in range(1, harmonics + 1):
                fh = f / h
                if fh < 20:
                    continue
                midi = 69 + 12 * np.log2(fh / 440.0)
                pc = int(round(midi)) % 12
                weight = 1.0 / h
                result[i, pc] += mag[j] ** 2 * weight
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    result /= norms
    return result
