"""Audio fingerprint database: content-addressable audio storage and retrieval.

Store and query audio by perceptual fingerprint (constellation map),
supporting duplicate detection, version tracking, and near-duplicate
clustering for large music catalogues.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FingerprintEntry:
    track_id: str
    hash_pairs: np.ndarray
    duration_s: float
    sample_rate: int = 44100
    bit_depth: int = 16


@dataclass
class MatchResult:
    query_id: str
    matched_id: str
    score: float
    offset_s: float
    confidence: float


def constellation_map(spectrogram: np.ndarray,
                      n_peaks: int = 15,
                      fan_out: int = 10) -> np.ndarray:
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(spectrogram, size=(3, 3))
    peaks = (spectrogram == local_max) & (spectrogram > np.percentile(spectrogram, 95))
    coords = np.argwhere(peaks)
    if len(coords) > n_peaks * spectrogram.shape[1] // 10:
        magnitudes = spectrogram[coords[:, 0], coords[:, 1]]
        top_idx = np.argsort(magnitudes)[-n_peaks * spectrogram.shape[1] // 10:]
        coords = coords[top_idx]
    coords = coords[coords[:, 1].argsort()]
    pairs = []
    for i in range(len(coords)):
        anchor = coords[i]
        targets = coords[i + 1:i + 1 + fan_out]
        for t in targets:
            dt = t[1] - anchor[1]
            if 1 <= dt <= 64:
                pair_hash = (int(anchor[0]) << 20) | (int(t[0]) << 10) | dt
                pairs.append([pair_hash, anchor[1]])
    return np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def fingerprint_similarity(fp_a: np.ndarray, fp_b: np.ndarray,
                           tolerance: int = 2) -> Tuple[float, float]:
    if len(fp_a) == 0 or len(fp_b) == 0:
        return 0.0, 0.0
    hashes_a = set(fp_a[:, 0].tolist())
    hashes_b = set(fp_b[:, 0].tolist())
    common = hashes_a & hashes_b
    jaccard = len(common) / max(len(hashes_a | hashes_b), 1)
    if len(common) == 0:
        return jaccard, 0.0
    offsets_a = {int(fp_a[i, 0]): int(fp_a[i, 1]) for i in range(len(fp_a))}
    offsets_b = {int(fp_b[i, 0]): int(fp_b[i, 1]) for i in range(len(fp_b))}
    deltas = [offsets_a[h] - offsets_b[h] for h in common if h in offsets_a and h in offsets_b]
    if not deltas:
        return jaccard, 0.0
    from collections import Counter
    delta_counts = Counter(deltas)
    best_count = max(delta_counts.values())
    temporal_coherence = best_count / len(deltas)
    return jaccard, temporal_coherence


def find_offset(fp_query: np.ndarray, fp_ref: np.ndarray,
                hop_s: float = 0.01) -> float:
    if len(fp_query) == 0 or len(fp_ref) == 0:
        return 0.0
    ref_map: Dict[int, List[int]] = {}
    for i in range(len(fp_ref)):
        h = int(fp_ref[i, 0])
        ref_map.setdefault(h, []).append(int(fp_ref[i, 1]))
    deltas: List[int] = []
    for i in range(len(fp_query)):
        h = int(fp_query[i, 0])
        if h in ref_map:
            for ref_t in ref_map[h]:
                deltas.append(int(fp_query[i, 1]) - ref_t)
    if not deltas:
        return 0.0
    from collections import Counter
    best_delta = Counter(deltas).most_common(1)[0][0]
    return best_delta * hop_s


def cluster_near_duplicates(fingerprints: List[np.ndarray],
                            threshold: float = 0.3) -> List[List[int]]:
    n = len(fingerprints)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            jac, _ = fingerprint_similarity(fingerprints[i], fingerprints[j])
            if jac >= threshold:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


def fingerprint_robustness(original: np.ndarray, degraded: np.ndarray) -> Dict[str, float]:
    jac, temp = fingerprint_similarity(original, degraded)
    hash_retention = 0.0
    if len(original) > 0:
        orig_set = set(original[:, 0].tolist())
        deg_set = set(degraded[:, 0].tolist())
        hash_retention = len(orig_set & deg_set) / max(len(orig_set), 1)
    return {
        "jaccard": jac,
        "temporal_coherence": temp,
        "hash_retention": hash_retention,
        "robust": jac > 0.15 and temp > 0.3,
    }


def energy_fingerprint(audio: np.ndarray, frame_size: int = 2048,
                       hop: int = 512) -> np.ndarray:
    n_frames = max(1, (len(audio) - frame_size) // hop + 1)
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_size]
        energy[i] = np.sum(frame ** 2) / len(frame)
    bands = np.zeros((n_frames, 4))
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_size]
        spectrum = np.abs(np.fft.rfft(frame))
        n = len(spectrum)
        bands[i, 0] = np.mean(spectrum[:n // 4] ** 2)
        bands[i, 1] = np.mean(spectrum[n // 4:n // 2] ** 2)
        bands[i, 2] = np.mean(spectrum[n // 2:3 * n // 4] ** 2)
        bands[i, 3] = np.mean(spectrum[3 * n // 4:] ** 2)
    return bands
