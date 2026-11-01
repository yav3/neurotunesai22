"""Repetition analysis: motif discovery, thumbnail extraction, structural compression."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Motif:
    start_frame: int
    length: int
    feature_hash: int = 0

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.length


@dataclass(frozen=True)
class MotifOccurrence:
    motif_id: int
    start_frame: int
    similarity: float


@dataclass
class RepetitionStructure:
    motifs: list[Motif] = field(default_factory=list)
    occurrences: list[MotifOccurrence] = field(default_factory=list)
    compression_ratio: float = 0.0


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = features / norms
    return normed @ normed.T


def lag_similarity(sim_matrix: np.ndarray, lag: int) -> float:
    n = sim_matrix.shape[0]
    if lag >= n or lag < 0:
        return 0.0
    diag = np.diag(sim_matrix, k=lag)
    return float(np.mean(diag))


def lag_spectrum(sim_matrix: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    n = sim_matrix.shape[0]
    if max_lag is None:
        max_lag = n // 2
    max_lag = min(max_lag, n - 1)
    spectrum = np.array([lag_similarity(sim_matrix, k) for k in range(max_lag + 1)])
    return spectrum


def find_repeated_segments(
    features: np.ndarray,
    min_length: int = 4,
    threshold: float = 0.85,
) -> list[tuple[int, int, int, float]]:
    n = features.shape[0]
    sim = cosine_similarity_matrix(features)
    segments: list[tuple[int, int, int, float]] = []

    for lag in range(min_length, n // 2):
        diag = np.diag(sim, k=lag)
        run_start = -1
        for i, v in enumerate(diag):
            if v >= threshold:
                if run_start < 0:
                    run_start = i
            else:
                if run_start >= 0 and (i - run_start) >= min_length:
                    mean_sim = float(np.mean(diag[run_start:i]))
                    segments.append((run_start, run_start + lag, i - run_start, mean_sim))
                run_start = -1
        if run_start >= 0 and (len(diag) - run_start) >= min_length:
            mean_sim = float(np.mean(diag[run_start:]))
            segments.append((run_start, run_start + lag, len(diag) - run_start, mean_sim))

    return segments


def structural_thumbnail(
    features: np.ndarray,
    thumbnail_length: int = 8,
) -> int:
    n = features.shape[0]
    if n <= thumbnail_length:
        return 0
    sim = cosine_similarity_matrix(features)
    best_start = 0
    best_score = -1.0
    for start in range(n - thumbnail_length):
        seg = features[start : start + thumbnail_length]
        seg_norm = np.linalg.norm(seg, axis=1, keepdims=True)
        seg_norm = np.maximum(seg_norm, 1e-12)
        seg_normed = seg / seg_norm
        feat_norm = np.linalg.norm(features, axis=1, keepdims=True)
        feat_norm = np.maximum(feat_norm, 1e-12)
        feat_normed = features / feat_norm
        cross_sim = feat_normed @ seg_normed.T
        score = float(np.mean(np.max(cross_sim, axis=1)))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start


def compression_ratio(features: np.ndarray, threshold: float = 0.9) -> float:
    n = features.shape[0]
    if n < 2:
        return 1.0
    sim = cosine_similarity_matrix(features)
    np.fill_diagonal(sim, 0.0)
    redundant = np.sum(sim > threshold)
    total_pairs = n * (n - 1)
    if total_pairs == 0:
        return 1.0
    return 1.0 - redundant / total_pairs


def build_repetition_structure(
    features: np.ndarray,
    min_length: int = 4,
    threshold: float = 0.85,
) -> RepetitionStructure:
    segments = find_repeated_segments(features, min_length, threshold)
    motifs: list[Motif] = []
    occurrences: list[MotifOccurrence] = []
    seen_starts: set[int] = set()

    for src, tgt, length, sim in segments:
        if src not in seen_starts:
            motif_id = len(motifs)
            motifs.append(Motif(start_frame=src, length=length))
            seen_starts.add(src)
            occurrences.append(MotifOccurrence(motif_id=motif_id, start_frame=tgt, similarity=sim))

    cr = compression_ratio(features, threshold)
    return RepetitionStructure(motifs=motifs, occurrences=occurrences, compression_ratio=cr)
