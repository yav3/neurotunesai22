"""Self-similarity matrix analysis for music structure detection."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normalized = features / norms
    return normalized @ normalized.T


def euclidean_distance_matrix(features: np.ndarray) -> np.ndarray:
    n = features.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((features[i] - features[j]) ** 2))
            D[i, j] = d
            D[j, i] = d
    return D


def distance_to_similarity(D: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return np.exp(-D ** 2 / (2 * sigma ** 2))


def recurrence_matrix(ssm: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (ssm >= threshold).astype(float)


def lag_similarity(ssm: np.ndarray, lag: int) -> np.ndarray:
    n = ssm.shape[0]
    if lag >= n:
        return np.array([])
    diag = np.array([ssm[i, i + lag] for i in range(n - lag)])
    return diag


def novelty_from_ssm(ssm: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    n = ssm.shape[0]
    half = kernel_size // 2
    novelty = np.zeros(n)
    for i in range(half, n - half):
        upper_left = ssm[i - half:i, i - half:i]
        lower_right = ssm[i:i + half, i:i + half]
        ul_mean = np.mean(upper_left)
        lr_mean = np.mean(lower_right)
        cross_block = ssm[i - half:i, i:i + half]
        cr_mean = np.mean(cross_block)
        novelty[i] = (ul_mean + lr_mean) / 2.0 - cr_mean
    return novelty


def pick_boundaries(novelty: np.ndarray, threshold: float = 0.0,
                     min_gap: int = 4) -> list[int]:
    boundaries = []
    for i in range(1, len(novelty) - 1):
        if novelty[i] > novelty[i - 1] and novelty[i] > novelty[i + 1]:
            if novelty[i] > threshold:
                if not boundaries or (i - boundaries[-1]) >= min_gap:
                    boundaries.append(i)
    return boundaries


@dataclass
class StructureSegment:
    start: int
    end: int
    label: str


def label_segments(ssm: np.ndarray, boundaries: list[int],
                    similarity_threshold: float = 0.6) -> list[StructureSegment]:
    all_bounds = [0] + sorted(boundaries) + [ssm.shape[0]]
    segments: list[StructureSegment] = []
    label_map: dict[int, str] = {}
    current_label = ord('A')

    for k in range(len(all_bounds) - 1):
        s, e = all_bounds[k], all_bounds[k + 1]
        matched = None
        for prev_idx, prev_label in label_map.items():
            ps, pe = all_bounds[prev_idx], all_bounds[prev_idx + 1]
            block = ssm[s:e, ps:pe]
            if block.size > 0 and np.mean(block) >= similarity_threshold:
                matched = prev_label
                break
        if matched is None:
            lbl = chr(current_label)
            current_label += 1
            label_map[k] = lbl
        else:
            lbl = matched
            label_map[k] = lbl
        segments.append(StructureSegment(start=s, end=e, label=lbl))
    return segments


def form_string(segments: list[StructureSegment]) -> str:
    return "".join(seg.label for seg in segments)


def repetition_ratio(ssm: np.ndarray, threshold: float = 0.7) -> float:
    n = ssm.shape[0]
    if n < 2:
        return 0.0
    upper = np.triu(ssm, k=1)
    count = np.sum(upper >= threshold)
    total = n * (n - 1) / 2
    return float(count / total)


def diagonal_strength(ssm: np.ndarray, lag: int) -> float:
    diag = lag_similarity(ssm, lag)
    if len(diag) == 0:
        return 0.0
    return float(np.mean(diag))


def find_repeated_sections(ssm: np.ndarray, min_length: int = 4,
                            threshold: float = 0.7) -> list[tuple[int, int, int]]:
    n = ssm.shape[0]
    sections = []
    for lag in range(1, n // 2):
        diag = lag_similarity(ssm, lag)
        run_start = None
        for i in range(len(diag)):
            if diag[i] >= threshold:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None and (i - run_start) >= min_length:
                    sections.append((run_start, i, lag))
                run_start = None
        if run_start is not None and (len(diag) - run_start) >= min_length:
            sections.append((run_start, len(diag), lag))
    return sections
