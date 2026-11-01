"""Song structure segmentation: SSM, novelty curves, boundary detection, labeling."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class SegmentType(Enum):
    INTRO = auto()
    VERSE = auto()
    CHORUS = auto()
    BRIDGE = auto()
    PRE_CHORUS = auto()
    POST_CHORUS = auto()
    INSTRUMENTAL = auto()
    OUTRO = auto()
    BREAKDOWN = auto()
    BUILD = auto()


@dataclass(frozen=True)
class Segment:
    segment_type: SegmentType
    start_time: float
    end_time: float
    confidence: float = 1.0
    label: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def self_similarity_matrix(features: np.ndarray, metric: str = "cosine") -> np.ndarray:
    n = features.shape[0]
    if metric == "cosine":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = features / norms
        ssm = normed @ normed.T
    else:
        diff = features[:, None, :] - features[None, :, :]
        ssm = -np.sqrt(np.sum(diff ** 2, axis=2))
    return ssm


def novelty_curve(ssm: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    n = ssm.shape[0]
    k = min(kernel_size, n // 2)
    if k < 1:
        return np.zeros(n)
    kernel = np.ones((2 * k, 2 * k))
    kernel[:k, :k] = 1
    kernel[k:, k:] = 1
    kernel[:k, k:] = -1
    kernel[k:, :k] = -1
    novelty = np.zeros(n)
    for i in range(k, n - k):
        patch = ssm[i - k:i + k, i - k:i + k]
        novelty[i] = np.sum(patch * kernel)
    novelty = np.maximum(novelty, 0)
    mx = np.max(novelty)
    if mx > 0:
        novelty /= mx
    return novelty


def detect_boundaries(
    novelty: np.ndarray,
    threshold: float = 0.3,
    min_segment_frames: int = 8,
) -> list[int]:
    boundaries = [0]
    for i in range(1, len(novelty) - 1):
        if novelty[i] > threshold and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
            if i - boundaries[-1] >= min_segment_frames:
                boundaries.append(i)
    return boundaries


def _cluster_segments(features: np.ndarray, boundaries: list[int], n_clusters: int = 4) -> list[int]:
    n_seg = len(boundaries)
    seg_features = []
    for i in range(n_seg):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < n_seg else features.shape[0]
        seg_features.append(np.mean(features[start:end], axis=0))
    seg_features = np.array(seg_features)
    n_clusters = min(n_clusters, n_seg)
    rng = np.random.RandomState(0)
    idx = rng.choice(n_seg, n_clusters, replace=False)
    centroids = seg_features[idx].copy()
    labels = np.zeros(n_seg, dtype=int)
    for _ in range(20):
        dists = np.zeros((n_seg, n_clusters))
        for c in range(n_clusters):
            diff = seg_features - centroids[c]
            dists[:, c] = np.sum(diff ** 2, axis=1)
        labels = np.argmin(dists, axis=1)
        for c in range(n_clusters):
            mask = labels == c
            if np.any(mask):
                centroids[c] = seg_features[mask].mean(axis=0)
    return labels.tolist()


_LABEL_MAP = {
    0: (SegmentType.VERSE, "A"),
    1: (SegmentType.CHORUS, "B"),
    2: (SegmentType.BRIDGE, "C"),
    3: (SegmentType.INSTRUMENTAL, "D"),
}


def label_segments(
    features: np.ndarray,
    boundaries: list[int],
    hop_time: float = 0.5,
    n_clusters: int = 4,
) -> list[Segment]:
    if not boundaries:
        return []
    cluster_ids = _cluster_segments(features, boundaries, n_clusters)
    segments = []
    n = len(boundaries)
    for i in range(n):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < n else features.shape[0]
        cid = cluster_ids[i]
        seg_type, label = _LABEL_MAP.get(cid, (SegmentType.VERSE, f"X{cid}"))
        segments.append(Segment(
            segment_type=seg_type,
            start_time=start * hop_time,
            end_time=end * hop_time,
            confidence=0.8,
            label=label,
        ))
    return segments


def repetition_structure(segments: list[Segment]) -> dict[str, list[int]]:
    structure: dict[str, list[int]] = {}
    for i, seg in enumerate(segments):
        key = seg.label
        if key not in structure:
            structure[key] = []
        structure[key].append(i)
    return structure


def recurrence_plot(features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    ssm = self_similarity_matrix(features, metric="cosine")
    return (ssm >= threshold).astype(np.float64)


def lag_matrix(ssm: np.ndarray) -> np.ndarray:
    n = ssm.shape[0]
    L = np.zeros((n, n))
    for lag in range(n):
        for i in range(n - lag):
            L[lag, i] = ssm[i, i + lag]
    return L
