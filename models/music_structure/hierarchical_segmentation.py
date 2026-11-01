"""Hierarchical music structure analysis: novelty detection, spectral clustering, multi-level segmentation."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist


@dataclass
class SegmentNode:
    start_time: float
    end_time: float
    label: int
    level: int = 0
    children: list["SegmentNode"] = field(default_factory=list)


def checkerboard_kernel(size: int) -> np.ndarray:
    k = np.ones((size, size))
    half = size // 2
    k[:half, :half] = -1
    k[half:, half:] = -1
    return k


def compute_novelty(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    n = ssm.shape[0]
    kernel = checkerboard_kernel(kernel_size)
    half = kernel_size // 2
    novelty = np.zeros(n)
    for i in range(n):
        r_lo = max(0, i - half)
        r_hi = min(n, i + half)
        c_lo = max(0, i - half)
        c_hi = min(n, i + half)
        patch = ssm[r_lo:r_hi, c_lo:c_hi]
        kr_lo = half - (i - r_lo)
        kr_hi = kr_lo + patch.shape[0]
        kc_lo = half - (i - c_lo)
        kc_hi = kc_lo + patch.shape[1]
        k_patch = kernel[kr_lo:kr_hi, kc_lo:kc_hi]
        novelty[i] = np.sum(patch * k_patch)
    novelty = np.maximum(novelty, 0)
    return novelty


def pick_peaks(novelty: np.ndarray, threshold: float = 0.0, min_distance: int = 4) -> np.ndarray:
    peaks = []
    n = len(novelty)
    for i in range(1, n - 1):
        if novelty[i] <= threshold:
            continue
        if novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def _build_ssm(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1e-10)
    normed = features / norms
    return normed @ normed.T


def _kmeans(data: np.ndarray, k: int, max_iter: int = 50, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    indices = rng.choice(n, size=min(k, n), replace=False)
    centers = data[indices].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = cdist(data, centers)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                centers[c] = data[mask].mean(axis=0)
    return labels


def laplacian_segmentation(features: np.ndarray, k_clusters: int = 4) -> np.ndarray:
    ssm = _build_ssm(features)
    ssm = (ssm + 1) / 2
    np.fill_diagonal(ssm, 0)
    D = np.diag(ssm.sum(axis=1))
    L = D - ssm
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L_norm)
    k = min(k_clusters, eigvecs.shape[1])
    embedding = eigvecs[:, :k]
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 1e-10, row_norms, 1e-10)
    embedding = embedding / row_norms
    labels = _kmeans(embedding, k)
    return labels


def build_hierarchy(features: np.ndarray, levels: list[int] | None = None) -> SegmentNode:
    n = features.shape[0]
    if levels is None:
        levels = [2, 4, 8]
    levels = sorted(levels)
    root = SegmentNode(start_time=0, end_time=float(n), label=0, level=0)
    prev_nodes = [root]
    for lv_idx, k in enumerate(levels):
        labels = laplacian_segmentation(features, k_clusters=k)
        segments = []
        i = 0
        while i < n:
            start = i
            lbl = labels[i]
            while i < n and labels[i] == lbl:
                i += 1
            seg = SegmentNode(start_time=float(start), end_time=float(i), label=int(lbl), level=lv_idx + 1)
            segments.append(seg)
        for parent in prev_nodes:
            for seg in segments:
                if seg.start_time >= parent.start_time and seg.end_time <= parent.end_time:
                    parent.children.append(seg)
        prev_nodes = segments
    return root


def segment_homogeneity(features: np.ndarray, boundaries: np.ndarray) -> float:
    bounds = sorted(set([0] + list(boundaries) + [len(features)]))
    scores = []
    for i in range(len(bounds) - 1):
        seg = features[bounds[i]:bounds[i + 1]]
        if len(seg) < 2:
            continue
        norms = np.linalg.norm(seg, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1e-10)
        normed = seg / norms
        sim = normed @ normed.T
        n_s = sim.shape[0]
        mask = np.triu(np.ones((n_s, n_s), dtype=bool), k=1)
        scores.append(float(sim[mask].mean()) if mask.sum() > 0 else 1.0)
    return float(np.mean(scores)) if scores else 0.0


def boundary_precision_recall(
    estimated: np.ndarray, reference: np.ndarray, tolerance: float = 3.0
) -> tuple[float, float, float]:
    est = np.asarray(estimated, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if len(est) == 0 and len(ref) == 0:
        return 1.0, 1.0, 1.0
    if len(est) == 0:
        return 0.0, 0.0, 0.0
    if len(ref) == 0:
        return 0.0, 0.0, 0.0
    tp_p = sum(1 for e in est if np.any(np.abs(ref - e) <= tolerance))
    tp_r = sum(1 for r in ref if np.any(np.abs(est - r) <= tolerance))
    precision = tp_p / len(est)
    recall = tp_r / len(ref)
    if precision + recall < 1e-10:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def form_string(labels: np.ndarray) -> str:
    mapping: dict[int, str] = {}
    current_char = ord("A")
    result = []
    prev = None
    for lbl in labels:
        lbl = int(lbl)
        if lbl not in mapping:
            mapping[lbl] = chr(current_char)
            current_char += 1
        ch = mapping[lbl]
        if ch != prev:
            result.append(ch)
            prev = ch
    return "".join(result)
