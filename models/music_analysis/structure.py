"""Structural analysis via self-similarity matrices and novelty curves."""
from __future__ import annotations
import numpy as np


class StructureAnalyzer:
    def __init__(self, kernel_size: int = 16):
        self.kernel_size = kernel_size

    def compute_ssm(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = features / norms
        return normed @ normed.T

    def novelty_curve(self, ssm: np.ndarray) -> np.ndarray:
        n = ssm.shape[0]
        k = min(self.kernel_size, n // 2)
        if k < 1:
            return np.zeros(n)
        kernel = np.ones((k, k))
        kernel[:k // 2, :k // 2] = 1
        kernel[k // 2:, k // 2:] = 1
        kernel[:k // 2, k // 2:] = -1
        kernel[k // 2:, :k // 2] = -1
        novelty = np.zeros(n)
        half = k // 2
        for i in range(half, n - half):
            patch = ssm[i - half:i + half, i - half:i + half]
            if patch.shape == kernel[:patch.shape[0], :patch.shape[1]].shape:
                novelty[i] = float(np.sum(patch * kernel[:patch.shape[0], :patch.shape[1]]))
        novelty = np.maximum(novelty, 0)
        mx = novelty.max()
        if mx > 1e-12:
            novelty /= mx
        return novelty

    def find_boundaries(self, ssm: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        nc = self.novelty_curve(ssm)
        peaks = []
        for i in range(1, len(nc) - 1):
            if nc[i] > nc[i - 1] and nc[i] > nc[i + 1] and nc[i] > threshold:
                peaks.append(i)
        return np.array(peaks, dtype=int)


class SectionLabeler:
    LABELS = ["intro", "verse", "chorus", "bridge", "outro"]

    def label_segments(self, features: np.ndarray, boundaries: np.ndarray) -> list[tuple[int, int, str]]:
        n = features.shape[0] if features.ndim >= 1 else 0
        if n == 0:
            return []
        bounds = sorted(set([0] + list(boundaries) + [n]))
        segments = []
        centroids = []
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            seg_feat = features[s:e]
            if seg_feat.ndim == 1:
                seg_feat = seg_feat.reshape(-1, 1)
            centroids.append(np.mean(seg_feat, axis=0))
            segments.append((s, e))
        labels = self._assign_labels(centroids, len(segments))
        return [(s, e, lbl) for (s, e), lbl in zip(segments, labels)]

    def _assign_labels(self, centroids: list[np.ndarray], n: int) -> list[str]:
        if n == 0:
            return []
        if n <= 2:
            return self.LABELS[:n]
        labels = [""] * n
        labels[0] = "intro"
        labels[-1] = "outro"
        if n == 3:
            labels[1] = "verse"
            return labels
        groups: dict[int, list[int]] = {}
        for i in range(1, n - 1):
            assigned = False
            for gid, members in groups.items():
                ref = centroids[members[0]]
                dist = float(np.linalg.norm(np.asarray(centroids[i]) - np.asarray(ref)))
                if dist < 0.5:
                    groups[gid].append(i)
                    assigned = True
                    break
            if not assigned:
                groups[len(groups)] = [i]
        label_cycle = ["verse", "chorus", "bridge"]
        for gi, (_, members) in enumerate(sorted(groups.items())):
            lbl = label_cycle[gi % len(label_cycle)]
            for idx in members:
                labels[idx] = lbl
        return labels


def compute_ssm(features: np.ndarray) -> np.ndarray:
    return StructureAnalyzer().compute_ssm(features)


def find_boundaries(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    return StructureAnalyzer(kernel_size=kernel_size).find_boundaries(ssm)
