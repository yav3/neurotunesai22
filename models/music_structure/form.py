"""Musical form analysis: section labeling, repetition detection, template matching."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SectionType(str, Enum):
    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    BRIDGE = "bridge"
    OUTRO = "outro"
    INSTRUMENTAL = "instrumental"
    PRE_CHORUS = "pre_chorus"
    UNKNOWN = "unknown"


@dataclass
class SectionLabel:
    start: int
    end: int
    label: str
    cluster_id: int
    repetition_count: int = 1


@dataclass
class FormTemplate:
    name: str
    pattern: list[str]
    match_score: float = 0.0


COMMON_TEMPLATES = [
    FormTemplate(name="verse-chorus", pattern=["A", "B", "A", "B"]),
    FormTemplate(name="AABA", pattern=["A", "A", "B", "A"]),
    FormTemplate(name="strophic", pattern=["A", "A", "A", "A"]),
    FormTemplate(name="through-composed", pattern=["A", "B", "C", "D"]),
    FormTemplate(name="rondo", pattern=["A", "B", "A", "C", "A"]),
    FormTemplate(name="verse-chorus-bridge", pattern=["A", "B", "A", "B", "C", "B"]),
    FormTemplate(name="binary", pattern=["A", "B"]),
    FormTemplate(name="ternary", pattern=["A", "B", "A"]),
]


def detect_repetitions(
    ssm: np.ndarray,
    min_length: int = 8,
) -> list[tuple[int, int, int]]:
    n = ssm.shape[0]
    repetitions = []
    for offset in range(1, n):
        diag = np.diag(ssm, k=offset)
        run_start = None
        run_len = 0
        for i, val in enumerate(diag):
            if val > 0.7:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_start is not None and run_len >= min_length:
                    repetitions.append((run_start, run_start + offset, run_len))
                run_start = None
                run_len = 0
        if run_start is not None and run_len >= min_length:
            repetitions.append((run_start, run_start + offset, run_len))
    return repetitions


class FormAnalyzer:
    """Analyze musical form by clustering segments and assigning labels."""

    def __init__(self, n_clusters: int = 4) -> None:
        self.n_clusters = n_clusters

    def label_sections(
        self,
        features: np.ndarray,
        boundaries: list[int],
    ) -> list[SectionLabel]:
        if len(boundaries) < 1:
            return []

        ends = list(boundaries[1:]) + [features.shape[0]]
        segments = []
        centroids = []
        for start, end in zip(boundaries, ends):
            seg_features = features[start:end]
            if len(seg_features) > 0:
                centroids.append(seg_features.mean(axis=0))
            else:
                centroids.append(np.zeros(features.shape[1]))
            segments.append((start, end))

        if not centroids:
            return []

        centroids = np.array(centroids)
        cluster_ids = self._cluster(centroids)

        labels = []
        cluster_counts: dict[int, int] = {}
        for (start, end), cid in zip(segments, cluster_ids):
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            letter = chr(ord("A") + (cid % 26))
            labels.append(
                SectionLabel(
                    start=start,
                    end=end,
                    label=letter,
                    cluster_id=int(cid),
                    repetition_count=cluster_counts[cid],
                )
            )

        return labels

    def _cluster(self, centroids: np.ndarray) -> list[int]:
        n = len(centroids)
        k = min(self.n_clusters, n)
        if k <= 1:
            return [0] * n

        rng = np.random.RandomState(42)
        indices = rng.choice(n, k, replace=False)
        centers = centroids[indices].copy()

        for _ in range(20):
            dists = np.zeros((n, k))
            for j in range(k):
                diff = centroids - centers[j]
                dists[:, j] = np.sum(diff ** 2, axis=1)
            assignments = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    new_centers[j] = centroids[mask].mean(axis=0)
                else:
                    new_centers[j] = centers[j]

            if np.allclose(new_centers, centers):
                break
            centers = new_centers

        return assignments.tolist()

    def identify_chorus(self, sections: list[SectionLabel]) -> list[int]:
        if not sections:
            return []
        label_counts: dict[str, int] = {}
        for s in sections:
            label_counts[s.label] = label_counts.get(s.label, 0) + 1
        most_common = max(label_counts, key=label_counts.get)
        return [i for i, s in enumerate(sections) if s.label == most_common]

    def match_template(
        self,
        sections: list[SectionLabel],
        templates: list[FormTemplate] | None = None,
    ) -> FormTemplate:
        if templates is None:
            templates = COMMON_TEMPLATES
        pattern = [s.label for s in sections]
        best = FormTemplate(name="unknown", pattern=pattern, match_score=0.0)

        for tmpl in templates:
            score = self._template_similarity(pattern, tmpl.pattern)
            if score > best.match_score:
                best = FormTemplate(name=tmpl.name, pattern=tmpl.pattern, match_score=score)

        return best

    @staticmethod
    def _template_similarity(actual: list[str], template: list[str]) -> float:
        if not actual or not template:
            return 0.0
        n = min(len(actual), len(template))
        actual_norm = []
        template_norm = []
        mapping_a: dict[str, int] = {}
        mapping_t: dict[str, int] = {}
        for a in actual[:n]:
            if a not in mapping_a:
                mapping_a[a] = len(mapping_a)
            actual_norm.append(mapping_a[a])
        for t in template[:n]:
            if t not in mapping_t:
                mapping_t[t] = len(mapping_t)
            template_norm.append(mapping_t[t])
        matches = sum(1 for a, t in zip(actual_norm, template_norm) if a == t)
        length_penalty = n / max(len(actual), len(template))
        return (matches / n) * length_penalty
