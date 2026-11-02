"""Musical form visualization: section diagrams, hierarchical trees, repetition maps."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FormSection:
    label: str
    start_beat: float
    end_beat: float
    level: int = 0
    parent_label: str = ""
    similarity_to_first: float = 1.0


@dataclass
class FormTree:
    label: str
    children: list[FormTree] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    @property
    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    def leaves(self) -> list[FormTree]:
        if not self.children:
            return [self]
        out: list[FormTree] = []
        for c in self.children:
            out.extend(c.leaves())
        return out


COMMON_FORMS: dict[str, list[str]] = {
    "binary": ["A", "B"],
    "ternary": ["A", "B", "A"],
    "rondo": ["A", "B", "A", "C", "A"],
    "aaba": ["A", "A", "B", "A"],
    "verse_chorus": ["V", "C", "V", "C"],
    "sonata": ["Exposition", "Development", "Recapitulation"],
    "strophic": ["A", "A", "A"],
    "through_composed": ["A", "B", "C", "D"],
    "twelve_bar_blues": ["I", "I", "I", "I", "IV", "IV", "I", "I", "V", "IV", "I", "V"],
}


def section_boundaries(labels: list[str], beats_per_section: float = 16.0) -> list[FormSection]:
    sections = []
    for i, label in enumerate(labels):
        first_idx = labels.index(label)
        sim = 1.0 if i == first_idx else 0.8
        sections.append(FormSection(
            label=label,
            start_beat=i * beats_per_section,
            end_beat=(i + 1) * beats_per_section,
            level=0,
            similarity_to_first=sim,
        ))
    return sections


def build_form_tree(sections: list[FormSection], piece_label: str = "Piece") -> FormTree:
    root = FormTree(piece_label, start=0.0,
                    end=sections[-1].end_beat if sections else 0.0)
    for sec in sections:
        child = FormTree(sec.label, start=sec.start_beat, end=sec.end_beat)
        root.children.append(child)
    return root


def repetition_matrix(features: np.ndarray) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.array([])
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1e-10)
    normed = features / norms
    return normed @ normed.T


def novelty_curve(self_sim: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    n = self_sim.shape[0]
    if n < kernel_size:
        return np.zeros(n)
    half = kernel_size // 2
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = 1
    kernel[half:, half:] = 1
    kernel[:half, half:] = -1
    kernel[half:, :half] = -1
    novelty = np.zeros(n)
    for i in range(half, n - half):
        patch = self_sim[i - half:i + half, i - half:i + half]
        novelty[i] = float(np.sum(patch * kernel))
    return np.maximum(novelty, 0)


def section_similarity_matrix(sections: list[FormSection]) -> np.ndarray:
    n = len(sections)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = 1.0 if sections[i].label == sections[j].label else 0.0
    return sim


def classify_form(labels: list[str]) -> str:
    for name, pattern in COMMON_FORMS.items():
        if labels == pattern:
            return name
    unique = []
    for lb in labels:
        if lb not in unique:
            unique.append(lb)
    if len(unique) == 1:
        return "strophic"
    if len(unique) == len(labels):
        return "through_composed"
    if labels[0] == labels[-1] and len(unique) == 2:
        return "ternary"
    return "free"
