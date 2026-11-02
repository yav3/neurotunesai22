"""Music tag taxonomy: hierarchical tags, PMI similarity, label propagation, folksonomy stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "TagNode", "TagSimilarity", "FolksonomyStats",
    "build_taxonomy", "tag_cooccurrence_matrix",
    "tag_pmi", "propagate_tags", "normalize_tag",
    "folksonomy_stats", "tag_hierarchy_depth",
]


@dataclass
class TagNode:
    name: str
    parent: Optional[str] = None
    depth: int = 0
    children: List[str] = field(default_factory=list)


@dataclass
class TagSimilarity:
    tag_a: str
    tag_b: str
    pmi: float
    cooccurrence_count: int


@dataclass
class FolksonomyStats:
    n_tags: int
    n_unique: int
    type_token_ratio: float
    long_tail_ratio: float
    entropy: float


SYNONYM_MAP: Dict[str, str] = {
    "hiphop": "hip-hop", "hip hop": "hip-hop",
    "rnb": "r&b", "r and b": "r&b", "rhythm and blues": "r&b",
    "edm": "electronic", "electronica": "electronic",
    "alt rock": "alternative rock", "alt-rock": "alternative rock",
    "lo-fi": "lofi", "lo fi": "lofi",
    "drum n bass": "drum and bass", "dnb": "drum and bass",
    "prog rock": "progressive rock",
}


def normalize_tag(tag: str) -> str:
    t = tag.strip().lower()
    return SYNONYM_MAP.get(t, t)


def build_taxonomy(edges: List[Tuple[str, str]]) -> Dict[str, TagNode]:
    nodes: Dict[str, TagNode] = {}
    for parent, child in edges:
        if parent not in nodes:
            nodes[parent] = TagNode(parent)
        if child not in nodes:
            nodes[child] = TagNode(child, parent=parent)
        else:
            nodes[child].parent = parent
        nodes[parent].children.append(child)
    for name, node in nodes.items():
        depth = 0
        current = name
        while nodes.get(current) and nodes[current].parent:
            current = nodes[current].parent
            depth += 1
            if depth > 100:
                break
        node.depth = depth
    return nodes


def tag_hierarchy_depth(taxonomy: Dict[str, TagNode]) -> int:
    if not taxonomy:
        return 0
    return max(n.depth for n in taxonomy.values())


def tag_cooccurrence_matrix(
    tagged_items: List[Set[str]], vocabulary: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    if vocabulary is None:
        all_tags: Set[str] = set()
        for tags in tagged_items:
            all_tags.update(tags)
        vocabulary = sorted(all_tags)
    tag_idx = {t: i for i, t in enumerate(vocabulary)}
    n = len(vocabulary)
    matrix = np.zeros((n, n), dtype=np.float64)
    for tags in tagged_items:
        present = [tag_idx[t] for t in tags if t in tag_idx]
        for i in present:
            for j in present:
                matrix[i, j] += 1.0
    return matrix, vocabulary


def tag_pmi(
    cooccurrence: np.ndarray, vocabulary: List[str], min_count: int = 2
) -> List[TagSimilarity]:
    n = cooccurrence.shape[0]
    total = np.sum(cooccurrence) / 2.0
    if total == 0:
        return []
    marginals = np.diag(cooccurrence).copy()
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            c_ij = cooccurrence[i, j]
            if c_ij < min_count:
                continue
            p_ij = c_ij / total
            p_i = marginals[i] / total
            p_j = marginals[j] / total
            denom = p_i * p_j
            if denom <= 0:
                continue
            pmi_val = float(np.log2(p_ij / denom))
            results.append(TagSimilarity(vocabulary[i], vocabulary[j], pmi_val, int(c_ij)))
    results.sort(key=lambda s: s.pmi, reverse=True)
    return results


def propagate_tags(
    known_tags: Dict[int, Set[str]],
    similarity_matrix: np.ndarray,
    n_items: int,
    n_iterations: int = 10,
    alpha: float = 0.5,
) -> Dict[int, Dict[str, float]]:
    all_tags: Set[str] = set()
    for tags in known_tags.values():
        all_tags.update(tags)
    tag_list = sorted(all_tags)
    tag_idx = {t: i for i, t in enumerate(tag_list)}
    n_tags = len(tag_list)
    labels = np.zeros((n_items, n_tags))
    for item_id, tags in known_tags.items():
        if 0 <= item_id < n_items:
            for t in tags:
                if t in tag_idx:
                    labels[item_id, tag_idx[t]] = 1.0
    initial = labels.copy()
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    W = similarity_matrix / row_sums
    for _ in range(n_iterations):
        labels = alpha * (W @ labels) + (1 - alpha) * initial
    result = {}
    for i in range(n_items):
        if np.max(labels[i]) > 0.01:
            scores = {}
            for j in range(n_tags):
                if labels[i, j] > 0.01:
                    scores[tag_list[j]] = float(labels[i, j])
            if scores:
                result[i] = scores
    return result


def folksonomy_stats(tag_counts: Dict[str, int]) -> FolksonomyStats:
    if not tag_counts:
        return FolksonomyStats(0, 0, 0.0, 0.0, 0.0)
    n_total = sum(tag_counts.values())
    n_unique = len(tag_counts)
    ttr = n_unique / max(n_total, 1)
    counts = np.array(sorted(tag_counts.values(), reverse=True), dtype=np.float64)
    cumulative = np.cumsum(counts) / counts.sum()
    long_tail_idx = np.searchsorted(cumulative, 0.8)
    long_tail_ratio = 1.0 - (long_tail_idx + 1) / n_unique if n_unique > 0 else 0.0
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))
    return FolksonomyStats(n_total, n_unique, ttr, long_tail_ratio, entropy)
