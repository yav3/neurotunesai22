"""Turn-based and simultaneous collaborative composition models."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContributionType(Enum):
    MELODY = "melody"
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    LYRICS = "lyrics"
    ARRANGEMENT = "arrangement"
    MIX = "mix"
    PRODUCTION = "production"


@dataclass
class Contribution:
    author_id: str
    ctype: ContributionType
    timestamp: float
    content_hash: str
    parent_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CompositionGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, Contribution] = {}
        self._children: dict[str, list[str]] = {}

    def add_contribution(self, contrib: Contribution) -> str:
        h = contrib.content_hash
        self._nodes[h] = contrib
        if contrib.parent_hash:
            self._children.setdefault(contrib.parent_hash, []).append(h)
        return h

    def fork(self, from_hash: str, new_contrib: Contribution) -> str:
        new_contrib.parent_hash = from_hash
        return self.add_contribution(new_contrib)

    def merge(self, hash_a: str, hash_b: str, merged_contrib: Contribution) -> str:
        combined = hashlib.sha256(f"{hash_a}:{hash_b}".encode()).hexdigest()[:16]
        merged_contrib.content_hash = combined
        merged_contrib.parent_hash = hash_a
        self.add_contribution(merged_contrib)
        self._children.setdefault(hash_b, []).append(combined)
        return combined

    def lineage(self, h: str) -> list[str]:
        chain = []
        current = h
        while current and current in self._nodes:
            chain.append(current)
            current = self._nodes[current].parent_hash
        return list(reversed(chain))

    def conflicts(self, hash_a: str, hash_b: str) -> list[str]:
        lin_a = set(self.lineage(hash_a))
        lin_b = set(self.lineage(hash_b))
        diverged = lin_a.symmetric_difference(lin_b)
        return sorted(diverged)

    @property
    def authors(self) -> set[str]:
        return {n.author_id for n in self._nodes.values()}

    @property
    def contributions(self) -> list[Contribution]:
        return list(self._nodes.values())


def attribution_weights(graph: CompositionGraph, method: str = "shapley_approx") -> dict[str, float]:
    authors = graph.authors
    if not authors:
        return {}
    contribs_by_author: dict[str, int] = {}
    for c in graph.contributions:
        contribs_by_author[c.author_id] = contribs_by_author.get(c.author_id, 0) + 1
    total = sum(contribs_by_author.values())
    if total == 0:
        return {a: 1.0 / len(authors) for a in authors}
    if method == "equal":
        return {a: 1.0 / len(authors) for a in authors}
    type_weights = {
        ContributionType.MELODY: 1.5,
        ContributionType.HARMONY: 1.2,
        ContributionType.LYRICS: 1.5,
        ContributionType.RHYTHM: 1.0,
        ContributionType.ARRANGEMENT: 1.0,
        ContributionType.MIX: 0.8,
        ContributionType.PRODUCTION: 0.8,
    }
    weighted: dict[str, float] = {}
    for c in graph.contributions:
        w = type_weights.get(c.ctype, 1.0)
        weighted[c.author_id] = weighted.get(c.author_id, 0.0) + w
    total_w = sum(weighted.values())
    if total_w == 0:
        return {a: 1.0 / len(authors) for a in authors}
    return {a: v / total_w for a, v in weighted.items()}


def style_consistency(features_sequence: list[dict[str, float]]) -> float:
    if len(features_sequence) < 2:
        return 1.0
    keys = set()
    for f in features_sequence:
        keys.update(f.keys())
    if not keys:
        return 1.0
    vectors = []
    sorted_keys = sorted(keys)
    for f in features_sequence:
        vectors.append([f.get(k, 0.0) for k in sorted_keys])
    arr = np.array(vectors)
    diffs = np.diff(arr, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    if len(distances) == 0:
        return 1.0
    mean_dist = float(np.mean(distances))
    return float(math.exp(-mean_dist))


import numpy as np
