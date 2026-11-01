"""Semantic similarity between musical concepts: distributional semantics,
conceptual metaphor mapping, taxonomic distance, feature-based similarity.

Pure numpy — no external NLP frameworks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ConceptType(str, Enum):
    GENRE = "genre"
    INSTRUMENT = "instrument"
    TECHNIQUE = "technique"
    EMOTION = "emotion"
    FORM = "form"
    TEXTURE = "texture"
    TEMPO = "tempo"
    DYNAMIC = "dynamic"


@dataclass
class MusicConcept:
    name: str
    concept_type: ConceptType
    features: Dict[str, float] = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)
    siblings: List[str] = field(default_factory=list)


@dataclass
class SimilarityResult:
    concept_a: str
    concept_b: str
    taxonomic: float
    feature_based: float
    distributional: float
    combined: float


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def feature_vector(concept: MusicConcept, feature_names: List[str]) -> np.ndarray:
    return np.array([concept.features.get(f, 0.0) for f in feature_names])


def feature_similarity(a: MusicConcept, b: MusicConcept,
                       feature_names: List[str]) -> float:
    va = feature_vector(a, feature_names)
    vb = feature_vector(b, feature_names)
    return _cosine_similarity(va, vb)


def taxonomic_distance(a: MusicConcept, b: MusicConcept,
                       taxonomy: Dict[str, List[str]]) -> float:
    def ancestors(name: str) -> List[str]:
        path = [name]
        current = name
        visited = {name}
        while current in taxonomy:
            parents = taxonomy[current]
            if not parents:
                break
            p = parents[0]
            if p in visited:
                break
            visited.add(p)
            path.append(p)
            current = p
        return path

    path_a = ancestors(a.name)
    path_b = ancestors(b.name)
    set_a = set(path_a)
    set_b = set(path_b)
    common = set_a & set_b

    if not common:
        return 1.0

    depth_a = min(path_a.index(c) for c in common)
    depth_b = min(path_b.index(c) for c in common)
    total_path = depth_a + depth_b
    max_depth = max(len(path_a), len(path_b))
    if max_depth == 0:
        return 0.0
    return total_path / (2.0 * max_depth)


def taxonomic_similarity(a: MusicConcept, b: MusicConcept,
                         taxonomy: Dict[str, List[str]]) -> float:
    return 1.0 - taxonomic_distance(a, b, taxonomy)


def wu_palmer_similarity(a: MusicConcept, b: MusicConcept,
                         taxonomy: Dict[str, List[str]]) -> float:
    def depth(name: str) -> int:
        d = 0
        current = name
        visited = {name}
        while current in taxonomy and taxonomy[current]:
            current = taxonomy[current][0]
            if current in visited:
                break
            visited.add(current)
            d += 1
        return d

    def lca_depth(na: str, nb: str) -> int:
        ancestors_a = set()
        current = na
        visited = {na}
        while True:
            ancestors_a.add(current)
            if current not in taxonomy or not taxonomy[current]:
                break
            current = taxonomy[current][0]
            if current in visited:
                break
            visited.add(current)

        current = nb
        visited2 = {nb}
        while True:
            if current in ancestors_a:
                return depth(current)
            if current not in taxonomy or not taxonomy[current]:
                break
            current = taxonomy[current][0]
            if current in visited2:
                break
            visited2.add(current)
        return 0

    da = depth(a.name)
    db = depth(b.name)
    dl = lca_depth(a.name, b.name)
    denom = da + db + 2 * dl
    if denom == 0:
        return 1.0
    return (2.0 * dl) / denom


def distributional_similarity(co_occurrence_a: np.ndarray,
                              co_occurrence_b: np.ndarray) -> float:
    return _cosine_similarity(co_occurrence_a, co_occurrence_b)


def pointwise_mutual_information(co_matrix: np.ndarray) -> np.ndarray:
    total = co_matrix.sum()
    if total < 1e-12:
        return np.zeros_like(co_matrix, dtype=np.float64)
    row_sums = co_matrix.sum(axis=1, keepdims=True)
    col_sums = co_matrix.sum(axis=0, keepdims=True)
    expected = (row_sums * col_sums) / total
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2(co_matrix / (expected + 1e-12) + 1e-12)
    pmi = np.maximum(pmi, 0.0)
    return pmi


def concept_embedding(pmi_matrix: np.ndarray, dim: int = 50) -> np.ndarray:
    U, s, _ = np.linalg.svd(pmi_matrix, full_matrices=False)
    k = min(dim, len(s))
    return U[:, :k] * np.sqrt(s[:k])


def embedding_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    return normed @ normed.T


def metaphor_mapping(source_relations: np.ndarray,
                     target_relations: np.ndarray) -> Tuple[float, np.ndarray]:
    n = source_relations.shape[0]
    m = target_relations.shape[0]
    k = min(n, m)

    source_flat = source_relations.flatten()
    target_flat = target_relations[:k, :k].flatten() if k < m else target_relations.flatten()

    if len(source_flat) != len(target_flat):
        min_len = min(len(source_flat), len(target_flat))
        source_flat = source_flat[:min_len]
        target_flat = target_flat[:min_len]

    alignment = _cosine_similarity(source_flat, target_flat)

    mapping = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s_row = source_relations[i, :]
            t_row = target_relations[j, :min(n, m)]
            if len(s_row) > len(t_row):
                s_row = s_row[:len(t_row)]
            elif len(t_row) > len(s_row):
                t_row = t_row[:len(s_row)]
            mapping[i, j] = _cosine_similarity(s_row, t_row)
    return alignment, mapping


def combined_similarity(a: MusicConcept, b: MusicConcept,
                        feature_names: List[str],
                        taxonomy: Dict[str, List[str]],
                        co_a: Optional[np.ndarray] = None,
                        co_b: Optional[np.ndarray] = None,
                        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
                        ) -> SimilarityResult:
    tax_sim = taxonomic_similarity(a, b, taxonomy)
    feat_sim = feature_similarity(a, b, feature_names)

    if co_a is not None and co_b is not None:
        dist_sim = distributional_similarity(co_a, co_b)
    else:
        dist_sim = 0.0
        weights = (weights[0] + weights[2] / 2, weights[1] + weights[2] / 2, 0.0)

    w_sum = sum(weights)
    if w_sum < 1e-12:
        comb = 0.0
    else:
        comb = (weights[0] * tax_sim + weights[1] * feat_sim +
                weights[2] * dist_sim) / w_sum

    return SimilarityResult(
        concept_a=a.name, concept_b=b.name,
        taxonomic=tax_sim, feature_based=feat_sim,
        distributional=dist_sim, combined=comb,
    )


def pairwise_concept_similarity(concepts: List[MusicConcept],
                                feature_names: List[str],
                                taxonomy: Dict[str, List[str]],
                                ) -> np.ndarray:
    n = len(concepts)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = combined_similarity(concepts[i], concepts[j],
                                    feature_names, taxonomy)
            sim[i, j] = r.combined
            sim[j, i] = r.combined
    return sim


def concept_clustering(sim_matrix: np.ndarray,
                       n_clusters: int = 5) -> np.ndarray:
    n = sim_matrix.shape[0]
    dist = 1.0 - sim_matrix
    np.fill_diagonal(dist, 0.0)

    labels = np.arange(n)
    cluster_map = {i: [i] for i in range(n)}
    active = set(range(n))

    while len(active) > n_clusters:
        best_dist = np.inf
        best_pair = (0, 0)
        active_list = sorted(active)
        for idx_i, i in enumerate(active_list):
            for j in active_list[idx_i + 1:]:
                d = dist[i, j]
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        ci, cj = best_pair
        for member in cluster_map[cj]:
            labels[member] = ci
        cluster_map[ci].extend(cluster_map[cj])
        del cluster_map[cj]
        active.remove(cj)

        for k in active:
            if k != ci:
                dist[ci, k] = min(dist[ci, k], dist[cj, k])
                dist[k, ci] = dist[ci, k]

    label_map = {}
    for new_label, old_label in enumerate(sorted(cluster_map.keys())):
        label_map[old_label] = new_label
    return np.array([label_map[labels[i]] for i in range(n)])
