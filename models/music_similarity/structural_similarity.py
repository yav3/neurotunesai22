"""Structural similarity: compare long-range form between pieces.

Quantify similarity of musical structures via section-graph alignment,
transition-matrix distance, form-type compatibility, and proportional
arc comparison for playlist sequencing and version detection.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class StructuralProfile:
    section_labels: List[str]
    section_durations: np.ndarray
    transition_matrix: np.ndarray


def section_label_edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def normalised_edit_similarity(a: List[str], b: List[str]) -> float:
    d = section_label_edit_distance(a, b)
    return 1.0 - d / max(len(a), len(b), 1)


def proportion_vector(durations: np.ndarray) -> np.ndarray:
    total = np.sum(durations)
    if total < 1e-10:
        return np.zeros_like(durations)
    return durations / total


def proportion_distance(a: np.ndarray, b: np.ndarray) -> float:
    pa = proportion_vector(a)
    pb = proportion_vector(b)
    n = max(len(pa), len(pb))
    va = np.zeros(n)
    vb = np.zeros(n)
    va[:len(pa)] = pa
    vb[:len(pb)] = pb
    return float(np.linalg.norm(va - vb))


def transition_matrix_from_labels(labels: List[str]) -> np.ndarray:
    unique = sorted(set(labels))
    idx = {l: i for i, l in enumerate(unique)}
    n = len(unique)
    mat = np.zeros((n, n))
    for i in range(len(labels) - 1):
        mat[idx[labels[i]], idx[labels[i + 1]]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums


def transition_matrix_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = max(a.shape[0], b.shape[0])
    ma = np.zeros((n, n))
    mb = np.zeros((n, n))
    ma[:a.shape[0], :a.shape[1]] = a
    mb[:b.shape[0], :b.shape[1]] = b
    return float(np.linalg.norm(ma - mb, "fro"))


def structural_similarity(profile_a: StructuralProfile,
                          profile_b: StructuralProfile,
                          w_label: float = 0.4,
                          w_prop: float = 0.3,
                          w_trans: float = 0.3) -> float:
    label_sim = normalised_edit_similarity(
        profile_a.section_labels, profile_b.section_labels)
    prop_dist = proportion_distance(
        profile_a.section_durations, profile_b.section_durations)
    prop_sim = 1.0 / (1.0 + prop_dist)
    trans_dist = transition_matrix_distance(
        profile_a.transition_matrix, profile_b.transition_matrix)
    trans_sim = 1.0 / (1.0 + trans_dist)
    return w_label * label_sim + w_prop * prop_sim + w_trans * trans_sim


def form_compatibility(labels_a: List[str], labels_b: List[str]) -> float:
    set_a = set(labels_a)
    set_b = set(labels_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = intersection / union
    len_ratio = min(len(labels_a), len(labels_b)) / max(len(labels_a), len(labels_b))
    return 0.5 * jaccard + 0.5 * len_ratio
