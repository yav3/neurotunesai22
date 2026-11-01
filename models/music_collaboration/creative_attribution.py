"""Creative attribution and contribution scoring for collaborative music.

Measures each collaborator's originality, influence graph, contribution
weight via edit distance, and novelty metrics for individual additions.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Contribution:
    collaborator_id: str
    timestamp: float
    content_hash: int
    n_notes: int
    novelty_score: float  # 0-1


@dataclass
class AttributionResult:
    collaborator_scores: Dict[str, float]
    influence_matrix: np.ndarray  # (n_collab, n_collab)
    collaborator_ids: List[str]


def edit_distance_notes(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """Levenshtein edit distance between two note sequences (MIDI pitches)."""
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


def contribution_weight(
    before: np.ndarray,
    after: np.ndarray,
    total_length: int,
) -> float:
    """Fraction of the final piece attributable to a single edit."""
    if total_length == 0:
        return 0.0
    dist = edit_distance_notes(before, after)
    return min(dist / max(total_length, 1), 1.0)


def novelty_score(
    new_content: np.ndarray,
    existing_content: np.ndarray,
    n_pitch_classes: int = 12,
) -> float:
    """Score how novel a contribution is relative to existing content.

    Uses pitch-class distribution divergence (KL-like).
    """
    if len(new_content) == 0:
        return 0.0
    hist_new = np.bincount(np.asarray(new_content, dtype=int) % n_pitch_classes, minlength=n_pitch_classes).astype(float)
    hist_new /= max(hist_new.sum(), 1)
    if len(existing_content) == 0:
        return 1.0
    hist_old = np.bincount(np.asarray(existing_content, dtype=int) % n_pitch_classes, minlength=n_pitch_classes).astype(float)
    hist_old /= max(hist_old.sum(), 1)
    # symmetric KL
    eps = 1e-10
    p = hist_new + eps
    q = hist_old + eps
    kl = 0.5 * np.sum(p * np.log(p / q)) + 0.5 * np.sum(q * np.log(q / p))
    return float(np.tanh(kl))


def build_influence_matrix(
    contributions: List[Contribution],
) -> Tuple[np.ndarray, List[str]]:
    """Build a collaborator influence matrix from sequential contributions.

    influence[i, j] = how much collaborator j's work built on i's.
    """
    ids = sorted(set(c.collaborator_id for c in contributions))
    n = len(ids)
    idx = {cid: i for i, cid in enumerate(ids)}
    matrix = np.zeros((n, n))

    sorted_contribs = sorted(contributions, key=lambda c: c.timestamp)
    for k in range(1, len(sorted_contribs)):
        cur = sorted_contribs[k]
        prev = sorted_contribs[k - 1]
        if cur.collaborator_id != prev.collaborator_id:
            i = idx[prev.collaborator_id]
            j = idx[cur.collaborator_id]
            matrix[i, j] += 1.0

    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    matrix /= col_sums
    return matrix, ids


def compute_attribution(
    contributions: List[Contribution],
) -> AttributionResult:
    """Compute overall attribution scores for all collaborators."""
    influence, ids = build_influence_matrix(contributions)
    idx = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)

    raw_scores = np.zeros(n)
    for c in contributions:
        i = idx[c.collaborator_id]
        raw_scores[i] += c.n_notes * (0.5 + 0.5 * c.novelty_score)

    total = raw_scores.sum()
    if total > 0:
        raw_scores /= total

    return AttributionResult(
        collaborator_scores={cid: float(raw_scores[i]) for cid, i in idx.items()},
        influence_matrix=influence,
        collaborator_ids=ids,
    )


def gini_coefficient(scores: np.ndarray) -> float:
    """Gini coefficient of contribution distribution; 0 = equal, 1 = monopoly."""
    if len(scores) < 2:
        return 0.0
    sorted_s = np.sort(scores)
    n = len(sorted_s)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_s) - (n + 1) * np.sum(sorted_s)) / (n * np.sum(sorted_s) + 1e-12))
