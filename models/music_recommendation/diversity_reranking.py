"""Diversity-aware recommendation re-ranking.

MMR (Maximal Marginal Relevance), DPP-based selection, intra-list
diversity metrics, and calibration for genre/mood coverage.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RankedItem:
    item_id: str
    relevance: float
    features: np.ndarray  # embedding / descriptor vector


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_rerank(
    candidates: List[RankedItem],
    k: int,
    lam: float = 0.5,
) -> List[RankedItem]:
    """Maximal Marginal Relevance re-ranking.

    lam = 1 → pure relevance; lam = 0 → pure diversity.
    """
    if k >= len(candidates):
        return list(candidates)

    selected: List[RankedItem] = []
    remaining = list(candidates)

    for _ in range(k):
        best_score = -np.inf
        best_idx = 0
        for i, cand in enumerate(remaining):
            rel = cand.relevance
            if selected:
                max_sim = max(cosine_sim(cand.features, s.features) for s in selected)
            else:
                max_sim = 0.0
            score = lam * rel - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


def intralist_diversity(items: List[RankedItem]) -> float:
    """Average pairwise distance in the selected list."""
    if len(items) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            total += 1.0 - cosine_sim(items[i].features, items[j].features)
            count += 1
    return total / count


def coverage_score(
    items: List[RankedItem],
    category_fn,
    all_categories: set,
) -> float:
    """Fraction of target categories represented in the recommendation list."""
    if not all_categories:
        return 1.0
    covered = set()
    for item in items:
        covered.add(category_fn(item))
    return len(covered & all_categories) / len(all_categories)


def calibration_error(
    items: List[RankedItem],
    target_distribution: Dict[str, float],
    category_fn,
) -> float:
    """KL-divergence between recommended category distribution and target."""
    eps = 1e-10
    cats = [category_fn(item) for item in items]
    n = len(cats)
    if n == 0:
        return 0.0
    freq: Dict[str, float] = {}
    for c in cats:
        freq[c] = freq.get(c, 0) + 1.0 / n

    all_cats = set(list(target_distribution.keys()) + list(freq.keys()))
    kl = 0.0
    for c in all_cats:
        p = target_distribution.get(c, eps)
        q = freq.get(c, eps)
        kl += p * np.log(p / q)
    return float(kl)


def dpp_quality_diversity_score(items: List[RankedItem]) -> float:
    """Log-determinant of the kernel matrix (DPP quality × diversity)."""
    if len(items) == 0:
        return 0.0
    n = len(items)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim = cosine_sim(items[i].features, items[j].features)
            L[i, j] = items[i].relevance * items[j].relevance * sim
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.maximum(eigvals, 1e-12)
    return float(np.sum(np.log(eigvals)))


def novelty_rerank(
    candidates: List[RankedItem],
    user_history: List[np.ndarray],
    k: int,
    alpha: float = 0.3,
) -> List[RankedItem]:
    """Re-rank by novelty: penalise items similar to user history."""
    scored = []
    for cand in candidates:
        if user_history:
            max_hist_sim = max(cosine_sim(cand.features, h) for h in user_history)
        else:
            max_hist_sim = 0.0
        adj_score = cand.relevance - alpha * max_hist_sim
        scored.append((adj_score, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]
