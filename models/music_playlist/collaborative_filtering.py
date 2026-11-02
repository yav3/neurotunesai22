"""Collaborative filtering for playlist recommendation and user taste modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "UserProfile", "ItemProfile", "RatingMatrix", "CFConfig",
    "build_rating_matrix", "user_similarity", "item_similarity",
    "user_based_predict", "item_based_predict",
    "matrix_factorization_als", "implicit_feedback_scores",
    "cold_start_fallback", "popularity_baseline",
    "ndcg_at_k", "precision_at_k", "recall_at_k",
    "mean_reciprocal_rank", "catalog_coverage",
]


@dataclass
class UserProfile:
    user_id: str
    listened: Dict[str, float] = field(default_factory=dict)
    skip_history: Dict[str, int] = field(default_factory=dict)


@dataclass
class ItemProfile:
    item_id: str
    features: np.ndarray = field(default_factory=lambda: np.zeros(0))
    popularity: float = 0.0


@dataclass
class RatingMatrix:
    matrix: np.ndarray
    user_ids: List[str]
    item_ids: List[str]


@dataclass
class CFConfig:
    n_factors: int = 32
    regularization: float = 0.01
    n_iterations: int = 20
    learning_rate: float = 0.01


def build_rating_matrix(
    profiles: List[UserProfile], all_items: List[str],
) -> RatingMatrix:
    user_ids = [p.user_id for p in profiles]
    item_map = {it: i for i, it in enumerate(all_items)}
    mat = np.zeros((len(profiles), len(all_items)))
    for u, prof in enumerate(profiles):
        for item_id, rating in prof.listened.items():
            if item_id in item_map:
                mat[u, item_map[item_id]] = rating
    return RatingMatrix(matrix=mat, user_ids=user_ids, item_ids=all_items)


def user_similarity(R: np.ndarray, metric: str = "cosine") -> np.ndarray:
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = R / norms
    if metric == "cosine":
        return normed @ normed.T
    elif metric == "pearson":
        means = R.mean(axis=1, keepdims=True)
        centered = R - means
        cn = np.linalg.norm(centered, axis=1, keepdims=True)
        cn = np.where(cn == 0, 1.0, cn)
        return (centered / cn) @ (centered / cn).T
    return normed @ normed.T


def item_similarity(R: np.ndarray) -> np.ndarray:
    return user_similarity(R.T)


def user_based_predict(
    R: np.ndarray, sim: np.ndarray, user_idx: int, item_idx: int, k: int = 20,
) -> float:
    scores = sim[user_idx].copy()
    scores[user_idx] = -np.inf
    neighbors = np.argsort(scores)[-k:]
    ratings = R[neighbors, item_idx]
    weights = scores[neighbors]
    mask = ratings > 0
    if not mask.any():
        return 0.0
    w = weights[mask]
    r = ratings[mask]
    denom = np.abs(w).sum()
    return float(w @ r / denom) if denom > 0 else 0.0


def item_based_predict(
    R: np.ndarray, sim_items: np.ndarray, user_idx: int, item_idx: int, k: int = 20,
) -> float:
    user_ratings = R[user_idx]
    rated = np.where(user_ratings > 0)[0]
    if len(rated) == 0:
        return 0.0
    sims = sim_items[item_idx, rated]
    top_k = np.argsort(sims)[-k:]
    w = sims[top_k]
    r = user_ratings[rated[top_k]]
    denom = np.abs(w).sum()
    return float(w @ r / denom) if denom > 0 else 0.0


def matrix_factorization_als(
    R: np.ndarray, config: Optional[CFConfig] = None, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = config or CFConfig()
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    U = rng.normal(0, 0.1, (n_users, cfg.n_factors))
    V = rng.normal(0, 0.1, (n_items, cfg.n_factors))
    mask = R > 0
    lam = cfg.regularization
    for _ in range(cfg.n_iterations):
        for u in range(n_users):
            rated = np.where(mask[u])[0]
            if len(rated) == 0:
                continue
            V_rated = V[rated]
            A = V_rated.T @ V_rated + lam * np.eye(cfg.n_factors)
            b = V_rated.T @ R[u, rated]
            U[u] = np.linalg.solve(A, b)
        for i in range(n_items):
            users_who_rated = np.where(mask[:, i])[0]
            if len(users_who_rated) == 0:
                continue
            U_rated = U[users_who_rated]
            A = U_rated.T @ U_rated + lam * np.eye(cfg.n_factors)
            b = U_rated.T @ R[users_who_rated, i]
            V[i] = np.linalg.solve(A, b)
    return U, V


def implicit_feedback_scores(
    listen_counts: np.ndarray, alpha: float = 40.0,
) -> np.ndarray:
    return 1.0 + alpha * np.log1p(listen_counts)


def cold_start_fallback(
    item_features: np.ndarray, known_items: np.ndarray,
    known_ratings: np.ndarray, k: int = 10,
) -> np.ndarray:
    norms_known = np.linalg.norm(known_items, axis=1, keepdims=True)
    norms_known = np.where(norms_known == 0, 1.0, norms_known)
    norms_new = np.linalg.norm(item_features, keepdims=True)
    norms_new = np.where(norms_new == 0, 1.0, norms_new)
    sims = (known_items / norms_known) @ (item_features / norms_new).T
    sims = sims.ravel()
    top_k = np.argsort(sims)[-k:]
    w = sims[top_k]
    denom = w.sum()
    return float(w @ known_ratings[top_k] / denom) if denom > 0 else 0.0


def popularity_baseline(R: np.ndarray) -> np.ndarray:
    counts = (R > 0).sum(axis=0).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def ndcg_at_k(relevant: np.ndarray, predicted_order: np.ndarray, k: int = 10) -> float:
    k = min(k, len(predicted_order))
    dcg = 0.0
    for i in range(k):
        rel = float(relevant[predicted_order[i]]) if predicted_order[i] < len(relevant) else 0.0
        dcg += rel / np.log2(i + 2)
    ideal_order = np.argsort(relevant)[::-1]
    idcg = 0.0
    for i in range(k):
        rel = float(relevant[ideal_order[i]]) if i < len(relevant) else 0.0
        idcg += rel / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevant_set: set, predicted: List, k: int = 10) -> float:
    top_k = predicted[:k]
    hits = sum(1 for p in top_k if p in relevant_set)
    return hits / k if k > 0 else 0.0


def recall_at_k(relevant_set: set, predicted: List, k: int = 10) -> float:
    top_k = predicted[:k]
    hits = sum(1 for p in top_k if p in relevant_set)
    return hits / len(relevant_set) if len(relevant_set) > 0 else 0.0


def mean_reciprocal_rank(relevant_set: set, predicted: List) -> float:
    for i, p in enumerate(predicted):
        if p in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def catalog_coverage(recommended_items: set, total_items: int) -> float:
    return len(recommended_items) / total_items if total_items > 0 else 0.0
