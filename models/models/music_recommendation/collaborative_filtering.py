"""Collaborative filtering for music recommendation: matrix factorization, k-NN, diversity."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UserItemMatrix:
    ratings: np.ndarray
    user_ids: List[str] = field(default_factory=list)
    item_ids: List[str] = field(default_factory=list)

    @property
    def n_users(self) -> int:
        return self.ratings.shape[0]

    @property
    def n_items(self) -> int:
        return self.ratings.shape[1]

    def density(self) -> float:
        if self.ratings.size == 0:
            return 0.0
        return float(np.count_nonzero(self.ratings) / self.ratings.size)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a != 0) & (b != 0)
    if mask.sum() < 2:
        return 0.0
    a_m = a[mask]
    b_m = b[mask]
    a_centered = a_m - a_m.mean()
    b_centered = b_m - b_m.mean()
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def adjusted_cosine_similarity(ratings: np.ndarray, item_i: int, item_j: int) -> float:
    col_i = ratings[:, item_i]
    col_j = ratings[:, item_j]
    mask = (col_i != 0) & (col_j != 0)
    if mask.sum() < 2:
        return 0.0
    user_means = np.array([ratings[u, ratings[u] != 0].mean() if np.any(ratings[u] != 0) else 0.0
                           for u in range(ratings.shape[0])])
    ci = col_i[mask] - user_means[mask]
    cj = col_j[mask] - user_means[mask]
    denom = np.linalg.norm(ci) * np.linalg.norm(cj)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(ci, cj) / denom)


def user_knn(matrix: UserItemMatrix, user_idx: int, k: int = 10,
             metric: str = "cosine") -> List[Tuple[int, float]]:
    sim_fn = cosine_similarity if metric == "cosine" else pearson_similarity
    sims = []
    for i in range(matrix.n_users):
        if i == user_idx:
            continue
        s = sim_fn(matrix.ratings[user_idx], matrix.ratings[i])
        sims.append((i, s))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


def predict_rating_user_knn(matrix: UserItemMatrix, user_idx: int, item_idx: int,
                            k: int = 10, metric: str = "cosine") -> float:
    neighbors = user_knn(matrix, user_idx, k, metric)
    num = 0.0
    denom = 0.0
    for neighbor_idx, sim in neighbors:
        r = matrix.ratings[neighbor_idx, item_idx]
        if r != 0 and sim > 0:
            num += sim * r
            denom += abs(sim)
    if denom < 1e-12:
        user_ratings = matrix.ratings[user_idx]
        nonzero = user_ratings[user_ratings != 0]
        return float(nonzero.mean()) if len(nonzero) > 0 else 0.0
    return float(num / denom)


def item_knn(matrix: UserItemMatrix, item_idx: int, k: int = 10) -> List[Tuple[int, float]]:
    sims = []
    for j in range(matrix.n_items):
        if j == item_idx:
            continue
        s = adjusted_cosine_similarity(matrix.ratings, item_idx, j)
        sims.append((j, s))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


@dataclass
class MFModel:
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray
    global_mean: float
    n_factors: int

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )

    def recommend(self, user_idx: int, n: int = 10,
                  exclude: Optional[set] = None) -> List[Tuple[int, float]]:
        scores = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.item_factors @ self.user_factors[user_idx]
        )
        exclude = exclude or set()
        ranked = [(j, float(scores[j])) for j in range(len(scores)) if j not in exclude]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:n]


def matrix_factorization(matrix: UserItemMatrix, n_factors: int = 20,
                         lr: float = 0.005, reg: float = 0.02,
                         n_epochs: int = 50) -> MFModel:
    n_u, n_i = matrix.n_users, matrix.n_items
    rng = np.random.default_rng(42)
    P = rng.normal(0, 0.1, (n_u, n_factors))
    Q = rng.normal(0, 0.1, (n_i, n_factors))
    bu = np.zeros(n_u)
    bi = np.zeros(n_i)

    nonzero = matrix.ratings[matrix.ratings != 0]
    mu = float(nonzero.mean()) if len(nonzero) > 0 else 0.0

    indices = list(zip(*np.nonzero(matrix.ratings)))
    for _ in range(n_epochs):
        rng.shuffle(indices)
        for u, i in indices:
            pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
            err = matrix.ratings[u, i] - pred
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            P_u_old = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * P_u_old - reg * Q[i])

    return MFModel(
        user_factors=P, item_factors=Q,
        user_bias=bu, item_bias=bi,
        global_mean=mu, n_factors=n_factors,
    )


def implicit_weight(confidence: float, alpha: float = 40.0) -> float:
    return 1.0 + alpha * confidence


def implicit_als_step(C: np.ndarray, P_pref: np.ndarray,
                      X: np.ndarray, Y: np.ndarray,
                      reg: float = 0.1) -> np.ndarray:
    n, k = X.shape
    X_new = np.zeros_like(X)
    YtY = Y.T @ Y
    for u in range(n):
        Cu = np.diag(C[u])
        A = YtY + Y.T @ Cu @ Y + reg * np.eye(k)
        b = Y.T @ Cu @ P_pref[u]
        X_new[u] = np.linalg.solve(A, b)
    return X_new


def cold_start_popularity(matrix: UserItemMatrix, n: int = 10) -> List[Tuple[int, float]]:
    popularity = np.count_nonzero(matrix.ratings, axis=0).astype(float)
    total = popularity.sum()
    if total > 0:
        popularity /= total
    ranked = [(j, float(popularity[j])) for j in range(matrix.n_items)]
    ranked.sort(key=lambda x: -x[1])
    return ranked[:n]


def cold_start_content_bridge(item_features: np.ndarray, new_item_features: np.ndarray,
                              item_ratings: np.ndarray, k: int = 5) -> float:
    sims = item_features @ new_item_features
    norms = np.linalg.norm(item_features, axis=1) * np.linalg.norm(new_item_features)
    norms = np.where(norms < 1e-12, 1.0, norms)
    sims = sims / norms
    top_k = np.argsort(sims)[-k:]
    weights = sims[top_k]
    w_sum = weights.sum()
    if w_sum < 1e-12:
        return float(item_ratings.mean())
    return float(np.dot(weights, item_ratings[top_k]) / w_sum)


def intra_list_diversity(item_features: np.ndarray, rec_indices: List[int]) -> float:
    if len(rec_indices) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(rec_indices)):
        for j in range(i + 1, len(rec_indices)):
            a = item_features[rec_indices[i]]
            b = item_features[rec_indices[j]]
            total += 1.0 - cosine_similarity(a, b)
            count += 1
    return total / count if count > 0 else 0.0


def serendipity_score(recommendations: List[int], popularity: np.ndarray,
                      relevance: np.ndarray) -> float:
    if len(recommendations) == 0:
        return 0.0
    scores = []
    max_pop = popularity.max() if popularity.max() > 0 else 1.0
    for idx in recommendations:
        unexpectedness = 1.0 - (popularity[idx] / max_pop)
        rel = relevance[idx] if relevance[idx] > 0 else 0.0
        scores.append(unexpectedness * rel)
    return float(np.mean(scores))


def diversity_rerank(scores: List[Tuple[int, float]], item_features: np.ndarray,
                     lam: float = 0.3, n: int = 10) -> List[Tuple[int, float]]:
    if len(scores) == 0:
        return []
    selected = [scores[0]]
    candidates = list(scores[1:])

    while len(selected) < n and candidates:
        best_score = -float("inf")
        best_idx = 0
        sel_indices = [s[0] for s in selected]
        for ci, (item_idx, rel) in enumerate(candidates):
            avg_dist = 0.0
            for si in sel_indices:
                avg_dist += 1.0 - cosine_similarity(
                    item_features[item_idx], item_features[si]
                )
            avg_dist /= len(sel_indices)
            combined = (1.0 - lam) * rel + lam * avg_dist
            if combined > best_score:
                best_score = combined
                best_idx = ci
        selected.append(candidates.pop(best_idx))

    return selected


def coverage(recommendations: List[List[int]], n_items: int) -> float:
    if n_items == 0:
        return 0.0
    seen = set()
    for rec_list in recommendations:
        seen.update(rec_list)
    return len(seen) / n_items


def novelty(recommendations: List[int], popularity: np.ndarray) -> float:
    if len(recommendations) == 0:
        return 0.0
    total = 0.0
    for idx in recommendations:
        p = popularity[idx]
        total += -math.log2(p) if p > 0 else 0.0
    return total / len(recommendations)
