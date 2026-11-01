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


# ---------------------------------------------------------------------------
# SVD-based factorization (Koren et al. 2009, Funk SVD / truncated SVD)
# ---------------------------------------------------------------------------

def normalize_matrix(ratings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-center rows (per-user mean subtraction). Returns (centered, means)."""
    means = np.zeros(ratings.shape[0])
    centered = ratings.copy().astype(float)
    for u in range(ratings.shape[0]):
        mask = ratings[u] != 0
        if mask.any():
            means[u] = ratings[u, mask].mean()
            centered[u, mask] -= means[u]
    return centered, means


def cosine_similarity_matrix(ratings: np.ndarray, axis: int = 0) -> np.ndarray:
    """Pairwise cosine similarity. axis=0 → between columns (items), axis=1 → rows (users)."""
    if axis == 1:
        M = ratings
    else:
        M = ratings.T
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = M / norms
    return normed @ normed.T


def svd_factorize(ratings: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated SVD on a (possibly sparse) rating matrix.

    Fills zeros with row means before decomposition so the zero entries
    don't collapse the latent space.  Returns (U_k, Sigma_k, Vt_k).
    """
    filled = ratings.copy().astype(float)
    for u in range(filled.shape[0]):
        mask = filled[u] != 0
        if mask.any():
            filled[u, ~mask] = filled[u, mask].mean()
    U, S, Vt = np.linalg.svd(filled, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k], S[:k], Vt[:k, :]


def svd_predict(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """Reconstruct full rating matrix from truncated SVD factors."""
    return U @ np.diag(S) @ Vt


def svd_recommend_top_n(U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
                        user_idx: int, original_ratings: np.ndarray,
                        n: int = 10) -> List[Tuple[int, float]]:
    """Top-N unrated items for a user from SVD reconstruction."""
    predicted = (U[user_idx] @ np.diag(S) @ Vt).ravel()
    rated_mask = original_ratings[user_idx] != 0
    predicted[rated_mask] = -np.inf
    top_indices = np.argsort(predicted)[::-1][:n]
    return [(int(j), float(predicted[j])) for j in top_indices if predicted[j] > -np.inf]


# ---------------------------------------------------------------------------
# Evaluation metrics (Herlocker et al. 2004, Shani & Gunawardana 2011)
# ---------------------------------------------------------------------------

def precision_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    """Precision@k: fraction of top-k recommendations that are relevant."""
    rec_k = recommended[:k]
    if len(rec_k) == 0:
        return 0.0
    return sum(1 for r in rec_k if r in relevant) / len(rec_k)


def recall_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    """Recall@k: fraction of relevant items found in top-k."""
    if len(relevant) == 0:
        return 0.0
    rec_k = recommended[:k]
    return sum(1 for r in rec_k if r in relevant) / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: set, k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg < 1e-12:
        return 0.0
    return dcg / idcg


def mean_average_precision(all_recommendations: List[List[int]],
                           all_relevant: List[set]) -> float:
    """MAP across multiple users."""
    if len(all_recommendations) == 0:
        return 0.0
    ap_sum = 0.0
    for rec, rel in zip(all_recommendations, all_relevant):
        if len(rel) == 0:
            continue
        hits = 0
        ap = 0.0
        for i, item in enumerate(rec):
            if item in rel:
                hits += 1
                ap += hits / (i + 1)
        ap_sum += ap / len(rel)
    return ap_sum / len(all_recommendations)


def hit_rate(all_recommendations: List[List[int]],
             all_relevant: List[set]) -> float:
    """Fraction of users for whom at least one recommendation is relevant."""
    if len(all_recommendations) == 0:
        return 0.0
    hits = sum(
        1 for rec, rel in zip(all_recommendations, all_relevant)
        if any(r in rel for r in rec)
    )
    return hits / len(all_recommendations)


def reciprocal_rank(recommended: List[int], relevant: set) -> float:
    """Reciprocal rank of the first relevant item."""
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(all_recommendations: List[List[int]],
                         all_relevant: List[set]) -> float:
    """MRR across users."""
    if len(all_recommendations) == 0:
        return 0.0
    return float(np.mean([
        reciprocal_rank(rec, rel) for rec, rel in zip(all_recommendations, all_relevant)
    ]))


def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Root mean squared error on rated entries."""
    mask = actuals != 0
    if not mask.any():
        return 0.0
    diff = predictions[mask] - actuals[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean absolute error on rated entries."""
    mask = actuals != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(predictions[mask] - actuals[mask])))
