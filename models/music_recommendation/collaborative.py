"""Collaborative filtering for music recommendation.

Matrix-factorization and neighbor-based approaches operating on
user-item interaction matrices.  Pure numpy — no external ML deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InteractionMatrix:
    """Sparse user×item interaction matrix stored in COO format."""

    user_ids: np.ndarray  # (nnz,) int
    item_ids: np.ndarray  # (nnz,) int
    values: np.ndarray  # (nnz,) float (play counts, ratings, etc.)
    n_users: int = 0
    n_items: int = 0

    def __post_init__(self) -> None:
        if self.n_users == 0:
            self.n_users = int(self.user_ids.max()) + 1 if len(self.user_ids) else 0
        if self.n_items == 0:
            self.n_items = int(self.item_ids.max()) + 1 if len(self.item_ids) else 0

    def to_dense(self) -> np.ndarray:
        mat = np.zeros((self.n_users, self.n_items))
        mat[self.user_ids, self.item_ids] = self.values
        return mat

    @staticmethod
    def from_dense(mat: np.ndarray) -> "InteractionMatrix":
        u, i = np.nonzero(mat)
        return InteractionMatrix(u, i, mat[u, i], mat.shape[0], mat.shape[1])


@dataclass
class MFModel:
    """Low-rank matrix factorization model: R ≈ U V^T + biases."""

    user_factors: np.ndarray  # (n_users, k)
    item_factors: np.ndarray  # (n_items, k)
    user_biases: np.ndarray  # (n_users,)
    item_biases: np.ndarray  # (n_items,)
    global_mean: float = 0.0

    def predict(self, user_id: int, item_id: int) -> float:
        return float(
            self.global_mean
            + self.user_biases[user_id]
            + self.item_biases[item_id]
            + self.user_factors[user_id] @ self.item_factors[item_id]
        )

    def predict_user(self, user_id: int) -> np.ndarray:
        return (
            self.global_mean
            + self.user_biases[user_id]
            + self.item_biases
            + self.item_factors @ self.user_factors[user_id]
        )

    def recommend(
        self, user_id: int, known_items: set[int] | None = None, n: int = 10
    ) -> list[tuple[int, float]]:
        scores = self.predict_user(user_id)
        if known_items:
            for i in known_items:
                if i < len(scores):
                    scores[i] = -np.inf
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]


def als_train(
    interactions: InteractionMatrix,
    k: int = 32,
    n_iter: int = 20,
    reg: float = 0.1,
    seed: int = 42,
) -> MFModel:
    """Alternating Least Squares for implicit/explicit feedback.

    Solves min_{U,V} ‖R − UV^T‖² + λ(‖U‖² + ‖V‖²) by alternating
    closed-form ridge regression on U and V.
    """
    rng = np.random.RandomState(seed)
    R = interactions.to_dense()
    n_u, n_i = R.shape
    global_mean = float(R[R > 0].mean()) if np.any(R > 0) else 0.0

    U = rng.randn(n_u, k) * 0.01
    V = rng.randn(n_i, k) * 0.01
    u_bias = np.zeros(n_u)
    i_bias = np.zeros(n_i)

    reg_eye = reg * np.eye(k)

    for _ in range(n_iter):
        # fix V, solve U
        VtV = V.T @ V + reg_eye
        for u in range(n_u):
            rated = np.where(R[u] > 0)[0]
            if len(rated) == 0:
                continue
            r_u = R[u, rated] - global_mean - i_bias[rated]
            V_u = V[rated]
            A = V_u.T @ V_u + reg * len(rated) * np.eye(k)
            b = V_u.T @ r_u
            U[u] = np.linalg.solve(A, b)
            u_bias[u] = float(np.mean(R[u, rated] - global_mean - i_bias[rated] - V[rated] @ U[u]))

        # fix U, solve V
        for i in range(n_i):
            users = np.where(R[:, i] > 0)[0]
            if len(users) == 0:
                continue
            r_i = R[users, i] - global_mean - u_bias[users]
            U_i = U[users]
            A = U_i.T @ U_i + reg * len(users) * np.eye(k)
            b = U_i.T @ r_i
            V[i] = np.linalg.solve(A, b)
            i_bias[i] = float(np.mean(R[users, i] - global_mean - u_bias[users] - U[users] @ V[i]))

    return MFModel(U, V, u_bias, i_bias, global_mean)


def user_knn_recommend(
    interactions: InteractionMatrix,
    user_id: int,
    k_neighbors: int = 10,
    n: int = 10,
) -> list[tuple[int, float]]:
    """User-based k-NN collaborative filtering."""
    R = interactions.to_dense()
    if user_id >= R.shape[0]:
        return []
    user_vec = R[user_id]
    norms = np.linalg.norm(R, axis=1)
    norms[norms < 1e-12] = 1.0
    sims = R @ user_vec / (norms * max(np.linalg.norm(user_vec), 1e-12))
    sims[user_id] = -np.inf
    neighbor_ids = np.argsort(sims)[::-1][:k_neighbors]
    neighbor_sims = sims[neighbor_ids]
    pos_mask = neighbor_sims > 0
    if not np.any(pos_mask):
        return []
    neighbor_ids = neighbor_ids[pos_mask]
    neighbor_sims = neighbor_sims[pos_mask]

    weighted_sum = neighbor_sims @ R[neighbor_ids]
    sim_sum = neighbor_sims.sum()
    predicted = weighted_sum / max(sim_sum, 1e-12)

    known = set(np.where(user_vec > 0)[0])
    for i in known:
        predicted[i] = -np.inf
    top_idx = np.argsort(predicted)[::-1][:n]
    return [(int(i), float(predicted[i])) for i in top_idx]


def item_knn_recommend(
    interactions: InteractionMatrix,
    user_id: int,
    k_neighbors: int = 10,
    n: int = 10,
) -> list[tuple[int, float]]:
    """Item-based k-NN collaborative filtering."""
    R = interactions.to_dense()
    if user_id >= R.shape[0]:
        return []
    user_vec = R[user_id]
    liked = np.where(user_vec > 0)[0]
    if len(liked) == 0:
        return []

    item_norms = np.linalg.norm(R.T, axis=1)
    item_norms[item_norms < 1e-12] = 1.0
    item_sim = (R.T @ R) / np.outer(item_norms, item_norms)

    scores = np.zeros(R.shape[1])
    for item in range(R.shape[1]):
        if user_vec[item] > 0:
            continue
        sims_to_liked = item_sim[item, liked]
        top_k = np.argsort(sims_to_liked)[::-1][:k_neighbors]
        top_sims = sims_to_liked[top_k]
        pos = top_sims > 0
        if np.any(pos):
            scores[item] = top_sims[pos] @ user_vec[liked[top_k[pos]]] / top_sims[pos].sum()

    known = set(liked)
    for i in known:
        scores[i] = -np.inf
    top_idx = np.argsort(scores)[::-1][:n]
    return [(int(i), float(scores[i])) for i in top_idx]


@dataclass
class PopularityRecommender:
    """Fallback: recommend most popular items a user hasn't seen."""

    popularity: np.ndarray = field(default_factory=lambda: np.array([]))

    @staticmethod
    def from_interactions(interactions: InteractionMatrix) -> "PopularityRecommender":
        R = interactions.to_dense()
        pop = (R > 0).sum(axis=0).astype(float)
        return PopularityRecommender(pop)

    def recommend(
        self, known_items: set[int] | None = None, n: int = 10
    ) -> list[tuple[int, float]]:
        scores = self.popularity.copy()
        if known_items:
            for i in known_items:
                if i < len(scores):
                    scores[i] = -np.inf
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]


def implicit_confidence(play_counts: np.ndarray, alpha: float = 40.0) -> np.ndarray:
    """Convert play counts to confidence weights: c_ui = 1 + α·r_ui."""
    return 1.0 + alpha * play_counts


def bm25_weight(interactions: InteractionMatrix, k1: float = 100.0, b: float = 0.8) -> InteractionMatrix:
    """Apply BM25 weighting to interaction values (TF-IDF variant for CF)."""
    R = interactions.to_dense()
    tf = R.copy()
    doc_len = tf.sum(axis=1)
    avg_dl = doc_len.mean() if doc_len.mean() > 0 else 1.0
    idf = np.log((R.shape[0] + 1) / ((R > 0).sum(axis=0) + 1))

    for u in range(R.shape[0]):
        dl = doc_len[u]
        denom = tf[u] + k1 * (1 - b + b * dl / avg_dl)
        denom[denom < 1e-12] = 1.0
        tf[u] = tf[u] * (k1 + 1) / denom * idf

    return InteractionMatrix.from_dense(tf)
