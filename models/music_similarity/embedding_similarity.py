"""Similarity in embedding spaces: aggregation, metrics, triplet evaluation, CCA."""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureAggregator:
    """Aggregate frame-level features to track-level representations."""

    def mean_std(self, frames: np.ndarray) -> np.ndarray:
        if frames.ndim == 1:
            return np.array([np.mean(frames), np.std(frames)])
        return np.concatenate([np.mean(frames, axis=0), np.std(frames, axis=0)])

    def gaussian_mixture(
        self, frames: np.ndarray, n_components: int = 4, max_iter: int = 50, seed: int = 42
    ) -> np.ndarray:
        if frames.ndim == 1:
            frames = frames.reshape(-1, 1)
        n, d = frames.shape
        if n < n_components:
            n_components = max(1, n)
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, n_components, replace=False)
        means = frames[indices].copy()
        covs = np.array([np.var(frames, axis=0) + 1e-6 for _ in range(n_components)])
        weights = np.ones(n_components) / n_components
        for _ in range(max_iter):
            resp = np.zeros((n, n_components))
            for k in range(n_components):
                diff = frames - means[k]
                exponent = -0.5 * np.sum(diff ** 2 / covs[k], axis=1)
                resp[:, k] = weights[k] * np.exp(exponent) / np.prod(np.sqrt(covs[k]) + 1e-30)
            row_sums = resp.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            resp /= row_sums
            for k in range(n_components):
                nk = resp[:, k].sum()
                if nk < 1e-10:
                    continue
                weights[k] = nk / n
                means[k] = (resp[:, k:k + 1].T @ frames).ravel() / nk
                diff = frames - means[k]
                covs[k] = (resp[:, k:k + 1].T @ (diff ** 2)).ravel() / nk + 1e-6
        return np.concatenate([means.ravel(), covs.ravel()])

    def temporal_pooling(
        self, frames: np.ndarray, n_segments: int = 4
    ) -> np.ndarray:
        if frames.ndim == 1:
            frames = frames.reshape(-1, 1)
        n = frames.shape[0]
        seg_size = max(1, n // n_segments)
        pooled = []
        for i in range(n_segments):
            start = i * seg_size
            end = min(start + seg_size, n)
            if start >= n:
                pooled.append(np.zeros(frames.shape[1]))
            else:
                pooled.append(np.mean(frames[start:end], axis=0))
        return np.concatenate(pooled)


@dataclass
class EmbeddingSimilarity:
    """Similarity metrics in embedding space."""

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def euclidean(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def mahalanobis(
        self, a: np.ndarray, b: np.ndarray, cov_inv: np.ndarray
    ) -> float:
        diff = a - b
        return float(np.sqrt(diff @ cov_inv @ diff))

    def pairwise_matrix(
        self, embeddings: np.ndarray, metric: str = "cosine"
    ) -> np.ndarray:
        n = embeddings.shape[0]
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if metric == "cosine":
                    s = self.cosine(embeddings[i], embeddings[j])
                elif metric == "euclidean":
                    s = -self.euclidean(embeddings[i], embeddings[j])
                else:
                    s = self.cosine(embeddings[i], embeddings[j])
                sim[i, j] = s
                sim[j, i] = s
        return sim


@dataclass
class TripletEvaluator:
    """Evaluate embedding quality via triplet constraints."""

    embeddings: np.ndarray
    labels: np.ndarray

    def triplet_accuracy(
        self, margin: float = 0.1, n_triplets: int = 1000, seed: int = 42
    ) -> float:
        rng = np.random.default_rng(seed)
        n = len(self.labels)
        if n < 3:
            return 0.0
        unique_labels = np.unique(self.labels)
        if len(unique_labels) < 2:
            return 0.0
        correct = 0
        total = 0
        for _ in range(n_triplets):
            anchor_idx = rng.integers(n)
            anchor_label = self.labels[anchor_idx]
            pos_indices = np.where(self.labels == anchor_label)[0]
            neg_indices = np.where(self.labels != anchor_label)[0]
            if len(pos_indices) < 2 or len(neg_indices) == 0:
                continue
            pos_idx = pos_indices[rng.integers(len(pos_indices))]
            while pos_idx == anchor_idx and len(pos_indices) > 1:
                pos_idx = pos_indices[rng.integers(len(pos_indices))]
            neg_idx = neg_indices[rng.integers(len(neg_indices))]
            d_pos = np.linalg.norm(self.embeddings[anchor_idx] - self.embeddings[pos_idx])
            d_neg = np.linalg.norm(self.embeddings[anchor_idx] - self.embeddings[neg_idx])
            if d_pos + margin < d_neg:
                correct += 1
            total += 1
        return correct / max(total, 1)

    def mean_average_precision(self) -> float:
        n = len(self.labels)
        if n < 2:
            return 0.0
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
        aps = []
        for i in range(n):
            sorted_idx = np.argsort(dists[i])
            relevant = self.labels[sorted_idx] == self.labels[i]
            relevant[sorted_idx == i] = False
            n_rel = np.sum(self.labels == self.labels[i]) - 1
            if n_rel == 0:
                continue
            precisions = []
            hits = 0
            for k in range(n):
                if sorted_idx[k] == i:
                    continue
                if relevant[k]:
                    hits += 1
                    precisions.append(hits / (len(precisions) + 1))
            if precisions:
                aps.append(np.mean(precisions))
        return float(np.mean(aps)) if aps else 0.0

    def recall_at_k(self, k: int = 10) -> float:
        n = len(self.labels)
        if n < 2:
            return 0.0
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
        recalls = []
        for i in range(n):
            sorted_idx = np.argsort(dists[i])
            neighbors = [j for j in sorted_idx if j != i][:k]
            n_rel = np.sum(self.labels == self.labels[i]) - 1
            if n_rel == 0:
                continue
            hits = sum(1 for j in neighbors if self.labels[j] == self.labels[i])
            recalls.append(hits / min(n_rel, k))
        return float(np.mean(recalls)) if recalls else 0.0


@dataclass
class CrossModalSimilarity:
    """Compare features across modalities using CCA."""

    def cca(
        self, X: np.ndarray, Y: np.ndarray, n_components: int = 5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = X.shape[0]
        dx, dy = X.shape[1], Y.shape[1]
        n_components = min(n_components, dx, dy, n)
        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)
        Cxx = (Xc.T @ Xc) / (n - 1) + 1e-6 * np.eye(dx)
        Cyy = (Yc.T @ Yc) / (n - 1) + 1e-6 * np.eye(dy)
        Cxy = (Xc.T @ Yc) / (n - 1)
        Cxx_inv = np.linalg.inv(Cxx)
        Cyy_inv = np.linalg.inv(Cyy)
        M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
        eigvals, eigvecs = np.linalg.eigh(M)
        idx = np.argsort(eigvals)[::-1][:n_components]
        correlations = np.sqrt(np.clip(eigvals[idx], 0, 1))
        Wx = eigvecs[:, idx]
        X_proj = Xc @ Wx
        Wy = Cyy_inv @ Cxy.T @ Wx
        norms = np.linalg.norm(Wy, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        Wy /= norms
        Y_proj = Yc @ Wy
        return correlations, X_proj, Y_proj

    def projection_similarity(
        self,
        audio_features: np.ndarray,
        text_features: np.ndarray,
        projection_dim: int = 32,
    ) -> float:
        n_components = min(projection_dim, audio_features.shape[1], text_features.shape[1])
        corrs, X_proj, Y_proj = self.cca(audio_features, text_features, n_components)
        similarities = []
        for i in range(X_proj.shape[0]):
            na = np.linalg.norm(X_proj[i])
            nb = np.linalg.norm(Y_proj[i])
            if na > 0 and nb > 0:
                similarities.append(np.dot(X_proj[i], Y_proj[i]) / (na * nb))
        return float(np.mean(similarities)) if similarities else 0.0
