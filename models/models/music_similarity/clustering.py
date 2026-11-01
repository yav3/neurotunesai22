"""Music clustering: k-means, hierarchical, cluster profiling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.music_similarity.embedding import EmbeddingSpace


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    centroid: np.ndarray
    member_ids: list[str]
    feature_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class ClusteringResult:
    labels: np.ndarray
    centroids: np.ndarray
    profiles: list[ClusterProfile]
    inertia: float = 0.0
    n_clusters: int = 0


class MusicClusterer:
    def kmeans(
        self,
        space: EmbeddingSpace,
        k: int = 5,
        max_iter: int = 100,
        seed: int | None = None,
    ) -> ClusteringResult:
        matrix, ids = space.to_matrix()
        if len(matrix) == 0 or k < 1:
            return ClusteringResult(
                labels=np.array([]),
                centroids=np.empty((0, 0)),
                profiles=[],
                n_clusters=0,
            )

        k = min(k, len(matrix))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(matrix), size=k, replace=False)
        centroids = matrix[idx].copy()

        labels = np.zeros(len(matrix), dtype=int)

        for _ in range(max_iter):
            dists = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for c in range(k):
                members = matrix[labels == c]
                if len(members) > 0:
                    centroids[c] = members.mean(axis=0)

        inertia = 0.0
        for c in range(k):
            members = matrix[labels == c]
            if len(members) > 0:
                inertia += np.sum((members - centroids[c]) ** 2)

        profiles = self._build_profiles(labels, centroids, ids, matrix, k)

        return ClusteringResult(
            labels=labels,
            centroids=centroids,
            profiles=profiles,
            inertia=float(inertia),
            n_clusters=k,
        )

    def hierarchical(
        self,
        space: EmbeddingSpace,
        k: int = 5,
    ) -> ClusteringResult:
        matrix, ids = space.to_matrix()
        if len(matrix) == 0 or k < 1:
            return ClusteringResult(
                labels=np.array([]),
                centroids=np.empty((0, 0)),
                profiles=[],
                n_clusters=0,
            )

        k = min(k, len(matrix))
        n = len(matrix)
        labels = np.arange(n)
        cluster_members: dict[int, list[int]] = {i: [i] for i in range(n)}
        next_id = n

        while len(cluster_members) > k:
            best_dist = float("inf")
            best_pair = (0, 0)
            cluster_ids = sorted(cluster_members.keys())

            for i_idx in range(len(cluster_ids)):
                for j_idx in range(i_idx + 1, len(cluster_ids)):
                    ci = cluster_ids[i_idx]
                    cj = cluster_ids[j_idx]
                    mi = matrix[cluster_members[ci]].mean(axis=0)
                    mj = matrix[cluster_members[cj]].mean(axis=0)
                    d = float(np.linalg.norm(mi - mj))
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ci, cj)

            ci, cj = best_pair
            merged = cluster_members[ci] + cluster_members[cj]
            del cluster_members[ci]
            del cluster_members[cj]
            cluster_members[next_id] = merged
            for m in merged:
                labels[m] = next_id
            next_id += 1

        final_map = {}
        for new_idx, cid in enumerate(sorted(cluster_members.keys())):
            for m in cluster_members[cid]:
                labels[m] = new_idx
            final_map[cid] = new_idx

        centroids = np.zeros((k, matrix.shape[1]))
        for c in range(k):
            members = matrix[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

        profiles = self._build_profiles(labels, centroids, ids, matrix, k)

        return ClusteringResult(
            labels=labels,
            centroids=centroids,
            profiles=profiles,
            n_clusters=k,
        )

    def _build_profiles(
        self,
        labels: np.ndarray,
        centroids: np.ndarray,
        ids: list[str],
        matrix: np.ndarray,
        k: int,
    ) -> list[ClusterProfile]:
        profiles = []
        for c in range(k):
            mask = labels == c
            member_ids = [ids[i] for i in range(len(ids)) if mask[i]]
            members = matrix[mask]
            summary = {}
            if len(members) > 0:
                for d in range(min(members.shape[1], 16)):
                    summary[f"dim_{d}_mean"] = float(members[:, d].mean())
                    summary[f"dim_{d}_std"] = float(members[:, d].std())

            profiles.append(
                ClusterProfile(
                    cluster_id=c,
                    size=int(mask.sum()),
                    centroid=centroids[c],
                    member_ids=member_ids,
                    feature_summary=summary,
                )
            )
        return profiles

    def silhouette_score(
        self, space: EmbeddingSpace, labels: np.ndarray
    ) -> float:
        matrix, _ = space.to_matrix()
        n = len(matrix)
        if n < 2 or len(set(labels)) < 2:
            return 0.0

        scores = []
        for i in range(n):
            own_cluster = labels[i]
            same = [j for j in range(n) if j != i and labels[j] == own_cluster]
            if not same:
                scores.append(0.0)
                continue
            a_i = np.mean([np.linalg.norm(matrix[i] - matrix[j]) for j in same])

            b_i = float("inf")
            for c in set(labels):
                if c == own_cluster:
                    continue
                other = [j for j in range(n) if labels[j] == c]
                if other:
                    d = np.mean([np.linalg.norm(matrix[i] - matrix[j]) for j in other])
                    b_i = min(b_i, d)

            if b_i == float("inf"):
                scores.append(0.0)
            else:
                scores.append((b_i - a_i) / max(a_i, b_i))

        return float(np.mean(scores))
