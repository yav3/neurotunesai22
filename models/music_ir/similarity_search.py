"""Similarity search: feature indexing, LSH, DTW alignment, cover song detection."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class FeatureIndex:
    def __init__(self, dim: int = 128) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []

    def add(self, track_id: str, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float64).flatten()
        if len(v) != self._dim:
            v = np.resize(v, self._dim)
        self._ids.append(track_id)
        self._vectors.append(v)

    def query(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        v = np.asarray(vector, dtype=np.float64).flatten()
        if len(v) != self._dim:
            v = np.resize(v, self._dim)
        mat = np.array(self._vectors)
        norms_mat = np.linalg.norm(mat, axis=1)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            return [(self._ids[i], 0.0) for i in range(min(k, len(self._ids)))]
        sims = mat @ v / (norms_mat * norm_v + 1e-12)
        top_k = min(k, len(sims))
        indices = np.argsort(sims)[::-1][:top_k]
        return [(self._ids[i], float(sims[i])) for i in indices]

    def __len__(self) -> int:
        return len(self._ids)


class LocalitySensitiveHash:
    def __init__(self, dim: int = 128, n_tables: int = 10, n_bits: int = 8) -> None:
        self._dim = dim
        self._n_tables = n_tables
        self._n_bits = n_bits
        self._planes: list[np.ndarray] = [
            np.random.randn(n_bits, dim) for _ in range(n_tables)
        ]
        self._tables: list[dict[int, list[str]]] = [
            defaultdict(list) for _ in range(n_tables)
        ]
        self._vectors: dict[str, np.ndarray] = {}

    def _hash(self, vector: np.ndarray, table_idx: int) -> int:
        proj = self._planes[table_idx] @ vector
        bits = (proj > 0).astype(int)
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    def add(self, track_id: str, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float64).flatten()
        if len(v) != self._dim:
            v = np.resize(v, self._dim)
        self._vectors[track_id] = v
        for i in range(self._n_tables):
            h = self._hash(v, i)
            self._tables[i][h].append(track_id)

    def query(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        v = np.asarray(vector, dtype=np.float64).flatten()
        if len(v) != self._dim:
            v = np.resize(v, self._dim)
        candidates: set[str] = set()
        for i in range(self._n_tables):
            h = self._hash(v, i)
            candidates.update(self._tables[i].get(h, []))
        if not candidates:
            return []
        norm_v = np.linalg.norm(v)
        results: list[tuple[str, float]] = []
        for cid in candidates:
            cv = self._vectors[cid]
            norm_c = np.linalg.norm(cv)
            if norm_v < 1e-12 or norm_c < 1e-12:
                sim = 0.0
            else:
                sim = float(np.dot(v, cv) / (norm_v * norm_c))
            results.append((cid, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def __len__(self) -> int:
        return len(self._vectors)


class SequenceAlignment:
    @staticmethod
    def dtw(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        n, m = len(seq_a), len(seq_b)
        if n == 0 or m == 0:
            return float("inf")
        a = np.asarray(seq_a)
        b = np.asarray(seq_b)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        n, m = a.shape[0], b.shape[0]
        dtw_mat = np.full((n + 1, m + 1), np.inf)
        dtw_mat[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
                dtw_mat[i, j] = cost + min(
                    dtw_mat[i - 1, j],
                    dtw_mat[i, j - 1],
                    dtw_mat[i - 1, j - 1],
                )
        return float(dtw_mat[n, m])

    @staticmethod
    def dtw_path(seq_a: np.ndarray, seq_b: np.ndarray) -> list[tuple[int, int]]:
        a = np.asarray(seq_a)
        b = np.asarray(seq_b)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        n, m = a.shape[0], b.shape[0]
        if n == 0 or m == 0:
            return []
        dtw_mat = np.full((n + 1, m + 1), np.inf)
        dtw_mat[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
                dtw_mat[i, j] = cost + min(
                    dtw_mat[i - 1, j],
                    dtw_mat[i, j - 1],
                    dtw_mat[i - 1, j - 1],
                )
        path: list[tuple[int, int]] = [(n - 1, m - 1)]
        i, j = n, m
        while i > 1 or j > 1:
            candidates: list[tuple[float, int, int]] = []
            if i > 1 and j > 1:
                candidates.append((dtw_mat[i - 1, j - 1], i - 1, j - 1))
            if i > 1:
                candidates.append((dtw_mat[i - 1, j], i - 1, j))
            if j > 1:
                candidates.append((dtw_mat[i, j - 1], i, j - 1))
            if not candidates:
                break
            _, i, j = min(candidates, key=lambda x: x[0])
            path.append((i - 1, j - 1))
        path.reverse()
        return path

    @staticmethod
    def smith_waterman(
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        match_score: float = 2.0,
        mismatch_penalty: float = -1.0,
        gap_penalty: float = -0.5,
        threshold: float = 0.1,
    ) -> tuple[float, list[tuple[int, int]]]:
        a = np.asarray(seq_a).flatten()
        b = np.asarray(seq_b).flatten()
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 0.0, []
        H = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sim = match_score if abs(a[i - 1] - b[j - 1]) < threshold else mismatch_penalty
                H[i, j] = max(
                    0,
                    H[i - 1, j - 1] + sim,
                    H[i - 1, j] + gap_penalty,
                    H[i, j - 1] + gap_penalty,
                )
        best_score = float(H.max())
        if best_score == 0:
            return 0.0, []
        i, j = np.unravel_index(np.argmax(H), H.shape)
        path: list[tuple[int, int]] = []
        while i > 0 and j > 0 and H[i, j] > 0:
            path.append((i - 1, j - 1))
            candidates = [
                (H[i - 1, j - 1], i - 1, j - 1),
                (H[i - 1, j], i - 1, j),
                (H[i, j - 1], i, j - 1),
            ]
            _, i, j = max(candidates, key=lambda x: x[0])
        path.reverse()
        return best_score, path


class CoverSongDetector:
    def __init__(self) -> None:
        self._alignment = SequenceAlignment()

    def compare(
        self,
        chroma_a: np.ndarray,
        chroma_b: np.ndarray,
    ) -> dict[str, float]:
        if chroma_a.size == 0 or chroma_b.size == 0:
            return {"dtw_distance": float("inf"), "oti_score": 0.0, "cover_score": 0.0}
        best_dist = float("inf")
        best_shift = 0
        a_mean = chroma_a.mean(axis=1) if chroma_a.ndim == 2 else chroma_a
        b_mean = chroma_b.mean(axis=1) if chroma_b.ndim == 2 else chroma_b
        for shift in range(12):
            rolled = np.roll(b_mean, shift)
            dist = float(np.linalg.norm(a_mean - rolled))
            if dist < best_dist:
                best_dist = dist
                best_shift = shift
        if chroma_b.ndim == 2:
            chroma_b_shifted = np.roll(chroma_b, best_shift, axis=0)
        else:
            chroma_b_shifted = np.roll(chroma_b, best_shift)
        if chroma_a.ndim == 2 and chroma_b_shifted.ndim == 2:
            a_seq = chroma_a.T
            b_seq = chroma_b_shifted.T
            n = min(64, a_seq.shape[0])
            m = min(64, b_seq.shape[0])
            dtw_dist = SequenceAlignment.dtw(a_seq[:n], b_seq[:m])
        else:
            dtw_dist = float(np.linalg.norm(
                chroma_a.flatten()[:128] - chroma_b_shifted.flatten()[:128]
            ))
        max_possible = max(1.0, dtw_dist)
        oti_score = max(0.0, 1.0 - best_dist / (np.sqrt(12) + 1e-12))
        cover_score = max(0.0, 1.0 - dtw_dist / (max_possible + 1e-12))
        return {
            "dtw_distance": dtw_dist,
            "oti_score": oti_score,
            "cover_score": float(0.5 * oti_score + 0.5 * cover_score),
        }
