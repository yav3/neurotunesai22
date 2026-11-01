"""Unsupervised music grouping: k-means, mood clusters, genre profiling."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from models.music_data.schema import TrackRecord


class MusicClusterer:
    def __init__(self, n_clusters: int = 5, max_iter: int = 100):
        self._k = n_clusters
        self._max_iter = max_iter
        self._centroids: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit(self, tracks: list[TrackRecord], seed: int | None = None) -> None:
        if not tracks:
            return
        rng = np.random.default_rng(seed)
        X = np.array([list(t.to_feature_dict().values()) for t in tracks], dtype=float)
        n = X.shape[0]
        k = min(self._k, n)
        idx = rng.choice(n, size=k, replace=False)
        centroids = X[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(self._max_iter):
            dists = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X])
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)
        self._centroids = centroids
        self._labels = labels

    def predict(self, track: TrackRecord) -> int:
        if self._centroids is None:
            return 0
        x = np.array(list(track.to_feature_dict().values()), dtype=float)
        dists = [np.linalg.norm(x - c) for c in self._centroids]
        return int(np.argmin(dists))

    @property
    def labels(self) -> np.ndarray | None:
        return self._labels


class MoodClusterer:
    def cluster(self, tracks: list[TrackRecord], n_clusters: int = 4, seed: int | None = None) -> dict[int, list[TrackRecord]]:
        if not tracks:
            return {}
        rng = np.random.default_rng(seed)
        vad = np.array([[t.valence, t.arousal, t.dominance] for t in tracks])
        k = min(n_clusters, len(tracks))
        idx = rng.choice(len(tracks), size=k, replace=False)
        centroids = vad[idx].copy()
        labels = np.zeros(len(tracks), dtype=int)
        for _ in range(50):
            dists = np.array([[np.linalg.norm(v - c) for c in centroids] for v in vad])
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                members = vad[labels == j]
                if len(members) > 0:
                    centroids[j] = members.mean(axis=0)
        groups: dict[int, list[TrackRecord]] = {}
        for i, t in enumerate(tracks):
            g = int(labels[i])
            groups.setdefault(g, []).append(t)
        return groups


@dataclass
class GenreProfile:
    genre: str = ""
    count: int = 0
    bpm_mean: float = 0.0
    bpm_std: float = 0.0
    valence_mean: float = 0.0
    energy_mean: float = 0.0


class GenreProfiler:
    def profile(self, tracks: list[TrackRecord]) -> dict[str, GenreProfile]:
        by_genre: dict[str, list[TrackRecord]] = {}
        for t in tracks:
            g = t.genre or "Unknown"
            by_genre.setdefault(g, []).append(t)
        profiles = {}
        for genre, ts in by_genre.items():
            bpms = [t.bpm for t in ts if t.bpm > 0]
            profiles[genre] = GenreProfile(
                genre=genre, count=len(ts),
                bpm_mean=float(np.mean(bpms)) if bpms else 0,
                bpm_std=float(np.std(bpms)) if bpms else 0,
                valence_mean=float(np.mean([t.valence for t in ts])),
                energy_mean=float(np.mean([t.energy for t in ts])),
            )
        return profiles


class ClusterLabel:
    MOOD_LABELS = {
        (True, True): "Excited/Happy",
        (True, False): "Calm/Content",
        (False, True): "Tense/Angry",
        (False, False): "Sad/Melancholic",
    }

    @staticmethod
    def label_mood_cluster(tracks: list[TrackRecord]) -> str:
        if not tracks:
            return "Unknown"
        avg_v = np.mean([t.valence for t in tracks])
        avg_a = np.mean([t.arousal for t in tracks])
        return ClusterLabel.MOOD_LABELS.get((avg_v >= 0.5, avg_a >= 0.5), "Mixed")

    @staticmethod
    def label_energy_cluster(tracks: list[TrackRecord]) -> str:
        if not tracks:
            return "Unknown"
        avg_e = np.mean([t.energy for t in tracks])
        if avg_e > 0.7:
            return "High Energy"
        elif avg_e > 0.4:
            return "Medium Energy"
        return "Low Energy"
