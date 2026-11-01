"""Similarity metrics for music tracks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SimilarityResult:
    score: float
    components: dict[str, float]


class SimilarityEngine:
    CAMELOT_COMPAT: dict[int, list[int]] = {
        i: [(i - 1) % 12, i, (i + 1) % 12] for i in range(12)
    }

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def manhattan(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(np.abs(a - b)))

    def weighted_feature_similarity(
        self,
        a: dict[str, float],
        b: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> SimilarityResult:
        keys = sorted(set(a.keys()) & set(b.keys()))
        if not keys:
            return SimilarityResult(score=0.0, components={})

        if weights is None:
            weights = {k: 1.0 for k in keys}

        total_w = sum(weights.get(k, 0.0) for k in keys)
        if total_w == 0:
            return SimilarityResult(score=0.0, components={})

        components = {}
        weighted_sum = 0.0
        for k in keys:
            diff = abs(a[k] - b[k])
            sim = max(0.0, 1.0 - diff)
            components[k] = sim
            w = weights.get(k, 0.0)
            weighted_sum += w * sim

        return SimilarityResult(
            score=weighted_sum / total_w,
            components=components,
        )

    def harmonic_compatibility(self, key_a: str, key_b: str) -> float:
        try:
            from models.music_playlist.transition import CAMELOT_NUMBERS
        except ImportError:
            return 0.5

        a_info = CAMELOT_NUMBERS.get(key_a)
        b_info = CAMELOT_NUMBERS.get(key_b)
        if a_info is None or b_info is None:
            return 0.0

        a_num, a_mode = a_info
        b_num, b_mode = b_info

        if a_num == b_num and a_mode == b_mode:
            return 1.0
        if a_num == b_num:
            return 0.8
        if abs(a_num - b_num) % 12 <= 1 or abs(a_num - b_num) % 12 >= 11:
            return 0.7
        dist = min(abs(a_num - b_num), 12 - abs(a_num - b_num))
        return max(0.0, 1.0 - dist * 0.15)

    def temporal_similarity(
        self,
        arc_a: list[float],
        arc_b: list[float],
    ) -> float:
        if not arc_a or not arc_b:
            return 0.0
        n = max(len(arc_a), len(arc_b))
        a = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arc_a)), arc_a)
        b = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arc_b)), arc_b)
        corr = np.corrcoef(a, b)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float((corr + 1.0) / 2.0)

    def composite_similarity(
        self,
        track_a: dict[str, Any],
        track_b: dict[str, Any],
        feature_weight: float = 0.4,
        harmonic_weight: float = 0.3,
        temporal_weight: float = 0.15,
        mood_weight: float = 0.15,
    ) -> SimilarityResult:
        feat_keys = ["bpm", "energy", "danceability", "acousticness", "valence"]
        fa = {k: float(track_a.get(k, 0)) for k in feat_keys}
        fb = {k: float(track_b.get(k, 0)) for k in feat_keys}
        feat_sim = self.weighted_feature_similarity(fa, fb).score

        harm_sim = self.harmonic_compatibility(
            str(track_a.get("key", "")),
            str(track_b.get("key", "")),
        )

        arc_a = track_a.get("emotion_arc", [])
        arc_b = track_b.get("emotion_arc", [])
        temp_sim = self.temporal_similarity(arc_a, arc_b) if arc_a and arc_b else 0.5

        mood_keys = ["valence", "arousal", "dominance"]
        ma = np.array([float(track_a.get(k, 0.5)) for k in mood_keys])
        mb = np.array([float(track_b.get(k, 0.5)) for k in mood_keys])
        mood_sim = max(0.0, 1.0 - float(np.linalg.norm(ma - mb)))

        total = (
            feature_weight * feat_sim
            + harmonic_weight * harm_sim
            + temporal_weight * temp_sim
            + mood_weight * mood_sim
        )

        return SimilarityResult(
            score=total,
            components={
                "feature": feat_sim,
                "harmonic": harm_sim,
                "temporal": temp_sim,
                "mood": mood_sim,
            },
        )
