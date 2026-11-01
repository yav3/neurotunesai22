"""Recommendation engines: content-based, mood-based, playlist continuation, contextual."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from models.music_data.schema import TrackRecord
from models.music_recommendation.similarity import CompositeSimilarity, EmotionalSimilarity


class ContentBasedRecommender:
    def __init__(self, similarity: CompositeSimilarity | None = None):
        self._sim = similarity or CompositeSimilarity()

    def recommend(self, seed: TrackRecord, candidates: list[TrackRecord], n: int = 10) -> list[tuple[TrackRecord, float]]:
        scored = [(t, self._sim.score(seed, t)) for t in candidates if t.id != seed.id]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


class MoodBasedRecommender:
    def recommend(
        self, target_valence: float, target_arousal: float, target_dominance: float,
        candidates: list[TrackRecord], n: int = 10,
    ) -> list[tuple[TrackRecord, float]]:
        scored = []
        for t in candidates:
            dist = np.sqrt(
                (t.valence - target_valence) ** 2
                + (t.arousal - target_arousal) ** 2
                + (t.dominance - target_dominance) ** 2
            )
            scored.append((t, float(dist)))
        scored.sort(key=lambda x: x[1])
        return scored[:n]


class PlaylistContinuationRecommender:
    def __init__(self):
        self._sim = CompositeSimilarity()

    def recommend(
        self, playlist: list[TrackRecord], candidates: list[TrackRecord], n: int = 5,
    ) -> list[tuple[TrackRecord, float]]:
        if not playlist:
            return [(t, 0.5) for t in candidates[:n]]
        last = playlist[-1]
        used_ids = {t.id for t in playlist}
        available = [t for t in candidates if t.id not in used_ids]
        scored = [(t, self._sim.score(last, t)) for t in available]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


@dataclass
class ContextualRecommender:
    preferred_genres: list[str] = field(default_factory=list)
    bpm_range: tuple[float, float] = (60, 180)
    energy_range: tuple[float, float] = (0.0, 1.0)

    def recommend(self, candidates: list[TrackRecord], n: int = 10) -> list[TrackRecord]:
        filtered = []
        for t in candidates:
            if self.preferred_genres and t.genre.lower() not in [g.lower() for g in self.preferred_genres]:
                continue
            if not (self.bpm_range[0] <= t.bpm <= self.bpm_range[1]):
                continue
            if not (self.energy_range[0] <= t.energy <= self.energy_range[1]):
                continue
            filtered.append(t)
        return filtered[:n]


class DiversityMixer:
    @staticmethod
    def diversify(
        ranked: list[tuple[TrackRecord, float]],
        window: int = 3,
    ) -> list[tuple[TrackRecord, float]]:
        if not ranked:
            return []
        result: list[tuple[TrackRecord, float]] = [ranked[0]]
        remaining = list(ranked[1:])
        while remaining and len(result) < len(ranked):
            recent_artists = {r[0].artist for r in result[-window:]}
            recent_genres = {r[0].genre for r in result[-window:]}
            best_idx, best_score = 0, -1.0
            for i, (t, s) in enumerate(remaining):
                penalty = 0.0
                if t.artist in recent_artists:
                    penalty += 0.3
                if t.genre in recent_genres:
                    penalty += 0.1
                adj = s - penalty
                if adj > best_score:
                    best_score = adj
                    best_idx = i
            result.append(remaining.pop(best_idx))
        return result
