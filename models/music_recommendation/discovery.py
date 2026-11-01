"""Music discovery: novelty scoring, exploration/exploitation, genre bridging, mood journeys."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from models.music_data.schema import TrackRecord
from models.music_recommendation.similarity import CompositeSimilarity, EmotionalSimilarity


class NoveltyScorer:
    def __init__(self):
        self._sim = CompositeSimilarity()

    def score(self, candidate: TrackRecord, history: list[TrackRecord]) -> float:
        if not history:
            return 1.0
        sims = [self._sim.score(candidate, h) for h in history]
        avg_sim = np.mean(sims)
        return float(1.0 - avg_sim)


class ExplorationExploitation:
    def __init__(self, epsilon: float = 0.2):
        self._epsilon = epsilon
        self._novelty = NoveltyScorer()

    def select(
        self,
        ranked: list[tuple[TrackRecord, float]],
        history: list[TrackRecord],
        seed: int | None = None,
    ) -> TrackRecord:
        if not ranked:
            raise ValueError("No candidates")
        rng = np.random.default_rng(seed)
        if rng.random() < self._epsilon:
            novelties = [(t, self._novelty.score(t, history)) for t, _ in ranked]
            novelties.sort(key=lambda x: x[1], reverse=True)
            return novelties[0][0]
        return ranked[0][0]

    def rerank(
        self,
        ranked: list[tuple[TrackRecord, float]],
        history: list[TrackRecord],
        alpha: float = 0.3,
    ) -> list[tuple[TrackRecord, float]]:
        reranked = []
        for t, score in ranked:
            novelty = self._novelty.score(t, history)
            adj = (1 - alpha) * score + alpha * novelty
            reranked.append((t, adj))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


class GenreBridging:
    def __init__(self):
        self._sim = CompositeSimilarity()

    def find_bridges(
        self,
        genre_a: str,
        genre_b: str,
        candidates: list[TrackRecord],
        n: int = 5,
    ) -> list[TrackRecord]:
        a_tracks = [t for t in candidates if t.genre.lower() == genre_a.lower()]
        b_tracks = [t for t in candidates if t.genre.lower() == genre_b.lower()]
        other = [t for t in candidates if t.genre.lower() not in (genre_a.lower(), genre_b.lower())]
        if not a_tracks or not b_tracks or not other:
            return []
        a_center = TrackRecord(
            valence=float(np.mean([t.valence for t in a_tracks])),
            arousal=float(np.mean([t.arousal for t in a_tracks])),
            energy=float(np.mean([t.energy for t in a_tracks])),
            bpm=float(np.mean([t.bpm for t in a_tracks])),
        )
        b_center = TrackRecord(
            valence=float(np.mean([t.valence for t in b_tracks])),
            arousal=float(np.mean([t.arousal for t in b_tracks])),
            energy=float(np.mean([t.energy for t in b_tracks])),
            bpm=float(np.mean([t.bpm for t in b_tracks])),
        )
        scored = []
        for t in other:
            sim_a = self._sim.score(t, a_center)
            sim_b = self._sim.score(t, b_center)
            bridge_score = min(sim_a, sim_b)
            scored.append((t, bridge_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:n]]


class MoodJourney:
    def plan(
        self,
        start_vad: tuple[float, float, float],
        end_vad: tuple[float, float, float],
        candidates: list[TrackRecord],
        n_steps: int = 8,
    ) -> list[TrackRecord]:
        if not candidates or n_steps < 2:
            return []
        waypoints = np.linspace(start_vad, end_vad, n_steps)
        journey: list[TrackRecord] = []
        used_ids: set[str] = set()
        for wp in waypoints:
            v, a, d = wp
            best_t, best_dist = None, float("inf")
            for t in candidates:
                if t.id in used_ids:
                    continue
                dist = np.sqrt((t.valence - v)**2 + (t.arousal - a)**2 + (t.dominance - d)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_t = t
            if best_t is not None:
                journey.append(best_t)
                used_ids.add(best_t.id)
        return journey
