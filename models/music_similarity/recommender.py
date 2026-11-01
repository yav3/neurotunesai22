"""Content-based music recommendation with diversity control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.music_similarity.embedding import EmbeddingSpace, TrackEmbedding
from models.music_similarity.similarity import SimilarityEngine


@dataclass
class Recommendation:
    track_id: str
    score: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ContentBasedRecommender:
    def __init__(self):
        self.engine = SimilarityEngine()

    def recommend_similar(
        self,
        seed_id: str,
        space: EmbeddingSpace,
        n: int = 10,
        exclude: set[str] | None = None,
        metric: str = "cosine",
    ) -> list[Recommendation]:
        seed = space.get(seed_id)
        if seed is None:
            return []
        exclude = (exclude or set()) | {seed_id}
        nearest = space.nearest(seed.vector, n=n, exclude=exclude, metric=metric)
        return [
            Recommendation(
                track_id=tid,
                score=max(0.0, 1.0 - dist),
                reason=f"similar to {seed_id}",
            )
            for tid, dist in nearest
        ]

    def recommend_from_seeds(
        self,
        seed_ids: list[str],
        space: EmbeddingSpace,
        n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[Recommendation]:
        seeds = [space.get(sid) for sid in seed_ids]
        seeds = [s for s in seeds if s is not None]
        if not seeds:
            return []

        centroid = np.mean([s.vector for s in seeds], axis=0)
        exclude = (exclude or set()) | set(seed_ids)
        nearest = space.nearest(centroid, n=n, exclude=exclude, metric="cosine")
        return [
            Recommendation(
                track_id=tid,
                score=max(0.0, 1.0 - dist),
                reason=f"similar to seed group ({len(seeds)} tracks)",
            )
            for tid, dist in nearest
        ]

    def recommend_mmr(
        self,
        seed_id: str,
        space: EmbeddingSpace,
        n: int = 10,
        lambda_: float = 0.7,
        candidate_pool: int = 50,
    ) -> list[Recommendation]:
        seed = space.get(seed_id)
        if seed is None:
            return []

        candidates = space.nearest(
            seed.vector, n=candidate_pool, exclude={seed_id}, metric="cosine"
        )
        if not candidates:
            return []

        selected: list[Recommendation] = []
        selected_vecs: list[np.ndarray] = []
        remaining = dict(candidates)

        for _ in range(min(n, len(remaining))):
            best_id = None
            best_mmr = -float("inf")

            for tid, dist in remaining.items():
                relevance = 1.0 - dist
                if selected_vecs:
                    emb = space.get(tid)
                    if emb is None:
                        continue
                    max_sim = max(
                        SimilarityEngine.cosine(emb.vector, sv) for sv in selected_vecs
                    )
                else:
                    max_sim = 0.0

                mmr = lambda_ * relevance - (1 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = tid

            if best_id is None:
                break

            emb = space.get(best_id)
            selected.append(
                Recommendation(
                    track_id=best_id,
                    score=best_mmr,
                    reason="mmr-diverse",
                )
            )
            if emb is not None:
                selected_vecs.append(emb.vector)
            del remaining[best_id]

        return selected


class PlaylistRecommender:
    def __init__(self):
        self.content_rec = ContentBasedRecommender()
        self.engine = SimilarityEngine()

    def generate_playlist(
        self,
        seed_id: str,
        space: EmbeddingSpace,
        track_data: dict[str, dict[str, Any]],
        n_tracks: int = 10,
        energy_target: list[float] | None = None,
        diversity: float = 0.3,
        seed: int | None = None,
    ) -> list[Recommendation]:
        rng = np.random.default_rng(seed)

        recs = self.content_rec.recommend_mmr(
            seed_id, space, n=n_tracks * 3, lambda_=1.0 - diversity
        )
        if not recs:
            return []

        if energy_target and len(energy_target) >= n_tracks:
            scored = []
            for rec in recs:
                td = track_data.get(rec.track_id, {})
                track_energy = float(td.get("energy", 0.5))
                scored.append((rec, track_energy))

            playlist: list[Recommendation] = []
            used = set()
            for i, target_e in enumerate(energy_target[:n_tracks]):
                best = None
                best_diff = float("inf")
                for rec, e in scored:
                    if rec.track_id in used:
                        continue
                    diff = abs(e - target_e)
                    if diff < best_diff:
                        best_diff = diff
                        best = rec
                if best:
                    playlist.append(best)
                    used.add(best.track_id)

            return playlist

        return recs[:n_tracks]

    def extend_playlist(
        self,
        current_ids: list[str],
        space: EmbeddingSpace,
        n_extend: int = 5,
    ) -> list[Recommendation]:
        if not current_ids:
            return []
        return self.content_rec.recommend_from_seeds(
            current_ids[-3:], space, n=n_extend, exclude=set(current_ids)
        )
