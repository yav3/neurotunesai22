"""User taste modeling: listening history, taste profiles, evolution, distance."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from models.music_data.schema import TrackRecord


@dataclass
class ListenEvent:
    track_id: str = ""
    timestamp: float = 0.0
    duration_listened_s: float = 0.0
    completed: bool = False


class ListeningHistory:
    def __init__(self):
        self._events: list[ListenEvent] = []

    def add(self, event: ListenEvent) -> None:
        self._events.append(event)

    def track_ids(self) -> list[str]:
        return [e.track_id for e in self._events]

    def play_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.track_id] = counts.get(e.track_id, 0) + 1
        return counts

    def recent(self, n: int = 20) -> list[ListenEvent]:
        return sorted(self._events, key=lambda e: e.timestamp, reverse=True)[:n]

    def __len__(self) -> int:
        return len(self._events)


@dataclass
class TasteProfile:
    preferred_genres: dict[str, float] = field(default_factory=dict)
    bpm_mean: float = 0.0
    bpm_std: float = 0.0
    energy_mean: float = 0.0
    valence_mean: float = 0.0
    arousal_mean: float = 0.0
    dominance_mean: float = 0.0
    top_artists: list[str] = field(default_factory=list)

    @classmethod
    def from_tracks(cls, tracks: list[TrackRecord]) -> TasteProfile:
        if not tracks:
            return cls()
        genres: dict[str, int] = {}
        for t in tracks:
            if t.genre:
                genres[t.genre] = genres.get(t.genre, 0) + 1
        total = sum(genres.values()) or 1
        genre_dist = {g: c / total for g, c in genres.items()}
        bpms = [t.bpm for t in tracks if t.bpm > 0]
        artists: dict[str, int] = {}
        for t in tracks:
            if t.artist:
                artists[t.artist] = artists.get(t.artist, 0) + 1
        top = sorted(artists, key=artists.get, reverse=True)[:10]
        return cls(
            preferred_genres=genre_dist,
            bpm_mean=float(np.mean(bpms)) if bpms else 0,
            bpm_std=float(np.std(bpms)) if bpms else 0,
            energy_mean=float(np.mean([t.energy for t in tracks])),
            valence_mean=float(np.mean([t.valence for t in tracks])),
            arousal_mean=float(np.mean([t.arousal for t in tracks])),
            dominance_mean=float(np.mean([t.dominance for t in tracks])),
            top_artists=top,
        )

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.bpm_mean / 200.0,
            self.energy_mean,
            self.valence_mean,
            self.arousal_mean,
            self.dominance_mean,
        ], dtype=float)


class TasteEvolution:
    def __init__(self):
        self._snapshots: list[tuple[float, TasteProfile]] = []

    def add_snapshot(self, timestamp: float, profile: TasteProfile) -> None:
        self._snapshots.append((timestamp, profile))

    def energy_trend(self) -> list[tuple[float, float]]:
        return [(ts, p.energy_mean) for ts, p in sorted(self._snapshots)]

    def valence_trend(self) -> list[tuple[float, float]]:
        return [(ts, p.valence_mean) for ts, p in sorted(self._snapshots)]

    def drift(self) -> float:
        if len(self._snapshots) < 2:
            return 0.0
        first = self._snapshots[0][1].to_vector()
        last = self._snapshots[-1][1].to_vector()
        return float(np.linalg.norm(last - first))


class TasteDistance:
    @staticmethod
    def distance(a: TasteProfile, b: TasteProfile) -> float:
        return float(np.linalg.norm(a.to_vector() - b.to_vector()))

    @staticmethod
    def genre_overlap(a: TasteProfile, b: TasteProfile) -> float:
        all_genres = set(a.preferred_genres) | set(b.preferred_genres)
        if not all_genres:
            return 1.0
        overlap = 0.0
        for g in all_genres:
            overlap += min(a.preferred_genres.get(g, 0), b.preferred_genres.get(g, 0))
        return float(overlap)
