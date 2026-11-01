"""In-memory music catalog for track management, search, and statistics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.music_data.schema import TrackRecord


@dataclass
class CatalogStats:
    total_tracks: int = 0
    total_duration_hours: float = 0.0
    bpm_range: tuple[float, float] = (0.0, 0.0)
    bpm_mean: float = 0.0
    key_distribution: dict[str, int] = field(default_factory=dict)
    genre_distribution: dict[str, int] = field(default_factory=dict)
    valence_mean: float = 0.0
    arousal_mean: float = 0.0
    energy_mean: float = 0.0
    artists: int = 0
    with_lyrics: int = 0
    languages: dict[str, int] = field(default_factory=dict)


class MusicCatalog:
    def __init__(self):
        self._tracks: dict[str, TrackRecord] = {}

    def add(self, track: TrackRecord) -> None:
        key = track.id or track.path or track.title
        if not key:
            raise ValueError("Track must have id, path, or title")
        self._tracks[key] = track

    def add_many(self, tracks: list[TrackRecord]) -> int:
        added = 0
        for t in tracks:
            try:
                self.add(t)
                added += 1
            except ValueError:
                continue
        return added

    def get(self, key: str) -> TrackRecord | None:
        return self._tracks.get(key)

    def all(self) -> list[TrackRecord]:
        return list(self._tracks.values())

    def __len__(self) -> int:
        return len(self._tracks)

    def search(
        self,
        genre: str | None = None,
        artist: str | None = None,
        bpm_range: tuple[float, float] | None = None,
        key: str | None = None,
        valence_range: tuple[float, float] | None = None,
        arousal_range: tuple[float, float] | None = None,
        energy_range: tuple[float, float] | None = None,
        has_lyrics: bool | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[TrackRecord]:
        results = []
        for t in self._tracks.values():
            if genre and genre.lower() not in t.genre.lower():
                continue
            if artist and artist.lower() not in t.artist.lower():
                continue
            if bpm_range and not (bpm_range[0] <= t.bpm <= bpm_range[1]):
                continue
            if key and t.key_str.lower() != key.lower():
                continue
            if valence_range and not (valence_range[0] <= t.valence <= valence_range[1]):
                continue
            if arousal_range and not (arousal_range[0] <= t.arousal <= arousal_range[1]):
                continue
            if energy_range and not (energy_range[0] <= t.energy <= energy_range[1]):
                continue
            if has_lyrics is not None and t.has_lyrics != has_lyrics:
                continue
            if tags and not any(tag in t.tags for tag in tags):
                continue
            results.append(t)
            if len(results) >= limit:
                break
        return results

    def nearest_by_vad(
        self,
        valence: float,
        arousal: float,
        dominance: float = 0.5,
        n: int = 5,
    ) -> list[tuple[TrackRecord, float]]:
        scored = []
        for t in self._tracks.values():
            dist = np.sqrt(
                (t.valence - valence) ** 2
                + (t.arousal - arousal) ** 2
                + (t.dominance - dominance) ** 2
            )
            scored.append((t, float(dist)))
        scored.sort(key=lambda x: x[1])
        return scored[:n]

    def stats(self) -> CatalogStats:
        tracks = self.all()
        if not tracks:
            return CatalogStats()

        bpms = [t.bpm for t in tracks if t.bpm > 0]
        keys = {}
        genres = {}
        artists = set()
        langs = {}
        with_lyrics = 0

        for t in tracks:
            if t.key_str:
                keys[t.key_str] = keys.get(t.key_str, 0) + 1
            if t.genre:
                genres[t.genre] = genres.get(t.genre, 0) + 1
            if t.artist:
                artists.add(t.artist)
            if t.has_lyrics:
                with_lyrics += 1
                if t.lyric_language:
                    langs[t.lyric_language] = langs.get(t.lyric_language, 0) + 1

        return CatalogStats(
            total_tracks=len(tracks),
            total_duration_hours=sum(t.duration_s for t in tracks) / 3600,
            bpm_range=(min(bpms) if bpms else 0, max(bpms) if bpms else 0),
            bpm_mean=float(np.mean(bpms)) if bpms else 0,
            key_distribution=keys,
            genre_distribution=genres,
            valence_mean=float(np.mean([t.valence for t in tracks])),
            arousal_mean=float(np.mean([t.arousal for t in tracks])),
            energy_mean=float(np.mean([t.energy for t in tracks])),
            artists=len(artists),
            with_lyrics=with_lyrics,
            languages=langs,
        )

    def to_feature_matrix(self) -> tuple[np.ndarray, list[str], list[str]]:
        tracks = self.all()
        if not tracks:
            return np.empty((0, 0)), [], []

        feature_names = list(tracks[0].to_feature_dict().keys())
        matrix = np.zeros((len(tracks), len(feature_names)))
        ids = []

        for i, t in enumerate(tracks):
            fd = t.to_feature_dict()
            for j, name in enumerate(feature_names):
                matrix[i, j] = fd.get(name, 0.0)
            ids.append(t.id or t.path or t.title)

        return matrix, ids, feature_names
