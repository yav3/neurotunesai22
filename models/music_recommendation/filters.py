"""Pre/post-filtering for music recommendation pipelines."""
from __future__ import annotations

from typing import Any, Protocol

from models.music_playlist.transition import CAMELOT_NUMBERS


class TrackFilter(Protocol):
    def accept(self, track: dict[str, Any]) -> bool: ...


class BPMFilter:
    def __init__(self, min_bpm: float = 0.0, max_bpm: float = 999.0):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def accept(self, track: dict[str, Any]) -> bool:
        bpm = track.get("bpm", 0.0)
        return self.min_bpm <= bpm <= self.max_bpm


class GenreFilter:
    def __init__(self, genres: list[str]):
        self.genres = {g.lower() for g in genres}

    def accept(self, track: dict[str, Any]) -> bool:
        g = track.get("genre", "").lower()
        return any(allowed in g for allowed in self.genres)


class KeyFilter:
    def __init__(self, compatible_keys: list[str]):
        self.keys = {k.lower() for k in compatible_keys}

    def accept(self, track: dict[str, Any]) -> bool:
        k = track.get("key", "").lower()
        return k in self.keys


class DurationFilter:
    def __init__(self, min_s: float = 0.0, max_s: float = 3600.0):
        self.min_s = min_s
        self.max_s = max_s

    def accept(self, track: dict[str, Any]) -> bool:
        d = track.get("duration_s", track.get("duration", 0.0))
        return self.min_s <= d <= self.max_s


class EnergyFilter:
    def __init__(self, min_energy: float = 0.0, max_energy: float = 1.0):
        self.min_energy = min_energy
        self.max_energy = max_energy

    def accept(self, track: dict[str, Any]) -> bool:
        e = track.get("energy", 0.0)
        return self.min_energy <= e <= self.max_energy


class ExclusionFilter:
    def __init__(self, track_ids: set[str] | list[str]):
        self.excluded = set(track_ids)

    def accept(self, track: dict[str, Any]) -> bool:
        return track.get("id", "") not in self.excluded


class CompositeFilter:
    def __init__(self, filters: list[Any] | None = None):
        self._filters: list[Any] = list(filters or [])

    def add(self, f: Any) -> "CompositeFilter":
        self._filters.append(f)
        return self

    def accept(self, track: dict[str, Any]) -> bool:
        return all(f.accept(track) for f in self._filters)

    def apply(self, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [t for t in tracks if self.accept(t)]
