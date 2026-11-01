"""Music catalog management and valuation."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter


@dataclass
class CatalogEntry:
    title: str
    artist: str
    genre: str = ""
    year: int = 0
    isrc: str = ""
    annual_streams: int = 0


def catalog_summary(entries: list[CatalogEntry]) -> dict:
    genres = Counter(e.genre for e in entries if e.genre)
    years = Counter(e.year for e in entries if e.year > 0)
    artists = Counter(e.artist for e in entries)
    total_streams = sum(e.annual_streams for e in entries)
    return {
        "total_tracks": len(entries),
        "genres": dict(genres.most_common(10)),
        "years": dict(years.most_common(10)),
        "top_artists": dict(artists.most_common(10)),
        "total_annual_streams": total_streams,
    }


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def duplicate_detection(
    entries: list[CatalogEntry], threshold: float = 0.8
) -> list[tuple[int, int, float]]:
    dupes = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            t_sim = _jaccard(_normalize(entries[i].title), _normalize(entries[j].title))
            a_sim = _jaccard(_normalize(entries[i].artist), _normalize(entries[j].artist))
            combined = 0.6 * t_sim + 0.4 * a_sim
            if combined >= threshold:
                dupes.append((i, j, combined))
    return dupes


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def catalog_valuation_heuristic(
    entries: list[CatalogEntry], multiplier: float = 12.0
) -> float:
    total_annual = sum(e.annual_streams * 0.004 for e in entries)
    return total_annual * multiplier
