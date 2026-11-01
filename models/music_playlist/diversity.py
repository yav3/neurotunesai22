"""Playlist diversity metrics: genre entropy, tempo spread, artist balance."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np


def genre_diversity(genres: list[str]) -> float:
    if not genres:
        return 0.0
    counts = Counter(genres)
    total = len(genres)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def tempo_diversity(tempos: list[float]) -> float:
    if len(tempos) < 2:
        return 0.0
    arr = np.array(tempos, dtype=float)
    mean = np.mean(arr)
    if mean < 1e-10:
        return 0.0
    return float(np.clip(np.std(arr) / mean, 0.0, 1.0))


def artist_diversity(artists: list[str]) -> float:
    if not artists:
        return 0.0
    unique = len(set(artists))
    return unique / len(artists)


def decade_spread(years: list[int]) -> float:
    if not years:
        return 0.0
    decades = set(y // 10 for y in years)
    return min(len(decades) / 5.0, 1.0)


def mood_coverage(
    moods: list[str], universe: list[str] | None = None,
) -> float:
    if not moods:
        return 0.0
    if universe is None:
        universe = [
            "happy", "sad", "energetic", "calm", "angry",
            "romantic", "melancholic", "uplifting", "dark", "peaceful",
        ]
    present = set(m.lower() for m in moods)
    covered = len(present & set(u.lower() for u in universe))
    return covered / max(len(universe), 1)


@dataclass(frozen=True)
class PlaylistDiversityReport:
    genre_score: float
    tempo_score: float
    artist_score: float
    decade_score: float
    mood_score: float
    overall_diversity: float


def compute_diversity(tracks: list[dict]) -> PlaylistDiversityReport:
    genres = [t.get("genre", "unknown") for t in tracks]
    tempos = [float(t.get("tempo", 120.0)) for t in tracks]
    artists = [t.get("artist", "unknown") for t in tracks]
    years = [int(t.get("year", 2000)) for t in tracks]
    moods = [t.get("mood", "") for t in tracks if t.get("mood")]

    gs = genre_diversity(genres)
    ts = tempo_diversity(tempos)
    ars = artist_diversity(artists)
    ds = decade_spread(years)
    ms = mood_coverage(moods)

    overall = 0.3 * gs + 0.15 * ts + 0.25 * ars + 0.15 * ds + 0.15 * ms
    return PlaylistDiversityReport(
        genre_score=gs, tempo_score=ts, artist_score=ars,
        decade_score=ds, mood_score=ms, overall_diversity=overall,
    )
