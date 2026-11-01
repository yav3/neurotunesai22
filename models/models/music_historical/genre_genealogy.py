"""Genre genealogy: family tree, distance metrics, fusion analysis."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Genre:
    name: str
    parent_genres: list[str] = field(default_factory=list)
    child_genres: list[str] = field(default_factory=list)
    origin_year: int = 0
    peak_year: int = 0
    region: str = ""
    defining_features: dict[str, float] = field(default_factory=dict)


_RAW_GENRES: dict[str, dict] = {
    "plainchant": {"origin": 600, "peak": 1000, "region": "Europe",
                   "features": {"tempo": 0.2, "harmony": 0.0, "rhythm": 0.1, "electric": 0.0, "vocal": 1.0}},
    "madrigal": {"origin": 1520, "peak": 1600, "region": "Italy", "parents": ["plainchant"],
                 "features": {"tempo": 0.3, "harmony": 0.4, "rhythm": 0.3, "electric": 0.0, "vocal": 0.9}},
    "opera": {"origin": 1597, "peak": 1800, "region": "Italy", "parents": ["madrigal"],
              "features": {"tempo": 0.4, "harmony": 0.6, "rhythm": 0.4, "electric": 0.0, "vocal": 0.9}},
    "baroque": {"origin": 1600, "peak": 1720, "region": "Europe", "parents": ["plainchant"],
                "features": {"tempo": 0.5, "harmony": 0.5, "rhythm": 0.5, "electric": 0.0, "vocal": 0.3}},
    "classical": {"origin": 1730, "peak": 1800, "region": "Europe", "parents": ["baroque"],
                  "features": {"tempo": 0.5, "harmony": 0.6, "rhythm": 0.5, "electric": 0.0, "vocal": 0.2}},
    "romantic": {"origin": 1800, "peak": 1880, "region": "Europe", "parents": ["classical"],
                 "features": {"tempo": 0.5, "harmony": 0.8, "rhythm": 0.5, "electric": 0.0, "vocal": 0.3}},
    "impressionism": {"origin": 1875, "peak": 1910, "region": "France", "parents": ["romantic"],
                      "features": {"tempo": 0.4, "harmony": 0.9, "rhythm": 0.3, "electric": 0.0, "vocal": 0.2}},
    "blues": {"origin": 1870, "peak": 1950, "region": "USA", "parents": [],
              "features": {"tempo": 0.4, "harmony": 0.4, "rhythm": 0.6, "electric": 0.2, "vocal": 0.8}},
    "ragtime": {"origin": 1895, "peak": 1920, "region": "USA", "parents": ["blues"],
                "features": {"tempo": 0.6, "harmony": 0.5, "rhythm": 0.8, "electric": 0.0, "vocal": 0.1}},
    "jazz": {"origin": 1900, "peak": 1960, "region": "USA", "parents": ["blues", "ragtime"],
             "features": {"tempo": 0.6, "harmony": 0.8, "rhythm": 0.8, "electric": 0.2, "vocal": 0.5}},
    "gospel": {"origin": 1920, "peak": 1960, "region": "USA", "parents": ["blues"],
               "features": {"tempo": 0.5, "harmony": 0.5, "rhythm": 0.6, "electric": 0.1, "vocal": 0.9}},
    "country": {"origin": 1920, "peak": 1970, "region": "USA", "parents": ["blues", "gospel"],
                "features": {"tempo": 0.5, "harmony": 0.4, "rhythm": 0.5, "electric": 0.3, "vocal": 0.8}},
    "swing": {"origin": 1930, "peak": 1945, "region": "USA", "parents": ["jazz"],
              "features": {"tempo": 0.7, "harmony": 0.6, "rhythm": 0.8, "electric": 0.1, "vocal": 0.5}},
    "bebop": {"origin": 1940, "peak": 1955, "region": "USA", "parents": ["jazz"],
              "features": {"tempo": 0.8, "harmony": 0.9, "rhythm": 0.9, "electric": 0.1, "vocal": 0.2}},
    "rhythm_and_blues": {"origin": 1940, "peak": 1960, "region": "USA", "parents": ["blues", "jazz", "gospel"],
                         "features": {"tempo": 0.6, "harmony": 0.5, "rhythm": 0.7, "electric": 0.4, "vocal": 0.8}},
    "rock_and_roll": {"origin": 1950, "peak": 1965, "region": "USA", "parents": ["rhythm_and_blues", "country"],
                      "features": {"tempo": 0.7, "harmony": 0.4, "rhythm": 0.7, "electric": 0.7, "vocal": 0.7}},
    "soul": {"origin": 1955, "peak": 1970, "region": "USA", "parents": ["rhythm_and_blues", "gospel"],
             "features": {"tempo": 0.5, "harmony": 0.6, "rhythm": 0.7, "electric": 0.4, "vocal": 0.9}},
    "rock": {"origin": 1960, "peak": 1975, "region": "USA/UK", "parents": ["rock_and_roll", "blues"],
             "features": {"tempo": 0.6, "harmony": 0.4, "rhythm": 0.7, "electric": 0.9, "vocal": 0.6}},
    "funk": {"origin": 1965, "peak": 1978, "region": "USA", "parents": ["soul", "jazz"],
             "features": {"tempo": 0.7, "harmony": 0.4, "rhythm": 0.9, "electric": 0.7, "vocal": 0.6}},
    "reggae": {"origin": 1968, "peak": 1980, "region": "Jamaica", "parents": ["rhythm_and_blues"],
               "features": {"tempo": 0.4, "harmony": 0.3, "rhythm": 0.8, "electric": 0.5, "vocal": 0.7}},
    "progressive_rock": {"origin": 1968, "peak": 1977, "region": "UK", "parents": ["rock", "classical"],
                         "features": {"tempo": 0.5, "harmony": 0.7, "rhythm": 0.7, "electric": 0.8, "vocal": 0.5}},
    "heavy_metal": {"origin": 1970, "peak": 1985, "region": "UK/USA", "parents": ["rock", "blues"],
                    "features": {"tempo": 0.7, "harmony": 0.3, "rhythm": 0.7, "electric": 1.0, "vocal": 0.6}},
    "disco": {"origin": 1972, "peak": 1979, "region": "USA", "parents": ["funk", "soul"],
              "features": {"tempo": 0.8, "harmony": 0.4, "rhythm": 0.9, "electric": 0.6, "vocal": 0.6}},
    "punk": {"origin": 1974, "peak": 1980, "region": "USA/UK", "parents": ["rock"],
             "features": {"tempo": 0.8, "harmony": 0.2, "rhythm": 0.7, "electric": 0.9, "vocal": 0.7}},
    "hip_hop": {"origin": 1973, "peak": 2020, "region": "USA", "parents": ["funk", "disco"],
                "features": {"tempo": 0.5, "harmony": 0.2, "rhythm": 0.9, "electric": 0.6, "vocal": 0.9}},
    "new_wave": {"origin": 1978, "peak": 1985, "region": "UK/USA", "parents": ["punk", "disco"],
                 "features": {"tempo": 0.7, "harmony": 0.4, "rhythm": 0.7, "electric": 0.8, "vocal": 0.7}},
    "synthpop": {"origin": 1978, "peak": 1985, "region": "UK", "parents": ["new_wave"],
                 "features": {"tempo": 0.7, "harmony": 0.4, "rhythm": 0.7, "electric": 1.0, "vocal": 0.6}},
    "house": {"origin": 1984, "peak": 2000, "region": "USA", "parents": ["disco", "synthpop"],
              "features": {"tempo": 0.8, "harmony": 0.3, "rhythm": 0.9, "electric": 0.9, "vocal": 0.3}},
    "techno": {"origin": 1985, "peak": 2000, "region": "USA", "parents": ["synthpop", "funk"],
               "features": {"tempo": 0.8, "harmony": 0.2, "rhythm": 0.9, "electric": 1.0, "vocal": 0.1}},
    "grunge": {"origin": 1986, "peak": 1994, "region": "USA", "parents": ["punk", "heavy_metal"],
               "features": {"tempo": 0.6, "harmony": 0.3, "rhythm": 0.7, "electric": 0.9, "vocal": 0.7}},
    "ambient": {"origin": 1978, "peak": 2010, "region": "UK", "parents": ["synthpop"],
                "features": {"tempo": 0.2, "harmony": 0.5, "rhythm": 0.1, "electric": 0.9, "vocal": 0.1}},
    "drum_and_bass": {"origin": 1991, "peak": 2005, "region": "UK", "parents": ["techno", "hip_hop"],
                      "features": {"tempo": 0.9, "harmony": 0.2, "rhythm": 1.0, "electric": 0.9, "vocal": 0.2}},
    "trip_hop": {"origin": 1991, "peak": 1998, "region": "UK", "parents": ["hip_hop", "ambient"],
                 "features": {"tempo": 0.4, "harmony": 0.4, "rhythm": 0.6, "electric": 0.8, "vocal": 0.5}},
    "neo_soul": {"origin": 1994, "peak": 2005, "region": "USA", "parents": ["soul", "hip_hop"],
                 "features": {"tempo": 0.4, "harmony": 0.6, "rhythm": 0.6, "electric": 0.5, "vocal": 0.9}},
    "edm": {"origin": 2000, "peak": 2015, "region": "Global", "parents": ["house", "techno"],
            "features": {"tempo": 0.8, "harmony": 0.3, "rhythm": 0.9, "electric": 1.0, "vocal": 0.3}},
    "indie_rock": {"origin": 1985, "peak": 2010, "region": "USA/UK", "parents": ["punk", "rock"],
                   "features": {"tempo": 0.6, "harmony": 0.5, "rhythm": 0.6, "electric": 0.7, "vocal": 0.7}},
    "trap": {"origin": 2003, "peak": 2020, "region": "USA", "parents": ["hip_hop"],
             "features": {"tempo": 0.5, "harmony": 0.2, "rhythm": 0.9, "electric": 0.8, "vocal": 0.8}},
    "lo_fi": {"origin": 2010, "peak": 2022, "region": "Global", "parents": ["hip_hop", "jazz", "ambient"],
              "features": {"tempo": 0.3, "harmony": 0.5, "rhythm": 0.5, "electric": 0.7, "vocal": 0.2}},
    "afrobeats": {"origin": 2005, "peak": 2025, "region": "Nigeria", "parents": ["hip_hop", "reggae", "funk"],
                  "features": {"tempo": 0.6, "harmony": 0.3, "rhythm": 0.9, "electric": 0.6, "vocal": 0.8}},
    "latin_pop": {"origin": 1990, "peak": 2020, "region": "Latin America", "parents": ["disco", "rock"],
                  "features": {"tempo": 0.7, "harmony": 0.4, "rhythm": 0.8, "electric": 0.6, "vocal": 0.8}},
    "k_pop": {"origin": 1992, "peak": 2025, "region": "South Korea", "parents": ["hip_hop", "edm", "rhythm_and_blues"],
              "features": {"tempo": 0.7, "harmony": 0.5, "rhythm": 0.8, "electric": 0.8, "vocal": 0.8}},
    "hyperpop": {"origin": 2018, "peak": 2024, "region": "Global", "parents": ["edm", "hip_hop", "punk"],
                 "features": {"tempo": 0.8, "harmony": 0.3, "rhythm": 0.8, "electric": 1.0, "vocal": 0.7}},
}

GENRE_TREE: dict[str, Genre] = {}

for _name, _info in _RAW_GENRES.items():
    GENRE_TREE[_name] = Genre(
        name=_name,
        parent_genres=_info.get("parents", []),
        origin_year=_info.get("origin", 0),
        peak_year=_info.get("peak", 0),
        region=_info.get("region", ""),
        defining_features=_info.get("features", {}),
    )

for _name, _genre in GENRE_TREE.items():
    for _parent in _genre.parent_genres:
        if _parent in GENRE_TREE:
            GENRE_TREE[_parent].child_genres.append(_name)


def _feature_vector(g: Genre) -> np.ndarray:
    keys = sorted(g.defining_features.keys())
    return np.array([g.defining_features.get(k, 0.0) for k in keys])


def genre_distance(a: str, b: str) -> float:
    ga, gb = GENRE_TREE.get(a), GENRE_TREE.get(b)
    if ga is None or gb is None:
        return float("inf")
    va, vb = _feature_vector(ga), _feature_vector(gb)
    if len(va) == 0 or len(vb) == 0:
        return float("inf")
    feat_dist = float(np.linalg.norm(va - vb))
    chain = genre_influence_chain(a, b) or genre_influence_chain(b, a)
    tree_dist = len(chain) - 1 if chain else 10.0
    return 0.5 * feat_dist + 0.5 * (tree_dist / 10.0)


def genre_influence_chain(source: str, target: str) -> list[str]:
    if source not in GENRE_TREE or target not in GENRE_TREE:
        return []
    visited: set[str] = set()
    queue: deque[list[str]] = deque([[target]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == source:
            return list(reversed(path))
        if node in visited:
            continue
        visited.add(node)
        g = GENRE_TREE.get(node)
        if g is None:
            continue
        for parent in g.parent_genres:
            if parent not in visited:
                queue.append(path + [parent])
    return []


def genres_by_era(start_year: int, end_year: int) -> list[Genre]:
    return [
        g for g in GENRE_TREE.values()
        if g.origin_year >= start_year and g.origin_year <= end_year
    ]


def genre_fusion_candidates(genre_name: str, max_distance: float = 0.6) -> list[tuple[str, float]]:
    if genre_name not in GENRE_TREE:
        return []
    candidates: list[tuple[str, float]] = []
    for other in GENRE_TREE:
        if other == genre_name:
            continue
        d = genre_distance(genre_name, other)
        if d <= max_distance:
            candidates.append((other, d))
    candidates.sort(key=lambda x: x[1])
    return candidates


def genre_innovation_index(genre_name: str) -> float:
    g = GENRE_TREE.get(genre_name)
    if g is None:
        return 0.0
    parent_features: list[np.ndarray] = []
    for p in g.parent_genres:
        pg = GENRE_TREE.get(p)
        if pg:
            parent_features.append(_feature_vector(pg))
    if not parent_features:
        return 1.0
    avg_parent = np.mean(parent_features, axis=0)
    own = _feature_vector(g)
    return float(np.linalg.norm(own - avg_parent))
