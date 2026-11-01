"""World rhythm traditions: tala, clave, compas, aksak, polyrhythm."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TalaPattern:
    name: str
    beats: int
    subdivisions: tuple[int, ...]
    accents: tuple[int, ...]
    tradition: str = "Indian"


_INDIAN_TALAS: dict[str, TalaPattern] = {
    "teental": TalaPattern("Teental", 16, (4, 4, 4, 4), (0, 4, 8, 12), "Hindustani"),
    "jhaptaal": TalaPattern("Jhaptaal", 10, (2, 3, 2, 3), (0, 2, 5, 7), "Hindustani"),
    "rupak": TalaPattern("Rupak", 7, (3, 2, 2), (0, 3, 5), "Hindustani"),
    "dadra": TalaPattern("Dadra", 6, (3, 3), (0, 3), "Hindustani"),
    "ektaal": TalaPattern("Ektaal", 12, (2, 2, 2, 2, 2, 2), (0, 2, 4, 6, 8, 10), "Hindustani"),
    "adi": TalaPattern("Adi", 8, (4, 2, 2), (0, 4, 6), "Carnatic"),
    "rupaka": TalaPattern("Rupaka", 4, (2, 2), (0, 2), "Carnatic"),
    "misra_chapu": TalaPattern("Misra Chapu", 7, (3, 2, 2), (0, 3, 5), "Carnatic"),
}


def indian_tala(name: str) -> TalaPattern:
    key = name.lower().replace(" ", "_")
    if key not in _INDIAN_TALAS:
        raise ValueError(f"Unknown tala: {name}. Available: {list(_INDIAN_TALAS.keys())}")
    return _INDIAN_TALAS[key]


_AFRICAN_TIMELINES: dict[str, tuple[int, ...]] = {
    "standard": (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0),
    "son_clave": (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0),
    "rumba_clave": (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0),
    "bembe": (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0),
    "agbadza": (1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0),
    "gahu": (1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0),
    "bossa_nova": (1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0),
}


def african_timeline(name: str) -> tuple[int, ...]:
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in _AFRICAN_TIMELINES:
        raise ValueError(f"Unknown timeline: {name}. Available: {list(_AFRICAN_TIMELINES.keys())}")
    return _AFRICAN_TIMELINES[key]


_FLAMENCO_COMPAS: dict[str, tuple[int, ...]] = {
    "bulerias": (0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1),
    "solea": (0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0),
    "siguiriyas": (0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0),
    "fandango": (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0),
    "tangos": (1, 0, 0, 0, 1, 0, 0, 0),
}


def flamenco_compas(palo: str) -> tuple[int, ...]:
    key = palo.lower().replace(" ", "_")
    if key not in _FLAMENCO_COMPAS:
        raise ValueError(f"Unknown palo: {palo}. Available: {list(_FLAMENCO_COMPAS.keys())}")
    return _FLAMENCO_COMPAS[key]


_AKSAK_PATTERNS: dict[str, tuple[int, ...]] = {
    "7_8_a": (2, 2, 3),
    "7_8_b": (3, 2, 2),
    "9_8_a": (2, 2, 2, 3),
    "9_8_b": (2, 3, 2, 2),
    "11_8": (2, 2, 3, 2, 2),
    "15_8": (2, 2, 2, 3, 2, 2, 2),
}


def aksak_pattern(name: str) -> tuple[int, ...]:
    key = name.lower().replace("/", "_").replace(" ", "_")
    if key not in _AKSAK_PATTERNS:
        raise ValueError(f"Unknown aksak: {name}. Available: {list(_AKSAK_PATTERNS.keys())}")
    return _AKSAK_PATTERNS[key]


def polyrhythm_matrix(patterns: list[tuple[int, ...]]) -> np.ndarray:
    if not patterns:
        return np.array([])
    max_len = max(len(p) for p in patterns)
    matrix = np.zeros((len(patterns), max_len), dtype=int)
    for i, p in enumerate(patterns):
        for j, v in enumerate(p):
            matrix[i, j] = v
    return matrix


def cross_rhythm_density(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    pa = np.zeros(max_len, dtype=int)
    pb = np.zeros(max_len, dtype=int)
    for i, v in enumerate(a):
        pa[i] = v
    for i, v in enumerate(b):
        pb[i] = v
    coincidences = np.sum((pa > 0) & (pb > 0))
    total_onsets = np.sum((pa > 0) | (pb > 0))
    if total_onsets == 0:
        return 0.0
    return 1.0 - coincidences / total_onsets
