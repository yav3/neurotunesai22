"""Flamenco: compás patterns, palos classification, Phrygian mode analysis."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


COMPAS_PATTERNS: dict[str, tuple[int, ...]] = {
    "bulerias":    (1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0),
    "solea":       (1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0),
    "alegrias":    (1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0),
    "seguiriya":   (1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0),
    "tangos":      (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    "fandango":    (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    "rumba":       (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    "tientos":     (1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
}

PHRYGIAN_INTERVALS = (0, 1, 3, 5, 7, 8, 10)
PHRYGIAN_DOMINANT_INTERVALS = (0, 1, 4, 5, 7, 8, 10)


@dataclass(frozen=True)
class Palo:
    name: str
    compas_beats: int
    meter: str
    mode: str
    typical_tempo_range: tuple[int, int]
    cadence: str = "andalusian"
    tags: tuple[str, ...] = ()


PALO_CATALOG: dict[str, Palo] = {
    "bulerias": Palo("bulerias", 12, "12/8", "phrygian_dominant", (180, 250), tags=("festero", "rapid")),
    "solea": Palo("solea", 12, "12/8", "phrygian", (80, 120), tags=("jondo", "solemn")),
    "alegrias": Palo("alegrias", 12, "12/8", "major", (120, 160), tags=("festero", "cadiz")),
    "seguiriya": Palo("seguiriya", 12, "12/8", "phrygian", (60, 90), tags=("jondo", "serious")),
    "tangos": Palo("tangos", 4, "4/4", "phrygian", (100, 140), tags=("festero",)),
    "fandango": Palo("fandango", 6, "3/4", "phrygian_dominant", (80, 140), tags=("libre",)),
    "rumba": Palo("rumba", 4, "4/4", "major", (120, 180), tags=("festero", "popular")),
    "tientos": Palo("tientos", 4, "4/4", "phrygian", (60, 90), tags=("jondo",)),
    "farruca": Palo("farruca", 4, "4/4", "minor", (80, 120), tags=("jondo", "galician")),
    "peteneras": Palo("peteneras", 12, "12/8", "phrygian", (80, 110), tags=("jondo", "rare")),
    "guajira": Palo("guajira", 12, "12/8", "major", (100, 130), tags=("ida_y_vuelta", "cuban")),
    "garrotin": Palo("garrotin", 4, "4/4", "major", (100, 150), tags=("festero",)),
}


def is_phrygian(pitches: list[int], root: int = 0) -> bool:
    """Check if a pitch set is Phrygian relative to root."""
    classes = frozenset((p - root) % 12 for p in pitches)
    return classes.issubset(frozenset(PHRYGIAN_INTERVALS))


def is_phrygian_dominant(pitches: list[int], root: int = 0) -> bool:
    classes = frozenset((p - root) % 12 for p in pitches)
    return classes.issubset(frozenset(PHRYGIAN_DOMINANT_INTERVALS))


def andalusian_cadence_score(chord_roots: list[int]) -> float:
    """Score how well a root progression matches the Andalusian cadence Am-G-F-E (iv-III-II-I).
    Returns [0, 1]."""
    if len(chord_roots) < 4:
        return 0.0
    pattern = np.array([0, -2, -4, -5])
    best = 0.0
    for start in range(len(chord_roots) - 3):
        window = np.array(chord_roots[start:start + 4])
        diffs = np.diff(window) % 12
        expected = np.diff(pattern) % 12
        matches = sum(1 for d, e in zip(diffs, expected) if d == e)
        best = max(best, matches / 3.0)
    return best


def classify_palo(tempo: float, meter: str, mode: str) -> list[tuple[str, float]]:
    """Match musical features to flamenco palos."""
    results = []
    for name, palo in PALO_CATALOG.items():
        score = 0.0
        lo, hi = palo.typical_tempo_range
        if lo <= tempo <= hi:
            score += 0.4
        elif abs(tempo - (lo + hi) / 2) < 40:
            score += 0.2
        if palo.meter == meter:
            score += 0.3
        if palo.mode == mode:
            score += 0.3
        if score > 0.3:
            results.append((name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def compas_correlation(onsets: np.ndarray, palo: str, beat_duration: float) -> float:
    """Correlation between onset pattern and compás template.
    onsets: times in seconds; beat_duration: compás beat duration."""
    if palo not in COMPAS_PATTERNS:
        return 0.0
    pattern = np.array(COMPAS_PATTERNS[palo], dtype=float)
    n_beats = len(pattern)
    if len(onsets) == 0 or beat_duration <= 0:
        return 0.0
    observed = np.zeros(n_beats)
    for t in onsets:
        beat_pos = int((t / beat_duration) % n_beats)
        observed[beat_pos] += 1.0
    if observed.max() > 0:
        observed /= observed.max()
    if pattern.max() > 0:
        pattern = pattern / pattern.max()
    return float(np.corrcoef(observed, pattern)[0, 1])


def rasgueado_density(onset_times: np.ndarray, window: float = 0.5) -> np.ndarray:
    """Detect rasgueado sections by onset density in sliding windows."""
    if len(onset_times) < 2:
        return np.array([])
    t_min, t_max = onset_times[0], onset_times[-1]
    n_windows = max(1, int((t_max - t_min) / window))
    densities = np.zeros(n_windows)
    for i in range(n_windows):
        t_lo = t_min + i * window
        t_hi = t_lo + window
        densities[i] = np.sum((onset_times >= t_lo) & (onset_times < t_hi))
    return densities
