"""Neo-Riemannian transformations: P/L/R operations, Tonnetz navigation.

Implements the three basic neo-Riemannian operations on triads (Parallel,
Leading-tone exchange, Relative) and their compositions, plus Tonnetz
coordinate geometry for visualizing harmonic motion as paths on a torus.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TriadQuality(Enum):
    MAJOR = "major"
    MINOR = "minor"


@dataclass(frozen=True)
class Triad:
    root: int
    quality: TriadQuality

    @property
    def pitch_classes(self) -> tuple[int, int, int]:
        r = self.root % 12
        if self.quality == TriadQuality.MAJOR:
            return (r, (r + 4) % 12, (r + 7) % 12)
        return (r, (r + 3) % 12, (r + 7) % 12)

    def __repr__(self) -> str:
        names = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]
        q = "" if self.quality == TriadQuality.MAJOR else "m"
        return f"{names[self.root % 12]}{q}"


# ── Core PLR operations ─────────────────────────────────────────────

def parallel(t: Triad) -> Triad:
    """P: toggle major/minor by moving the third (C → Cm, Cm → C)."""
    if t.quality == TriadQuality.MAJOR:
        return Triad(root=t.root, quality=TriadQuality.MINOR)
    return Triad(root=t.root, quality=TriadQuality.MAJOR)


def leading_tone(t: Triad) -> Triad:
    """L: leading-tone exchange. Moves the note not shared by the relative triad.

    Major: root moves down a semitone (C → Em).
    Minor: fifth moves up a semitone (Cm → Ab).
    """
    if t.quality == TriadQuality.MAJOR:
        new_root = (t.root - 1) % 12
        return Triad(root=(t.root + 4) % 12, quality=TriadQuality.MINOR)
    return Triad(root=(t.root + 3) % 12, quality=TriadQuality.MAJOR)


def relative(t: Triad) -> Triad:
    """R: relative transformation (C → Am, Am → C)."""
    if t.quality == TriadQuality.MAJOR:
        return Triad(root=(t.root + 9) % 12, quality=TriadQuality.MINOR)
    return Triad(root=(t.root + 3) % 12, quality=TriadQuality.MAJOR)


# ── Composite transformations ───────────────────────────────────────

NR_OPS = {"P": parallel, "L": leading_tone, "R": relative}


def apply_sequence(t: Triad, ops: str) -> list[Triad]:
    """Apply a sequence of P/L/R operations, returning all intermediate triads.

    ``ops`` is a string like "PRL" meaning P then R then L.
    """
    path = [t]
    current = t
    for ch in ops.upper():
        fn = NR_OPS.get(ch)
        if fn is None:
            raise ValueError(f"Unknown operation: {ch}")
        current = fn(current)
        path.append(current)
    return path


def hexatonic_cycle(t: Triad) -> list[Triad]:
    """The hexatonic (PL) cycle starting from triad t.

    Alternates P and L; returns to the start after 6 steps.
    """
    return apply_sequence(t, "PLPLPLPL")


def octatonic_cycle(t: Triad) -> list[Triad]:
    """The octatonic (PR) cycle starting from triad t.

    Alternates P and R; returns to the start after 8 steps.
    """
    return apply_sequence(t, "PRPRPRPR")


# ── Tonnetz coordinates ─────────────────────────────────────────────

def tonnetz_coordinates(pc: int) -> tuple[float, float]:
    """Map a pitch class to Tonnetz (x, y) coordinates.

    x-axis: perfect fifths; y-axis: major thirds.
    The Tonnetz is a quotient of Z^2; we embed it in R^2
    via the standard fifth/third generators.
    """
    fifth_steps = 0
    third_steps = 0
    target = pc % 12
    best = (0, 0, 12)
    for f in range(-6, 7):
        for t in range(-2, 3):
            candidate = (7 * f + 4 * t) % 12
            dist = abs(f) + abs(t)
            if candidate == target and dist < best[2]:
                best = (f, t, dist)
    return (float(best[0]), float(best[1]))


def triad_tonnetz_centroid(t: Triad) -> tuple[float, float]:
    """Centroid of a triad's three pitch classes on the Tonnetz."""
    pcs = t.pitch_classes
    coords = [tonnetz_coordinates(pc) for pc in pcs]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (sum(xs) / 3.0, sum(ys) / 3.0)


def tonnetz_distance(t1: Triad, t2: Triad) -> float:
    """Euclidean distance between triad centroids on the Tonnetz."""
    c1 = triad_tonnetz_centroid(t1)
    c2 = triad_tonnetz_centroid(t2)
    return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))


# ── Voice-leading parsimony ─────────────────────────────────────────

def voice_leading_distance(t1: Triad, t2: Triad) -> int:
    """Total semitone displacement of the most parsimonious voice leading.

    Uses the sum of minimum pitch-class distances between ordered chord tones.
    """
    pcs1 = sorted(t1.pitch_classes)
    pcs2 = sorted(t2.pitch_classes)
    total = 0
    for a, b in zip(pcs1, pcs2):
        d = abs(a - b)
        total += min(d, 12 - d)
    return total


def plr_distance(t1: Triad, t2: Triad, max_depth: int = 6) -> int:
    """Minimum number of P/L/R steps to reach t2 from t1.

    BFS over the PLR graph; returns -1 if unreachable within max_depth.
    """
    if t1.root % 12 == t2.root % 12 and t1.quality == t2.quality:
        return 0
    visited = {(t1.root % 12, t1.quality)}
    frontier = [t1]
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for t in frontier:
            for fn in (parallel, leading_tone, relative):
                nt = fn(t)
                key = (nt.root % 12, nt.quality)
                if key == (t2.root % 12, t2.quality):
                    return depth
                if key not in visited:
                    visited.add(key)
                    next_frontier.append(nt)
        frontier = next_frontier
    return -1


# ── Harmonic path analysis ──────────────────────────────────────────

@dataclass(frozen=True)
class HarmonicPathStats:
    n_chords: int
    total_vl_distance: int
    mean_vl_distance: float
    total_tonnetz_distance: float
    mean_tonnetz_distance: float
    quality_changes: int


def analyze_harmonic_path(triads: list[Triad]) -> HarmonicPathStats:
    """Analyze a sequence of triads for voice-leading and Tonnetz motion."""
    n = len(triads)
    if n < 2:
        return HarmonicPathStats(n, 0, 0.0, 0.0, 0.0, 0)
    vl_dists = [voice_leading_distance(triads[i], triads[i + 1]) for i in range(n - 1)]
    tn_dists = [tonnetz_distance(triads[i], triads[i + 1]) for i in range(n - 1)]
    qc = sum(1 for i in range(n - 1) if triads[i].quality != triads[i + 1].quality)
    return HarmonicPathStats(
        n_chords=n,
        total_vl_distance=sum(vl_dists),
        mean_vl_distance=float(np.mean(vl_dists)),
        total_tonnetz_distance=float(sum(tn_dists)),
        mean_tonnetz_distance=float(np.mean(tn_dists)),
        quality_changes=qc,
    )
