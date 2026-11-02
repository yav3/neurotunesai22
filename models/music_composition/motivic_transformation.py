"""Motivic development techniques.

Implements classical motivic transformations: retrograde, inversion,
augmentation, diminution, fragmentation, sequence, and similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class Motif:
    pitches: List[int]
    durations: List[float]
    label: str = ""


def retrograde(motif: Motif) -> Motif:
    return Motif(
        pitches=motif.pitches[::-1],
        durations=motif.durations[::-1],
        label=f"{motif.label}_R",
    )


def inversion(motif: Motif, axis: int | None = None) -> Motif:
    if not motif.pitches:
        return Motif(pitches=[], durations=list(motif.durations), label=f"{motif.label}_I")
    ax = axis if axis is not None else motif.pitches[0]
    inverted = [2 * ax - p for p in motif.pitches]
    return Motif(pitches=inverted, durations=list(motif.durations), label=f"{motif.label}_I")


def retrograde_inversion(motif: Motif, axis: int | None = None) -> Motif:
    inv = inversion(motif, axis)
    ri = retrograde(inv)
    ri.label = f"{motif.label}_RI"
    return ri


def augmentation(motif: Motif, factor: float = 2.0) -> Motif:
    return Motif(
        pitches=list(motif.pitches),
        durations=[d * factor for d in motif.durations],
        label=f"{motif.label}_aug{factor}",
    )


def diminution(motif: Motif, factor: float = 2.0) -> Motif:
    return Motif(
        pitches=list(motif.pitches),
        durations=[d / factor for d in motif.durations],
        label=f"{motif.label}_dim{factor}",
    )


def transposition(motif: Motif, semitones: int) -> Motif:
    return Motif(
        pitches=[p + semitones for p in motif.pitches],
        durations=list(motif.durations),
        label=f"{motif.label}_T{semitones}",
    )


def fragmentation(motif: Motif, start: int, end: int) -> Motif:
    return Motif(
        pitches=motif.pitches[start:end],
        durations=motif.durations[start:end],
        label=f"{motif.label}_frag",
    )


def sequence(motif: Motif, interval: int, n_repetitions: int) -> List[Motif]:
    """Sequential transposition (e.g., Beethoven's Fifth opening development)."""
    result = [motif]
    for i in range(1, n_repetitions):
        result.append(transposition(motif, interval * i))
    return result


def motivic_similarity(a: Motif, b: Motif) -> float:
    """Edit-distance-based similarity in [0,1] between two motifs."""
    pa, pb = a.pitches, b.pitches
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    m, n = len(pa), len(pb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pa[i - 1] == pb[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    max_len = max(m, n)
    return 1.0 - dp[m][n] / max_len


def interval_sequence(motif: Motif) -> List[int]:
    """Extract the interval sequence (successive pitch differences)."""
    if len(motif.pitches) < 2:
        return []
    return [motif.pitches[i + 1] - motif.pitches[i] for i in range(len(motif.pitches) - 1)]


def contour(motif: Motif) -> List[int]:
    """Morris contour: +1 for ascending, -1 for descending, 0 for repeated."""
    intervals = interval_sequence(motif)
    return [1 if iv > 0 else (-1 if iv < 0 else 0) for iv in intervals]


def contour_similarity(a: Motif, b: Motif) -> float:
    """Compare contour shapes: fraction of matching contour directions."""
    ca, cb = contour(a), contour(b)
    if not ca or not cb:
        return 0.0
    n = min(len(ca), len(cb))
    matches = sum(1 for i in range(n) if ca[i] == cb[i])
    return matches / n


TRANSFORMS = {
    "R": retrograde,
    "I": inversion,
    "RI": retrograde_inversion,
    "aug": augmentation,
    "dim": diminution,
}


def develop_motif(
    motif: Motif,
    techniques: List[str],
    seed: int | None = None,
) -> List[Motif]:
    """Chain multiple named transformations on a motif."""
    rng = np.random.default_rng(seed)
    results = [motif]
    current = motif
    for tech in techniques:
        if tech in ("R", "I", "RI"):
            current = TRANSFORMS[tech](current)
        elif tech.startswith("T"):
            semitones = int(tech[1:]) if len(tech) > 1 else int(rng.integers(-5, 6))
            current = transposition(current, semitones)
        elif tech == "aug":
            current = augmentation(current)
        elif tech == "dim":
            current = diminution(current)
        elif tech == "frag":
            mid = max(1, len(current.pitches) // 2)
            current = fragmentation(current, 0, mid)
        results.append(current)
    return results


def permute_rhythm(motif: Motif, seed: int = 0) -> Motif:
    """Shuffle durations while keeping pitches fixed."""
    rng = np.random.default_rng(seed)
    shuffled = list(motif.durations)
    rng.shuffle(shuffled)
    return Motif(pitches=list(motif.pitches), durations=shuffled, label=f"{motif.label}_rperm")
