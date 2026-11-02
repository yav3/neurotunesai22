"""Motivic development techniques for improvisation and composition."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Motif:
    pitches: list[int]
    durations: list[float]
    velocities: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.velocities:
            self.velocities = [80] * len(self.pitches)


def transpose_motif(motif: Motif, semitones: int) -> Motif:
    return Motif(
        pitches=[p + semitones for p in motif.pitches],
        durations=list(motif.durations),
        velocities=list(motif.velocities),
    )


def invert_motif(motif: Motif, axis: int | None = None) -> Motif:
    if not motif.pitches:
        return Motif([], list(motif.durations), list(motif.velocities))
    ax = axis if axis is not None else motif.pitches[0]
    return Motif(
        pitches=[2 * ax - p for p in motif.pitches],
        durations=list(motif.durations),
        velocities=list(motif.velocities),
    )


def retrograde_motif(motif: Motif) -> Motif:
    return Motif(
        pitches=list(reversed(motif.pitches)),
        durations=list(reversed(motif.durations)),
        velocities=list(reversed(motif.velocities)),
    )


def augment_motif(motif: Motif, factor: float = 2.0) -> Motif:
    return Motif(
        pitches=list(motif.pitches),
        durations=[d * factor for d in motif.durations],
        velocities=list(motif.velocities),
    )


def diminish_motif(motif: Motif, factor: float = 0.5) -> Motif:
    return augment_motif(motif, factor)


def sequence_motif(motif: Motif, intervals: list[int]) -> list[Motif]:
    result = [motif]
    cumulative = 0
    for iv in intervals:
        cumulative += iv
        result.append(transpose_motif(motif, cumulative))
    return result


def motivic_similarity(a: Motif, b: Motif) -> float:
    if not a.pitches or not b.pitches:
        return 0.0
    intervals_a = [a.pitches[i + 1] - a.pitches[i] for i in range(len(a.pitches) - 1)]
    intervals_b = [b.pitches[i + 1] - b.pitches[i] for i in range(len(b.pitches) - 1)]
    if not intervals_a or not intervals_b:
        return 0.0

    contour_a = [1 if iv > 0 else (-1 if iv < 0 else 0) for iv in intervals_a]
    contour_b = [1 if iv > 0 else (-1 if iv < 0 else 0) for iv in intervals_b]

    min_len = min(len(contour_a), len(contour_b))
    matches = sum(1 for i in range(min_len) if contour_a[i] == contour_b[i])
    contour_sim = matches / max(len(contour_a), len(contour_b))

    min_iv = min(len(intervals_a), len(intervals_b))
    iv_diffs = [abs(intervals_a[i] - intervals_b[i]) for i in range(min_iv)]
    max_diff = max(max(abs(iv) for iv in intervals_a + intervals_b), 1)
    interval_sim = 1.0 - sum(iv_diffs) / (min_iv * max_diff)

    len_ratio = min(len(a.pitches), len(b.pitches)) / max(len(a.pitches), len(b.pitches))

    return 0.4 * contour_sim + 0.4 * interval_sim + 0.2 * len_ratio
