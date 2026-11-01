"""Melodic development techniques: motivic transformation, sequence, inversion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Motif:
    pitches: tuple[int, ...]
    durations: tuple[float, ...]

    @property
    def intervals(self) -> tuple[int, ...]:
        return tuple(self.pitches[i + 1] - self.pitches[i]
                     for i in range(len(self.pitches) - 1))

    @property
    def contour(self) -> tuple[int, ...]:
        return tuple(
            1 if self.pitches[i + 1] > self.pitches[i]
            else (-1 if self.pitches[i + 1] < self.pitches[i] else 0)
            for i in range(len(self.pitches) - 1)
        )


def transpose(motif: Motif, semitones: int) -> Motif:
    return Motif(
        pitches=tuple(p + semitones for p in motif.pitches),
        durations=motif.durations,
    )


def invert(motif: Motif, axis: int | None = None) -> Motif:
    if not motif.pitches:
        return motif
    ax = axis if axis is not None else motif.pitches[0]
    return Motif(
        pitches=tuple(2 * ax - p for p in motif.pitches),
        durations=motif.durations,
    )


def retrograde(motif: Motif) -> Motif:
    return Motif(
        pitches=tuple(reversed(motif.pitches)),
        durations=tuple(reversed(motif.durations)),
    )


def retrograde_inversion(motif: Motif, axis: int | None = None) -> Motif:
    return retrograde(invert(motif, axis))


def augmentation(motif: Motif, factor: float = 2.0) -> Motif:
    return Motif(
        pitches=motif.pitches,
        durations=tuple(d * factor for d in motif.durations),
    )


def diminution(motif: Motif, factor: float = 2.0) -> Motif:
    return Motif(
        pitches=motif.pitches,
        durations=tuple(d / factor for d in motif.durations),
    )


def sequence(motif: Motif, steps: int, interval: int = 2) -> list[Motif]:
    result: list[Motif] = [motif]
    for i in range(1, steps):
        result.append(transpose(motif, interval * i))
    return result


def tonal_sequence(motif: Motif, scale_degrees: Sequence[int],
                   steps: int, direction: int = 1) -> list[Motif]:
    result: list[Motif] = [motif]
    current_offset = 0
    for i in range(1, steps):
        idx = (i * direction) % len(scale_degrees)
        current_offset += scale_degrees[idx] - scale_degrees[idx - 1] if idx > 0 else scale_degrees[0]
        result.append(transpose(motif, current_offset))
    return result


def fragment(motif: Motif, start: int = 0, end: int | None = None) -> Motif:
    e = end if end is not None else len(motif.pitches)
    return Motif(
        pitches=motif.pitches[start:e],
        durations=motif.durations[start:e],
    )


def extend_by_repetition(motif: Motif, repetitions: int = 2,
                          variation_prob: float = 0.0,
                          rng: np.random.Generator | None = None) -> Motif:
    if rng is None:
        rng = np.random.default_rng()
    pitches: list[int] = list(motif.pitches)
    durations: list[float] = list(motif.durations)
    for _ in range(repetitions - 1):
        for j, p in enumerate(motif.pitches):
            if rng.random() < variation_prob:
                pitches.append(p + rng.choice([-2, -1, 1, 2]))
            else:
                pitches.append(p)
            durations.append(motif.durations[j])
    return Motif(pitches=tuple(pitches), durations=tuple(durations))


def contour_similarity(a: Motif, b: Motif) -> float:
    ca, cb = a.contour, b.contour
    if not ca and not cb:
        return 1.0
    min_len = min(len(ca), len(cb))
    if min_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if ca[i] == cb[i])
    return matches / min_len


def interval_similarity(a: Motif, b: Motif) -> float:
    ia, ib = a.intervals, b.intervals
    if not ia and not ib:
        return 1.0
    min_len = min(len(ia), len(ib))
    if min_len == 0:
        return 0.0
    diffs = [abs(ia[i] - ib[i]) for i in range(min_len)]
    max_diff = max(max(diffs), 1)
    return 1.0 - np.mean(diffs) / max_diff


def develop_theme(motif: Motif, techniques: Sequence[str],
                  rng: np.random.Generator | None = None) -> list[Motif]:
    if rng is None:
        rng = np.random.default_rng()
    result: list[Motif] = [motif]
    current = motif
    for tech in techniques:
        if tech == "transpose":
            current = transpose(current, int(rng.choice([-5, -3, -2, 2, 3, 5, 7])))
        elif tech == "invert":
            current = invert(current)
        elif tech == "retrograde":
            current = retrograde(current)
        elif tech == "augment":
            current = augmentation(current)
        elif tech == "diminish":
            current = diminution(current)
        elif tech == "fragment":
            n = len(current.pitches)
            if n > 2:
                current = fragment(current, 0, max(2, n // 2))
        elif tech == "sequence":
            seq = sequence(current, 3, int(rng.choice([-3, -2, 2, 3])))
            result.extend(seq[1:])
            current = seq[-1]
            continue
        result.append(current)
    return result
