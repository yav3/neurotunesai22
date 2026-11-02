"""Twelve-tone / serial composition: row operations, matrices, combinatoriality."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

__all__ = [
    "ToneRow", "RowForm", "RowTransformation",
    "validate_row", "prime", "retrograde", "inversion",
    "retrograde_inversion", "transpose_row", "row_matrix",
    "interval_vector", "all_interval_row", "combinatoriality",
    "derive_row",
]


@dataclass
class ToneRow:
    pitches: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.pitches:
            self.pitches = [p % 12 for p in self.pitches]


class RowForm(Enum):
    PRIME = "P"
    RETROGRADE = "R"
    INVERSION = "I"
    RETROGRADE_INVERSION = "RI"


@dataclass
class RowTransformation:
    form: RowForm = RowForm.PRIME
    transposition: int = 0
    pitches: List[int] = field(default_factory=list)


def validate_row(pitches: List[int]) -> bool:
    if len(pitches) != 12:
        return False
    pcs = [p % 12 for p in pitches]
    return len(set(pcs)) == 12


def prime(row: ToneRow) -> List[int]:
    return list(row.pitches)


def retrograde(row: ToneRow) -> List[int]:
    return list(reversed(row.pitches))


def inversion(row: ToneRow) -> List[int]:
    if not row.pitches:
        return []
    first = row.pitches[0]
    return [(2 * first - p) % 12 for p in row.pitches]


def retrograde_inversion(row: ToneRow) -> List[int]:
    inv = inversion(row)
    return list(reversed(inv))


def transpose_row(pitches: List[int], semitones: int) -> List[int]:
    return [(p + semitones) % 12 for p in pitches]


def row_matrix(row: ToneRow) -> List[List[int]]:
    if not validate_row(row.pitches):
        return []
    inv = inversion(row)
    matrix = []
    for i in range(12):
        transposition = (inv[i] - row.pitches[0]) % 12
        matrix.append(transpose_row(row.pitches, transposition))
    return matrix


def interval_vector(row: ToneRow) -> List[int]:
    pcs = [p % 12 for p in row.pitches]
    iv = [0] * 6
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            diff = abs(pcs[i] - pcs[j])
            ic = min(diff, 12 - diff)
            if 1 <= ic <= 6:
                iv[ic - 1] += 1
    return iv


def all_interval_row() -> ToneRow:
    import random as _rng

    target_intervals = set(range(1, 12))
    for _ in range(10000):
        perm = list(range(12))
        _rng.shuffle(perm)
        intervals = set()
        valid = True
        for i in range(11):
            diff = (perm[i + 1] - perm[i]) % 12
            if diff in intervals or diff == 0:
                valid = False
                break
            intervals.add(diff)
        if valid and intervals == target_intervals:
            return ToneRow(pitches=perm)
    return ToneRow(pitches=list(range(12)))


def combinatoriality(row_a: ToneRow, row_b: ToneRow) -> bool:
    if len(row_a.pitches) != 12 or len(row_b.pitches) != 12:
        return False
    hex_a = set(row_a.pitches[:6])
    hex_b = set(row_b.pitches[:6])
    return len(hex_a & hex_b) == 0


def derive_row(hexachord: List[int]) -> ToneRow:
    pcs = [p % 12 for p in hexachord]
    if len(set(pcs)) != 6:
        return ToneRow(pitches=list(range(12)))
    complement = [p for p in range(12) if p not in set(pcs)]
    return ToneRow(pitches=pcs + complement)
