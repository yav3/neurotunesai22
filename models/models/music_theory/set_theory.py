"""Pitch-class set theory — Forte numbers, interval vectors, transformations, set-class catalog."""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


def to_pitch_class(pitch: int) -> int:
    return pitch % 12


def pitch_class_set(pitches: list[int]) -> FrozenSet[int]:
    return frozenset(to_pitch_class(p) for p in pitches)


def normal_form(pcs: FrozenSet[int]) -> tuple[int, ...]:
    if not pcs:
        return ()
    members = sorted(pcs)
    n = len(members)
    if n == 1:
        return (members[0],)
    best: tuple[int, ...] | None = None
    for rotation in range(n):
        candidate = tuple(members[(rotation + i) % n] for i in range(n))
        adjusted = tuple((c - candidate[0]) % 12 for c in candidate)
        if best is None or adjusted < best:
            best = adjusted
    return best


def prime_form(pcs: FrozenSet[int]) -> tuple[int, ...]:
    nf = normal_form(pcs)
    inv = frozenset((12 - pc) % 12 for pc in pcs)
    nf_inv = normal_form(inv)
    return min(nf, nf_inv)


def interval_vector(pcs: FrozenSet[int]) -> tuple[int, int, int, int, int, int]:
    members = sorted(pcs)
    vec = [0] * 6
    n = len(members)
    for i in range(n):
        for j in range(i + 1, n):
            diff = (members[j] - members[i]) % 12
            ic = min(diff, 12 - diff)
            if 1 <= ic <= 6:
                vec[ic - 1] += 1
    return (vec[0], vec[1], vec[2], vec[3], vec[4], vec[5])


def transpose(pcs: FrozenSet[int], n: int) -> FrozenSet[int]:
    return frozenset((pc + n) % 12 for pc in pcs)


def invert(pcs: FrozenSet[int]) -> FrozenSet[int]:
    return frozenset((12 - pc) % 12 for pc in pcs)


def complement(pcs: FrozenSet[int]) -> FrozenSet[int]:
    return frozenset(range(12)) - pcs


def is_subset(a: FrozenSet[int], b: FrozenSet[int]) -> bool:
    return a.issubset(b)


def z_related(a: FrozenSet[int], b: FrozenSet[int]) -> bool:
    return interval_vector(a) == interval_vector(b) and prime_form(a) != prime_form(b)


@dataclass(frozen=True)
class SetClass:
    prime: tuple[int, ...]
    forte_number: str
    iv: tuple[int, int, int, int, int, int]
    cardinality: int


FORTE_CATALOG_3: dict[str, tuple[int, ...]] = {
    "3-1": (0, 1, 2),
    "3-2": (0, 1, 3),
    "3-3": (0, 1, 4),
    "3-4": (0, 1, 5),
    "3-5": (0, 1, 6),
    "3-6": (0, 2, 4),
    "3-7": (0, 2, 5),
    "3-8": (0, 2, 6),
    "3-9": (0, 2, 7),
    "3-10": (0, 3, 6),
    "3-11": (0, 3, 7),
    "3-12": (0, 4, 8),
}

FORTE_CATALOG_4: dict[str, tuple[int, ...]] = {
    "4-1": (0, 1, 2, 3),
    "4-2": (0, 1, 2, 4),
    "4-3": (0, 1, 3, 4),
    "4-4": (0, 1, 2, 5),
    "4-5": (0, 1, 2, 6),
    "4-6": (0, 1, 2, 7),
    "4-7": (0, 1, 4, 5),
    "4-8": (0, 1, 5, 6),
    "4-9": (0, 1, 6, 7),
    "4-10": (0, 2, 3, 5),
    "4-11": (0, 1, 3, 5),
    "4-12": (0, 2, 3, 6),
    "4-13": (0, 1, 3, 6),
    "4-14": (0, 2, 3, 7),
    "4-16": (0, 1, 5, 7),
    "4-17": (0, 3, 4, 7),
    "4-18": (0, 1, 4, 7),
    "4-19": (0, 1, 4, 8),
    "4-20": (0, 1, 5, 8),
    "4-21": (0, 2, 4, 6),
    "4-22": (0, 2, 4, 7),
    "4-23": (0, 2, 5, 7),
    "4-24": (0, 2, 4, 8),
    "4-25": (0, 2, 6, 8),
    "4-26": (0, 3, 5, 8),
    "4-27": (0, 2, 5, 8),
    "4-28": (0, 3, 6, 9),
    "4-29": (0, 1, 3, 7),
}


def lookup_forte(pcs: FrozenSet[int]) -> str | None:
    pf = prime_form(pcs)
    n = len(pf)
    catalog = FORTE_CATALOG_3 if n == 3 else FORTE_CATALOG_4 if n == 4 else {}
    for forte_num, entry in catalog.items():
        if entry == pf:
            return forte_num
    return None


def similarity_index(a: FrozenSet[int], b: FrozenSet[int]) -> float:
    iv_a = interval_vector(a)
    iv_b = interval_vector(b)
    shared = sum(min(x, y) for x, y in zip(iv_a, iv_b))
    total = sum(iv_a) + sum(iv_b)
    return 2 * shared / total if total > 0 else 1.0
