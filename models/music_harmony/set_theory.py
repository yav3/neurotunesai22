"""Pitch-class set theory: Forte numbers, normal/prime form, interval vectors, Z-relations."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np


@dataclass(frozen=True)
class PCSet:
    pcs: tuple[int, ...]

    @property
    def cardinality(self) -> int:
        return len(self.pcs)

    def __repr__(self) -> str:
        return f"PCSet({list(self.pcs)})"


def to_pc_set(pitches: list[int]) -> PCSet:
    return PCSet(tuple(sorted(set(p % 12 for p in pitches))))


def transpose(pcs: PCSet, n: int) -> PCSet:
    return PCSet(tuple(sorted((p + n) % 12 for p in pcs.pcs)))


def invert(pcs: PCSet, axis: int = 0) -> PCSet:
    return PCSet(tuple(sorted((axis - p) % 12 for p in pcs.pcs)))


def complement(pcs: PCSet) -> PCSet:
    return PCSet(tuple(p for p in range(12) if p not in pcs.pcs))


def normal_form(pcs: PCSet) -> PCSet:
    if pcs.cardinality == 0:
        return pcs
    members = sorted(set(pcs.pcs))
    n = len(members)
    if n == 1:
        return PCSet((0,))
    best = None
    for rotation in range(n):
        rotated = [members[(rotation + i) % n] for i in range(n)]
        shifted = [(p - rotated[0]) % 12 for p in rotated]
        t = tuple(shifted)
        if best is None or t < best:
            best = t
    return PCSet(best) if best is not None else pcs


def prime_form(pcs: PCSet) -> PCSet:
    nf = normal_form(pcs)
    inv_nf = normal_form(invert(pcs))
    return PCSet(min(nf.pcs, inv_nf.pcs))


def interval_vector(pcs: PCSet) -> tuple[int, ...]:
    ic_counts = [0] * 6
    members = list(pcs.pcs)
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            diff = (members[j] - members[i]) % 12
            ic = min(diff, 12 - diff)
            if 1 <= ic <= 6:
                ic_counts[ic - 1] += 1
    return tuple(ic_counts)


def is_z_related(a: PCSet, b: PCSet) -> bool:
    if a.cardinality != b.cardinality:
        return False
    if prime_form(a) == prime_form(b):
        return False
    return interval_vector(a) == interval_vector(b)


def is_subset(sub: PCSet, sup: PCSet) -> bool:
    return set(sub.pcs).issubset(set(sup.pcs))


def is_superset(sup: PCSet, sub: PCSet) -> bool:
    return is_subset(sub, sup)


def common_tones_under_transposition(pcs: PCSet, n: int) -> int:
    transposed = set((p + n) % 12 for p in pcs.pcs)
    return len(set(pcs.pcs) & transposed)


def similarity_index(a: PCSet, b: PCSet) -> float:
    if a.cardinality == 0 or b.cardinality == 0:
        return 0.0
    iv_a = np.array(interval_vector(a), dtype=float)
    iv_b = np.array(interval_vector(b), dtype=float)
    na = np.linalg.norm(iv_a)
    nb = np.linalg.norm(iv_b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(iv_a, iv_b) / (na * nb))


FORTE_CATALOG_3: dict[str, tuple[int, ...]] = {
    "3-1":  (0, 1, 2),
    "3-2":  (0, 1, 3),
    "3-3":  (0, 1, 4),
    "3-4":  (0, 1, 5),
    "3-5":  (0, 1, 6),
    "3-6":  (0, 2, 4),
    "3-7":  (0, 2, 5),
    "3-8":  (0, 2, 6),
    "3-9":  (0, 2, 7),
    "3-10": (0, 3, 6),
    "3-11": (0, 3, 7),
    "3-12": (0, 4, 8),
}

FORTE_CATALOG_4: dict[str, tuple[int, ...]] = {
    "4-1":  (0, 1, 2, 3),
    "4-2":  (0, 1, 2, 4),
    "4-3":  (0, 1, 3, 4),
    "4-4":  (0, 1, 2, 5),
    "4-5":  (0, 1, 2, 6),
    "4-6":  (0, 1, 2, 7),
    "4-7":  (0, 1, 4, 5),
    "4-8":  (0, 1, 5, 6),
    "4-9":  (0, 1, 6, 7),
    "4-z15": (0, 1, 4, 6),
    "4-z29": (0, 1, 3, 7),
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
}


def lookup_forte(pcs: PCSet) -> str:
    pf = prime_form(pcs)
    catalog: dict[str, tuple[int, ...]] = {}
    if pf.cardinality == 3:
        catalog = FORTE_CATALOG_3
    elif pf.cardinality == 4:
        catalog = FORTE_CATALOG_4
    for name, form in catalog.items():
        if pf.pcs == form:
            return name
    return f"?-{pf.cardinality}"


def all_subsets(pcs: PCSet, size: int) -> list[PCSet]:
    return [PCSet(tuple(sorted(combo))) for combo in combinations(pcs.pcs, size)]
