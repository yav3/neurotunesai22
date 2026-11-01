"""Neo-Riemannian and Lewinian transformational theory.

PLR operations on triads, Tonnetz navigation, hexatonic/octatonic systems,
GIS framework. Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class Quality(Enum):
    MAJOR = "major"
    MINOR = "minor"


@dataclass(frozen=True)
class Triad:
    root: int   # pitch class 0..11
    quality: Quality

    @property
    def pcs(self) -> frozenset[int]:
        if self.quality == Quality.MAJOR:
            return frozenset({self.root, (self.root + 4) % 12, (self.root + 7) % 12})
        return frozenset({self.root, (self.root + 3) % 12, (self.root + 7) % 12})

    def __repr__(self) -> str:
        names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        q = "" if self.quality == Quality.MAJOR else "m"
        return f"{names[self.root]}{q}"


# ---------------------------------------------------------------------------
# PLR operations
# ---------------------------------------------------------------------------

def parallel(t: Triad) -> Triad:
    """P: change quality, keep root (C major ↔ C minor)."""
    new_q = Quality.MINOR if t.quality == Quality.MAJOR else Quality.MAJOR
    return Triad(t.root, new_q)


def relative(t: Triad) -> Triad:
    """R: relative major/minor (C major ↔ A minor)."""
    if t.quality == Quality.MAJOR:
        return Triad((t.root + 9) % 12, Quality.MINOR)
    return Triad((t.root + 3) % 12, Quality.MAJOR)


def leading_tone(t: Triad) -> Triad:
    """L: leading-tone exchange (C major ↔ E minor)."""
    if t.quality == Quality.MAJOR:
        return Triad((t.root + 4) % 12, Quality.MINOR)
    return Triad((t.root + 8) % 12, Quality.MAJOR)


def nebenverwandt(t: Triad) -> Triad:
    """N (= RP): nebenverwandt."""
    return parallel(relative(t))


def slide(t: Triad) -> Triad:
    """S (= LPR): slide relation."""
    return relative(parallel(leading_tone(t)))


def hexpole(t: Triad) -> Triad:
    """H (= LP): hexatonic pole — maximally distant within a hexatonic system."""
    return parallel(leading_tone(t))


OPERATIONS = {
    "P": parallel,
    "R": relative,
    "L": leading_tone,
    "N": nebenverwandt,
    "S": slide,
    "H": hexpole,
}


def apply_path(t: Triad, ops: Sequence[str]) -> list[Triad]:
    """Apply a sequence of operation names and return the trajectory."""
    path = [t]
    current = t
    for op_name in ops:
        fn = OPERATIONS[op_name.upper()]
        current = fn(current)
        path.append(current)
    return path


# ---------------------------------------------------------------------------
# Tonnetz
# ---------------------------------------------------------------------------

def tonnetz_coords(t: Triad) -> tuple[float, float]:
    """Map a triad to (x, y) in the Tonnetz (fifths × major-thirds axes)."""
    fifth_steps = (t.root * 7) % 12
    third_steps = (t.root * 4) % 12
    x = float(fifth_steps)
    y = float(third_steps)
    if t.quality == Quality.MINOR:
        y -= 0.5
    return (x, y)


def tonnetz_distance(a: Triad, b: Triad) -> float:
    """Euclidean distance in Tonnetz space."""
    ax, ay = tonnetz_coords(a)
    bx, by = tonnetz_coords(b)
    dx = min(abs(ax - bx), 12 - abs(ax - bx))
    dy = min(abs(ay - by), 12 - abs(ay - by))
    return float(np.sqrt(dx**2 + dy**2))


def plr_distance(a: Triad, b: Triad) -> int:
    """Minimum number of P, L, R moves to reach *b* from *a* (BFS)."""
    if a == b:
        return 0
    from collections import deque
    visited: set[tuple[int, str]] = set()
    queue: deque[tuple[Triad, int]] = deque([(a, 0)])
    visited.add((a.root, a.quality.value))

    while queue:
        current, depth = queue.popleft()
        if depth > 12:
            break
        for fn in (parallel, leading_tone, relative):
            nxt = fn(current)
            key = (nxt.root, nxt.quality.value)
            if key not in visited:
                if nxt == b:
                    return depth + 1
                visited.add(key)
                queue.append((nxt, depth + 1))
    return -1


# ---------------------------------------------------------------------------
# Hexatonic and octatonic systems
# ---------------------------------------------------------------------------

def hexatonic_system(t: Triad) -> list[Triad]:
    """Return the 6 triads in the hexatonic system containing *t* (cycle of LP)."""
    cycle: list[Triad] = [t]
    current = t
    for _ in range(5):
        current = parallel(leading_tone(current)) if len(cycle) % 2 == 1 else leading_tone(parallel(current))
        if current == t:
            break
        cycle.append(current)
    return cycle


def octatonic_pool(t: Triad) -> list[Triad]:
    """Return triads in the octatonic pool containing *t* (cycle of PR)."""
    cycle: list[Triad] = [t]
    current = t
    for _ in range(7):
        current = relative(parallel(current)) if len(cycle) % 2 == 1 else parallel(relative(current))
        if current == t:
            break
        cycle.append(current)
    return cycle


def weitzmann_region(aug_triad_root: int) -> list[Triad]:
    """The 6 triads surrounding an augmented triad in the Tonnetz."""
    aug = frozenset({aug_triad_root, (aug_triad_root + 4) % 12, (aug_triad_root + 8) % 12})
    region: list[Triad] = []
    for pc in aug:
        region.append(Triad(pc, Quality.MAJOR))
        region.append(Triad(pc, Quality.MINOR))
    return region


# ---------------------------------------------------------------------------
# Generalized Interval System (GIS)
# ---------------------------------------------------------------------------

@dataclass
class GIS:
    """A GIS: (S, IVLS, int) where S is a set of elements, IVLS a group, int the interval function."""
    elements: list[int]
    modulus: int = 12

    def interval(self, a: int, b: int) -> int:
        """int(a, b) = b - a mod modulus."""
        return (b - a) % self.modulus

    def transpose_elem(self, s: int, n: int) -> int:
        return (s + n) % self.modulus

    def invert_elem(self, s: int, index: int = 0) -> int:
        return (2 * index - s) % self.modulus

    def interval_sequence(self, seq: Sequence[int]) -> list[int]:
        return [self.interval(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
