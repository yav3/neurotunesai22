"""Neo-Riemannian transformations: P, L, R, tonnetz geometry, hexatonic systems."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Triad:
    root: int
    quality: str

    @property
    def pitch_classes(self) -> tuple[int, int, int]:
        r = self.root % 12
        if self.quality == "major":
            return (r, (r + 4) % 12, (r + 7) % 12)
        return (r, (r + 3) % 12, (r + 7) % 12)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triad):
            return NotImplemented
        return self.root % 12 == other.root % 12 and self.quality == other.quality

    def __hash__(self) -> int:
        return hash((self.root % 12, self.quality))


def P(triad: Triad) -> Triad:
    return Triad(triad.root, "minor" if triad.quality == "major" else "major")


def L(triad: Triad) -> Triad:
    if triad.quality == "major":
        return Triad((triad.root + 4) % 12, "minor")
    return Triad((triad.root - 4) % 12, "major")


def R(triad: Triad) -> Triad:
    if triad.quality == "major":
        return Triad((triad.root + 9) % 12, "minor")
    return Triad((triad.root + 3) % 12, "major")


TRANSFORMS = {"P": P, "L": L, "R": R}


def apply_chain(triad: Triad, chain: str) -> list[Triad]:
    path = [triad]
    current = triad
    for ch in chain:
        fn = TRANSFORMS.get(ch.upper())
        if fn is None:
            raise ValueError(f"Unknown transform: {ch}")
        current = fn(current)
        path.append(current)
    return path


def tonnetz_coordinates(triad: Triad) -> tuple[float, float]:
    r = triad.root % 12
    x = (r * 7) % 12
    y = (r * 4) % 12
    if triad.quality == "minor":
        y = (y - 1) % 12
    return (float(x), float(y))


def tonnetz_distance(a: Triad, b: Triad) -> float:
    ax, ay = tonnetz_coordinates(a)
    bx, by = tonnetz_coordinates(b)
    dx = min(abs(ax - bx), 12 - abs(ax - bx))
    dy = min(abs(ay - by), 12 - abs(ay - by))
    return float(np.sqrt(dx ** 2 + dy ** 2))


def _canonical(t: Triad) -> tuple[int, str]:
    return (t.root % 12, t.quality)


def shortest_path(source: Triad, target: Triad, max_depth: int = 6) -> str | None:
    if _canonical(source) == _canonical(target):
        return ""
    from collections import deque
    queue: deque[tuple[Triad, str]] = deque()
    visited: set[tuple[int, str]] = {_canonical(source)}
    for name, fn in TRANSFORMS.items():
        nxt = fn(source)
        key = _canonical(nxt)
        if key == _canonical(target):
            return name
        if key not in visited:
            visited.add(key)
            queue.append((nxt, name))
    while queue:
        current, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for name, fn in TRANSFORMS.items():
            nxt = fn(current)
            key = _canonical(nxt)
            new_path = path + name
            if key == _canonical(target):
                return new_path
            if key not in visited:
                visited.add(key)
                queue.append((nxt, new_path))
    return None


HEXATONIC_SYSTEMS = {
    "northern": [Triad(0, "major"), Triad(0, "minor"), Triad(8, "major"),
                 Triad(8, "minor"), Triad(4, "major"), Triad(4, "minor")],
    "southern": [Triad(1, "major"), Triad(1, "minor"), Triad(9, "major"),
                 Triad(9, "minor"), Triad(5, "major"), Triad(5, "minor")],
    "eastern":  [Triad(2, "major"), Triad(2, "minor"), Triad(10, "major"),
                 Triad(10, "minor"), Triad(6, "major"), Triad(6, "minor")],
    "western":  [Triad(3, "major"), Triad(3, "minor"), Triad(11, "major"),
                 Triad(11, "minor"), Triad(7, "major"), Triad(7, "minor")],
}


def hexatonic_system_for(triad: Triad) -> str | None:
    key = _canonical(triad)
    for name, members in HEXATONIC_SYSTEMS.items():
        for m in members:
            if _canonical(m) == key:
                return name
    return None


def hexatonic_pole(triad: Triad) -> Triad:
    return apply_chain(triad, "PLP")[-1]


def nebenverwandt(triad: Triad) -> Triad:
    return apply_chain(triad, "RLP")[-1]


def slide(triad: Triad) -> Triad:
    return apply_chain(triad, "PLR")[-1]
