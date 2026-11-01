"""Rhythm systems: tala, usul, clave, Euclidean rhythms, polyrhythms."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class TimeSystem:
    name: str
    beats: int
    subdivision: int
    accents: list[int] = field(default_factory=list)
    region: str = ""

    def accent_pattern(self) -> list[bool]:
        total = self.beats * self.subdivision
        return [i in self.accents for i in range(total)]


# ---------------------------------------------------------------------------
# Indian Tala
# ---------------------------------------------------------------------------

@dataclass
class Tala:
    name: str
    beats: int
    subdivisions: list[int] = field(default_factory=list)
    bols: list[str] = field(default_factory=list)
    sam: int = 0
    khali: int = 0

    def total_matras(self) -> int:
        return sum(self.subdivisions) if self.subdivisions else self.beats

    def vibhag_boundaries(self) -> list[int]:
        boundaries = [0]
        for s in self.subdivisions:
            boundaries.append(boundaries[-1] + s)
        return boundaries

    def bol_cycle(self, n_cycles: int = 1) -> list[str]:
        if not self.bols:
            return []
        return (self.bols * n_cycles)[:self.total_matras() * n_cycles]


TINTAL = Tala("Tintal", 16, [4, 4, 4, 4],
              ["Dha", "Dhin", "Dhin", "Dha", "Dha", "Dhin", "Dhin", "Dha",
               "Dha", "Tin", "Tin", "Ta", "Ta", "Dhin", "Dhin", "Dha"],
              sam=0, khali=8)
JHAPTAL = Tala("Jhaptal", 10, [2, 3, 2, 3],
               ["Dhi", "Na", "Dhi", "Dhi", "Na", "Ti", "Na", "Dhi", "Dhi", "Na"],
               sam=0, khali=5)
RUPAK = Tala("Rupak", 7, [3, 2, 2],
             ["Tin", "Tin", "Na", "Dhi", "Na", "Dhi", "Na"],
             sam=0, khali=0)
EKTAL = Tala("Ektal", 12, [2, 2, 2, 2, 2, 2],
             ["Dhin", "Dhin", "Dha", "Ge", "Ti", "Ra", "Ka", "Ti",
              "Dha", "Ge", "Ti", "Ra"],
             sam=0, khali=2)
DADRA = Tala("Dadra", 6, [3, 3],
             ["Dha", "Dhi", "Na", "Dha", "Tin", "Na"],
             sam=0, khali=3)
KEHERWA = Tala("Keherwa", 8, [4, 4],
               ["Dha", "Ge", "Na", "Ti", "Na", "Ka", "Dhi", "Na"],
               sam=0, khali=4)

TALA_REGISTRY: list[Tala] = [TINTAL, JHAPTAL, RUPAK, EKTAL, DADRA, KEHERWA]


# ---------------------------------------------------------------------------
# Ottoman / Arabic Usul
# ---------------------------------------------------------------------------

@dataclass
class Usul:
    name: str
    beats: int
    time_signature: str
    pattern: list[str] = field(default_factory=list)

    def onset_mask(self) -> list[bool]:
        return [p.lower() in ("düm", "dum", "tek", "teke", "te") for p in self.pattern]

    def strong_beats(self) -> list[int]:
        return [i for i, p in enumerate(self.pattern) if p.lower() in ("düm", "dum")]


DUYEK = Usul("Düyek", 8, "8/8", ["Düm", ".", "Tek", ".", "Tek", ".", "Düm", "Tek"])
AKSAK = Usul("Aksak", 9, "9/8", ["Düm", ".", "Tek", ".", "Tek", "Düm", ".", "Tek", "."])
CIFTETELLI = Usul("Çiftetelli", 8, "8/8", ["Düm", ".", "Düm", ".", "Tek", ".", "Tek", "."])
KARSILAMA = Usul("Karsilama", 9, "9/8", ["Düm", ".", "Tek", "Düm", ".", "Tek", "Düm", "Tek", "."])

USUL_REGISTRY: list[Usul] = [DUYEK, AKSAK, CIFTETELLI, KARSILAMA]


# ---------------------------------------------------------------------------
# Afro-Cuban Clave
# ---------------------------------------------------------------------------

@dataclass
class ClaveSyncopation:
    name: str
    pattern: list[int]
    n_steps: int = 16

    def onset_times(self) -> list[int]:
        return [i for i, v in enumerate(self.pattern) if v == 1]

    def syncopation_score(self) -> float:
        onsets = self.onset_times()
        if len(onsets) < 2:
            return 0.0
        ioi = np.diff(onsets)
        return float(np.std(ioi) / (np.mean(ioi) + 1e-12))

    def as_binary(self) -> np.ndarray:
        return np.array(self.pattern, dtype=np.int8)


SON_32 = ClaveSyncopation("Son 3-2", [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0])
SON_23 = ClaveSyncopation("Son 2-3", [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
RUMBA = ClaveSyncopation("Rumba", [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
BOSSA_NOVA = ClaveSyncopation("Bossa Nova", [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0])

CLAVE_REGISTRY: list[ClaveSyncopation] = [SON_32, SON_23, RUMBA, BOSSA_NOVA]


# ---------------------------------------------------------------------------
# Euclidean Rhythm (Bjorklund algorithm)
# ---------------------------------------------------------------------------

class EuclideanRhythm:
    @staticmethod
    def generate(k: int, n: int) -> list[int]:
        if k <= 0 or n <= 0:
            return [0] * max(n, 0)
        k = min(k, n)
        # Bjorklund algorithm
        groups: list[list[int]] = [[1] for _ in range(k)] + [[0] for _ in range(n - k)]
        while True:
            remainder = len(groups) - k
            if remainder <= 1:
                break
            new_groups: list[list[int]] = []
            moves = min(k, remainder)
            for i in range(moves):
                new_groups.append(groups[i] + groups[k + i])
            for i in range(moves, k):
                new_groups.append(groups[i])
            for i in range(k + moves, len(groups)):
                new_groups.append(groups[i])
            groups = new_groups
            k = moves + max(0, k - moves)
        result: list[int] = []
        for g in groups:
            result.extend(g)
        return result

    @staticmethod
    def rotation(pattern: list[int], shift: int) -> list[int]:
        if not pattern:
            return []
        n = len(pattern)
        shift = shift % n
        return pattern[shift:] + pattern[:shift]


# ---------------------------------------------------------------------------
# Polyrhythm Generator
# ---------------------------------------------------------------------------

class PolyrhythmGenerator:
    @staticmethod
    def generate(k: int, m: int, resolution: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if resolution <= 0:
            resolution = k * m
        grid_k = np.zeros(resolution, dtype=np.int8)
        grid_m = np.zeros(resolution, dtype=np.int8)
        for i in range(k):
            idx = int(round(i * resolution / k)) % resolution
            grid_k[idx] = 1
        for i in range(m):
            idx = int(round(i * resolution / m)) % resolution
            grid_m[idx] = 1
        return grid_k, grid_m

    @staticmethod
    def coincidence_points(k: int, m: int, resolution: int = 0) -> list[int]:
        gk, gm = PolyrhythmGenerator.generate(k, m, resolution)
        return [i for i in range(len(gk)) if gk[i] == 1 and gm[i] == 1]

    @staticmethod
    def combined(k: int, m: int, resolution: int = 0) -> np.ndarray:
        gk, gm = PolyrhythmGenerator.generate(k, m, resolution)
        return np.clip(gk + gm, 0, 1)
