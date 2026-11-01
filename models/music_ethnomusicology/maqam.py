"""Arabic/Turkish maqam system: quarter-tone scales, jins, modulation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Maqam:
    name: str
    family: str
    jins_lower: tuple[int, ...]
    jins_upper: tuple[int, ...]

    @property
    def intervals(self) -> tuple[int, ...]:
        return self.jins_lower + self.jins_upper

    @property
    def total_cents(self) -> int:
        return sum(self.intervals)


MAQAM_CATALOG: dict[str, Maqam] = {
    "Rast":        Maqam("Rast", "Rast", (4, 3, 3), (4, 4, 2, 2)),
    "Bayati":      Maqam("Bayati", "Bayati", (3, 3, 4), (4, 2, 4, 4)),
    "Hijaz":       Maqam("Hijaz", "Hijaz", (2, 6, 2), (4, 2, 4, 4)),
    "Saba":        Maqam("Saba", "Saba", (3, 3, 2), (4, 2, 4, 4)),
    "Nahawand":    Maqam("Nahawand", "Nahawand", (4, 2, 4), (2, 4, 4, 2)),
    "Ajam":        Maqam("Ajam", "Ajam", (4, 4, 2), (4, 4, 2, 2)),
    "Kurd":        Maqam("Kurd", "Kurd", (2, 4, 4), (2, 4, 4, 2)),
    "Sikah":       Maqam("Sikah", "Sikah", (3, 4, 3), (4, 3, 4, 3)),
    "Huzam":       Maqam("Huzam", "Sikah", (3, 4, 2), (5, 2, 4, 4)),
    "Jiharkah":    Maqam("Jiharkah", "Rast", (4, 4, 2), (4, 3, 3, 4)),
    "Suznak":      Maqam("Suznak", "Rast", (4, 3, 3), (2, 6, 2, 2)),
    "Nawa_Athar":  Maqam("Nawa Athar", "Nawa Athar", (4, 2, 6), (2, 4, 2, 4)),
    "Athar_Kurd":  Maqam("Athar Kurd", "Kurd", (2, 4, 6), (2, 4, 2, 4)),
    "Nikriz":      Maqam("Nikriz", "Nikriz", (4, 2, 6), (2, 4, 4, 2)),
    "Farahfaza":   Maqam("Farahfaza", "Bayati", (3, 3, 4), (2, 4, 4, 2)),
}


def maqam_modulation_distance(m1: Maqam, m2: Maqam) -> float:
    i1 = np.array(m1.intervals, dtype=float)
    i2 = np.array(m2.intervals, dtype=float)
    max_len = max(len(i1), len(i2))
    p1 = np.zeros(max_len)
    p2 = np.zeros(max_len)
    p1[:len(i1)] = i1
    p2[:len(i2)] = i2
    return float(np.sqrt(np.sum((p1 - p2) ** 2)) / max_len)


def classify_maqam(intervals: list[int]) -> list[tuple[Maqam, float]]:
    query = np.array(intervals, dtype=float)
    scored = []
    for maqam in MAQAM_CATALOG.values():
        ref = np.array(maqam.intervals, dtype=float)
        max_len = max(len(query), len(ref))
        q = np.zeros(max_len)
        r = np.zeros(max_len)
        q[:len(query)] = query
        r[:len(ref)] = ref
        dist = float(np.sqrt(np.sum((q - r) ** 2)))
        max_possible = max_len * 6.0
        sim = max(0.0, 1.0 - dist / max_possible)
        if sim > 0.3:
            scored.append((maqam, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
