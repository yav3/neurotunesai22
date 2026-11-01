"""World music scales: maqam, raga, gamelan, and cross-cultural scale analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CulturalScale:
    name: str
    intervals: list[int]
    tradition: str = ""
    tonic_name: str = ""
    mood: str = ""

    @property
    def degree_count(self) -> int:
        return len(self.intervals) + 1

    @property
    def octave_span(self) -> int:
        return sum(self.intervals)

    def pitches(self, root: int = 60) -> list[int]:
        result = [root]
        for iv in self.intervals:
            result.append(result[-1] + iv)
        return result

    def ascending(self, root: int = 60) -> list[int]:
        return self.pitches(root)

    def descending(self, root: int = 60) -> list[int]:
        return list(reversed(self.pitches(root)))

    def contains_pitch(self, pitch: int, root: int = 60) -> bool:
        pcs = set((p - root) % 12 for p in self.pitches(root))
        return (pitch - root) % 12 in pcs


MAQAM_SCALES: dict[str, CulturalScale] = {
    "rast": CulturalScale("Rast", [2, 2, 1, 2, 2, 2, 1], "Arabic", mood="joyful"),
    "bayati": CulturalScale("Bayati", [1, 2, 2, 2, 1, 2, 2], "Arabic", mood="melancholy"),
    "hijaz": CulturalScale("Hijaz", [1, 3, 1, 2, 1, 2, 2], "Arabic", mood="exotic"),
    "saba": CulturalScale("Saba", [1, 2, 1, 3, 1, 2, 2], "Arabic", mood="sorrow"),
    "nahawand": CulturalScale("Nahawand", [2, 1, 2, 2, 1, 2, 2], "Arabic", mood="romantic"),
    "kurd": CulturalScale("Kurd", [1, 2, 2, 2, 1, 2, 2], "Arabic/Turkish", mood="contemplative"),
    "ajam": CulturalScale("Ajam", [2, 2, 1, 2, 2, 2, 1], "Arabic", mood="bright"),
    "sikah": CulturalScale("Sikah", [2, 1, 3, 1, 2, 1, 2], "Arabic", mood="spiritual"),
}

RAGA_SCALES: dict[str, CulturalScale] = {
    "bilawal": CulturalScale("Bilawal", [2, 2, 1, 2, 2, 2, 1], "Hindustani", mood="serene"),
    "yaman": CulturalScale("Yaman", [2, 2, 2, 1, 2, 2, 1], "Hindustani", mood="devotional"),
    "bhairav": CulturalScale("Bhairav", [1, 3, 1, 2, 1, 3, 1], "Hindustani", mood="solemn"),
    "todi": CulturalScale("Todi", [1, 2, 2, 2, 1, 3, 1], "Hindustani", mood="pathos"),
    "malkauns": CulturalScale("Malkauns", [3, 2, 2, 3, 2], "Hindustani", mood="midnight"),
    "bhairavi": CulturalScale("Bhairavi", [1, 2, 2, 2, 1, 2, 2], "Hindustani", mood="morning"),
    "kafi": CulturalScale("Kafi", [2, 1, 2, 2, 2, 1, 2], "Hindustani", mood="rainy"),
    "purvi": CulturalScale("Purvi", [1, 3, 1, 2, 1, 3, 1], "Hindustani", mood="sunset"),
    "marwa": CulturalScale("Marwa", [1, 3, 2, 1, 2, 2, 1], "Hindustani", mood="twilight"),
    "shankarabharanam": CulturalScale("Shankarabharanam", [2, 2, 1, 2, 2, 2, 1], "Carnatic", mood="auspicious"),
    "kalyani": CulturalScale("Kalyani", [2, 2, 2, 1, 2, 2, 1], "Carnatic", mood="happy"),
    "kharaharapriya": CulturalScale("Kharaharapriya", [2, 1, 2, 2, 2, 1, 2], "Carnatic", mood="devotional"),
    "mayamalavagowla": CulturalScale("Mayamalavagowla", [1, 3, 1, 2, 1, 3, 1], "Carnatic", mood="grandeur"),
}

GAMELAN_SCALES: dict[str, CulturalScale] = {
    "slendro": CulturalScale("Slendro", [2, 3, 2, 2, 3], "Javanese", mood="calm"),
    "pelog_lima": CulturalScale("Pelog Lima", [1, 2, 4, 1, 4], "Javanese", mood="solemn"),
    "pelog_barang": CulturalScale("Pelog Barang", [2, 1, 4, 1, 4], "Javanese", mood="bright"),
    "pelog_nem": CulturalScale("Pelog Nem", [1, 2, 4, 3, 2], "Javanese", mood="stately"),
}


class ScaleDatabase:
    def __init__(self) -> None:
        self._scales: dict[str, CulturalScale] = {}
        for name, sc in MAQAM_SCALES.items():
            self._scales[f"maqam:{name}"] = sc
        for name, sc in RAGA_SCALES.items():
            self._scales[f"raga:{name}"] = sc
        for name, sc in GAMELAN_SCALES.items():
            self._scales[f"gamelan:{name}"] = sc

    def add(self, key: str, scale: CulturalScale) -> None:
        self._scales[key] = scale

    def get(self, key: str) -> CulturalScale | None:
        return self._scales.get(key)

    def search_by_tradition(self, tradition: str) -> list[CulturalScale]:
        t = tradition.lower()
        return [s for s in self._scales.values() if t in s.tradition.lower()]

    def search_by_mood(self, mood: str) -> list[CulturalScale]:
        m = mood.lower()
        return [s for s in self._scales.values() if m in s.mood.lower()]

    def all_scales(self) -> list[CulturalScale]:
        return list(self._scales.values())


def interval_pattern(pitches: list[int]) -> list[int]:
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def scale_similarity(scale_a: CulturalScale, scale_b: CulturalScale) -> float:
    iv_a = np.array(scale_a.intervals, dtype=float)
    iv_b = np.array(scale_b.intervals, dtype=float)
    if len(iv_a) != len(iv_b):
        max_len = max(len(iv_a), len(iv_b))
        a_pad = np.zeros(max_len)
        b_pad = np.zeros(max_len)
        a_pad[:len(iv_a)] = iv_a
        b_pad[:len(iv_b)] = iv_b
        iv_a, iv_b = a_pad, b_pad
    na = np.linalg.norm(iv_a)
    nb = np.linalg.norm(iv_b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(iv_a, iv_b) / (na * nb))
