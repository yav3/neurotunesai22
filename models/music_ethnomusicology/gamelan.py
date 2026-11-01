"""Indonesian gamelan: slendro/pelog tuning, pathet, kotekan interlocking."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

SLENDRO_INTERVALS = (240, 240, 240, 240, 240)
PELOG_INTERVALS = (120, 270, 130, 280, 120, 270, 130)


@dataclass(frozen=True)
class GamelanScale:
    name: str
    tuning_system: str
    cent_values: tuple[float, ...]

    @property
    def n_tones(self) -> int:
        return len(self.cent_values)

    def frequency_ratios(self, base_freq: float = 220.0) -> list[float]:
        return [base_freq * 2 ** (c / 1200.0) for c in self.cent_values]


GAMELAN_SCALES: dict[str, GamelanScale] = {
    "slendro": GamelanScale("slendro", "slendro", (0, 240, 480, 720, 960)),
    "pelog_bem": GamelanScale("pelog_bem", "pelog", (0, 120, 390, 520, 800)),
    "pelog_barang": GamelanScale("pelog_barang", "pelog", (0, 270, 400, 680, 800)),
    "pelog_full": GamelanScale("pelog_full", "pelog", (0, 120, 390, 520, 680, 800, 1070)),
}

PATHET_MAP: dict[str, dict[str, int]] = {
    "pathet_nem":   {"dasar": 0, "gulu": 1, "dada": 2, "lima": 3, "nem": 4},
    "pathet_sanga": {"dasar": 0, "gulu": 1, "dada": 2, "lima": 3, "nem": 4},
    "pathet_manyura": {"dasar": 0, "gulu": 1, "dada": 2, "lima": 3, "nem": 4},
}


def interlocking_pattern(melody: list[int], n_tones: int = 5) -> tuple[list[int], list[int]]:
    polos = []
    sangsih = []
    for i, note in enumerate(melody):
        if i % 2 == 0:
            polos.append(note)
            sangsih.append((note + 1) % n_tones)
        else:
            polos.append((note - 1) % n_tones)
            sangsih.append(note)
    return polos, sangsih


def colotomic_structure(n_beats: int, form: str = "lancaran") -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {"gong": [], "kenong": [], "kempul": [], "ketuk": []}
    if form == "lancaran":
        cycle = min(n_beats, 16)
        for i in range(0, n_beats, cycle):
            positions["gong"].append(min(i + cycle - 1, n_beats - 1))
            if cycle >= 4:
                positions["kenong"].append(i + cycle // 2 - 1)
            if cycle >= 8:
                positions["kempul"].append(i + cycle // 4 - 1)
                positions["kempul"].append(i + 3 * cycle // 4 - 1)
            for j in range(0, cycle, max(cycle // 4, 1)):
                positions["ketuk"].append(i + j)
    elif form == "ketawang":
        cycle = min(n_beats, 32)
        for i in range(0, n_beats, cycle):
            positions["gong"].append(min(i + cycle - 1, n_beats - 1))
            positions["kenong"].append(i + cycle // 2 - 1)
            for j in range(0, cycle, cycle // 4):
                positions["ketuk"].append(i + j)
    return positions
