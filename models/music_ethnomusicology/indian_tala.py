"""Indian tala system — Hindustani and Carnatic rhythmic frameworks."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TalaFamily(Enum):
    HINDUSTANI = "hindustani"
    CARNATIC = "carnatic"


@dataclass
class TalaBol:
    """A single rhythmic syllable (bol)."""
    syllable: str
    weight: float = 1.0  # sam=3.0, taali=2.0, khali=0.5, regular=1.0
    is_sam: bool = False
    is_taali: bool = False
    is_khali: bool = False


@dataclass
class Tala:
    name: str
    family: TalaFamily
    beats: int
    vibhags: list[int]  # beats per vibhag
    taali_positions: list[int]  # 1-indexed vibhag positions with clap
    khali_positions: list[int]  # 1-indexed vibhag positions with wave
    bol_pattern: list[str] = field(default_factory=list)


HINDUSTANI_TALAS: dict[str, Tala] = {
    "teental": Tala("Teental", TalaFamily.HINDUSTANI, 16,
                    [4, 4, 4, 4], [1, 2, 4], [3],
                    ["dha", "dhin", "dhin", "dha",
                     "dha", "dhin", "dhin", "dha",
                     "dha", "tin", "tin", "ta",
                     "ta", "dhin", "dhin", "dha"]),
    "jhaptaal": Tala("Jhaptaal", TalaFamily.HINDUSTANI, 10,
                     [2, 3, 2, 3], [1, 2, 4], [3],
                     ["dhi", "na", "dhi", "dhi", "na",
                      "ti", "na", "dhi", "dhi", "na"]),
    "ektaal": Tala("Ektaal", TalaFamily.HINDUSTANI, 12,
                   [2, 2, 2, 2, 2, 2], [1, 2, 5, 6], [3, 4]),
    "rupaktaal": Tala("Rupak Taal", TalaFamily.HINDUSTANI, 7,
                      [3, 2, 2], [2, 3], [1]),
    "dadra": Tala("Dadra", TalaFamily.HINDUSTANI, 6,
                  [3, 3], [1], [2],
                  ["dha", "dhi", "na", "dha", "tu", "na"]),
}


@dataclass
class CarnaticTala:
    name: str
    anga_sequence: list[str]  # laghu, drutam, anudrutam
    laghu_jati: int = 4  # chatusra=4, tisra=3, misra=7, khanda=5, sankirna=9
    aksharas: int = 0

    def compute_aksharas(self) -> int:
        total = 0
        for anga in self.anga_sequence:
            if anga == "laghu":
                total += self.laghu_jati
            elif anga == "drutam":
                total += 2
            elif anga == "anudrutam":
                total += 1
        return total


CARNATIC_SAPTA_TALAS: dict[str, list[str]] = {
    "dhruva": ["laghu", "drutam", "laghu", "laghu"],
    "matya": ["laghu", "drutam", "laghu"],
    "rupaka": ["drutam", "laghu"],
    "jhampa": ["laghu", "anudrutam", "drutam"],
    "triputa": ["laghu", "drutam", "drutam"],
    "ata": ["laghu", "laghu", "drutam", "drutam"],
    "eka": ["laghu"],
}


def build_carnatic_tala(tala_name: str, jati: int = 4) -> CarnaticTala:
    if tala_name not in CARNATIC_SAPTA_TALAS:
        raise ValueError(f"Unknown tala: {tala_name}")
    angas = CARNATIC_SAPTA_TALAS[tala_name]
    ct = CarnaticTala(name=tala_name, anga_sequence=angas, laghu_jati=jati)
    ct.aksharas = ct.compute_aksharas()
    return ct


def tala_complexity(tala: Tala) -> float:
    """Rhythmic complexity: normalized by beat asymmetry and vibhag count."""
    n = len(tala.vibhags)
    if n <= 1:
        return 0.0
    mean_v = sum(tala.vibhags) / n
    variance = sum((v - mean_v) ** 2 for v in tala.vibhags) / n
    asymmetry = (variance ** 0.5) / (mean_v + 1e-10)
    return min(1.0, asymmetry * n / tala.beats)


def layakari_subdivision(base_pattern: list[str], speed_multiple: int) -> list[str]:
    """Generate layakari (speed variation) by subdividing each bol."""
    if speed_multiple < 1:
        return base_pattern
    return [bol for bol in base_pattern for _ in range(speed_multiple)]


def tihai_pattern(phrase: list[str], rest_beats: int = 0) -> list[str]:
    """Construct a tihai: phrase repeated 3 times with optional rests between."""
    rest = ["—"] * rest_beats
    return phrase + rest + phrase + rest + phrase
