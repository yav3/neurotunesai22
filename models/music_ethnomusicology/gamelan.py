"""Indonesian gamelan: slendro/pelog tuning, pathet, colotomy, kotekan, irama."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


SLENDRO_CENTS = (0, 240, 480, 720, 960)
PELOG_CENTS = (0, 120, 270, 400, 540, 670, 800)
SLENDRO_INTERVALS = (240, 240, 240, 240, 240)
PELOG_INTERVALS = (120, 150, 130, 140, 130, 130, 200)


class ScaleType(Enum):
    SLENDRO = "slendro"
    PELOG = "pelog"


class Pathet(Enum):
    NEM = "nem"
    SANGA = "sanga"
    MANYURA = "manyura"
    LIMA = "lima"
    BARANG = "barang"


class IramaLevel(Enum):
    LANCAR = 1
    TANGGUNG = 2
    DADOS = 4
    WILED = 8
    RANGKEP = 16


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


@dataclass(frozen=True)
class ColotomyStructure:
    form: str
    cycle_beats: int
    gong_positions: tuple[int, ...]
    kenong_positions: tuple[int, ...]
    kempul_positions: tuple[int, ...]
    ketuk_positions: tuple[int, ...]


@dataclass(frozen=True)
class KotekanPattern:
    polos: tuple[int, ...]
    sangsih: tuple[int, ...]
    resultant: tuple[int, ...]


GAMELAN_SCALES: dict[str, GamelanScale] = {
    "slendro": GamelanScale("slendro", "slendro", SLENDRO_CENTS),
    "pelog_bem": GamelanScale("pelog_bem", "pelog", (0, 120, 390, 520, 800)),
    "pelog_barang": GamelanScale("pelog_barang", "pelog", (0, 270, 400, 680, 800)),
    "pelog_full": GamelanScale("pelog_full", "pelog", PELOG_CENTS),
}

PATHET_MAP: dict[str, dict[str, int]] = {
    "pathet_nem":     {"dasar": 2, "gulu": 5, "dada": 6, "lima": 1, "nem": 2},
    "pathet_sanga":   {"dasar": 5, "gulu": 6, "dada": 1, "lima": 2, "nem": 3},
    "pathet_manyura": {"dasar": 6, "gulu": 1, "dada": 2, "lima": 3, "nem": 5},
}

COLOTOMY: dict[str, ColotomyStructure] = {
    "lancaran": ColotomyStructure(
        "lancaran", 16,
        gong_positions=(15,),
        kenong_positions=(3, 7, 11, 15),
        kempul_positions=(5, 9, 13),
        ketuk_positions=tuple(range(0, 16, 2)),
    ),
    "ketawang": ColotomyStructure(
        "ketawang", 16,
        gong_positions=(15,),
        kenong_positions=(7, 15),
        kempul_positions=(11,),
        ketuk_positions=(3, 7, 11, 15),
    ),
    "ladrang": ColotomyStructure(
        "ladrang", 32,
        gong_positions=(31,),
        kenong_positions=(7, 15, 23, 31),
        kempul_positions=(11, 19, 27),
        ketuk_positions=tuple(range(3, 32, 4)),
    ),
    "gending": ColotomyStructure(
        "gending", 64,
        gong_positions=(63,),
        kenong_positions=(15, 31, 47, 63),
        kempul_positions=(23, 39, 55),
        ketuk_positions=tuple(range(7, 64, 8)),
    ),
}


def gamelan_interval_cents(note_a: int, note_b: int,
                           scale_type: ScaleType = ScaleType.SLENDRO) -> float:
    cents = SLENDRO_CENTS if scale_type == ScaleType.SLENDRO else PELOG_CENTS
    n = len(cents)
    a_c = cents[note_a % n]
    b_c = cents[note_b % n]
    diff = b_c - a_c
    if diff < 0:
        diff += 1200
    return float(diff)


def pathet_classification(melody: list[int], scale_type: ScaleType = ScaleType.SLENDRO) -> Pathet:
    if not melody:
        return Pathet.NEM
    final = melody[-1] % 5
    first = melody[0] % 5
    if final in (1, 2):
        return Pathet.NEM
    if final == 0 or first == 0:
        return Pathet.SANGA
    return Pathet.MANYURA


def colotomy_structure(n_beats: int, form: str = "lancaran") -> dict[str, list[int]]:
    if form in COLOTOMY:
        template = COLOTOMY[form]
        cycle = template.cycle_beats
        result: dict[str, list[int]] = {"gong": [], "kenong": [], "kempul": [], "ketuk": []}
        for offset in range(0, n_beats, cycle):
            for g in template.gong_positions:
                pos = offset + g
                if pos < n_beats:
                    result["gong"].append(pos)
            for k in template.kenong_positions:
                pos = offset + k
                if pos < n_beats:
                    result["kenong"].append(pos)
            for k in template.kempul_positions:
                pos = offset + k
                if pos < n_beats:
                    result["kempul"].append(pos)
            for k in template.ketuk_positions:
                pos = offset + k
                if pos < n_beats:
                    result["ketuk"].append(pos)
        return result
    return {"gong": [n_beats - 1] if n_beats > 0 else [], "kenong": [], "kempul": [], "ketuk": []}


def kotekan_pattern(melody: list[int], n_tones: int = 5) -> KotekanPattern:
    polos = []
    sangsih = []
    resultant = []
    for i, note in enumerate(melody):
        if i % 2 == 0:
            p = note
            s = (note + 1) % n_tones
        else:
            p = (note - 1) % n_tones
            s = note
        polos.append(p)
        sangsih.append(s)
        resultant.append(note)
    return KotekanPattern(tuple(polos), tuple(sangsih), tuple(resultant))


def interlocking_pattern(melody: list[int], n_tones: int = 5) -> tuple[list[int], list[int]]:
    kp = kotekan_pattern(melody, n_tones)
    return list(kp.polos), list(kp.sangsih)


def balungan_density(melody: list[int]) -> float:
    if len(melody) <= 1:
        return 0.0
    changes = sum(1 for i in range(1, len(melody)) if melody[i] != melody[i - 1])
    return changes / (len(melody) - 1)


def irama_level(ratio: float) -> IramaLevel:
    if ratio <= 1.5:
        return IramaLevel.LANCAR
    if ratio <= 3:
        return IramaLevel.TANGGUNG
    if ratio <= 6:
        return IramaLevel.DADOS
    if ratio <= 12:
        return IramaLevel.WILED
    return IramaLevel.RANGKEP


def garap_elaboration(balungan: list[int], level: IramaLevel,
                      n_tones: int = 5) -> list[int]:
    mult = level.value
    result = []
    for i, note in enumerate(balungan):
        result.append(note)
        next_note = balungan[(i + 1) % len(balungan)]
        for j in range(1, mult):
            alpha = j / mult
            interp = round(note * (1 - alpha) + next_note * alpha) % n_tones
            result.append(interp)
    return result
