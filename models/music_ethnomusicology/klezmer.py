"""Klezmer music theory: modes/shteygers, ornaments, dance rhythms."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np


class KlezmerMode(Enum):
    FREYGISH = "freygish"
    MI_SHEBERAKH = "mi_sheberakh"
    AHAVA_RABA = "ahava_raba"
    ADONOY_MOLOKH = "adonoy_molokh"
    MOGEN_OVOS = "mogen_ovos"


KLEZMER_INTERVALS: dict[KlezmerMode, tuple[int, ...]] = {
    KlezmerMode.FREYGISH:       (0, 1, 4, 5, 7, 8, 10),
    KlezmerMode.MI_SHEBERAKH:   (0, 2, 3, 6, 7, 9, 10),
    KlezmerMode.AHAVA_RABA:     (0, 1, 4, 5, 7, 8, 10),
    KlezmerMode.ADONOY_MOLOKH:  (0, 2, 4, 5, 7, 9, 11),
    KlezmerMode.MOGEN_OVOS:     (0, 2, 3, 5, 7, 8, 11),
}


class KlezmerOrnament(Enum):
    KREKHTS = "krekhts"
    TSHOK = "tshok"
    DREYDLEKH = "dreydlekh"
    KNEYTSHN = "kneytshn"
    KNEYTSH = "kneytsh"


class DanceType(Enum):
    BULGAR = "bulgar"
    HORA = "hora"
    ZHOK = "zhok"
    FREYLEKHS = "freylekhs"
    KHOSIDL = "khosidl"
    TERKISH = "terkish"
    SHER = "sher"


DANCE_METERS: dict[DanceType, tuple[int, int]] = {
    DanceType.BULGAR: (2, 4),
    DanceType.HORA: (3, 8),
    DanceType.ZHOK: (3, 8),
    DanceType.FREYLEKHS: (2, 4),
    DanceType.KHOSIDL: (2, 4),
    DanceType.TERKISH: (4, 4),
    DanceType.SHER: (2, 4),
}

DANCE_TEMPO_RANGE: dict[DanceType, tuple[int, int]] = {
    DanceType.BULGAR: (100, 160),
    DanceType.HORA: (60, 100),
    DanceType.ZHOK: (80, 120),
    DanceType.FREYLEKHS: (120, 180),
    DanceType.KHOSIDL: (50, 80),
    DanceType.TERKISH: (80, 120),
    DanceType.SHER: (100, 140),
}


def classify_klezmer_mode(pitches: list[int], tonic: int = 0) -> KlezmerMode:
    pcs = set((p - tonic) % 12 for p in pitches)
    best = KlezmerMode.FREYGISH
    best_score = -1
    for mode, intervals in KLEZMER_INTERVALS.items():
        score = len(pcs & set(intervals))
        if score > best_score:
            best_score = score
            best = mode
    return best


def has_augmented_second(pitches: list[int]) -> bool:
    for i in range(len(pitches) - 1):
        if abs(pitches[i + 1] - pitches[i]) == 3:
            return True
    return False


@dataclass(frozen=True)
class KlezmerPhrase:
    notes: tuple[int, ...]
    mode: KlezmerMode
    ornaments: tuple[KlezmerOrnament, ...] = ()


def generate_krekhts(note: int, depth_cents: float = 50.0) -> list[float]:
    t = np.linspace(0, 1, 20)
    contour = note - (depth_cents / 100.0) * np.sin(np.pi * t)
    return contour.tolist()


def classify_dance(time_sig: tuple[int, int], tempo_bpm: float) -> DanceType:
    num, denom = time_sig
    if num == 3 and denom == 8:
        return DanceType.HORA if tempo_bpm < 100 else DanceType.ZHOK
    if tempo_bpm < 80:
        return DanceType.KHOSIDL
    if tempo_bpm > 140:
        return DanceType.FREYLEKHS
    return DanceType.BULGAR


def freygish_scale(tonic: int = 0) -> list[int]:
    return [(tonic + iv) % 12 for iv in KLEZMER_INTERVALS[KlezmerMode.FREYGISH]]


def mode_tension(pitches: list[int], mode: KlezmerMode, tonic: int = 0) -> float:
    intervals = set(KLEZMER_INTERVALS[mode])
    out_of_mode = sum(1 for p in pitches if (p - tonic) % 12 not in intervals)
    return out_of_mode / max(len(pitches), 1)
