"""Timbral compatibility and instrument combination scoring."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class InstrumentFamily(Enum):
    STRINGS = "strings"
    WOODWINDS = "woodwinds"
    BRASS = "brass"
    PERCUSSION = "percussion"
    KEYBOARDS = "keyboards"
    VOICE = "voice"
    PLUCKED = "plucked"
    ELECTRONIC = "electronic"


@dataclass
class InstrumentProfile:
    """Timbral profile of an instrument."""
    name: str
    family: InstrumentFamily
    brightness: float  # 0-1
    warmth: float  # 0-1
    attack_sharpness: float  # 0-1
    sustain_ability: float  # 0-1
    low_midi: int = 21
    high_midi: int = 108

    @property
    def range_semitones(self) -> int:
        return self.high_midi - self.low_midi


# Common instrument profiles
COMMON_INSTRUMENTS: dict[str, InstrumentProfile] = {
    "violin": InstrumentProfile("violin", InstrumentFamily.STRINGS, 0.7, 0.6, 0.5, 0.9, 55, 103),
    "cello": InstrumentProfile("cello", InstrumentFamily.STRINGS, 0.4, 0.8, 0.4, 0.9, 36, 76),
    "flute": InstrumentProfile("flute", InstrumentFamily.WOODWINDS, 0.9, 0.3, 0.6, 0.7, 60, 96),
    "clarinet": InstrumentProfile("clarinet", InstrumentFamily.WOODWINDS, 0.5, 0.7, 0.4, 0.8, 50, 91),
    "trumpet": InstrumentProfile("trumpet", InstrumentFamily.BRASS, 0.8, 0.4, 0.8, 0.7, 55, 82),
    "french_horn": InstrumentProfile("french_horn", InstrumentFamily.BRASS, 0.5, 0.8, 0.3, 0.8, 41, 77),
    "piano": InstrumentProfile("piano", InstrumentFamily.KEYBOARDS, 0.6, 0.5, 0.7, 0.5, 21, 108),
    "guitar": InstrumentProfile("guitar", InstrumentFamily.PLUCKED, 0.5, 0.6, 0.6, 0.4, 40, 84),
    "soprano_voice": InstrumentProfile("soprano_voice", InstrumentFamily.VOICE, 0.7, 0.5, 0.3, 0.9, 60, 84),
    "bass_voice": InstrumentProfile("bass_voice", InstrumentFamily.VOICE, 0.3, 0.8, 0.3, 0.8, 40, 64),
}


def timbral_distance(a: InstrumentProfile, b: InstrumentProfile) -> float:
    """Euclidean distance in timbral space (brightness, warmth, attack, sustain)."""
    return math.sqrt(
        (a.brightness - b.brightness) ** 2
        + (a.warmth - b.warmth) ** 2
        + (a.attack_sharpness - b.attack_sharpness) ** 2
        + (a.sustain_ability - b.sustain_ability) ** 2
    )


def timbral_compatibility(a: InstrumentProfile, b: InstrumentProfile) -> float:
    """Compatibility score in [0, 1]. Higher = better blend."""
    d = timbral_distance(a, b)
    max_d = 2.0  # maximum possible distance in 4D unit cube
    return 1.0 - d / max_d


def range_overlap(a: InstrumentProfile, b: InstrumentProfile) -> int:
    """Overlapping semitone range between two instruments."""
    lo = max(a.low_midi, b.low_midi)
    hi = min(a.high_midi, b.high_midi)
    return max(0, hi - lo)


def register_complement(a: InstrumentProfile, b: InstrumentProfile) -> float:
    """How well two instruments cover different registers. 1 = no overlap, 0 = identical range."""
    overlap = range_overlap(a, b)
    union = (max(a.high_midi, b.high_midi) - min(a.low_midi, b.low_midi))
    if union == 0:
        return 0.0
    return 1.0 - overlap / union


def combination_score(instruments: list[InstrumentProfile]) -> float:
    """Overall score for an instrument combination (mean pairwise compatibility)."""
    if len(instruments) < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(len(instruments)):
        for j in range(i + 1, len(instruments)):
            total += timbral_compatibility(instruments[i], instruments[j])
            count += 1
    return total / count


def family_diversity(instruments: list[InstrumentProfile]) -> float:
    """Fraction of distinct instrument families represented."""
    if not instruments:
        return 0.0
    families = {inst.family for inst in instruments}
    total_families = len(InstrumentFamily)
    return len(families) / total_families


def suggest_complement(
    existing: list[InstrumentProfile],
    candidates: list[InstrumentProfile] | None = None,
) -> InstrumentProfile | None:
    """Suggest the best complementary instrument from candidates."""
    if candidates is None:
        candidates = list(COMMON_INSTRUMENTS.values())
    existing_names = {e.name for e in existing}
    candidates = [c for c in candidates if c.name not in existing_names]
    if not candidates:
        return None
    best = max(candidates, key=lambda c: combination_score(existing + [c]))
    return best


def blend_brightness(instruments: list[InstrumentProfile]) -> float:
    """Average brightness of an ensemble."""
    if not instruments:
        return 0.0
    return sum(i.brightness for i in instruments) / len(instruments)


def total_range(instruments: list[InstrumentProfile]) -> tuple[int, int]:
    """Combined pitch range (lowest, highest MIDI) of an ensemble."""
    if not instruments:
        return (0, 0)
    lo = min(i.low_midi for i in instruments)
    hi = max(i.high_midi for i in instruments)
    return (lo, hi)
