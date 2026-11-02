"""Orchestral doubling rules and instrument combination scoring.

Governs when unison/octave doublings are effective vs. muddy,
spacing rules per register, and dynamic balance across sections.
Pure numeric — instrument data is inline, no external DB needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import numpy as np


class InstrumentFamily(Enum):
    WOODWIND = "woodwind"
    BRASS = "brass"
    STRING = "string"
    PERCUSSION = "percussion"
    KEYBOARD = "keyboard"
    VOICE = "voice"


@dataclass
class InstrumentSpec:
    name: str
    family: InstrumentFamily
    low: int
    high: int
    transposition: int = 0
    dynamic_power: float = 1.0


ORCHESTRA_DB: Dict[str, InstrumentSpec] = {
    "flute": InstrumentSpec("flute", InstrumentFamily.WOODWIND, 60, 96, dynamic_power=0.6),
    "oboe": InstrumentSpec("oboe", InstrumentFamily.WOODWIND, 58, 91, dynamic_power=0.7),
    "clarinet": InstrumentSpec("clarinet", InstrumentFamily.WOODWIND, 50, 94, transposition=-2, dynamic_power=0.65),
    "bassoon": InstrumentSpec("bassoon", InstrumentFamily.WOODWIND, 34, 75, dynamic_power=0.7),
    "horn": InstrumentSpec("horn", InstrumentFamily.BRASS, 34, 77, transposition=-7, dynamic_power=0.8),
    "trumpet": InstrumentSpec("trumpet", InstrumentFamily.BRASS, 55, 82, transposition=-2, dynamic_power=0.9),
    "trombone": InstrumentSpec("trombone", InstrumentFamily.BRASS, 40, 72, dynamic_power=0.95),
    "tuba": InstrumentSpec("tuba", InstrumentFamily.BRASS, 28, 58, dynamic_power=1.0),
    "violin": InstrumentSpec("violin", InstrumentFamily.STRING, 55, 100, dynamic_power=0.5),
    "viola": InstrumentSpec("viola", InstrumentFamily.STRING, 48, 88, dynamic_power=0.45),
    "cello": InstrumentSpec("cello", InstrumentFamily.STRING, 36, 76, dynamic_power=0.55),
    "double_bass": InstrumentSpec("double_bass", InstrumentFamily.STRING, 28, 60, transposition=-12, dynamic_power=0.6),
    "piano": InstrumentSpec("piano", InstrumentFamily.KEYBOARD, 21, 108, dynamic_power=0.85),
}


class DoublingType(Enum):
    UNISON = "unison"
    OCTAVE = "octave"
    TWO_OCTAVE = "two_octave"
    PARTIAL = "partial"


@dataclass
class DoublingScore:
    blend: float
    power_balance: float
    register_clarity: float
    overall: float
    warnings: List[str] = field(default_factory=list)


def fits_range(instrument: str, pitch: int) -> bool:
    spec = ORCHESTRA_DB.get(instrument)
    if spec is None:
        return True
    concert = pitch + spec.transposition
    return spec.low <= concert <= spec.high


def register_weight(pitch: int) -> float:
    if pitch < 40:
        return 0.6
    if pitch < 55:
        return 0.8
    if pitch < 72:
        return 1.0
    if pitch < 84:
        return 0.9
    return 0.7


def blend_score(inst_a: str, inst_b: str) -> float:
    a = ORCHESTRA_DB.get(inst_a)
    b = ORCHESTRA_DB.get(inst_b)
    if a is None or b is None:
        return 0.5
    if a.family == b.family:
        return 0.9
    cross_family = {
        frozenset({InstrumentFamily.STRING, InstrumentFamily.WOODWIND}): 0.75,
        frozenset({InstrumentFamily.BRASS, InstrumentFamily.WOODWIND}): 0.55,
        frozenset({InstrumentFamily.STRING, InstrumentFamily.BRASS}): 0.5,
        frozenset({InstrumentFamily.KEYBOARD, InstrumentFamily.STRING}): 0.7,
    }
    key = frozenset({a.family, b.family})
    return cross_family.get(key, 0.5)


def power_balance(instruments: Sequence[str]) -> float:
    powers = []
    for name in instruments:
        spec = ORCHESTRA_DB.get(name)
        powers.append(spec.dynamic_power if spec else 0.5)
    if len(powers) < 2:
        return 1.0
    arr = np.array(powers)
    ratio = arr.min() / arr.max() if arr.max() > 0 else 1.0
    return float(ratio)


def spacing_penalty(pitches: Sequence[int]) -> float:
    if len(pitches) < 2:
        return 0.0
    s = sorted(pitches)
    penalty = 0.0
    for i in range(len(s) - 1):
        gap = s[i + 1] - s[i]
        if s[i] < 48 and gap < 7:
            penalty += (7 - gap) * 0.1
        elif s[i] < 48 and gap > 19:
            penalty += (gap - 19) * 0.05
    return min(penalty, 1.0)


def evaluate_doubling(
    instruments: Sequence[str],
    pitches: Sequence[int],
    doubling_type: DoublingType = DoublingType.UNISON,
) -> DoublingScore:
    warnings: list[str] = []
    for inst, p in zip(instruments, pitches):
        if not fits_range(inst, p):
            warnings.append(f"{inst} at MIDI {p} is out of range")

    bl = 1.0
    if len(instruments) >= 2:
        scores = []
        for i in range(len(instruments)):
            for j in range(i + 1, len(instruments)):
                scores.append(blend_score(instruments[i], instruments[j]))
        bl = float(np.mean(scores))

    pb = power_balance(instruments)
    sp = spacing_penalty(pitches)
    rc = 1.0 - sp

    rw = np.mean([register_weight(p) for p in pitches]) if pitches else 1.0
    overall = 0.35 * bl + 0.25 * pb + 0.25 * rc + 0.15 * float(rw)

    if doubling_type == DoublingType.UNISON and any(p < 48 for p in pitches):
        warnings.append("unison doubling in low register may sound muddy")
        overall *= 0.85

    return DoublingScore(
        blend=bl,
        power_balance=pb,
        register_clarity=rc,
        overall=overall,
        warnings=warnings,
    )


def suggest_doubling(
    melody_pitch: int,
    lead_instrument: str,
    candidates: Sequence[str] | None = None,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    if candidates is None:
        candidates = list(ORCHESTRA_DB.keys())
    results = []
    for c in candidates:
        if c == lead_instrument:
            continue
        if not fits_range(c, melody_pitch):
            continue
        sc = evaluate_doubling([lead_instrument, c], [melody_pitch, melody_pitch])
        results.append((c, sc.overall))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def section_balance(
    section_instruments: Dict[str, int],
) -> Dict[str, float]:
    total_power = 0.0
    section_powers: Dict[str, float] = {}
    for name, count in section_instruments.items():
        spec = ORCHESTRA_DB.get(name)
        p = (spec.dynamic_power if spec else 0.5) * math.sqrt(count)
        section_powers[name] = p
        total_power += p
    if total_power == 0:
        return {k: 0.0 for k in section_instruments}
    return {k: v / total_power for k, v in section_powers.items()}
