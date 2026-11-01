"""Orchestration rules: doublings, voicing constraints, balance, register effectiveness."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InstrumentDoubling:
    instruments: list[str]
    blend_quality: float
    typical_use: str


COMMON_DOUBLINGS: dict[str, InstrumentDoubling] = {
    "flute_violin_unison": InstrumentDoubling(
        ["flute", "violin"], 0.92, "melodic reinforcement, bright"
    ),
    "oboe_violin_unison": InstrumentDoubling(
        ["oboe", "violin"], 0.85, "expressive melody, slightly nasal edge"
    ),
    "clarinet_viola_unison": InstrumentDoubling(
        ["clarinet_bb", "viola"], 0.88, "warm, blended middle voice"
    ),
    "horn_cello_octave": InstrumentDoubling(
        ["french_horn", "cello"], 0.90, "rich octave doubling, heroic"
    ),
    "oboe_clarinet_third": InstrumentDoubling(
        ["oboe", "clarinet_bb"], 0.80, "parallel thirds, pastoral"
    ),
    "trumpet_trombone_octave": InstrumentDoubling(
        ["trumpet_bb", "trombone"], 0.87, "brass chorale reinforcement"
    ),
    "flute_clarinet_unison": InstrumentDoubling(
        ["flute", "clarinet_bb"], 0.83, "light, airy"
    ),
    "bassoon_cello_unison": InstrumentDoubling(
        ["bassoon", "cello"], 0.91, "warm bass, very homogeneous"
    ),
    "piccolo_flute_octave": InstrumentDoubling(
        ["piccolo", "flute"], 0.88, "sparkling high octave"
    ),
    "tuba_contrabass_unison": InstrumentDoubling(
        ["tuba", "double_bass"], 0.86, "foundation, heavy bass"
    ),
}


@dataclass
class VoicingConstraint:
    min_spacing_semitones: int
    max_spacing_semitones: int
    register: str


VOICING_RULES: dict[str, VoicingConstraint] = {
    "close": VoicingConstraint(1, 5, "any"),
    "open": VoicingConstraint(5, 12, "any"),
    "drop2": VoicingConstraint(3, 14, "middle"),
    "spread": VoicingConstraint(7, 24, "full"),
    "brass_section": VoicingConstraint(2, 7, "middle"),
    "string_choir": VoicingConstraint(3, 19, "full"),
}

POWER_DB: dict[str, float] = {
    "trumpet_bb": 0.0,
    "trombone": -1.0,
    "french_horn": -3.0,
    "tuba": -2.0,
    "flute": -8.0,
    "oboe": -6.0,
    "clarinet_bb": -7.0,
    "bassoon": -6.0,
    "violin": -10.0,
    "viola": -11.0,
    "cello": -9.0,
    "double_bass": -10.0,
}


def score_voicing(pitches: list[int], constraints: VoicingConstraint) -> float:
    if len(pitches) < 2:
        return 1.0
    sorted_p = sorted(pitches)
    violations = 0
    total = len(sorted_p) - 1
    for i in range(total):
        gap = sorted_p[i + 1] - sorted_p[i]
        if gap < constraints.min_spacing_semitones or gap > constraints.max_spacing_semitones:
            violations += 1
    return 1.0 - violations / max(total, 1)


def suggest_doublings(instrument: str, context: str = "melody") -> list[InstrumentDoubling]:
    results = []
    for d in COMMON_DOUBLINGS.values():
        if instrument in d.instruments and context.lower() in d.typical_use.lower():
            results.append(d)
    if not results:
        for d in COMMON_DOUBLINGS.values():
            if instrument in d.instruments:
                results.append(d)
    return sorted(results, key=lambda x: -x.blend_quality)


def orchestral_balance(
    instruments: list[str], dynamics: list[float],
) -> dict[str, float]:
    if not instruments:
        return {"total_db": 0.0, "balance_score": 0.0}
    effective = []
    for inst, dyn in zip(instruments, dynamics):
        base = POWER_DB.get(inst, -8.0)
        effective.append(base + 20.0 * np.log10(max(dyn, 0.01)))
    total = 10.0 * np.log10(sum(10 ** (e / 10) for e in effective))
    spread = max(effective) - min(effective)
    balance = max(0.0, 1.0 - spread / 20.0)
    return {"total_db": float(total), "balance_score": float(balance), "spread_db": float(spread)}


REGISTER_CURVES: dict[str, tuple[int, int, int, int]] = {
    "violin": (55, 65, 90, 100),
    "viola": (48, 58, 82, 89),
    "cello": (36, 48, 72, 79),
    "flute": (60, 68, 88, 93),
    "oboe": (58, 65, 82, 88),
    "clarinet_bb": (50, 60, 84, 91),
    "trumpet_bb": (55, 62, 75, 79),
    "french_horn": (34, 48, 68, 72),
    "trombone": (40, 48, 65, 69),
}


def register_effectiveness(instrument: str, midi_pitch: int) -> float:
    curve = REGISTER_CURVES.get(instrument)
    if curve is None:
        return 0.5
    low, sweet_low, sweet_high, high = curve
    if midi_pitch < low or midi_pitch > high:
        return 0.0
    if sweet_low <= midi_pitch <= sweet_high:
        return 1.0
    if midi_pitch < sweet_low:
        return (midi_pitch - low) / max(sweet_low - low, 1)
    return (high - midi_pitch) / max(high - sweet_high, 1)
