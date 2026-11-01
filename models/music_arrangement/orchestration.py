"""Orchestration: instrument ranges, doublings, balance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentRange:
    name: str
    midi_low: int
    midi_high: int
    transposition: int = 0
    family: str = "other"


ORCHESTRA_RANGES: dict[str, InstrumentRange] = {
    "piccolo":      InstrumentRange("piccolo", 74, 108, 12, "woodwind"),
    "flute":        InstrumentRange("flute", 60, 96, 0, "woodwind"),
    "oboe":         InstrumentRange("oboe", 58, 91, 0, "woodwind"),
    "clarinet_bb":  InstrumentRange("clarinet_bb", 50, 94, 2, "woodwind"),
    "bassoon":      InstrumentRange("bassoon", 34, 75, 0, "woodwind"),
    "french_horn":  InstrumentRange("french_horn", 34, 77, 7, "brass"),
    "trumpet_bb":   InstrumentRange("trumpet_bb", 55, 82, 2, "brass"),
    "trombone":     InstrumentRange("trombone", 40, 72, 0, "brass"),
    "tuba":         InstrumentRange("tuba", 28, 58, 0, "brass"),
    "violin":       InstrumentRange("violin", 55, 103, 0, "strings"),
    "viola":        InstrumentRange("viola", 48, 91, 0, "strings"),
    "cello":        InstrumentRange("cello", 36, 76, 0, "strings"),
    "double_bass":  InstrumentRange("double_bass", 28, 67, 12, "strings"),
    "timpani":      InstrumentRange("timpani", 40, 55, 0, "percussion"),
    "piano":        InstrumentRange("piano", 21, 108, 0, "keyboard"),
    "harp":         InstrumentRange("harp", 24, 103, 0, "strings"),
}


def fits_range(pitch: int, instrument: str) -> bool:
    r = ORCHESTRA_RANGES.get(instrument)
    if r is None:
        return False
    return r.midi_low <= pitch <= r.midi_high


def suggest_transposition(pitch: int, instrument: str) -> int | None:
    r = ORCHESTRA_RANGES.get(instrument)
    if r is None:
        return None
    if pitch < r.midi_low:
        octaves_up = ((r.midi_low - pitch) + 11) // 12
        return octaves_up * 12
    if pitch > r.midi_high:
        octaves_down = ((pitch - r.midi_high) + 11) // 12
        return -octaves_down * 12
    return 0


def doubling_score(pitches_a: list[int], pitches_b: list[int]) -> float:
    if not pitches_a or not pitches_b:
        return 0.0
    pcs_a = {p % 12 for p in pitches_a}
    pcs_b = {p % 12 for p in pitches_b}
    overlap = len(pcs_a & pcs_b)
    return overlap / max(len(pcs_a | pcs_b), 1)


def orchestral_balance(part_assignments: dict[str, list[int]]) -> dict[str, float]:
    family_notes: dict[str, int] = {}
    total = 0
    for inst, pitches in part_assignments.items():
        r = ORCHESTRA_RANGES.get(inst)
        fam = r.family if r else "other"
        family_notes[fam] = family_notes.get(fam, 0) + len(pitches)
        total += len(pitches)
    if total == 0:
        return {}
    return {k: v / total for k, v in family_notes.items()}
