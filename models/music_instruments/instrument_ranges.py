"""Instrument range definitions and range-based utilities."""
from dataclasses import dataclass
import numpy as np


@dataclass
class InstrumentRange:
    name: str
    lowest_note: int
    highest_note: int
    comfortable_low: int
    comfortable_high: int
    transposing: bool = False
    transposition_semitones: int = 0


INSTRUMENT_RANGES: dict[str, InstrumentRange] = {
    "piano": InstrumentRange("Piano", 21, 108, 28, 96),
    "violin": InstrumentRange("Violin", 55, 103, 55, 93),
    "viola": InstrumentRange("Viola", 48, 93, 48, 84),
    "cello": InstrumentRange("Cello", 36, 84, 36, 76),
    "double_bass": InstrumentRange("Double Bass", 28, 67, 28, 60, True, 12),
    "flute": InstrumentRange("Flute", 60, 96, 62, 91),
    "oboe": InstrumentRange("Oboe", 58, 91, 60, 86),
    "clarinet_bb": InstrumentRange("Clarinet in Bb", 50, 94, 52, 89, True, 2),
    "bassoon": InstrumentRange("Bassoon", 34, 75, 36, 70),
    "french_horn": InstrumentRange("French Horn", 34, 77, 40, 72, True, 7),
    "trumpet_bb": InstrumentRange("Trumpet in Bb", 55, 82, 57, 79, True, 2),
    "trombone": InstrumentRange("Trombone", 40, 72, 43, 67),
    "tuba": InstrumentRange("Tuba", 28, 58, 30, 55),
    "soprano": InstrumentRange("Soprano Voice", 60, 84, 62, 79),
    "alto": InstrumentRange("Alto Voice", 55, 77, 57, 74),
    "tenor": InstrumentRange("Tenor Voice", 48, 72, 50, 69),
    "baritone": InstrumentRange("Baritone Voice", 45, 67, 47, 64),
    "bass_voice": InstrumentRange("Bass Voice", 40, 64, 43, 60),
    "guitar": InstrumentRange("Guitar", 40, 88, 40, 79, True, 12),
    "harp": InstrumentRange("Harp", 24, 103, 28, 96),
    "piccolo": InstrumentRange("Piccolo", 74, 108, 76, 103, True, -12),
    "alto_saxophone": InstrumentRange("Alto Saxophone", 49, 80, 51, 78, True, 9),
    "tenor_saxophone": InstrumentRange("Tenor Saxophone", 44, 76, 46, 73, True, 14),
    "marimba": InstrumentRange("Marimba", 45, 96, 48, 91),
}


def note_in_range(instrument: str, midi_note: int) -> bool:
    r = INSTRUMENT_RANGES.get(instrument)
    if r is None:
        return False
    return r.lowest_note <= midi_note <= r.highest_note


def comfortable_range(instrument: str) -> tuple[int, int]:
    r = INSTRUMENT_RANGES.get(instrument)
    if r is None:
        return (0, 0)
    return (r.comfortable_low, r.comfortable_high)


def transpose_for_instrument(notes: np.ndarray, target_instrument: str) -> np.ndarray:
    r = INSTRUMENT_RANGES.get(target_instrument)
    if r is None:
        return notes.copy()
    out = notes.copy().astype(float)
    mid_target = (r.comfortable_low + r.comfortable_high) / 2.0
    mid_notes = np.mean(out) if len(out) > 0 else mid_target
    shift_octaves = round((mid_target - mid_notes) / 12.0)
    out += shift_octaves * 12
    return out


def find_instruments_for_range(low_midi: int, high_midi: int) -> list[str]:
    result = []
    for name, r in INSTRUMENT_RANGES.items():
        if r.lowest_note <= low_midi and r.highest_note >= high_midi:
            result.append(name)
    return result


def range_overlap(inst_a: str, inst_b: str) -> float:
    ra = INSTRUMENT_RANGES.get(inst_a)
    rb = INSTRUMENT_RANGES.get(inst_b)
    if ra is None or rb is None:
        return 0.0
    lo = max(ra.lowest_note, rb.lowest_note)
    hi = min(ra.highest_note, rb.highest_note)
    if lo >= hi:
        return 0.0
    overlap = hi - lo
    union = max(ra.highest_note, rb.highest_note) - min(ra.lowest_note, rb.lowest_note)
    return overlap / union if union > 0 else 0.0
