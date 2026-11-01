"""Comprehensive MIDI pitch ranges for 60+ instruments including world instruments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InstrumentRange:
    name: str
    midi_low: int
    midi_high: int
    practical_low: int = 0
    practical_high: int = 127
    dynamic_range_db: float = 40.0
    standard_tuning: tuple[int, ...] = ()
    family: str = "other"

    def __post_init__(self) -> None:
        if self.practical_low == 0:
            self.practical_low = self.midi_low
        if self.practical_high == 127:
            self.practical_high = self.midi_high


INSTRUMENT_RANGES: dict[str, InstrumentRange] = {
    "violin": InstrumentRange("violin", 55, 105, 55, 100, 45, (55, 62, 69, 76), "strings"),
    "viola": InstrumentRange("viola", 48, 93, 48, 89, 42, (48, 55, 62, 69), "strings"),
    "cello": InstrumentRange("cello", 36, 84, 36, 79, 45, (36, 43, 50, 57), "strings"),
    "double_bass": InstrumentRange("double_bass", 28, 67, 28, 60, 40, (28, 33, 38, 43), "strings"),
    "acoustic_guitar": InstrumentRange("acoustic_guitar", 40, 88, 40, 84, 35, (40, 45, 50, 55, 59, 64), "strings"),
    "electric_guitar": InstrumentRange("electric_guitar", 40, 88, 40, 88, 30, (40, 45, 50, 55, 59, 64), "strings"),
    "bass_guitar": InstrumentRange("bass_guitar", 28, 67, 28, 60, 30, (28, 33, 38, 43), "strings"),
    "harp": InstrumentRange("harp", 24, 103, 24, 100, 40, (), "strings"),
    "banjo": InstrumentRange("banjo", 50, 79, 50, 76, 25, (50, 55, 59, 62, 67), "strings"),
    "mandolin": InstrumentRange("mandolin", 55, 93, 55, 88, 25, (55, 62, 69, 76), "strings"),
    "ukulele": InstrumentRange("ukulele", 60, 84, 60, 81, 20, (67, 60, 64, 69), "strings"),
    "piano": InstrumentRange("piano", 21, 108, 21, 108, 60, (), "keyboard"),
    "harpsichord": InstrumentRange("harpsichord", 29, 89, 29, 89, 15, (), "keyboard"),
    "organ": InstrumentRange("organ", 24, 108, 24, 108, 25, (), "keyboard"),
    "celesta": InstrumentRange("celesta", 60, 108, 60, 100, 30, (), "keyboard"),
    "accordion": InstrumentRange("accordion", 36, 96, 41, 89, 35, (), "keyboard"),
    "flute": InstrumentRange("flute", 60, 96, 60, 93, 35, (), "woodwind"),
    "piccolo": InstrumentRange("piccolo", 74, 108, 74, 105, 30, (), "woodwind"),
    "oboe": InstrumentRange("oboe", 58, 91, 58, 88, 35, (), "woodwind"),
    "english_horn": InstrumentRange("english_horn", 52, 81, 52, 79, 30, (), "woodwind"),
    "clarinet_bb": InstrumentRange("clarinet_bb", 50, 94, 50, 91, 40, (), "woodwind"),
    "bass_clarinet": InstrumentRange("bass_clarinet", 38, 77, 38, 74, 38, (), "woodwind"),
    "bassoon": InstrumentRange("bassoon", 34, 75, 34, 72, 35, (), "woodwind"),
    "contrabassoon": InstrumentRange("contrabassoon", 22, 53, 22, 50, 30, (), "woodwind"),
    "saxophone_soprano": InstrumentRange("saxophone_soprano", 56, 87, 56, 85, 35, (), "woodwind"),
    "saxophone_alto": InstrumentRange("saxophone_alto", 49, 80, 49, 78, 40, (), "woodwind"),
    "saxophone_tenor": InstrumentRange("saxophone_tenor", 44, 75, 44, 73, 40, (), "woodwind"),
    "saxophone_bari": InstrumentRange("saxophone_bari", 36, 68, 36, 66, 38, (), "woodwind"),
    "trumpet_bb": InstrumentRange("trumpet_bb", 55, 82, 55, 79, 35, (), "brass"),
    "french_horn": InstrumentRange("french_horn", 34, 77, 34, 72, 40, (), "brass"),
    "trombone": InstrumentRange("trombone", 40, 72, 40, 69, 42, (), "brass"),
    "bass_trombone": InstrumentRange("bass_trombone", 34, 67, 34, 62, 40, (), "brass"),
    "tuba": InstrumentRange("tuba", 28, 58, 28, 55, 38, (), "brass"),
    "euphonium": InstrumentRange("euphonium", 34, 72, 34, 69, 38, (), "brass"),
    "timpani": InstrumentRange("timpani", 40, 55, 40, 55, 35, (), "percussion"),
    "xylophone": InstrumentRange("xylophone", 65, 108, 65, 105, 30, (), "percussion"),
    "marimba": InstrumentRange("marimba", 45, 96, 45, 93, 35, (), "percussion"),
    "vibraphone": InstrumentRange("vibraphone", 53, 89, 53, 89, 30, (), "percussion"),
    "glockenspiel": InstrumentRange("glockenspiel", 79, 108, 79, 105, 25, (), "percussion"),
    "tubular_bells": InstrumentRange("tubular_bells", 60, 77, 60, 77, 20, (), "percussion"),
    "steel_pan": InstrumentRange("steel_pan", 48, 96, 53, 89, 25, (), "percussion"),
    "sitar": InstrumentRange("sitar", 36, 84, 40, 79, 35, (36, 43, 48, 55, 60, 64, 67), "world"),
    "erhu": InstrumentRange("erhu", 52, 88, 55, 84, 30, (52, 59), "world"),
    "koto": InstrumentRange("koto", 36, 84, 40, 79, 30, (), "world"),
    "shamisen": InstrumentRange("shamisen", 45, 81, 48, 77, 25, (45, 52, 57), "world"),
    "oud": InstrumentRange("oud", 36, 72, 38, 69, 25, (36, 43, 48, 52, 55, 60), "world"),
    "bouzouki": InstrumentRange("bouzouki", 43, 79, 45, 76, 25, (43, 50, 55, 62), "world"),
    "balalaika": InstrumentRange("balalaika", 52, 84, 55, 79, 20, (52, 64, 64), "world"),
    "pipa": InstrumentRange("pipa", 45, 93, 48, 88, 30, (45, 52, 57, 64), "world"),
    "guzheng": InstrumentRange("guzheng", 36, 96, 40, 93, 30, (), "world"),
    "sarangi": InstrumentRange("sarangi", 40, 84, 43, 79, 35, (40, 47, 54), "world"),
    "gamelan_saron": InstrumentRange("gamelan_saron", 60, 72, 60, 72, 15, (), "world"),
    "gamelan_gender": InstrumentRange("gamelan_gender", 48, 84, 48, 79, 20, (), "world"),
    "djembe": InstrumentRange("djembe", 0, 0, 0, 0, 25, (), "world_perc"),
    "tabla": InstrumentRange("tabla", 0, 0, 0, 0, 30, (), "world_perc"),
    "cajon": InstrumentRange("cajon", 0, 0, 0, 0, 20, (), "world_perc"),
    "taiko": InstrumentRange("taiko", 0, 0, 0, 0, 35, (), "world_perc"),
    "darbuka": InstrumentRange("darbuka", 0, 0, 0, 0, 20, (), "world_perc"),
    "congas": InstrumentRange("congas", 0, 0, 0, 0, 25, (), "world_perc"),
    "bongos": InstrumentRange("bongos", 0, 0, 0, 0, 20, (), "world_perc"),
    "didgeridoo": InstrumentRange("didgeridoo", 29, 41, 29, 38, 15, (), "world"),
    "shakuhachi": InstrumentRange("shakuhachi", 62, 91, 62, 86, 30, (), "world"),
    "bansuri": InstrumentRange("bansuri", 55, 91, 57, 86, 25, (), "world"),
}


def range_overlap(inst_a: str, inst_b: str) -> tuple[int, int] | None:
    a = INSTRUMENT_RANGES.get(inst_a)
    b = INSTRUMENT_RANGES.get(inst_b)
    if a is None or b is None:
        return None
    lo = max(a.practical_low, b.practical_low)
    hi = min(a.practical_high, b.practical_high)
    if lo > hi:
        return None
    return (lo, hi)


def tessitura(inst: str, percentile_low: float = 0.25, percentile_high: float = 0.75) -> tuple[int, int]:
    r = INSTRUMENT_RANGES.get(inst)
    if r is None:
        return (60, 72)
    span = r.practical_high - r.practical_low
    return (
        r.practical_low + int(span * percentile_low),
        r.practical_low + int(span * percentile_high),
    )


def ensemble_range(instruments: list[str]) -> tuple[int, int]:
    lows = []
    highs = []
    for name in instruments:
        r = INSTRUMENT_RANGES.get(name)
        if r and r.practical_low > 0:
            lows.append(r.practical_low)
        if r and r.practical_high > 0:
            highs.append(r.practical_high)
    if not lows or not highs:
        return (21, 108)
    return (min(lows), max(highs))


def instruments_in_range(midi_low: int, midi_high: int) -> list[str]:
    result = []
    for name, r in INSTRUMENT_RANGES.items():
        if r.practical_low <= midi_high and r.practical_high >= midi_low:
            result.append(name)
    return result
