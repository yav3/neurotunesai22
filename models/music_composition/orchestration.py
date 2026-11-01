"""Instrument ranges, orchestration rules, voice assignment."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class InstrumentRange:
    name: str
    min_pitch: int
    max_pitch: int
    transposition: int = 0
    family: str = "other"


INSTRUMENTS: dict[str, InstrumentRange] = {
    "violin": InstrumentRange("violin", 55, 103, 0, "strings"),
    "viola": InstrumentRange("viola", 48, 93, 0, "strings"),
    "cello": InstrumentRange("cello", 36, 76, 0, "strings"),
    "contrabass": InstrumentRange("contrabass", 28, 67, 0, "strings"),
    "flute": InstrumentRange("flute", 60, 96, 0, "woodwinds"),
    "oboe": InstrumentRange("oboe", 58, 91, 0, "woodwinds"),
    "clarinet": InstrumentRange("clarinet", 50, 94, -2, "woodwinds"),
    "bassoon": InstrumentRange("bassoon", 34, 75, 0, "woodwinds"),
    "trumpet": InstrumentRange("trumpet", 55, 82, -2, "brass"),
    "french_horn": InstrumentRange("french_horn", 34, 77, -7, "brass"),
    "trombone": InstrumentRange("trombone", 40, 72, 0, "brass"),
    "tuba": InstrumentRange("tuba", 28, 58, 0, "brass"),
    "piano": InstrumentRange("piano", 21, 108, 0, "keyboard"),
    "guitar": InstrumentRange("guitar", 40, 84, 0, "plucked"),
    "harp": InstrumentRange("harp", 24, 103, 0, "plucked"),
    "timpani": InstrumentRange("timpani", 40, 55, 0, "percussion"),
    "marimba": InstrumentRange("marimba", 45, 96, 0, "percussion"),
    "xylophone": InstrumentRange("xylophone", 65, 108, 0, "percussion"),
    "voice_soprano": InstrumentRange("voice_soprano", 60, 84, 0, "voice"),
    "voice_alto": InstrumentRange("voice_alto", 53, 77, 0, "voice"),
    "voice_tenor": InstrumentRange("voice_tenor", 48, 72, 0, "voice"),
    "voice_bass": InstrumentRange("voice_bass", 40, 64, 0, "voice"),
}


class Orchestrator:
    def __init__(self, instruments: list[InstrumentRange] | None = None):
        self._instruments = instruments or list(INSTRUMENTS.values())

    def fits_range(self, pitch: int, instrument: InstrumentRange) -> bool:
        concert_pitch = pitch + instrument.transposition
        return instrument.min_pitch <= concert_pitch <= instrument.max_pitch

    def find_instruments(self, pitch: int) -> list[InstrumentRange]:
        return [inst for inst in self._instruments if self.fits_range(pitch, inst)]

    def range_violations(self, pitches: list[int], instrument: InstrumentRange) -> list[int]:
        return [i for i, p in enumerate(pitches) if not self.fits_range(p, instrument)]

    def assign_voices(
        self, pitches: list[int], instruments: list[InstrumentRange] | None = None,
    ) -> dict[str, list[int]]:
        insts = instruments or self._instruments
        assignment: dict[str, list[int]] = {inst.name: [] for inst in insts}
        for p in sorted(pitches):
            best = None
            best_margin = -1
            for inst in insts:
                if self.fits_range(p, inst):
                    cp = p + inst.transposition
                    margin = min(cp - inst.min_pitch, inst.max_pitch - cp)
                    if margin > best_margin:
                        best_margin = margin
                        best = inst
            if best is not None:
                assignment[best.name].append(p)
        return assignment

    def family_balance(self, assignment: dict[str, list[int]]) -> dict[str, int]:
        families: dict[str, int] = {}
        for inst_name, pitches in assignment.items():
            if pitches:
                inst = next((i for i in self._instruments if i.name == inst_name), None)
                if inst:
                    families[inst.family] = families.get(inst.family, 0) + len(pitches)
        return families


def fits_range(pitch: int, instrument: InstrumentRange) -> bool:
    return Orchestrator().fits_range(pitch, instrument)


def assign_voices(pitches: list[int], instruments: list[InstrumentRange]) -> dict[str, list[int]]:
    return Orchestrator(instruments).assign_voices(pitches, instruments)
