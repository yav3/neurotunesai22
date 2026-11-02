"""Instrument taxonomy following Hornbostel-Sachs classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class InstrumentFamily(Enum):
    IDIOPHONE = "idiophone"
    MEMBRANOPHONE = "membranophone"
    CHORDOPHONE = "chordophone"
    AEROPHONE = "aerophone"
    ELECTROPHONE = "electrophone"


class InstrumentClass(Enum):
    WOODWIND = "woodwind"
    BRASS = "brass"
    STRING_BOWED = "string_bowed"
    STRING_PLUCKED = "string_plucked"
    STRING_STRUCK = "string_struck"
    PERCUSSION_PITCHED = "percussion_pitched"
    PERCUSSION_UNPITCHED = "percussion_unpitched"
    KEYBOARD = "keyboard"
    VOICE = "voice"
    ELECTRONIC = "electronic"


@dataclass
class Instrument:
    name: str
    family: InstrumentFamily
    instrument_class: InstrumentClass
    midi_low: int
    midi_high: int
    transposition: int = 0
    clef: str = "treble"
    standard_tuning: list[int] = field(default_factory=list)


INSTRUMENT_DATABASE: list[Instrument] = [
    Instrument("violin", InstrumentFamily.CHORDOPHONE, InstrumentClass.STRING_BOWED, 55, 103, standard_tuning=[55, 62, 69, 76]),
    Instrument("viola", InstrumentFamily.CHORDOPHONE, InstrumentClass.STRING_BOWED, 48, 93, clef="alto", standard_tuning=[48, 55, 62, 69]),
    Instrument("cello", InstrumentFamily.CHORDOPHONE, InstrumentClass.STRING_BOWED, 36, 84, clef="bass", standard_tuning=[36, 43, 50, 57]),
    Instrument("double bass", InstrumentFamily.CHORDOPHONE, InstrumentClass.STRING_BOWED, 28, 67, clef="bass", standard_tuning=[28, 33, 38, 43]),
    Instrument("acoustic guitar", InstrumentFamily.CHORDOPHONE, InstrumentClass.STRING_PLUCKED, 40, 88, standard_tuning=[40, 45, 50, 55, 59, 64]),
    Instrument("piano", InstrumentFamily.CHORDOPHONE, InstrumentClass.STRING_STRUCK, 21, 108, clef="grand"),
    Instrument("flute", InstrumentFamily.AEROPHONE, InstrumentClass.WOODWIND, 60, 96),
    Instrument("oboe", InstrumentFamily.AEROPHONE, InstrumentClass.WOODWIND, 58, 91),
    Instrument("clarinet Bb", InstrumentFamily.AEROPHONE, InstrumentClass.WOODWIND, 50, 94, transposition=-2),
    Instrument("bassoon", InstrumentFamily.AEROPHONE, InstrumentClass.WOODWIND, 34, 75, clef="bass"),
    Instrument("trumpet Bb", InstrumentFamily.AEROPHONE, InstrumentClass.BRASS, 55, 82, transposition=-2),
    Instrument("french horn F", InstrumentFamily.AEROPHONE, InstrumentClass.BRASS, 34, 77, transposition=-7),
    Instrument("trombone", InstrumentFamily.AEROPHONE, InstrumentClass.BRASS, 40, 72, clef="bass"),
    Instrument("tuba", InstrumentFamily.AEROPHONE, InstrumentClass.BRASS, 28, 58, clef="bass"),
    Instrument("timpani", InstrumentFamily.MEMBRANOPHONE, InstrumentClass.PERCUSSION_PITCHED, 40, 55, clef="bass"),
    Instrument("xylophone", InstrumentFamily.IDIOPHONE, InstrumentClass.PERCUSSION_PITCHED, 65, 108),
    Instrument("marimba", InstrumentFamily.IDIOPHONE, InstrumentClass.PERCUSSION_PITCHED, 45, 96),
    Instrument("snare drum", InstrumentFamily.MEMBRANOPHONE, InstrumentClass.PERCUSSION_UNPITCHED, 0, 0),
    Instrument("bass drum", InstrumentFamily.MEMBRANOPHONE, InstrumentClass.PERCUSSION_UNPITCHED, 0, 0),
    Instrument("synthesizer", InstrumentFamily.ELECTROPHONE, InstrumentClass.ELECTRONIC, 0, 127),
]


def instruments_by_family(family: InstrumentFamily) -> list[Instrument]:
    return [i for i in INSTRUMENT_DATABASE if i.family == family]


def pitch_range(name: str) -> tuple[int, int]:
    for inst in INSTRUMENT_DATABASE:
        if inst.name.lower() == name.lower():
            return (inst.midi_low, inst.midi_high)
    return (0, 127)


def transposition_interval(name: str) -> int:
    for inst in INSTRUMENT_DATABASE:
        if inst.name.lower() == name.lower():
            return inst.transposition
    return 0


def is_transposing(name: str) -> bool:
    return transposition_interval(name) != 0


def concert_pitch_offset(name: str) -> int:
    return -transposition_interval(name)
