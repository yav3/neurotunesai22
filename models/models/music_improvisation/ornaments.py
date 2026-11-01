"""Musical ornaments — trills, mordents, turns, grace notes, tremolo, glissando."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OrnamentType(Enum):
    TRILL = "trill"
    MORDENT = "mordent"
    TURN = "turn"
    GRACE_NOTE = "grace_note"
    APPOGGIATURA = "appoggiatura"
    ACCIACCATURA = "acciaccatura"
    TREMOLO = "tremolo"
    GLISSANDO = "glissando"


@dataclass
class Ornament:
    ornament_type: OrnamentType
    pitches: list[int]
    beat_duration: float = 0.25


def trill(main_pitch: int, interval: int = 2, n_notes: int = 6) -> list[int]:
    return [main_pitch if i % 2 == 0 else main_pitch + interval for i in range(n_notes)]


def mordent(main_pitch: int, upper: bool = True, interval: int = 2) -> list[int]:
    neighbor = main_pitch + interval if upper else main_pitch - interval
    return [main_pitch, neighbor, main_pitch]


def turn(main_pitch: int, upper_interval: int = 2, lower_interval: int = 1) -> list[int]:
    return [
        main_pitch + upper_interval,
        main_pitch,
        main_pitch - lower_interval,
        main_pitch,
    ]


def grace_note(main_pitch: int, approach_from: int = -2) -> list[int]:
    return [main_pitch + approach_from, main_pitch]


def appoggiatura(main_pitch: int, above: bool = True, interval: int = 2) -> list[int]:
    app = main_pitch + interval if above else main_pitch - interval
    return [app, main_pitch]


def acciaccatura(main_pitch: int, above: bool = True, interval: int = 1) -> list[int]:
    acc = main_pitch + interval if above else main_pitch - interval
    return [acc, main_pitch]


def tremolo(pitch: int, repetitions: int = 4) -> list[int]:
    return [pitch] * repetitions


def glissando_pitches(start: int, end: int) -> list[int]:
    step = 1 if end >= start else -1
    return list(range(start, end + step, step))


def ornament_duration(orn: Ornament) -> float:
    return orn.beat_duration


def apply_ornament_timing(orn: Ornament) -> list[float]:
    n = len(orn.pitches)
    if n <= 1:
        return [0.0]
    note_dur = orn.beat_duration / n
    return [i * note_dur for i in range(n)]
