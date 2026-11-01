"""Musical ornamentation: realization of trills, mordents, turns, jazz bends.

Models ornament types across Baroque, Classical, Romantic, and Jazz styles
with period-appropriate realization rules, density metrics, and automatic
ornamentation of melodies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class OrnamentType(Enum):
    TRILL = "trill"
    MORDENT = "mordent"
    INVERTED_MORDENT = "inverted_mordent"
    TURN = "turn"
    APPOGGIATURA = "appoggiatura"
    ACCIACCATURA = "acciaccatura"
    GLISSANDO = "glissando"
    PORTAMENTO = "portamento"
    VIBRATO = "vibrato"
    TREMOLO = "tremolo"
    GRUPETTO = "grupetto"


@dataclass
class Ornament:
    type: OrnamentType
    anchor_pitch: int
    duration_fraction: float = 0.25
    speed: float = 1.0
    intensity: float = 0.7
    auxiliary_interval: int = 2


@dataclass
class MicroEvent:
    pitch: int
    duration: float


def realize_trill(anchor: int, auxiliary_interval: int = 2,
                  n_alternations: int = 4, note_dur: float = 0.0625) -> list[MicroEvent]:
    events = []
    upper = anchor + auxiliary_interval
    for i in range(n_alternations):
        events.append(MicroEvent(anchor if i % 2 == 0 else upper, note_dur))
    return events


def realize_mordent(anchor: int, inverted: bool = False,
                    interval: int = 2, note_dur: float = 0.0625) -> list[MicroEvent]:
    aux = anchor + interval if not inverted else anchor - interval
    return [MicroEvent(anchor, note_dur), MicroEvent(aux, note_dur), MicroEvent(anchor, note_dur)]


def realize_turn(anchor: int, interval: int = 2,
                 note_dur: float = 0.0625) -> list[MicroEvent]:
    return [
        MicroEvent(anchor + interval, note_dur),
        MicroEvent(anchor, note_dur),
        MicroEvent(anchor - interval, note_dur),
        MicroEvent(anchor, note_dur),
    ]


def realize_appoggiatura(anchor: int, approach_interval: int = 2,
                         grace_dur: float = 0.125, main_dur: float = 0.125) -> list[MicroEvent]:
    return [MicroEvent(anchor + approach_interval, grace_dur), MicroEvent(anchor, main_dur)]


def realize_acciaccatura(anchor: int, approach_interval: int = 1,
                         grace_dur: float = 0.03125, main_dur: float = 0.21875) -> list[MicroEvent]:
    return [MicroEvent(anchor - approach_interval, grace_dur), MicroEvent(anchor, main_dur)]


def realize_ornament(ornament: Ornament, context_tempo: float = 120.0) -> list[MicroEvent]:
    base_dur = 60.0 / context_tempo * ornament.duration_fraction / ornament.speed
    t = ornament.type
    if t == OrnamentType.TRILL:
        return realize_trill(ornament.anchor_pitch, ornament.auxiliary_interval,
                             n_alternations=max(2, int(4 * ornament.speed)), note_dur=base_dur)
    if t == OrnamentType.MORDENT:
        return realize_mordent(ornament.anchor_pitch, inverted=False,
                               interval=ornament.auxiliary_interval, note_dur=base_dur)
    if t == OrnamentType.INVERTED_MORDENT:
        return realize_mordent(ornament.anchor_pitch, inverted=True,
                               interval=ornament.auxiliary_interval, note_dur=base_dur)
    if t == OrnamentType.TURN:
        return realize_turn(ornament.anchor_pitch, ornament.auxiliary_interval, base_dur)
    if t == OrnamentType.APPOGGIATURA:
        return realize_appoggiatura(ornament.anchor_pitch, ornament.auxiliary_interval,
                                    base_dur, base_dur * 2)
    if t == OrnamentType.ACCIACCATURA:
        return realize_acciaccatura(ornament.anchor_pitch, ornament.auxiliary_interval,
                                    base_dur * 0.25, base_dur * 1.75)
    if t == OrnamentType.GRUPETTO:
        return realize_turn(ornament.anchor_pitch, ornament.auxiliary_interval, base_dur)
    return [MicroEvent(ornament.anchor_pitch, base_dur * 4)]


# ---------------------------------------------------------------------------
# Style-specific ornament sets
# ---------------------------------------------------------------------------

BAROQUE_ORNAMENT_TABLE: dict[str, list[OrnamentType]] = {
    "strong_beat": [OrnamentType.TRILL, OrnamentType.APPOGGIATURA],
    "weak_beat": [OrnamentType.MORDENT, OrnamentType.TURN],
    "cadence": [OrnamentType.TRILL, OrnamentType.TURN],
    "ascending_step": [OrnamentType.MORDENT],
    "descending_step": [OrnamentType.INVERTED_MORDENT],
}

JAZZ_ORNAMENT_TABLE: dict[str, list[OrnamentType]] = {
    "approach": [OrnamentType.ACCIACCATURA, OrnamentType.GLISSANDO],
    "sustain": [OrnamentType.VIBRATO],
    "release": [OrnamentType.PORTAMENTO, OrnamentType.GLISSANDO],
    "accent": [OrnamentType.ACCIACCATURA],
}


@dataclass
class OrnamentDensity:
    ornaments_per_beat: float
    style: str

    @property
    def appropriate(self) -> bool:
        ranges = {
            "baroque": (0.1, 0.5),
            "classical": (0.05, 0.3),
            "romantic": (0.02, 0.2),
            "jazz": (0.1, 0.6),
        }
        lo, hi = ranges.get(self.style.lower(), (0.0, 1.0))
        return lo <= self.ornaments_per_beat <= hi


def compute_density(n_ornaments: int, n_beats: int, style: str = "baroque") -> OrnamentDensity:
    density = n_ornaments / max(n_beats, 1)
    return OrnamentDensity(density, style)


def apply_ornaments(melody: list[int], durations: list[float],
                    style: str = "baroque", density: float = 0.2,
                    rng: np.random.RandomState | None = None
                    ) -> list[tuple[list[MicroEvent], float]]:
    """Auto-ornament a melody according to style conventions.

    Returns list of (micro_events, original_duration) per note.
    """
    if rng is None:
        rng = np.random.RandomState()
    table = BAROQUE_ORNAMENT_TABLE if style.lower() == "baroque" else JAZZ_ORNAMENT_TABLE
    result = []
    for i, (pitch, dur) in enumerate(zip(melody, durations)):
        if rng.rand() > density:
            result.append(([MicroEvent(pitch, dur)], dur))
            continue
        beat_pos = "strong_beat" if i % 2 == 0 else "weak_beat"
        if i == len(melody) - 1:
            beat_pos = "cadence"
        candidates = table.get(beat_pos, table.get("approach", [OrnamentType.MORDENT]))
        chosen = candidates[rng.randint(len(candidates))]
        orn = Ornament(chosen, pitch, duration_fraction=0.25)
        events = realize_ornament(orn, context_tempo=120.0)
        result.append((events, dur))
    return result
