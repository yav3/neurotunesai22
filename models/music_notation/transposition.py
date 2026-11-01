"""Transposition utilities: pitch transposition, instrument transposition, enharmonic spelling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .abc_parser import ABCNote, ABCTune, SEMITONE_MAP
from .musicxml import MXLNote, MXLScore, STEP_TO_SEMITONE


CHROMATIC_SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHROMATIC_FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

SHARP_KEYS = {"C", "G", "D", "A", "E", "B", "F#", "C#",
              "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m"}
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb",
             "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm", "Abm"}

KEY_FIFTHS = {
    "Cb": -7, "Gb": -6, "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2,
    "F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5,
    "F#": 6, "C#": 7,
}

FIFTHS_TO_KEY_MAJOR = {v: k for k, v in KEY_FIFTHS.items()}
FIFTHS_TO_KEY_MINOR = {
    -7: "Abm", -6: "Ebm", -5: "Bbm", -4: "Fm", -3: "Cm",
    -2: "Gm", -1: "Dm", 0: "Am", 1: "Em", 2: "Bm",
    3: "F#m", 4: "C#m", 5: "G#m", 6: "D#m", 7: "A#m",
}


def _midi_to_step_octave(midi: int, prefer_sharps: bool = True) -> tuple[str, int, int]:
    octave = (midi // 12) - 1
    pc = midi % 12
    natural_pcs = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}
    if pc in natural_pcs:
        return natural_pcs[pc], octave, 0
    if prefer_sharps:
        sharp_map = {1: ("C", 1), 3: ("D", 1), 6: ("F", 1), 8: ("G", 1), 10: ("A", 1)}
        step, acc = sharp_map[pc]
        return step, octave, acc
    flat_map = {1: ("D", -1), 3: ("E", -1), 6: ("G", -1), 8: ("A", -1), 10: ("B", -1)}
    step, acc = flat_map[pc]
    return step, octave, acc


class Transposer:
    def __init__(self, semitones: int, prefer_sharps: bool = True):
        self.semitones = semitones
        self.prefer_sharps = prefer_sharps

    def transpose_abc_note(self, note: ABCNote) -> ABCNote:
        if note.is_rest:
            return ABCNote(
                pitch=note.pitch, duration=note.duration,
                is_rest=True, octave=note.octave,
            )
        midi = note.midi_number() + self.semitones
        step, octave, acc = _midi_to_step_octave(midi, self.prefer_sharps)
        return ABCNote(
            pitch=step, duration=note.duration,
            accidental=acc, octave=octave,
        )

    def transpose_mxl_note(self, note: MXLNote) -> MXLNote:
        if note.is_rest:
            return MXLNote(
                is_rest=True, duration=note.duration,
                note_type=note.note_type, voice=note.voice,
                staff=note.staff, dot=note.dot,
            )
        midi = note.midi_number() + self.semitones
        step, octave, acc = _midi_to_step_octave(midi, self.prefer_sharps)
        return MXLNote(
            step=step, octave=octave, duration=note.duration,
            note_type=note.note_type, dot=note.dot,
            accidental=acc, voice=note.voice, staff=note.staff,
            is_chord=note.is_chord, lyrics=note.lyrics,
        )

    def transpose_tune(self, tune: ABCTune) -> ABCTune:
        new_notes = [self.transpose_abc_note(n) for n in tune.notes]
        new_key = self._transpose_key_name(tune.header.key)
        from .abc_parser import ABCHeader
        new_header = ABCHeader(
            title=tune.header.title, composer=tune.header.composer,
            meter=tune.header.meter, key=new_key,
            tempo=tune.header.tempo,
            reference_number=tune.header.reference_number,
            unit_note_length=tune.header.unit_note_length,
        )
        return ABCTune(header=new_header, notes=new_notes)

    def transpose_score(self, score: MXLScore) -> MXLScore:
        from .musicxml import MXLPart, MXLMeasure
        new_parts = []
        for part in score.parts:
            new_measures = []
            for measure in part.measures:
                new_notes = [self.transpose_mxl_note(n) for n in measure.notes]
                new_key = None
                if measure.key_fifths is not None:
                    new_key = measure.key_fifths + self._fifths_offset()
                    new_key = max(-7, min(7, new_key))
                nm = MXLMeasure(
                    number=measure.number, notes=new_notes,
                    time_sig=measure.time_sig, key_fifths=new_key,
                    clef=measure.clef, tempo=measure.tempo,
                )
                new_measures.append(nm)
            new_parts.append(MXLPart(
                part_id=part.part_id, part_name=part.part_name,
                instrument=part.instrument, measures=new_measures,
            ))
        return MXLScore(title=score.title, composer=score.composer, parts=new_parts)

    def _transpose_key_name(self, key: str) -> str:
        is_minor = key.endswith("m") or key.endswith("min")
        base = key.replace("min", "").replace("m", "").strip()
        fifths = KEY_FIFTHS.get(base, 0)
        new_fifths = fifths + self._fifths_offset()
        new_fifths = max(-7, min(7, new_fifths))
        if is_minor:
            return FIFTHS_TO_KEY_MINOR.get(new_fifths, "Am")
        return FIFTHS_TO_KEY_MAJOR.get(new_fifths, "C")

    def _fifths_offset(self) -> int:
        semis = self.semitones % 12
        fifths_map = {0: 0, 1: -5, 2: 2, 3: -3, 4: 4, 5: -1,
                      6: 6, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}
        return fifths_map.get(semis, 0)


INSTRUMENT_TRANSPOSITIONS: dict[str, int] = {
    "Bb": -2,
    "Eb": -9,
    "F": -7,
    "A": -3,
    "C": 0,
    "D": -10,
    "G": -5,
}


class InstrumentTransposition:
    def __init__(self, instrument_key: str = "Bb"):
        self.instrument_key = instrument_key
        self.offset = INSTRUMENT_TRANSPOSITIONS.get(instrument_key, 0)

    def concert_to_written(self, note: MXLNote) -> MXLNote:
        t = Transposer(-self.offset)
        return t.transpose_mxl_note(note)

    def written_to_concert(self, note: MXLNote) -> MXLNote:
        t = Transposer(self.offset)
        return t.transpose_mxl_note(note)

    def concert_to_written_score(self, score: MXLScore) -> MXLScore:
        t = Transposer(-self.offset)
        return t.transpose_score(score)

    def written_to_concert_score(self, score: MXLScore) -> MXLScore:
        t = Transposer(self.offset)
        return t.transpose_score(score)


class KeySignatureTransposer:
    @staticmethod
    def transpose(key: str, semitones: int) -> str:
        t = Transposer(semitones)
        return t._transpose_key_name(key)

    @staticmethod
    def fifths_count(key: str) -> int:
        base = key.replace("min", "").replace("m", "").strip()
        return KEY_FIFTHS.get(base, 0)

    @staticmethod
    def key_from_fifths(fifths: int, minor: bool = False) -> str:
        fifths = max(-7, min(7, fifths))
        if minor:
            return FIFTHS_TO_KEY_MINOR.get(fifths, "Am")
        return FIFTHS_TO_KEY_MAJOR.get(fifths, "C")


class EnharmonicSpeller:
    SHARP_PREFERENCE = {
        "C": set(), "G": {"F#"}, "D": {"F#", "C#"},
        "A": {"F#", "C#", "G#"}, "E": {"F#", "C#", "G#", "D#"},
        "B": {"F#", "C#", "G#", "D#", "A#"},
    }
    FLAT_PREFERENCE = {
        "F": {"Bb"}, "Bb": {"Bb", "Eb"},
        "Eb": {"Bb", "Eb", "Ab"}, "Ab": {"Bb", "Eb", "Ab", "Db"},
        "Db": {"Bb", "Eb", "Ab", "Db", "Gb"},
    }

    def __init__(self, key: str = "C"):
        self.key = key
        self.prefer_sharps = key in SHARP_KEYS or key not in FLAT_KEYS

    def spell(self, midi: int) -> tuple[str, int, int]:
        return _midi_to_step_octave(midi, self.prefer_sharps)

    def spell_note_name(self, midi: int) -> str:
        step, octave, acc = self.spell(midi)
        acc_str = ""
        if acc > 0:
            acc_str = "#" * acc
        elif acc < 0:
            acc_str = "b" * abs(acc)
        return f"{step}{acc_str}{octave}"

    def respell(self, step: str, accidental: int, octave: int) -> tuple[str, int, int]:
        midi = 12 * (octave + 1) + STEP_TO_SEMITONE.get(step, 0) + accidental
        return self.spell(midi)
