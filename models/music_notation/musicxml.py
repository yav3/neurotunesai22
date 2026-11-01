"""MusicXML parser and writer using xml.etree."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


DURATION_TYPE_TICKS: dict[str, int] = {
    "whole": 4096, "half": 2048, "quarter": 1024, "eighth": 512,
    "16th": 256, "32nd": 128, "64th": 64,
}

STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass
class MXLNote:
    step: str = "C"
    octave: int = 4
    duration: int = 1
    note_type: str = "quarter"
    dot: bool = False
    accidental: int = 0
    voice: int = 1
    staff: int = 1
    is_rest: bool = False
    is_chord: bool = False
    lyrics: str = ""

    def midi_number(self) -> int:
        if self.is_rest:
            return -1
        base = STEP_TO_SEMITONE.get(self.step, 0)
        return 12 * (self.octave + 1) + base + self.accidental

    def ticks(self) -> int:
        base = DURATION_TYPE_TICKS.get(self.note_type, 1024)
        if self.dot:
            base = base + base // 2
        return base


@dataclass
class MXLMeasure:
    number: int = 1
    notes: list[MXLNote] = field(default_factory=list)
    time_sig: Optional[tuple[int, int]] = None
    key_fifths: Optional[int] = None
    clef: Optional[str] = None
    tempo: Optional[int] = None

    def note_count(self) -> int:
        return sum(1 for n in self.notes if not n.is_rest)

    def rest_count(self) -> int:
        return sum(1 for n in self.notes if n.is_rest)

    def total_duration(self) -> int:
        return sum(n.duration for n in self.notes if not n.is_chord)


@dataclass
class MXLPart:
    part_id: str = "P1"
    part_name: str = "Piano"
    instrument: str = "Piano"
    measures: list[MXLMeasure] = field(default_factory=list)

    def note_count(self) -> int:
        return sum(m.note_count() for m in self.measures)

    def measure_count(self) -> int:
        return len(self.measures)


@dataclass
class MXLScore:
    title: str = ""
    composer: str = ""
    parts: list[MXLPart] = field(default_factory=list)

    def total_notes(self) -> int:
        return sum(p.note_count() for p in self.parts)

    def total_measures(self) -> int:
        return max((p.measure_count() for p in self.parts), default=0)


class MXLParser:
    def parse(self, xml_text: str) -> MXLScore:
        root = ET.fromstring(xml_text)
        score = MXLScore()

        work = root.find("work")
        if work is not None:
            wt = work.find("work-title")
            if wt is not None and wt.text:
                score.title = wt.text

        ident = root.find("identification")
        if ident is not None:
            for creator in ident.findall("creator"):
                if creator.get("type") == "composer" and creator.text:
                    score.composer = creator.text

        movement = root.find("movement-title")
        if movement is not None and movement.text and not score.title:
            score.title = movement.text

        part_list = root.find("part-list")
        part_names: dict[str, str] = {}
        if part_list is not None:
            for sp in part_list.findall("score-part"):
                pid = sp.get("id", "")
                pn = sp.find("part-name")
                part_names[pid] = pn.text if pn is not None and pn.text else pid

        for part_el in root.findall("part"):
            pid = part_el.get("id", "P1")
            part = MXLPart(
                part_id=pid,
                part_name=part_names.get(pid, pid),
                instrument=part_names.get(pid, pid),
            )
            for measure_el in part_el.findall("measure"):
                measure = self._parse_measure(measure_el)
                part.measures.append(measure)
            score.parts.append(part)

        return score

    def _parse_measure(self, el: ET.Element) -> MXLMeasure:
        num = int(el.get("number", "1"))
        measure = MXLMeasure(number=num)

        attrs = el.find("attributes")
        if attrs is not None:
            time_el = attrs.find("time")
            if time_el is not None:
                beats = time_el.find("beats")
                beat_type = time_el.find("beat-type")
                if beats is not None and beat_type is not None:
                    measure.time_sig = (int(beats.text), int(beat_type.text))
            key_el = attrs.find("key")
            if key_el is not None:
                fifths = key_el.find("fifths")
                if fifths is not None:
                    measure.key_fifths = int(fifths.text)
            clef_el = attrs.find("clef")
            if clef_el is not None:
                sign = clef_el.find("sign")
                if sign is not None:
                    measure.clef = sign.text

        direction = el.find("direction")
        if direction is not None:
            sound = direction.find("sound")
            if sound is not None and sound.get("tempo"):
                measure.tempo = int(float(sound.get("tempo")))

        for note_el in el.findall("note"):
            note = self._parse_note(note_el)
            measure.notes.append(note)

        return measure

    def _parse_note(self, el: ET.Element) -> MXLNote:
        note = MXLNote()

        if el.find("rest") is not None:
            note.is_rest = True
        else:
            pitch_el = el.find("pitch")
            if pitch_el is not None:
                step = pitch_el.find("step")
                octave = pitch_el.find("octave")
                alter = pitch_el.find("alter")
                if step is not None:
                    note.step = step.text
                if octave is not None:
                    note.octave = int(octave.text)
                if alter is not None:
                    note.accidental = int(float(alter.text))

        if el.find("chord") is not None:
            note.is_chord = True

        dur = el.find("duration")
        if dur is not None:
            note.duration = int(dur.text)

        ntype = el.find("type")
        if ntype is not None:
            note.note_type = ntype.text

        if el.find("dot") is not None:
            note.dot = True

        voice = el.find("voice")
        if voice is not None:
            note.voice = int(voice.text)

        staff = el.find("staff")
        if staff is not None:
            note.staff = int(staff.text)

        lyric = el.find("lyric")
        if lyric is not None:
            text_el = lyric.find("text")
            if text_el is not None and text_el.text:
                note.lyrics = text_el.text

        return note


class MXLWriter:
    def write(self, score: MXLScore) -> str:
        root = ET.Element("score-partwise", version="4.0")

        if score.title:
            work = ET.SubElement(root, "work")
            ET.SubElement(work, "work-title").text = score.title

        if score.composer:
            ident = ET.SubElement(root, "identification")
            creator = ET.SubElement(ident, "creator", type="composer")
            creator.text = score.composer

        part_list = ET.SubElement(root, "part-list")
        for part in score.parts:
            sp = ET.SubElement(part_list, "score-part", id=part.part_id)
            ET.SubElement(sp, "part-name").text = part.part_name

        for part in score.parts:
            part_el = ET.SubElement(root, "part", id=part.part_id)
            for measure in part.measures:
                self._write_measure(part_el, measure)

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _write_measure(self, parent: ET.Element, measure: MXLMeasure) -> None:
        mel = ET.SubElement(parent, "measure", number=str(measure.number))

        if measure.time_sig or measure.key_fifths is not None or measure.clef:
            attrs = ET.SubElement(mel, "attributes")
            if measure.key_fifths is not None:
                key = ET.SubElement(attrs, "key")
                ET.SubElement(key, "fifths").text = str(measure.key_fifths)
            if measure.time_sig:
                time = ET.SubElement(attrs, "time")
                ET.SubElement(time, "beats").text = str(measure.time_sig[0])
                ET.SubElement(time, "beat-type").text = str(measure.time_sig[1])
            if measure.clef:
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = measure.clef

        if measure.tempo:
            direction = ET.SubElement(mel, "direction")
            ET.SubElement(direction, "sound", tempo=str(measure.tempo))

        for note in measure.notes:
            self._write_note(mel, note)

    def _write_note(self, parent: ET.Element, note: MXLNote) -> None:
        nel = ET.SubElement(parent, "note")

        if note.is_chord:
            ET.SubElement(nel, "chord")

        if note.is_rest:
            ET.SubElement(nel, "rest")
        else:
            pitch = ET.SubElement(nel, "pitch")
            ET.SubElement(pitch, "step").text = note.step
            if note.accidental != 0:
                ET.SubElement(pitch, "alter").text = str(note.accidental)
            ET.SubElement(pitch, "octave").text = str(note.octave)

        ET.SubElement(nel, "duration").text = str(note.duration)
        ET.SubElement(nel, "voice").text = str(note.voice)
        ET.SubElement(nel, "type").text = note.note_type

        if note.dot:
            ET.SubElement(nel, "dot")

        if note.staff != 1:
            ET.SubElement(nel, "staff").text = str(note.staff)

        if note.lyrics:
            lyric = ET.SubElement(nel, "lyric")
            ET.SubElement(lyric, "text").text = note.lyrics
