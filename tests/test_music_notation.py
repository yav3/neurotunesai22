"""Tests for the music_notation module."""
import math
import pytest
import numpy as np

from models.music_notation.abc_parser import (
    ABCHeader, ABCNote, ABCTune, ABCParser, ABCWriter, SEMITONE_MAP,
)
from models.music_notation.musicxml import (
    MXLNote, MXLMeasure, MXLPart, MXLScore, MXLParser, MXLWriter,
)
from models.music_notation.score_analysis import (
    ScoreStatistics, VoiceLeading, MelodicAnalysis, HarmonicRhythm,
)
from models.music_notation.staff_layout import (
    Clef, StaffLine, NoteHead, Beam, StaffLayout,
)
from models.music_notation.transposition import (
    Transposer, InstrumentTransposition, KeySignatureTransposer,
    EnharmonicSpeller,
)


# --- ABC Parser ---

class TestABCNote:
    def test_midi_c4(self):
        n = ABCNote(pitch="C", octave=4, accidental=0)
        assert n.midi_number() == 60

    def test_midi_a4(self):
        n = ABCNote(pitch="A", octave=4)
        assert n.midi_number() == 69

    def test_midi_sharp(self):
        n = ABCNote(pitch="F", octave=4, accidental=1)
        assert n.midi_number() == 66

    def test_midi_rest(self):
        n = ABCNote(pitch="z", is_rest=True)
        assert n.midi_number() == -1

    def test_frequency_a4(self):
        n = ABCNote(pitch="A", octave=4)
        assert abs(n.frequency() - 440.0) < 0.01

    def test_frequency_rest(self):
        n = ABCNote(pitch="z", is_rest=True)
        assert n.frequency() == 0.0


class TestABCParser:
    SIMPLE_ABC = """X: 1
T: Test Tune
C: J.S. Bach
M: 4/4
L: 1/8
Q: 120
K: C
C D E F | G A B c |
"""

    def test_parse_header(self):
        parser = ABCParser()
        tune = parser.parse(self.SIMPLE_ABC)
        assert tune.header.title == "Test Tune"
        assert tune.header.composer == "J.S. Bach"
        assert tune.header.meter == "4/4"
        assert tune.header.key == "C"
        assert tune.header.tempo == 120

    def test_parse_notes(self):
        parser = ABCParser()
        tune = parser.parse(self.SIMPLE_ABC)
        assert tune.note_count() >= 8
        assert tune.notes[0].pitch == "C"
        assert tune.notes[0].octave == 4

    def test_parse_lowercase_higher_octave(self):
        abc = "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\nc d e\n"
        parser = ABCParser()
        tune = parser.parse(abc)
        assert tune.notes[0].octave == 5

    def test_parse_rest(self):
        abc = "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\nC z D\n"
        parser = ABCParser()
        tune = parser.parse(abc)
        rests = [n for n in tune.notes if n.is_rest]
        assert len(rests) >= 1

    def test_parse_accidental(self):
        abc = "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\n^F _B\n"
        parser = ABCParser()
        tune = parser.parse(abc)
        assert tune.notes[0].accidental == 1
        assert tune.notes[1].accidental == -1

    def test_parse_duration_multiplier(self):
        abc = "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\nC2 D/2\n"
        parser = ABCParser()
        tune = parser.parse(abc)
        assert tune.notes[0].duration == 0.25
        assert abs(tune.notes[1].duration - 0.0625) < 1e-9

    def test_parse_octave_marks(self):
        abc = "X:1\nT:Test\nM:4/4\nL:1/8\nK:C\nC, C c c'\n"
        parser = ABCParser()
        tune = parser.parse(abc)
        assert tune.notes[0].octave == 3
        assert tune.notes[1].octave == 4
        assert tune.notes[2].octave == 5
        assert tune.notes[3].octave == 6


class TestABCWriter:
    def test_roundtrip(self):
        abc = "X:1\nT:Roundtrip\nM:4/4\nL:1/8\nK:C\nC D E F\n"
        parser = ABCParser()
        tune = parser.parse(abc)
        writer = ABCWriter()
        output = writer.write(tune)
        assert "T: Roundtrip" in output
        assert "K: C" in output

    def test_write_rest(self):
        tune = ABCTune()
        tune.notes = [ABCNote(pitch="z", duration=0.125, is_rest=True)]
        writer = ABCWriter()
        out = writer.write(tune)
        assert "z" in out

    def test_write_accidental(self):
        tune = ABCTune()
        tune.notes = [ABCNote(pitch="F", duration=0.125, accidental=1, octave=4)]
        writer = ABCWriter()
        out = writer.write(tune)
        assert "^F" in out


class TestABCTune:
    def test_total_duration(self):
        tune = ABCTune()
        tune.notes = [
            ABCNote(pitch="C", duration=1.0),
            ABCNote(pitch="D", duration=0.5),
        ]
        assert tune.total_duration() == 1.5

    def test_pitch_range(self):
        tune = ABCTune()
        tune.notes = [
            ABCNote(pitch="C", octave=4),
            ABCNote(pitch="G", octave=5),
        ]
        lo, hi = tune.pitch_range()
        assert lo == 60
        assert hi == 79


# --- MusicXML ---

SAMPLE_MXL = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <work><work-title>Test Score</work-title></work>
  <identification>
    <creator type="composer">Test Composer</creator>
  </identification>
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>1</voice>
      </note>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>1</voice>
      </note>
      <note>
        <pitch><step>G</step><octave>4</octave></pitch>
        <duration>1</duration><type>quarter</type><voice>1</voice>
      </note>
      <note>
        <rest/><duration>1</duration><type>quarter</type><voice>1</voice>
      </note>
    </measure>
  </part>
</score-partwise>"""


class TestMXLNote:
    def test_midi_c4(self):
        n = MXLNote(step="C", octave=4)
        assert n.midi_number() == 60

    def test_midi_rest(self):
        n = MXLNote(is_rest=True)
        assert n.midi_number() == -1

    def test_ticks_quarter(self):
        n = MXLNote(note_type="quarter")
        assert n.ticks() == 1024

    def test_ticks_dotted_half(self):
        n = MXLNote(note_type="half", dot=True)
        assert n.ticks() == 3072


class TestMXLParser:
    def test_parse_title(self):
        score = MXLParser().parse(SAMPLE_MXL)
        assert score.title == "Test Score"

    def test_parse_composer(self):
        score = MXLParser().parse(SAMPLE_MXL)
        assert score.composer == "Test Composer"

    def test_parse_parts(self):
        score = MXLParser().parse(SAMPLE_MXL)
        assert len(score.parts) == 1
        assert score.parts[0].part_name == "Piano"

    def test_parse_notes(self):
        score = MXLParser().parse(SAMPLE_MXL)
        m = score.parts[0].measures[0]
        assert m.note_count() == 3
        assert m.rest_count() == 1

    def test_parse_attributes(self):
        score = MXLParser().parse(SAMPLE_MXL)
        m = score.parts[0].measures[0]
        assert m.time_sig == (4, 4)
        assert m.key_fifths == 0
        assert m.clef == "G"


class TestMXLWriter:
    def test_roundtrip(self):
        score = MXLParser().parse(SAMPLE_MXL)
        xml_out = MXLWriter().write(score)
        score2 = MXLParser().parse(xml_out)
        assert score2.title == "Test Score"
        assert score2.total_notes() == 3

    def test_write_contains_elements(self):
        score = MXLParser().parse(SAMPLE_MXL)
        xml_out = MXLWriter().write(score)
        assert "<score-partwise" in xml_out
        assert "<part-name>Piano</part-name>" in xml_out


# --- Score Analysis ---

class TestScoreStatistics:
    def test_from_score(self):
        score = MXLParser().parse(SAMPLE_MXL)
        stats = ScoreStatistics.from_score(score)
        assert stats.note_count == 3
        assert stats.rest_count == 1
        assert stats.rest_fraction == 0.25
        assert stats.pitch_range() > 0

    def test_density(self):
        score = MXLParser().parse(SAMPLE_MXL)
        stats = ScoreStatistics.from_score(score)
        assert len(stats.density_per_measure) == 1
        assert stats.density_per_measure[0] == 3.0


class TestVoiceLeading:
    def _n(self, step, octave):
        return MXLNote(step=step, octave=octave)

    def test_parallel_fifths(self):
        v1 = [self._n("C", 4), self._n("D", 4)]
        v2 = [self._n("G", 4), self._n("A", 4)]
        pf = VoiceLeading.detect_parallel_fifths(v1, v2)
        assert len(pf) == 1

    def test_no_parallel_fifths(self):
        v1 = [self._n("C", 4), self._n("D", 4)]
        v2 = [self._n("E", 4), self._n("A", 4)]
        pf = VoiceLeading.detect_parallel_fifths(v1, v2)
        assert len(pf) == 0

    def test_voice_crossing(self):
        upper = [self._n("C", 4), self._n("D", 3)]
        lower = [self._n("G", 3), self._n("G", 3)]
        crossings = VoiceLeading.detect_voice_crossing(upper, lower)
        assert 1 in crossings

    def test_spacing(self):
        voices = [
            [self._n("C", 5), self._n("D", 5)],
            [self._n("C", 4), self._n("D", 4)],
        ]
        sp = VoiceLeading.spacing_analysis(voices)
        assert len(sp) == 2
        assert sp[0][0] == 12


class TestMelodicAnalysis:
    def _n(self, step, octave):
        return MXLNote(step=step, octave=octave)

    def test_interval_histogram(self):
        notes = [self._n("C", 4), self._n("E", 4), self._n("G", 4)]
        hist = MelodicAnalysis.interval_histogram(notes)
        assert 4 in hist
        assert 3 in hist

    def test_contour_ascending(self):
        notes = [self._n("C", 4), self._n("D", 4), self._n("E", 4),
                 self._n("F", 4), self._n("G", 4), self._n("A", 4)]
        assert MelodicAnalysis.contour(notes) == "ascending"

    def test_contour_descending(self):
        notes = [self._n("A", 4), self._n("G", 4), self._n("F", 4),
                 self._n("E", 4), self._n("D", 4), self._n("C", 4)]
        assert MelodicAnalysis.contour(notes) == "descending"

    def test_motivic_patterns(self):
        notes = [self._n("C", 4), self._n("D", 4), self._n("E", 4),
                 self._n("C", 4), self._n("D", 4), self._n("E", 4)]
        patterns = MelodicAnalysis.motivic_patterns(notes, n_gram=2)
        assert len(patterns) >= 1
        assert patterns[0][1] >= 2

    def test_leap_step_ratio(self):
        notes = [self._n("C", 4), self._n("G", 4), self._n("A", 4)]
        ratio = MelodicAnalysis.leap_step_ratio(notes)
        assert ratio == 0.5


class TestHarmonicRhythm:
    def _measure(self, steps):
        notes = [MXLNote(step=s, octave=4, duration=1) for s in steps]
        return MXLMeasure(notes=notes)

    def test_chord_change_rate(self):
        m = self._measure(["C", "C", "E", "G"])
        rates = HarmonicRhythm.chord_change_rate([m])
        assert rates[0] >= 2

    def test_harmonic_density(self):
        m = self._measure(["C", "E", "G", "B"])
        density = HarmonicRhythm.harmonic_density([m])
        assert density == 4.0

    def test_average_chord_duration(self):
        m = self._measure(["C", "C", "E", "E"])
        avg = HarmonicRhythm.average_chord_duration([m])
        assert avg == 2.0


# --- Staff Layout ---

class TestStaffLayout:
    def test_compute_y_middle_c_treble(self):
        layout = StaffLayout(clef=Clef.TREBLE)
        staff = StaffLine(y_position=0.0)
        y = layout.compute_y("B", 4, staff)
        assert y == staff.middle_line_y()

    def test_ledger_lines_high(self):
        layout = StaffLayout(clef=Clef.TREBLE)
        staff = StaffLine(y_position=0.0)
        y = layout.compute_y("C", 6, staff)
        ledgers = layout.needs_ledger_lines(y, staff)
        assert len(ledgers) >= 1

    def test_stem_direction(self):
        layout = StaffLayout(clef=Clef.TREBLE)
        staff = StaffLine(y_position=0.0)
        y_high = layout.compute_y("A", 5, staff)
        y_low = layout.compute_y("C", 4, staff)
        assert layout.stem_direction(y_high, staff) == "down"
        assert layout.stem_direction(y_low, staff) == "up"

    def test_layout_notes(self):
        layout = StaffLayout(clef=Clef.TREBLE)
        staff = StaffLine(y_position=0.0)
        notes = [("C", 4, 0, "quarter"), ("E", 4, 0, "quarter"), ("G", 4, 0, "quarter")]
        heads = layout.layout_notes(notes, staff)
        assert len(heads) == 3
        assert heads[0].y_position < heads[1].y_position < heads[2].y_position

    def test_beam_grouping(self):
        heads = [
            NoteHead(duration_type="eighth"),
            NoteHead(duration_type="eighth"),
            NoteHead(duration_type="quarter"),
            NoteHead(duration_type="eighth"),
            NoteHead(duration_type="eighth"),
        ]
        beams = StaffLayout.group_beams(heads, beam_grouping=2)
        assert len(beams) == 2
        assert beams[0].length() == 2


class TestStaffLine:
    def test_line_positions(self):
        staff = StaffLine(y_position=100.0)
        assert len(staff.line_positions) == 5
        assert staff.line_positions[0] == 100.0
        assert staff.line_positions[4] == 140.0

    def test_middle_line(self):
        staff = StaffLine(y_position=0.0)
        assert staff.middle_line_y() == 20.0


# --- Transposition ---

class TestTransposer:
    def test_transpose_up_octave(self):
        t = Transposer(12)
        note = ABCNote(pitch="C", octave=4)
        out = t.transpose_abc_note(note)
        assert out.pitch == "C"
        assert out.octave == 5

    def test_transpose_up_fifth(self):
        t = Transposer(7)
        note = ABCNote(pitch="C", octave=4)
        out = t.transpose_abc_note(note)
        assert out.pitch == "G"
        assert out.octave == 4

    def test_transpose_rest_unchanged(self):
        t = Transposer(5)
        note = ABCNote(pitch="z", is_rest=True)
        out = t.transpose_abc_note(note)
        assert out.is_rest

    def test_transpose_tune_key(self):
        tune = ABCTune()
        tune.header = ABCHeader(key="C")
        tune.notes = [ABCNote(pitch="C", octave=4, duration=0.125)]
        t = Transposer(2)
        out = t.transpose_tune(tune)
        assert out.header.key == "D"

    def test_transpose_mxl_note(self):
        t = Transposer(4)
        n = MXLNote(step="C", octave=4, duration=1, note_type="quarter")
        out = t.transpose_mxl_note(n)
        assert out.step == "E"
        assert out.octave == 4


class TestInstrumentTransposition:
    def test_bb_concert_to_written(self):
        it = InstrumentTransposition("Bb")
        concert_c = MXLNote(step="C", octave=4)
        written = it.concert_to_written(concert_c)
        assert written.midi_number() == 62

    def test_bb_written_to_concert(self):
        it = InstrumentTransposition("Bb")
        written_c = MXLNote(step="C", octave=4)
        concert = it.written_to_concert(written_c)
        assert concert.midi_number() == 58

    def test_f_horn(self):
        it = InstrumentTransposition("F")
        concert_c = MXLNote(step="C", octave=4)
        written = it.concert_to_written(concert_c)
        assert written.midi_number() == 67


class TestKeySignatureTransposer:
    def test_transpose_c_up_fifth(self):
        assert KeySignatureTransposer.transpose("C", 7) == "G"

    def test_transpose_g_up_whole(self):
        assert KeySignatureTransposer.transpose("G", 2) == "A"

    def test_fifths_count(self):
        assert KeySignatureTransposer.fifths_count("G") == 1
        assert KeySignatureTransposer.fifths_count("F") == -1
        assert KeySignatureTransposer.fifths_count("C") == 0

    def test_key_from_fifths(self):
        assert KeySignatureTransposer.key_from_fifths(0) == "C"
        assert KeySignatureTransposer.key_from_fifths(1) == "G"
        assert KeySignatureTransposer.key_from_fifths(-1) == "F"
        assert KeySignatureTransposer.key_from_fifths(0, minor=True) == "Am"


class TestEnharmonicSpeller:
    def test_sharp_key(self):
        speller = EnharmonicSpeller("G")
        step, octave, acc = speller.spell(66)
        assert step == "F" and acc == 1

    def test_flat_key(self):
        speller = EnharmonicSpeller("Eb")
        step, octave, acc = speller.spell(63)
        assert step == "E" and acc == -1

    def test_spell_note_name(self):
        speller = EnharmonicSpeller("D")
        name = speller.spell_note_name(66)
        assert name == "F#4"

    def test_respell(self):
        speller = EnharmonicSpeller("Bb")
        step, octave, acc = speller.respell("A", 1, 4)
        assert step == "B" and acc == -1
