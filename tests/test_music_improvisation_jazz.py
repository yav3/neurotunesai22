"""Tests for jazz harmony module."""
import numpy as np
import pytest
from models.music_improvisation.jazz_harmony import (
    ChordTone, chord_tones, available_tensions, guide_tone_line,
    approach_note, ii_V_I_voicings, tritone_substitution,
    backdoor_progression, coltrane_changes, tension_resolution_score,
)


def test_chord_tones_maj7():
    ct = chord_tones(ChordTone(root=0, quality="maj7"))
    np.testing.assert_array_equal(ct, [0, 4, 7, 11])


def test_chord_tones_min7():
    ct = chord_tones(ChordTone(root=0, quality="min7"))
    np.testing.assert_array_equal(ct, [0, 3, 7, 10])


def test_tensions_nonempty():
    t = available_tensions(ChordTone(root=0, quality="dom7"), scale_root=0, scale_type="mixolydian")
    assert len(t) > 0


def test_guide_tone_smooth():
    chords = [ChordTone(2, "min7"), ChordTone(7, "dom7"), ChordTone(0, "maj7")]
    line = guide_tone_line(chords)
    assert len(line) == 3
    for pc in line:
        assert 0 <= pc < 12


def test_approach_chromatic():
    notes = approach_note(5, "chromatic")
    assert notes[-1] == 5
    assert notes[0] == 4


def test_approach_enclosure():
    notes = approach_note(5, "enclosure")
    assert len(notes) == 3
    assert notes[-1] == 5


def test_ii_V_I_three_chords():
    voicings = ii_V_I_voicings(0, "major")
    assert len(voicings) == 3
    for v in voicings:
        assert len(v) == 4


def test_tritone_sub():
    c = ChordTone(root=7, quality="dom7")
    sub = tritone_substitution(c)
    assert sub.root == 1
    assert sub.quality == "dom7"


def test_backdoor():
    target = ChordTone(root=0, quality="maj7")
    prog = backdoor_progression(target)
    assert len(prog) == 2
    assert prog[0].root == 10


def test_coltrane_six_chords():
    start = ChordTone(root=0, quality="maj7")
    changes = coltrane_changes(start)
    assert len(changes) == 6
    roots = [c.root for c in changes]
    assert 0 in roots and 4 in roots and 8 in roots


def test_tension_chord_tone():
    s = tension_resolution_score(0, ChordTone(0, "maj7"))
    assert s == 1.0


def test_tension_avoid():
    s = tension_resolution_score(1, ChordTone(0, "maj7"))
    assert s < 0.5


def test_tension_available():
    ct = ChordTone(0, "dom7")
    tensions = available_tensions(ct, 0, "mixolydian")
    if len(tensions) > 0:
        s = tension_resolution_score(int(tensions[0]), ct)
        assert s >= 0.7


def test_guide_tone_empty():
    assert guide_tone_line([]) == []
