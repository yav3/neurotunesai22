from __future__ import annotations

import pytest

from models.music_theory.scales import (
    CHROMATIC_NOTES, SCALE_INTERVALS, ScaleLibrary,
)
from models.music_theory.intervals import (
    INTERVAL_TABLE, IntervalCalculator,
)
from models.music_theory.chords import ChordLibrary
from models.music_theory.progressions import (
    COMMON_PROGRESSIONS, CAMELOT_MAP,
    ProgressionAnalyzer, CamelotWheel,
)
from models.music_theory.rhythm import (
    COMMON_METERS, GROOVE_TEMPLATES, RhythmAnalyzer,
)


class TestScales:

    def test_c_major_notes(self):
        s = ScaleLibrary.get_scale("C", "major")
        assert s.notes == ("C", "D", "E", "F", "G", "A", "B")

    def test_a_natural_minor(self):
        s = ScaleLibrary.get_scale("A", "natural_minor")
        assert s.notes == ("A", "B", "C", "D", "E", "F", "G")

    def test_all_12_keys_major(self):
        for root in CHROMATIC_NOTES:
            s = ScaleLibrary.get_scale(root, "major")
            assert len(s.notes) == 7
            assert s.notes[0] == root

    def test_all_scale_types_build(self):
        for st in SCALE_INTERVALS:
            s = ScaleLibrary.get_scale("C", st)
            assert len(s.notes) == len(SCALE_INTERVALS[st])

    def test_pentatonic_5_notes(self):
        assert len(ScaleLibrary.get_scale("C", "pentatonic_major").notes) == 5

    def test_blues_6_notes(self):
        assert len(ScaleLibrary.get_scale("C", "blues").notes) == 6

    def test_chromatic_12_notes(self):
        assert len(ScaleLibrary.get_scale("C", "chromatic").notes) == 12

    def test_modes_returns_7(self):
        modes = ScaleLibrary.get_modes("C")
        assert len(modes) == 7

    def test_relative_minor_of_c(self):
        assert ScaleLibrary.relative_minor("C") == "A"

    def test_relative_major_of_a(self):
        assert ScaleLibrary.relative_major("A") == "C"

    def test_relative_symmetry(self):
        for root in CHROMATIC_NOTES:
            minor = ScaleLibrary.relative_minor(root)
            assert ScaleLibrary.relative_major(minor) == root

    def test_enharmonic(self):
        assert ScaleLibrary.enharmonic_equivalent("Db") == "C#"

    def test_key_signature_c(self):
        assert ScaleLibrary.key_signature("C") == 0

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            ScaleLibrary.get_scale("C", "nonexistent")

    def test_all_scales_count(self):
        assert len(ScaleLibrary.all_scales()) == 12 * len(SCALE_INTERVALS)

    def test_frozen(self):
        s = ScaleLibrary.get_scale("C", "major")
        with pytest.raises(AttributeError):
            s.root = "D"


class TestIntervals:

    def test_unison(self):
        assert INTERVAL_TABLE[0].quality == "perfect"

    def test_octave(self):
        assert INTERVAL_TABLE[12].semitones == 12

    def test_perfect_fifth_consonant(self):
        assert IntervalCalculator.is_consonant(INTERVAL_TABLE[7])

    def test_tritone_dissonant(self):
        assert not IntervalCalculator.is_consonant(INTERVAL_TABLE[6])

    def test_interval_c_g(self):
        assert IntervalCalculator.interval_between("C", "G").semitones == 7

    def test_inversion_sums_12(self):
        for s in range(1, 12):
            inv = IntervalCalculator.invert(INTERVAL_TABLE[s])
            assert s + inv.semitones == 12

    def test_transpose_c_7(self):
        assert IntervalCalculator.transpose("C", 7) == "G"

    def test_midi_c4(self):
        assert IntervalCalculator.note_to_midi("C", 4) == 60

    def test_midi_roundtrip(self):
        note, octave = IntervalCalculator.midi_to_note(60)
        assert note == "C" and octave == 4


class TestChords:

    def test_c_major(self):
        assert ChordLibrary.build_chord("C", "major").notes == ("C", "E", "G")

    def test_c_minor(self):
        c = ChordLibrary.build_chord("C", "minor")
        assert c.notes == ("C", "D#", "G")

    def test_dom7_4_notes(self):
        assert len(ChordLibrary.build_chord("G", "dom7").notes) == 4

    def test_detect_c_major(self):
        matches = ChordLibrary.detect_chord(["C", "E", "G"])
        assert any(m.root == "C" and m.quality == "major" for m in matches)

    def test_voice_leading_positive(self):
        c = ChordLibrary.build_chord("C", "major")
        g = ChordLibrary.build_chord("G", "major")
        assert ChordLibrary.voice_leading_cost(c, g) > 0

    def test_common_tones(self):
        c = ChordLibrary.build_chord("C", "major")
        am = ChordLibrary.build_chord("A", "minor")
        assert len(ChordLibrary.common_tones(c, am)) > 0

    def test_unknown_quality_raises(self):
        with pytest.raises(ValueError):
            ChordLibrary.build_chord("C", "nonexistent")


class TestProgressions:

    def test_progressions_count(self):
        assert len(COMMON_PROGRESSIONS) >= 15

    def test_i_v_vi_iv_in_c(self):
        prog = ProgressionAnalyzer.build_progression("C", ["I", "V", "vi", "IV"])
        roots = [c.root for c in prog.chords]
        assert roots == ["C", "G", "A", "F"]

    def test_authentic_cadence(self):
        g = ChordLibrary.build_chord("G", "major")
        c = ChordLibrary.build_chord("C", "major")
        assert ProgressionAnalyzer.detect_cadence(g, c, "C") == "authentic"

    def test_half_cadence(self):
        c = ChordLibrary.build_chord("C", "major")
        g = ChordLibrary.build_chord("G", "major")
        assert ProgressionAnalyzer.detect_cadence(c, g, "C") == "half"

    def test_plagal_cadence(self):
        f = ChordLibrary.build_chord("F", "major")
        c = ChordLibrary.build_chord("C", "major")
        assert ProgressionAnalyzer.detect_cadence(f, c, "C") == "plagal"

    def test_functional_analysis(self):
        prog = ProgressionAnalyzer.build_progression("C", ["I", "IV", "V", "I"])
        funcs = ProgressionAnalyzer.functional_analysis(prog)
        assert funcs == ["tonic", "subdominant", "dominant", "tonic"]

    def test_camelot_24_keys(self):
        assert len(CAMELOT_MAP) == 24

    def test_camelot_self_compatible(self):
        assert CamelotWheel.is_compatible("C major", "C major")

    def test_camelot_neighbors(self):
        assert len(CamelotWheel.compatible_keys("C major")) >= 3

    def test_camelot_distant_incompatible(self):
        assert not CamelotWheel.is_compatible("C major", "F# major")


class TestRhythm:

    def test_4_4_simple(self):
        assert COMMON_METERS["4/4"].feel == "simple"

    def test_6_8_compound(self):
        assert COMMON_METERS["6/8"].feel == "compound"

    def test_7_8_irregular(self):
        assert COMMON_METERS["7/8"].feel == "irregular"

    def test_templates_count(self):
        assert len(GROOVE_TEMPLATES) >= 10

    def test_swing_ratio_gt_1(self):
        assert RhythmAnalyzer.swing_ratio(GROOVE_TEMPLATES["swing"]) > 1.0

    def test_straight_ratio_near_1(self):
        r = RhythmAnalyzer.swing_ratio(GROOVE_TEMPLATES["straight"])
        assert 0.9 <= r <= 1.1

    def test_syncopation_reggae(self):
        assert RhythmAnalyzer.syncopation_score(GROOVE_TEMPLATES["reggae"]) > 0.0

    def test_detect_meter(self):
        ts = RhythmAnalyzer.detect_meter([0.0, 0.5, 1.0, 1.5])
        assert ts.feel in ("simple", "compound", "irregular")

    def test_groove_similarity_identical(self):
        p = GROOVE_TEMPLATES["straight"]
        assert RhythmAnalyzer.groove_similarity(p, p) == pytest.approx(1.0)

    def test_groove_similarity_different(self):
        sim = RhythmAnalyzer.groove_similarity(
            GROOVE_TEMPLATES["straight"], GROOVE_TEMPLATES["reggae"]
        )
        assert 0.0 <= sim < 1.0
