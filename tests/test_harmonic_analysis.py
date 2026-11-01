"""Tests for harmonic analysis module."""
import numpy as np
import pytest
from models.music_analysis.harmonic_analysis import (
    chroma_from_pitches, identify_chord, to_roman_numeral,
    classify_function, analyze_progression, functional_tension_curve,
    detect_cadence, harmonic_rhythm, chord_complexity,
    progression_complexity, modulation_points, ChordLabel,
    CHORD_TEMPLATES, template_matching_chord,
)


class TestChroma:
    def test_c_major(self):
        chroma = chroma_from_pitches(np.array([60, 64, 67]))
        assert chroma[0] > 0  # C
        assert chroma[4] > 0  # E
        assert chroma[7] > 0  # G

    def test_empty(self):
        chroma = chroma_from_pitches(np.array([]))
        assert np.all(chroma == 0)

    def test_normalized(self):
        chroma = chroma_from_pitches(np.array([60, 60, 64, 67]))
        assert abs(chroma.sum() - 1.0) < 1e-10


class TestChordIdentification:
    def test_c_major(self):
        chord = identify_chord(np.array([60, 64, 67]))
        assert chord.root == 0
        assert chord.quality == "maj"

    def test_a_minor(self):
        chord = identify_chord(np.array([57, 60, 64]))
        assert chord.root == 9  # A
        assert chord.quality == "min"

    def test_g_dom7(self):
        chord = identify_chord(np.array([55, 59, 62, 65]))
        assert chord.quality in ("dom7", "min7", "maj")

    def test_pitch_classes(self):
        chord = ChordLabel(root=0, quality="maj")
        pcs = chord.pitch_classes
        assert 0 in pcs and 4 in pcs and 7 in pcs


class TestRomanNumeral:
    def test_I_in_C(self):
        chord = ChordLabel(root=0, quality="maj")
        rn = to_roman_numeral(chord, key=0)
        assert rn.degree == "I"

    def test_V_in_C(self):
        chord = ChordLabel(root=7, quality="maj")
        rn = to_roman_numeral(chord, key=0)
        assert rn.degree == "V"

    def test_iv_in_A_minor(self):
        chord = ChordLabel(root=2, quality="min")
        rn = to_roman_numeral(chord, key=9, mode="minor")
        assert rn.degree.lower() in ("iv", "v")


class TestFunction:
    def test_tonic(self):
        chord = ChordLabel(root=0, quality="maj")
        rn = to_roman_numeral(chord, key=0)
        func = classify_function(rn)
        assert func.name == "tonic"

    def test_dominant(self):
        chord = ChordLabel(root=7, quality="maj")
        rn = to_roman_numeral(chord, key=0)
        func = classify_function(rn)
        assert func.name == "dominant"


class TestProgression:
    def test_basic(self):
        chords = [
            np.array([60, 64, 67]),  # C
            np.array([65, 69, 72]),  # F
            np.array([67, 71, 74]),  # G
            np.array([60, 64, 67]),  # C
        ]
        times = np.array([0, 1, 2, 3])
        steps = analyze_progression(chords, times, key=0)
        assert len(steps) == 4
        assert steps[0].function.name == "tonic"

    def test_tension_curve(self):
        chords = [np.array([60, 64, 67])] * 4
        times = np.arange(4)
        steps = analyze_progression(chords, times, key=0)
        curve = functional_tension_curve(steps)
        assert len(curve) == 4


class TestCadence:
    def test_authentic(self):
        chords = [
            np.array([60, 64, 67]),  # I
            np.array([67, 71, 74]),  # V
            np.array([60, 64, 67]),  # I
        ]
        steps = analyze_progression(chords, np.arange(3), key=0)
        cadences = detect_cadence(steps)
        types = [c[1] for c in cadences]
        assert "authentic" in types


class TestHarmonicRhythm:
    def test_basic(self):
        chords = [np.array([60, 64, 67])] * 4
        times = np.array([0.0, 1.0, 2.0, 3.0])
        steps = analyze_progression(chords, times, key=0)
        hr = harmonic_rhythm(steps)
        assert len(hr) == 3
        np.testing.assert_allclose(hr, [1.0, 1.0, 1.0])


class TestComplexity:
    def test_triad_simple(self):
        chord = ChordLabel(root=0, quality="maj")
        c = chord_complexity(chord)
        assert 0 <= c <= 1

    def test_dim7_more_complex(self):
        maj = ChordLabel(root=0, quality="maj")
        dim7 = ChordLabel(root=0, quality="dim7")
        assert chord_complexity(dim7) >= chord_complexity(maj)

    def test_progression(self):
        chords = [np.array([60, 64, 67]), np.array([65, 69, 72])]
        steps = analyze_progression(chords, np.arange(2), key=0)
        c = progression_complexity(steps)
        assert 0 <= c <= 1


class TestModulation:
    def test_no_modulation(self):
        chords = [np.array([60, 64, 67])] * 10
        steps = analyze_progression(chords, np.arange(10), key=0)
        mods = modulation_points(steps)
        assert len(mods) == 0
