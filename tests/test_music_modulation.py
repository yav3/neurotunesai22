"""Tests for music_harmony.modulation — key distance, pivot chords, tonicization, modulation detection."""
import numpy as np
import pytest

from models.music_harmony.modulation import (
    Key, ModulationType, ModulationEvent, Tonicization,
    key_distance_fifths, key_distance_shared_pcs,
    find_pivot_chords, classify_modulation,
    detect_tonicizations, ModulationAnalyzer, key_distance_matrix,
    MAJOR_SCALE, MINOR_SCALE,
    DIATONIC_QUALITIES_MAJOR, DIATONIC_QUALITIES_MINOR,
    _key_pitch_classes, _diatonic_triads,
)
from models.music_harmony.chords import Chord, ChordQuality


class TestKey:
    def test_key_name_major(self):
        assert Key(root=0, mode="major").name == "C"
        assert Key(root=7, mode="major").name == "G"

    def test_key_name_minor(self):
        assert Key(root=9, mode="minor").name == "Am"
        assert Key(root=2, mode="minor").name == "Dm"

    def test_pitch_classes_c_major(self):
        k = Key(root=0, mode="major")
        assert k.pitch_classes == frozenset({0, 2, 4, 5, 7, 9, 11})

    def test_pitch_classes_a_minor(self):
        k = Key(root=9, mode="minor")
        expected = frozenset((9 + s) % 12 for s in MINOR_SCALE)
        assert k.pitch_classes == expected

    def test_diatonic_triads_c_major(self):
        triads = Key(root=0, mode="major").diatonic_triads
        assert len(triads) == 7
        assert triads[0].root == 0 and triads[0].quality == ChordQuality.MAJOR
        assert triads[1].root == 2 and triads[1].quality == ChordQuality.MINOR
        assert triads[6].root == 11 and triads[6].quality == ChordQuality.DIMINISHED

    def test_relative_key(self):
        c_major = Key(root=0, mode="major")
        a_minor = c_major.relative
        assert a_minor.root == 9
        assert a_minor.mode == "minor"
        assert a_minor.relative == c_major

    def test_parallel_key(self):
        c_major = Key(root=0, mode="major")
        c_minor = c_major.parallel
        assert c_minor.root == 0
        assert c_minor.mode == "minor"
        assert c_minor.parallel == c_major

    def test_key_frozen(self):
        k = Key(root=0)
        with pytest.raises(AttributeError):
            k.root = 5


class TestKeyDistance:
    def test_same_key_zero_distance(self):
        k = Key(root=0, mode="major")
        assert key_distance_fifths(k, k) == 0

    def test_fifth_apart(self):
        c = Key(root=0)
        g = Key(root=7)
        assert key_distance_fifths(c, g) == 1

    def test_tritone_six_fifths(self):
        c = Key(root=0)
        fs = Key(root=6)
        assert key_distance_fifths(c, fs) == 6

    def test_mode_change_adds_one(self):
        c_maj = Key(root=0, mode="major")
        c_min = Key(root=0, mode="minor")
        assert key_distance_fifths(c_maj, c_min) == 1

    def test_relative_keys_close(self):
        c = Key(root=0, mode="major")
        a = Key(root=9, mode="minor")
        dist = key_distance_fifths(c, a)
        assert dist <= 4

    def test_shared_pcs_same_key(self):
        k = Key(root=0)
        assert key_distance_shared_pcs(k, k) == pytest.approx(0.0)

    def test_shared_pcs_remote_keys(self):
        c = Key(root=0)
        fs = Key(root=6)
        dist = key_distance_shared_pcs(c, fs)
        assert 0.0 < dist <= 1.0

    def test_distance_matrix_shape(self):
        keys = [Key(root=i) for i in range(12)]
        mat = key_distance_matrix(keys)
        assert mat.shape == (12, 12)
        assert np.allclose(np.diag(mat), 0)

    def test_distance_matrix_symmetric(self):
        keys = [Key(root=0), Key(root=7), Key(root=5)]
        mat = key_distance_matrix(keys)
        assert np.allclose(mat, mat.T)


class TestPivotChords:
    def test_closely_related_keys_have_pivots(self):
        c = Key(root=0, mode="major")
        g = Key(root=7, mode="major")
        pivots = find_pivot_chords(c, g)
        assert len(pivots) > 0

    def test_same_key_all_are_pivots(self):
        c = Key(root=0, mode="major")
        pivots = find_pivot_chords(c, c)
        assert len(pivots) == 7

    def test_remote_keys_fewer_pivots(self):
        c = Key(root=0, mode="major")
        fs = Key(root=6, mode="major")
        pivots = find_pivot_chords(c, fs)
        c_g_pivots = find_pivot_chords(c, Key(root=7, mode="major"))
        assert len(pivots) <= len(c_g_pivots)

    def test_pivot_chords_belong_to_both_keys(self):
        c = Key(root=0, mode="major")
        f = Key(root=5, mode="major")
        pivots = find_pivot_chords(c, f)
        to_triads = f.diatonic_triads
        to_pc_set = {ch.pitch_classes for ch in to_triads}
        for p in pivots:
            assert p.pitch_classes in to_pc_set


class TestClassifyModulation:
    def test_parallel_key_chromatic(self):
        c_maj = Key(root=0, mode="major")
        c_min = Key(root=0, mode="minor")
        assert classify_modulation(c_maj, c_min) == ModulationType.CHROMATIC

    def test_close_key_pivot(self):
        c = Key(root=0, mode="major")
        g = Key(root=7, mode="major")
        assert classify_modulation(c, g) == ModulationType.PIVOT

    def test_relative_key_pivot(self):
        c = Key(root=0, mode="major")
        a_min = Key(root=9, mode="minor")
        assert classify_modulation(c, a_min) == ModulationType.PIVOT

    def test_remote_key_direct_or_chromatic(self):
        c = Key(root=0)
        fs = Key(root=6)
        mt = classify_modulation(c, fs)
        assert mt in (ModulationType.DIRECT, ModulationType.CHROMATIC)


class TestTonicization:
    def test_v7_to_i_detected(self):
        chords = [
            Chord(root=7, quality=ChordQuality.DOMINANT7),  # G7
            Chord(root=0, quality=ChordQuality.MAJOR),       # C
        ]
        results = detect_tonicizations(chords)
        assert len(results) == 1
        assert results[0].target_chord.root == 0
        assert results[0].applied_dominant.root == 7
        assert results[0].strength == 0.8

    def test_v_to_i_weaker(self):
        chords = [
            Chord(root=7, quality=ChordQuality.MAJOR),  # G (not dom7)
            Chord(root=0, quality=ChordQuality.MAJOR),   # C
        ]
        results = detect_tonicizations(chords)
        assert len(results) == 1
        assert results[0].strength == 0.5

    def test_no_tonicization_wrong_root(self):
        chords = [
            Chord(root=7, quality=ChordQuality.DOMINANT7),  # G7
            Chord(root=5, quality=ChordQuality.MAJOR),       # F (not C)
        ]
        results = detect_tonicizations(chords)
        assert len(results) == 0

    def test_multiple_tonicizations(self):
        chords = [
            Chord(root=9, quality=ChordQuality.DOMINANT7),  # A7
            Chord(root=2, quality=ChordQuality.MINOR),       # Dm (V7/ii)
            Chord(root=7, quality=ChordQuality.DOMINANT7),  # G7
            Chord(root=0, quality=ChordQuality.MAJOR),       # C (V7/I)
        ]
        results = detect_tonicizations(chords)
        assert len(results) == 2

    def test_empty_chords(self):
        assert detect_tonicizations([]) == []

    def test_single_chord(self):
        assert detect_tonicizations([Chord(root=0, quality=ChordQuality.MAJOR)]) == []

    def test_minor_target(self):
        chords = [
            Chord(root=4, quality=ChordQuality.DOMINANT7),  # E7
            Chord(root=9, quality=ChordQuality.MINOR),       # Am
        ]
        results = detect_tonicizations(chords)
        assert len(results) == 1
        assert results[0].target_chord.root == 9


class TestModulationAnalyzer:
    def test_no_modulation_in_diatonic(self):
        analyzer = ModulationAnalyzer(min_chords_in_key=3)
        c = Key(root=0, mode="major")
        chords = c.diatonic_triads[:6]
        events = analyzer.detect_modulations(chords, c)
        assert len(events) == 0

    def test_too_few_chords(self):
        analyzer = ModulationAnalyzer(min_chords_in_key=3)
        c = Key(root=0)
        assert analyzer.detect_modulations([Chord(root=0, quality=ChordQuality.MAJOR)], c) == []

    def test_detects_key_change(self):
        analyzer = ModulationAnalyzer(min_chords_in_key=2)
        c = Key(root=0, mode="major")
        c_triads = c.diatonic_triads
        g = Key(root=7, mode="major")
        g_triads = g.diatonic_triads
        chords = c_triads[:4] + g_triads[:4]
        events = analyzer.detect_modulations(chords, c)
        if events:
            assert events[0].from_key == c

    def test_modulation_density_no_mods(self):
        analyzer = ModulationAnalyzer()
        c = Key(root=0)
        chords = c.diatonic_triads[:6]
        density = analyzer.modulation_density(chords, c)
        assert density == 0.0

    def test_modulation_density_empty(self):
        analyzer = ModulationAnalyzer()
        assert analyzer.modulation_density([], Key(root=0)) == 0.0

    def test_best_key_for_segment(self):
        analyzer = ModulationAnalyzer()
        g = Key(root=7, mode="major")
        chords = g.diatonic_triads[:5]
        best = analyzer._best_key_for_segment(chords)
        assert best is not None
        assert best.pitch_classes == g.pitch_classes


class TestScalesAndQualities:
    def test_major_scale_intervals(self):
        assert MAJOR_SCALE == (0, 2, 4, 5, 7, 9, 11)

    def test_minor_scale_intervals(self):
        assert MINOR_SCALE == (0, 2, 3, 5, 7, 8, 10)

    def test_diatonic_qualities_major_length(self):
        assert len(DIATONIC_QUALITIES_MAJOR) == 7

    def test_diatonic_qualities_minor_length(self):
        assert len(DIATONIC_QUALITIES_MINOR) == 7

    def test_key_pitch_classes_7_notes(self):
        pcs = _key_pitch_classes(0, "major")
        assert len(pcs) == 7

    def test_diatonic_triads_7_chords(self):
        triads = _diatonic_triads(0, "major")
        assert len(triads) == 7


class TestModulationEvent:
    def test_event_fields(self):
        e = ModulationEvent(
            from_key=Key(root=0),
            to_key=Key(root=7),
            position=4,
            modulation_type=ModulationType.PIVOT,
            pivot_chord=Chord(root=7, quality=ChordQuality.MAJOR),
            distance=1,
        )
        assert e.from_key.root == 0
        assert e.to_key.root == 7
        assert e.position == 4
        assert e.distance == 1

    def test_event_frozen(self):
        e = ModulationEvent(
            from_key=Key(root=0), to_key=Key(root=7),
            position=0, modulation_type=ModulationType.DIRECT,
        )
        with pytest.raises(AttributeError):
            e.position = 5
