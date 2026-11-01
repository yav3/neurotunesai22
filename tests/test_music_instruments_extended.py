"""Tests for instrument ranges and articulation sub-modules."""
import numpy as np
import pytest
from models.music_instruments.instrument_ranges import (
    INSTRUMENT_RANGES, note_in_range, comfortable_range,
    transpose_for_instrument, find_instruments_for_range, range_overlap,
)
from models.music_instruments.articulation import (
    ArticulationType, ARTICULATION_PROFILES, apply_articulation,
    detect_articulation, articulation_compatibility,
)

SR = 16000


class TestInstrumentRanges:
    def test_piano_range(self):
        assert note_in_range("piano", 60)
        assert note_in_range("piano", 21)
        assert not note_in_range("piano", 20)

    def test_violin_range(self):
        assert note_in_range("violin", 55)
        assert not note_in_range("violin", 54)

    def test_unknown_instrument(self):
        assert not note_in_range("theremin", 60)

    def test_comfortable_range_piano(self):
        lo, hi = comfortable_range("piano")
        assert lo == 28
        assert hi == 96

    def test_comfortable_range_unknown(self):
        assert comfortable_range("theremin") == (0, 0)

    def test_transpose_for_instrument(self):
        notes = np.array([80, 82, 84, 86, 88])
        out = transpose_for_instrument(notes, "cello")
        assert np.all(out <= 84)

    def test_transpose_preserves_intervals(self):
        notes = np.array([60.0, 64.0, 67.0])
        out = transpose_for_instrument(notes, "violin")
        intervals = np.diff(out)
        np.testing.assert_allclose(intervals, [4.0, 3.0])

    def test_find_instruments_for_range_c4_c5(self):
        result = find_instruments_for_range(60, 72)
        assert "piano" in result
        assert "violin" in result

    def test_find_instruments_narrow(self):
        result = find_instruments_for_range(21, 108)
        assert "piano" in result
        assert "violin" not in result

    def test_range_overlap_symmetric(self):
        o1 = range_overlap("violin", "viola")
        o2 = range_overlap("viola", "violin")
        assert abs(o1 - o2) < 1e-10

    def test_range_overlap_bounds(self):
        o = range_overlap("piano", "violin")
        assert 0.0 <= o <= 1.0

    def test_range_overlap_same(self):
        o = range_overlap("piano", "piano")
        assert abs(o - 1.0) < 0.01

    def test_at_least_20_instruments(self):
        assert len(INSTRUMENT_RANGES) >= 20


class TestArticulation:
    def test_all_profiles_present(self):
        for at in ArticulationType:
            assert at in ARTICULATION_PROFILES

    def test_apply_staccato_preserves_length(self):
        sig = np.sin(np.linspace(0, 2 * np.pi, SR))
        out = apply_articulation(sig, SR, ArticulationType.STACCATO)
        assert len(out) == len(sig)

    def test_apply_legato(self):
        sig = np.ones(SR)
        out = apply_articulation(sig, SR, ArticulationType.LEGATO)
        assert len(out) == SR

    def test_apply_tremolo_modulates(self):
        sig = np.ones(SR)
        out = apply_articulation(sig, SR, ArticulationType.TREMOLO)
        assert np.std(out) > 0.1

    def test_apply_empty_signal(self):
        out = apply_articulation(np.array([]), SR, ArticulationType.LEGATO)
        assert len(out) == 0

    def test_detect_returns_valid(self):
        sig = np.random.default_rng(42).standard_normal(SR)
        results = detect_articulation(sig, SR)
        for t, art, conf in results:
            assert t >= 0
            assert isinstance(art, ArticulationType)
            assert 0 <= conf <= 1.0

    def test_compatibility_pizzicato_violin(self):
        c = articulation_compatibility("violin", ArticulationType.PIZZICATO)
        assert c == 1.0

    def test_compatibility_pizzicato_flute(self):
        c = articulation_compatibility("flute", ArticulationType.PIZZICATO)
        assert c < 0.5

    def test_compatibility_legato_any(self):
        c = articulation_compatibility("piano", ArticulationType.LEGATO)
        assert c > 0

    def test_compatibility_flutter_tongue_woodwind(self):
        c = articulation_compatibility("flute", ArticulationType.FLUTTER_TONGUE)
        assert c == 1.0
