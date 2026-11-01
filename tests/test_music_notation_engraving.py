"""Tests for notation engraving algorithms."""
import numpy as np
import pytest

from models.music_notation.engraving import (
    StaffMetrics, compute_note_spacing, stem_direction, beam_slope,
    collision_detect, lyric_hyphenation_points, page_break_penalty,
    accidental_stacking, tie_arc_height, optimal_system_breaks,
)


class TestNoteSpacing:
    def test_proportional(self):
        durations = np.array([0.25, 0.5, 1.0])
        spacing = compute_note_spacing(durations)
        assert len(spacing) == 3
        assert spacing[2] > spacing[1] > spacing[0]

    def test_empty(self):
        assert len(compute_note_spacing(np.array([]))) == 0

    def test_positive(self):
        spacing = compute_note_spacing(np.array([0.125, 0.25, 0.5, 1.0, 2.0]))
        assert np.all(spacing > 0)


class TestStemDirection:
    def test_above_middle(self):
        dirs = stem_direction(np.array([75, 80]))
        assert np.all(dirs == -1)

    def test_below_middle(self):
        dirs = stem_direction(np.array([60, 65]))
        assert np.all(dirs == 1)

    def test_empty(self):
        assert len(stem_direction(np.array([]))) == 0


class TestBeamSlope:
    def test_ascending(self):
        positions = np.array([60.0, 62.0, 64.0, 66.0])
        slope = beam_slope(positions)
        assert slope > 0

    def test_flat(self):
        positions = np.array([60.0, 60.0, 60.0])
        assert beam_slope(positions) == pytest.approx(0.0)

    def test_clamped(self):
        positions = np.array([60.0, 70.0])
        slope = beam_slope(positions)
        assert abs(slope) <= 0.5

    def test_single(self):
        assert beam_slope(np.array([60.0])) == 0.0


class TestCollisionDetect:
    def test_overlapping(self):
        boxes = [(0, 0, 2, 2), (1, 1, 3, 3)]
        assert len(collision_detect(boxes)) == 1

    def test_no_overlap(self):
        boxes = [(0, 0, 1, 1), (2, 2, 3, 3)]
        assert len(collision_detect(boxes)) == 0

    def test_empty(self):
        assert collision_detect([]) == []


class TestLyricHyphenation:
    def test_alignment(self):
        syllables = ["hel", "lo", "world"]
        x_pos = np.array([1.0, 2.0, 4.0])
        result = lyric_hyphenation_points(syllables, x_pos)
        np.testing.assert_array_equal(result, x_pos)


class TestPageBreak:
    def test_ideal(self):
        measures = [4, 4, 4]
        assert page_break_penalty(measures, ideal=4) == pytest.approx(0.0)

    def test_uneven(self):
        measures = [2, 6]
        assert page_break_penalty(measures, ideal=4) > 0.0

    def test_empty(self):
        assert page_break_penalty([]) == 0.0


class TestAccidentalStacking:
    def test_no_conflict(self):
        columns = accidental_stacking(np.array([60, 64, 68]))
        assert np.all(columns == 0)

    def test_adjacent_conflict(self):
        columns = accidental_stacking(np.array([60, 61]))
        assert np.max(columns) >= 1


class TestTieArc:
    def test_min_height(self):
        assert tie_arc_height(0.0) == 0.5

    def test_grows_with_distance(self):
        assert tie_arc_height(10.0) > tie_arc_height(1.0)


class TestSystemBreaks:
    def test_fits_in_one(self):
        widths = np.array([2.0, 2.0, 2.0])
        assert optimal_system_breaks(widths, page_width=10.0) == []

    def test_needs_break(self):
        widths = np.array([5.0, 5.0, 5.0])
        breaks = optimal_system_breaks(widths, page_width=8.0)
        assert len(breaks) >= 1

    def test_empty(self):
        assert optimal_system_breaks(np.array([]), page_width=10.0) == []
