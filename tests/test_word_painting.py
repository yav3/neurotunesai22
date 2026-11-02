"""Tests for word painting module."""
import numpy as np
import pytest
from models.music_linguistics.word_painting import (
    detect_painting_words, direction_to_contour, dynamic_curve_from_gestures,
    suggest_painting, painting_density, melodic_interval_from_direction,
    tempo_adjustment, detect_word_repetition, onomatopoeia_gestures,
    DIRECTION_WORDS, SPEED_WORDS,
)


class TestDetection:
    def test_rise(self):
        gestures = detect_painting_words("I will rise up to heaven")
        directions = [g for g in gestures if g.category == "direction"]
        assert len(directions) >= 2
        assert all(g.direction == 1 for g in directions)

    def test_fall(self):
        gestures = detect_painting_words("fall down to the ground")
        directions = [g for g in gestures if g.category == "direction"]
        assert len(directions) >= 2
        assert all(g.direction == -1 for g in directions)

    def test_speed(self):
        gestures = detect_painting_words("run fast and then rest")
        speed = [g for g in gestures if g.category == "speed"]
        assert len(speed) >= 2

    def test_dynamic(self):
        gestures = detect_painting_words("whisper soft then shout loud")
        dynamic = [g for g in gestures if g.category == "dynamic"]
        assert len(dynamic) >= 2

    def test_empty(self):
        assert detect_painting_words("") == []

    def test_no_painting_words(self):
        gestures = detect_painting_words("the table is brown")
        assert len(gestures) == 0


class TestContour:
    def test_ascending(self):
        gestures = detect_painting_words("rise soar fly")
        contour = direction_to_contour(gestures, n_notes=8)
        assert contour[-1] > contour[0]

    def test_descending(self):
        gestures = detect_painting_words("fall sink drop")
        contour = direction_to_contour(gestures, n_notes=8)
        assert contour[-1] < contour[0]

    def test_empty_flat(self):
        contour = direction_to_contour([], n_notes=8)
        assert np.all(contour == 0)


class TestDynamicCurve:
    def test_loud(self):
        gestures = detect_painting_words("thunder roar shout")
        curve = dynamic_curve_from_gestures(gestures)
        assert np.mean(curve) > 0.7

    def test_soft(self):
        gestures = detect_painting_words("whisper quiet hush")
        curve = dynamic_curve_from_gestures(gestures)
        assert np.mean(curve) < 0.3


class TestSuggest:
    def test_basic(self):
        result = suggest_painting("rise up to the sky and shout loud")
        assert len(result.gestures) > 0
        assert result.melodic_contour is not None
        assert result.dynamic_curve is not None

    def test_density_metric(self):
        d = painting_density("rise up to the sky and fall down below")
        assert d > 0.3

    def test_no_painting(self):
        d = painting_density("the table is brown and square")
        assert d == 0.0


class TestMelodicInterval:
    def test_up(self):
        iv = melodic_interval_from_direction(1, 0.9)
        assert iv > 0

    def test_down(self):
        iv = melodic_interval_from_direction(-1, 0.5)
        assert iv < 0


class TestTempo:
    def test_fast(self):
        gestures = detect_painting_words("rush fast hurry")
        t = tempo_adjustment(gestures)
        assert t > 1.0

    def test_slow(self):
        gestures = detect_painting_words("slow calm rest")
        t = tempo_adjustment(gestures)
        assert t < 1.0

    def test_none(self):
        assert tempo_adjustment([]) == 1.0


class TestRepetition:
    def test_detect(self):
        text = "love love love love love forever"
        reps = detect_word_repetition(text, min_count=3)
        assert len(reps) >= 1
        assert reps[0].word == "love"
        assert reps[0].count == 5

    def test_no_repetition(self):
        reps = detect_word_repetition("one two three four five")
        assert len(reps) == 0


class TestOnomatopoeia:
    def test_bang(self):
        gestures = onomatopoeia_gestures("bang crash boom")
        assert len(gestures) == 3

    def test_none(self):
        assert onomatopoeia_gestures("the cat sat") == []
