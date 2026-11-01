"""Tests for world dance-rhythm pattern analysis."""
import numpy as np
import pytest

from models.music_cultural.dance_rhythm import (
    DancePattern, DANCE_CATALOG, pattern_to_binary, accent_to_weight,
    pattern_similarity, classify_dance, generate_onset_times,
    syncopation_score, swing_feel_from_pattern, tempo_in_range,
)


class TestCatalog:
    def test_catalog_nonempty(self):
        assert len(DANCE_CATALOG) >= 15

    def test_waltz_triple(self):
        w = DANCE_CATALOG["waltz"]
        assert w.time_signature == (3, 4)
        assert 1 in w.accents

    def test_samba_duple(self):
        s = DANCE_CATALOG["samba"]
        assert s.time_signature[0] in (2, 4)

    def test_all_have_onsets(self):
        for name, pat in DANCE_CATALOG.items():
            assert len(pat.onsets) > 0, f"{name} has no onsets"
            assert len(pat.accents) > 0, f"{name} has no accents"


class TestBinaryConversion:
    def test_waltz(self):
        w = DANCE_CATALOG["waltz"]
        b = pattern_to_binary(w)
        assert b[0] == 1
        assert len(b) == w.cycle_pulses

    def test_round_trip(self):
        pat = DANCE_CATALOG["reggae_one_drop"]
        b = pattern_to_binary(pat)
        assert sum(b) == len(pat.onsets)


class TestAccentWeight:
    def test_accent_positions_present(self):
        pat = DANCE_CATALOG["waltz"]
        weights = accent_to_weight(pat)
        assert len(weights) == pat.cycle_pulses
        assert weights[pat.accents[0] - 1] > 0


class TestSimilarity:
    def test_identical(self):
        p = DANCE_CATALOG["waltz"]
        assert pattern_similarity(p, p) == pytest.approx(1.0)

    def test_different(self):
        a = DANCE_CATALOG["waltz"]
        b = DANCE_CATALOG["samba"]
        sim = pattern_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_symmetric(self):
        a = DANCE_CATALOG["tango"]
        b = DANCE_CATALOG["polka"]
        assert pattern_similarity(a, b) == pytest.approx(pattern_similarity(b, a))


class TestClassify:
    def test_waltz_tempo(self):
        w = DANCE_CATALOG["waltz"]
        onsets = pattern_to_binary(w)
        result = classify_dance(onsets, tempo_bpm=90.0)
        assert result[0].name.lower() == "waltz"

    def test_returns_sorted(self):
        pat = DANCE_CATALOG["samba"]
        onsets = pattern_to_binary(pat)
        result = classify_dance(onsets, tempo_bpm=100.0)
        scores = [r.confidence for r in result]
        assert scores == sorted(scores, reverse=True)


class TestOnsetTimes:
    def test_length(self):
        pat = DANCE_CATALOG["waltz"]
        times = generate_onset_times(pat, tempo_bpm=90.0, n_bars=2)
        assert len(times) == 2 * len(pat.onsets)

    def test_monotonic(self):
        pat = DANCE_CATALOG["samba"]
        times = generate_onset_times(pat, tempo_bpm=100.0, n_bars=3)
        assert all(times[i] < times[i + 1] for i in range(len(times) - 1))


class TestSyncopation:
    def test_on_beat_low(self):
        pat = DancePattern("on_beat", "test", (4, 4), 4, (1, 2, 3, 4), (1,), (120, 120), 0.0)
        assert syncopation_score(pat) == pytest.approx(0.0)

    def test_offbeat_higher(self):
        off = DancePattern("off", "test", (4, 4), 8, (2, 4, 6, 8), (2,), (120, 120), 0.0)
        on = DancePattern("on", "test", (4, 4), 8, (1, 3, 5, 7), (1,), (120, 120), 0.0)
        assert syncopation_score(off) > syncopation_score(on)


class TestTempoRange:
    def test_in_range(self):
        pat = DANCE_CATALOG["waltz"]
        lo, hi = pat.tempo_range_bpm
        mid = (lo + hi) / 2
        assert tempo_in_range(pat, mid)

    def test_out_of_range(self):
        pat = DANCE_CATALOG["waltz"]
        assert not tempo_in_range(pat, 300.0)
