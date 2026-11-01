"""Tests for session recording and polymetric analysis modules."""
import math
import numpy as np
import pytest

from models.music_collaboration.session_recording import (
    EventType, SessionEvent, SessionRecording,
    record_event, events_in_range, player_activity,
    activity_density, player_contribution, interaction_matrix,
    session_fingerprint, merge_recordings, tempo_changes, note_events_only,
)
from models.music_rhythm.polymetric import (
    Meter, polymetric_cycle_length, coincidence_points,
    coincidence_density, metric_dissonance, hemiola_ratio,
    is_hemiola, beat_pattern, combined_pattern,
    metric_complexity, grouping_conflict,
)


# ── Session Recording ────────────────────────────────────────────────

class TestSessionRecording:
    def _make_recording(self):
        rec = SessionRecording(session_id="s1", start_time=0.0)
        for i in range(10):
            record_event(rec, SessionEvent(
                timestamp=float(i), event_type=EventType.NOTE_ON,
                player_id="alice" if i % 2 == 0 else "bob",
            ))
        return rec

    def test_duration(self):
        rec = self._make_recording()
        assert rec.duration == 9.0

    def test_player_ids(self):
        rec = self._make_recording()
        assert rec.player_ids == {"alice", "bob"}

    def test_event_count(self):
        rec = self._make_recording()
        assert rec.event_count == 10

    def test_events_in_range(self):
        rec = self._make_recording()
        sub = events_in_range(rec, 2.0, 5.0)
        assert all(2.0 <= e.timestamp <= 5.0 for e in sub)
        assert len(sub) == 4

    def test_player_activity(self):
        rec = self._make_recording()
        alice = player_activity(rec, "alice")
        assert all(e.player_id == "alice" for e in alice)
        assert len(alice) == 5

    def test_activity_density(self):
        rec = self._make_recording()
        d = activity_density(rec, window_sec=3.0)
        assert len(d) == 3
        assert sum(d) == 10

    def test_player_contribution(self):
        rec = self._make_recording()
        c = player_contribution(rec)
        assert c["alice"] == pytest.approx(0.5)
        assert c["bob"] == pytest.approx(0.5)

    def test_empty_contribution(self):
        rec = SessionRecording(session_id="empty")
        assert player_contribution(rec) == {}

    def test_interaction_matrix(self):
        rec = self._make_recording()
        mat = interaction_matrix(rec, window_sec=1.5)
        assert isinstance(mat, dict)
        key = tuple(sorted(["alice", "bob"]))
        assert key in mat
        assert mat[key] > 0

    def test_fingerprint_deterministic(self):
        rec = self._make_recording()
        fp1 = session_fingerprint(rec)
        fp2 = session_fingerprint(rec)
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_merge_preserves_events(self):
        r1 = SessionRecording(session_id="a", start_time=0.0)
        record_event(r1, SessionEvent(0.5, EventType.NOTE_ON, "p1"))
        r2 = SessionRecording(session_id="b", start_time=1.0)
        record_event(r2, SessionEvent(1.5, EventType.NOTE_ON, "p2"))
        merged = merge_recordings(r1, r2)
        assert merged.event_count == 2
        assert merged.start_time == 0.0

    def test_tempo_changes(self):
        rec = SessionRecording(session_id="t")
        record_event(rec, SessionEvent(0.0, EventType.TEMPO_CHANGE, "sys", {"bpm": 120.0}))
        record_event(rec, SessionEvent(4.0, EventType.TEMPO_CHANGE, "sys", {"bpm": 140.0}))
        record_event(rec, SessionEvent(2.0, EventType.NOTE_ON, "p1"))
        tc = tempo_changes(rec)
        assert len(tc) == 2
        assert tc[0] == (0.0, 120.0)
        assert tc[1] == (4.0, 140.0)

    def test_note_events_only(self):
        rec = SessionRecording(session_id="n")
        record_event(rec, SessionEvent(0.0, EventType.NOTE_ON, "p1"))
        record_event(rec, SessionEvent(0.5, EventType.CHAT, "p2"))
        record_event(rec, SessionEvent(1.0, EventType.NOTE_OFF, "p1"))
        notes = note_events_only(rec)
        assert len(notes) == 2


# ── Polymetric Analysis ──────────────────────────────────────────────

class TestMeter:
    def test_pulse_count(self):
        m = Meter(beats=4, subdivision=3)
        assert m.pulse_count == 12

    def test_default_subdivision(self):
        m = Meter(beats=3)
        assert m.pulse_count == 3


class TestPolymetricCycle:
    def test_3_against_4(self):
        m1 = Meter(beats=3)
        m2 = Meter(beats=4)
        assert polymetric_cycle_length(m1, m2) == 12

    def test_same_meter(self):
        m = Meter(beats=4)
        assert polymetric_cycle_length(m, m) == 4

    def test_coprime(self):
        m1 = Meter(beats=5)
        m2 = Meter(beats=7)
        assert polymetric_cycle_length(m1, m2) == 35


class TestCoincidence:
    def test_same_meter_all_coincide(self):
        m = Meter(beats=4)
        pts = coincidence_points(m, m)
        assert pts == [0]

    def test_3_4_coincidence(self):
        m1 = Meter(beats=3)
        m2 = Meter(beats=4)
        pts = coincidence_points(m1, m2)
        assert 0 in pts

    def test_density_same_meter(self):
        m = Meter(beats=4)
        assert coincidence_density(m, m) == pytest.approx(1.0 / 4.0)

    def test_dissonance_same(self):
        m = Meter(beats=4)
        d = metric_dissonance(m, m)
        assert 0.0 <= d <= 1.0


class TestHemiola:
    def test_3_2_hemiola(self):
        m1 = Meter(beats=3)
        m2 = Meter(beats=2)
        assert is_hemiola(m1, m2) is True
        assert hemiola_ratio(m1, m2) == (3, 2)

    def test_not_hemiola(self):
        m1 = Meter(beats=5)
        m2 = Meter(beats=4)
        assert is_hemiola(m1, m2) is False


class TestBeatPattern:
    def test_pattern_length(self):
        m = Meter(beats=3)
        p = beat_pattern(m, 12)
        assert len(p) == 12
        assert p[0] is True
        assert p[3] is True

    def test_combined_pattern(self):
        m1 = Meter(beats=3)
        m2 = Meter(beats=4)
        cp = combined_pattern(m1, m2)
        assert len(cp) == 12
        assert cp[0] == (True, True)


class TestMetricComplexity:
    def test_single_meter(self):
        assert metric_complexity([Meter(beats=4)]) == 0.0

    def test_two_meters(self):
        c = metric_complexity([Meter(beats=3), Meter(beats=4)])
        assert 0.0 < c <= 1.0

    def test_three_meters(self):
        c = metric_complexity([Meter(beats=3), Meter(beats=4), Meter(beats=5)])
        assert 0.0 < c <= 1.0


class TestGroupingConflict:
    def test_identical_has_cross_pairs(self):
        a = [0.0, 0.5, 1.0]
        c = grouping_conflict(a, a)
        assert 0.0 <= c <= 1.0

    def test_empty(self):
        assert grouping_conflict([], [1.0]) == 0.0

    def test_high_conflict(self):
        a = [0.0, 1.0, 2.0]
        b = [0.5, 1.5, 2.5]
        c = grouping_conflict(a, b, tolerance=0.01)
        assert c > 0.5
