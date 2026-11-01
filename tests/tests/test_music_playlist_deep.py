"""Deep tests for playlist analytics and generation."""
import numpy as np
import pytest

from models.music_playlist.playlist_analytics import (
    playlist_stats, PlaylistStats,
    coherence_score, surprise_curve, listening_fatigue,
    tempo_smoothness, energy_arc_fit,
    compute_flow, FlowScore,
)
from models.music_playlist.playlist_generation import (
    TrackCandidate, GenerationRules,
    generate_playlist, fill_gap, split_playlist,
)


# ── Analytics ────────────────────────────────────────────────────────

class TestPlaylistStats:
    def test_empty(self):
        s = playlist_stats(np.array([]), np.array([]), np.array([]))
        assert s.n_tracks == 0
        assert s.duration_sec == 0.0

    def test_basic(self):
        tempos = np.array([120.0, 130.0, 125.0])
        energies = np.array([0.5, 0.7, 0.6])
        durations = np.array([200.0, 210.0, 190.0])
        s = playlist_stats(tempos, energies, durations)
        assert s.n_tracks == 3
        assert s.duration_sec == pytest.approx(600.0)
        assert s.mean_tempo == pytest.approx(125.0)
        assert s.tempo_range == (120.0, 130.0)

    def test_key_entropy(self):
        tempos = np.array([120.0, 120.0])
        energies = np.array([0.5, 0.5])
        durations = np.array([200.0, 200.0])
        keys = np.array([0, 0])
        s = playlist_stats(tempos, energies, durations, keys)
        assert s.key_entropy == 0.0

    def test_key_entropy_uniform(self):
        tempos = np.ones(12) * 120
        energies = np.ones(12) * 0.5
        durations = np.ones(12) * 200
        keys = np.arange(12)
        s = playlist_stats(tempos, energies, durations, keys)
        assert s.key_entropy == pytest.approx(np.log2(12), abs=0.01)


class TestCoherence:
    def test_identical_vectors(self):
        f = np.tile([1.0, 0.5, 0.3], (5, 1))
        assert coherence_score(f) == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors(self):
        f = np.eye(4)
        assert coherence_score(f) == pytest.approx(0.0, abs=0.01)

    def test_single_track(self):
        assert coherence_score(np.array([[1.0, 2.0]])) == 1.0

    def test_range(self):
        rng = np.random.default_rng(42)
        f = rng.standard_normal((10, 5))
        c = coherence_score(f)
        assert -1.0 <= c <= 1.0


class TestSurpriseCurve:
    def test_identical_tracks(self):
        f = np.tile([1.0, 0.0], (5, 1))
        s = surprise_curve(f)
        assert s[0] == 0.0
        assert np.allclose(s[1:], 0.0, atol=1e-10)

    def test_orthogonal_transitions(self):
        f = np.eye(3)
        s = surprise_curve(f)
        assert s[0] == 0.0
        assert s[1] == pytest.approx(1.0, abs=0.01)

    def test_single(self):
        assert len(surprise_curve(np.array([[1.0]]))) == 1

    def test_empty(self):
        assert len(surprise_curve(np.zeros((0, 3)))) == 0


class TestListeningFatigue:
    def test_all_low(self):
        assert listening_fatigue(np.array([0.1, 0.2, 0.3])) == 0.0

    def test_all_high(self):
        assert listening_fatigue(np.array([0.9, 0.95, 0.85])) == 1.0

    def test_single(self):
        assert listening_fatigue(np.array([0.9])) == 0.0


class TestTempoSmoothness:
    def test_constant(self):
        assert tempo_smoothness(np.array([120.0, 120.0, 120.0])) == 1.0

    def test_large_jumps(self):
        ts = tempo_smoothness(np.array([60.0, 180.0, 60.0]))
        assert ts < 0.5

    def test_single(self):
        assert tempo_smoothness(np.array([120.0])) == 1.0


class TestEnergyArcFit:
    def test_perfect_hill(self):
        n = 20
        t = np.linspace(0, 1, n)
        energies = np.sin(np.pi * t)
        r2 = energy_arc_fit(energies, "hill")
        assert r2 == pytest.approx(1.0, abs=0.01)

    def test_perfect_ramp(self):
        n = 20
        energies = np.linspace(0, 1, n)
        r2 = energy_arc_fit(energies, "ramp")
        assert r2 == pytest.approx(1.0, abs=0.01)

    def test_short(self):
        assert energy_arc_fit(np.array([0.5]), "hill") == 0.0


class TestComputeFlow:
    def test_returns_flow_score(self):
        rng = np.random.default_rng(0)
        features = rng.standard_normal((10, 4))
        tempos = rng.uniform(100, 140, 10)
        energies = rng.uniform(0.3, 0.8, 10)
        flow = compute_flow(features, tempos, energies)
        assert isinstance(flow, FlowScore)
        assert 0.0 <= flow.overall <= 1.0


# ── Generation ───────────────────────────────────────────────────────

def _make_pool(n=20, seed=42):
    rng = np.random.default_rng(seed)
    return [
        TrackCandidate(
            track_id=i,
            tempo=rng.uniform(80, 160),
            energy=rng.uniform(0.2, 0.9),
            key=int(rng.integers(0, 12)),
            duration_sec=rng.uniform(180, 300),
        )
        for i in range(n)
    ]


class TestGeneratePlaylist:
    def test_nonempty(self):
        pool = _make_pool()
        rules = GenerationRules(max_tracks=10)
        result = generate_playlist(pool, rules)
        assert 1 <= len(result) <= 10

    def test_empty_pool(self):
        assert generate_playlist([], GenerationRules()) == []

    def test_respects_max_tracks(self):
        pool = _make_pool(50)
        rules = GenerationRules(max_tracks=5)
        result = generate_playlist(pool, rules)
        assert len(result) <= 5

    def test_seed_track(self):
        pool = _make_pool()
        rules = GenerationRules(seed_track_id=5, max_tracks=10)
        result = generate_playlist(pool, rules)
        assert result[0].track_id == 5

    def test_no_duplicates(self):
        pool = _make_pool()
        result = generate_playlist(pool, GenerationRules(max_tracks=15))
        ids = [t.track_id for t in result]
        assert len(ids) == len(set(ids))

    def test_duration_limit(self):
        pool = _make_pool(50)
        rules = GenerationRules(target_duration_sec=600, max_tracks=50)
        result = generate_playlist(pool, rules)
        total = sum(t.duration_sec for t in result)
        assert total <= 900  # some overshoot from last track is OK


class TestFillGap:
    def test_suggests_candidates(self):
        pool = _make_pool(20)
        current = pool[:5]
        remaining = pool[5:]
        suggestions = fill_gap(current, remaining, position=2)
        assert len(suggestions) <= 5
        assert all(s.track_id not in {t.track_id for t in current} for s in suggestions)

    def test_empty_pool(self):
        current = _make_pool(5)
        assert fill_gap(current, [], position=2) == []

    def test_invalid_position(self):
        pool = _make_pool(5)
        assert fill_gap(pool[:3], pool, position=-1) == []


class TestSplitPlaylist:
    def test_single_segment(self):
        pool = _make_pool(3)
        for t in pool:
            t.duration_sec = 100.0
        segments = split_playlist(pool, max_duration_sec=500)
        assert len(segments) == 1

    def test_multiple_segments(self):
        pool = _make_pool(10)
        for t in pool:
            t.duration_sec = 300.0
        segments = split_playlist(pool, max_duration_sec=700)
        assert len(segments) >= 2
        for seg in segments:
            assert len(seg) > 0

    def test_empty(self):
        assert split_playlist([], max_duration_sec=1000) == []
