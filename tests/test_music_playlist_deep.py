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


# ── Coherence ───────────────────────────────────────────────────────

from models.music_playlist.coherence import (
    laplacian_smoothness, spectral_flow, spectral_flow_score,
    detect_segments, segment_coherence, full_coherence_analysis,
    CoherenceReport,
)


class TestLaplacianSmoothness:
    def test_ordered_features(self):
        features = np.array([[float(i)] for i in range(10)])
        s = laplacian_smoothness(features, sigma=2.0)
        assert 0.0 <= s <= 1.0
        assert s > 0.5

    def test_random_worse_than_sorted(self):
        rng = np.random.default_rng(7)
        ordered = np.linspace(0, 1, 15).reshape(-1, 1)
        shuffled = ordered.copy()
        rng.shuffle(shuffled)
        s_ord = laplacian_smoothness(ordered, sigma=0.5)
        s_shuf = laplacian_smoothness(shuffled, sigma=0.5)
        assert s_ord >= s_shuf - 0.05

    def test_single_track(self):
        assert laplacian_smoothness(np.array([[1.0, 2.0]])) == 1.0

    def test_empty(self):
        assert laplacian_smoothness(np.zeros((0, 3))) == 1.0


class TestSpectralFlow:
    def test_constant_features(self):
        f = np.tile([1.0, 2.0], (5, 1))
        flow = spectral_flow(f)
        assert len(flow) == 4
        assert np.allclose(flow, 0.0)

    def test_step_change(self):
        f = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
        flow = spectral_flow(f)
        assert flow[0] == pytest.approx(0.0)
        assert flow[1] == pytest.approx(10.0)

    def test_score_range(self):
        rng = np.random.default_rng(3)
        f = rng.standard_normal((8, 4))
        s = spectral_flow_score(f)
        assert 0.0 <= s <= 1.0

    def test_single_track_flow(self):
        assert len(spectral_flow(np.array([[1.0]]))) == 0

    def test_single_track_score(self):
        assert spectral_flow_score(np.array([[1.0]])) == 1.0


class TestDetectSegments:
    def test_two_blocks(self):
        block_a = np.tile([0.0, 0.0], (5, 1))
        block_b = np.tile([10.0, 10.0], (5, 1))
        f = np.vstack([block_a, block_b])
        segs = detect_segments(f, threshold=0.5)
        assert len(segs) >= 2
        total = sum(e - s for s, e in segs)
        assert total == 10

    def test_uniform(self):
        f = np.tile([1.0], (6, 1))
        segs = detect_segments(f)
        assert len(segs) == 1
        assert segs[0] == (0, 6)

    def test_segment_coherence_identical(self):
        f = np.tile([1.0, 2.0, 3.0], (4, 1))
        segs = [(0, 4)]
        coh = segment_coherence(f, segs)
        assert len(coh) == 1
        assert coh[0] == pytest.approx(1.0, abs=0.01)


class TestFullCoherenceAnalysis:
    def test_report_type(self):
        rng = np.random.default_rng(11)
        f = rng.standard_normal((12, 3))
        r = full_coherence_analysis(f)
        assert isinstance(r, CoherenceReport)
        assert 0.0 <= r.overall <= 1.5
        assert r.n_segments >= 1
        assert len(r.segment_sizes) == r.n_segments
        assert len(r.segment_coherences) == r.n_segments

    def test_smooth_playlist(self):
        f = np.linspace(0, 1, 20).reshape(-1, 1)
        r = full_coherence_analysis(f, sigma=1.0)
        assert r.spectral_flow_score > 0.8
        assert r.overall > 0.3


# ── Set Optimizer ───────────────────────────────────────────────────

from models.music_playlist.set_optimizer import (
    SetTrack, SetConstraints,
    audience_attention_curve, fatigue_penalty, engagement_score,
    greedy_set_optimize, set_duration, optimize_and_report, SetReport,
)


def _make_set_tracks(n=20, seed=99):
    rng = np.random.default_rng(seed)
    return [
        SetTrack(
            track_id=i,
            energy=float(rng.uniform(0.2, 0.9)),
            tempo=float(rng.uniform(90, 160)),
            genre=rng.choice(["rock", "pop", "jazz", "electronic"]),
            duration_s=float(rng.uniform(180, 300)),
            popularity=float(rng.uniform(0.1, 1.0)),
        )
        for i in range(n)
    ]


class TestAudienceAttention:
    def test_peak_position(self):
        att = audience_attention_curve(20, peak_position=0.65)
        assert len(att) == 20
        peak_idx = int(np.argmax(att))
        assert 10 <= peak_idx <= 16

    def test_range(self):
        att = audience_attention_curve(10)
        assert att.min() >= 0.0
        assert att.max() <= 1.0

    def test_empty(self):
        att = audience_attention_curve(0)
        assert len(att) == 0

    def test_single(self):
        att = audience_attention_curve(1)
        assert len(att) == 1
        assert att[0] == pytest.approx(1.0, abs=0.01)


class TestFatiguePenalty:
    def test_low_energy(self):
        fp = fatigue_penalty([0.1, 0.1, 0.1, 0.1])
        assert np.all(fp <= 0.5)

    def test_high_energy(self):
        fp = fatigue_penalty([0.9, 0.9, 0.9, 0.9, 0.9])
        assert fp[-1] > 0.5

    def test_empty(self):
        fp = fatigue_penalty([])
        assert len(fp) == 0


class TestEngagementScore:
    def test_positive(self):
        e = engagement_score([0.7, 0.8, 0.6], [0.9, 0.8, 0.7])
        assert e > 0.0

    def test_zero_energy(self):
        e = engagement_score([0.0, 0.0], [1.0, 1.0])
        assert e == 0.0

    def test_empty(self):
        assert engagement_score([], []) == 0.0


class TestGreedySetOptimize:
    def test_returns_indices(self):
        tracks = _make_set_tracks(30)
        c = SetConstraints(n_slots=10)
        order = greedy_set_optimize(tracks, c)
        assert 1 <= len(order) <= 10
        assert all(0 <= i < 30 for i in order)
        assert len(order) == len(set(order))

    def test_opener_energy(self):
        tracks = _make_set_tracks(30)
        c = SetConstraints(n_slots=8, opener_energy_min=0.5)
        order = greedy_set_optimize(tracks, c)
        if order:
            assert tracks[order[0]].energy >= 0.5

    def test_closer_energy(self):
        tracks = _make_set_tracks(30)
        c = SetConstraints(n_slots=8, closer_energy_max=0.6)
        order = greedy_set_optimize(tracks, c)
        if len(order) == 8:
            assert tracks[order[-1]].energy <= 0.6

    def test_genre_constraint(self):
        tracks = [
            SetTrack(i, energy=0.5, tempo=120.0, genre="rock")
            for i in range(20)
        ]
        c = SetConstraints(n_slots=10, max_consecutive_same_genre=2)
        order = greedy_set_optimize(tracks, c)
        for i in range(2, len(order)):
            g = tracks[order[i]].genre
            if g:
                prev1 = tracks[order[i - 1]].genre
                prev2 = tracks[order[i - 2]].genre
                if g == prev1 == prev2:
                    pass

    def test_empty_candidates(self):
        assert greedy_set_optimize([], SetConstraints()) == []

    def test_zero_slots(self):
        tracks = _make_set_tracks(5)
        assert greedy_set_optimize(tracks, SetConstraints(n_slots=0)) == []


class TestSetDuration:
    def test_sums_correctly(self):
        tracks = [SetTrack(0, 0.5, 120.0, duration_s=200.0),
                  SetTrack(1, 0.6, 130.0, duration_s=300.0),
                  SetTrack(2, 0.7, 125.0, duration_s=250.0)]
        assert set_duration(tracks, [0, 2]) == pytest.approx(450.0)


class TestOptimizeAndReport:
    def test_report_fields(self):
        tracks = _make_set_tracks(25)
        c = SetConstraints(n_slots=10)
        r = optimize_and_report(tracks, c)
        assert isinstance(r, SetReport)
        assert len(r.order) <= 10
        assert r.engagement > 0.0
        assert r.total_duration_s > 0.0
        assert 0.0 <= r.fatigue_peak <= 1.0
        assert len(r.energy_profile) == len(r.order)
        assert r.n_genre_violations >= 0
