"""Tests for the music_playlist module: transitions, energy curves, optimizer."""
from __future__ import annotations

import pytest
import numpy as np

from models.music_playlist.transition import (
    TransitionType,
    TransitionScore,
    TransitionScorer,
    CAMELOT_NUMBERS,
)
from models.music_playlist.energy_curve import (
    EnergyCurve,
    EnergyCurveShape,
    EnergyCurveDesigner,
)
from models.music_playlist.playlist_optimizer import (
    TrackSlot,
    PlaylistConstraints,
    PlaylistOptimizer,
)


# ---------------------------------------------------------------------------
# Transition scoring
# ---------------------------------------------------------------------------

class TestCamelotWheel:
    def test_all_positions(self):
        assert len(CAMELOT_NUMBERS) == 25  # 24 + C# minor alias

    def test_key_names_present(self):
        assert "A minor" in CAMELOT_NUMBERS
        assert "C major" in CAMELOT_NUMBERS
        assert "Ab minor" in CAMELOT_NUMBERS

    def test_parse_tuple(self):
        assert CAMELOT_NUMBERS["A minor"] == (8, "A")
        assert CAMELOT_NUMBERS["A major"] == (11, "B")


class TestTransitionScorer:
    @pytest.fixture
    def scorer(self):
        return TransitionScorer()

    def _track(self, bpm=128.0, key="A minor", energy=0.7, valence=0.5, arousal=0.5):
        return {"bpm": bpm, "key": key, "energy": energy, "valence": valence, "arousal": arousal}

    def test_same_key_perfect(self, scorer):
        a = self._track(bpm=128, key="A minor", energy=0.7)
        b = self._track(bpm=128, key="A minor", energy=0.7)
        s = scorer.score(a, b)
        assert s.overall > 0.8
        assert s.transition_type == TransitionType.SAME_KEY

    def test_adjacent_key(self, scorer):
        a = self._track(key="A minor")
        b = self._track(key="E minor")
        s = scorer.score(a, b)
        assert s.harmonic > 0.6

    def test_parallel_key(self, scorer):
        a = self._track(key="A minor")
        b = self._track(key="C major")
        s = scorer.score(a, b)
        assert s.harmonic > 0.5

    def test_incompatible_key(self, scorer):
        a = self._track(key="Ab minor")
        b = self._track(key="F major")
        s = scorer.score(a, b)
        assert s.harmonic < 0.5

    def test_tempo_match(self, scorer):
        a = self._track(bpm=128)
        b = self._track(bpm=130)
        s = scorer.score(a, b)
        assert s.tempo > 0.8

    def test_double_time(self, scorer):
        a = self._track(bpm=128)
        b = self._track(bpm=64)
        s = scorer.score(a, b)
        assert s.bpm_ratio >= 0.5

    def test_energy_boost(self, scorer):
        a = self._track(energy=0.4)
        b = self._track(energy=0.8)
        s = scorer.score(a, b)
        assert s.energy > 0.3

    def test_mood_similar(self, scorer):
        a = self._track(valence=0.8, arousal=0.7)
        b = self._track(valence=0.75, arousal=0.65)
        s = scorer.score(a, b)
        assert s.mood > 0.8

    def test_score_range(self, scorer):
        a = self._track()
        b = self._track(bpm=200, key="6B", energy=0.1, valence=0.1)
        s = scorer.score(a, b)
        assert 0.0 <= s.overall <= 1.0

    def test_rank_transitions(self, scorer):
        anchor = self._track(bpm=128, key="8A", energy=0.7)
        cands = [
            self._track(bpm=128, key="8A", energy=0.7),
            self._track(bpm=90, key="3B", energy=0.2),
            self._track(bpm=130, key="9A", energy=0.75),
        ]
        ranked = scorer.rank_transitions(anchor, cands)
        assert len(ranked) == 3
        scores = [s.overall for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Energy curves
# ---------------------------------------------------------------------------

class TestEnergyCurveDesigner:
    @pytest.fixture
    def designer(self):
        return EnergyCurveDesigner()

    @pytest.mark.parametrize("shape", list(EnergyCurveShape))
    def test_all_shapes_valid(self, designer, shape):
        if shape == EnergyCurveShape.CUSTOM:
            return
        c = designer.design(shape, 10, 0.2, 1.0)
        assert c.n_slots == 10
        assert len(c.values) == 10
        assert all(0.0 <= v <= 1.2 for v in c.values)

    def test_linear_up_monotone(self, designer):
        c = designer.design(EnergyCurveShape.LINEAR_UP, 8, 0.2, 1.0)
        for i in range(len(c.values) - 1):
            assert c.values[i] <= c.values[i + 1] + 1e-9

    def test_linear_down_monotone(self, designer):
        c = designer.design(EnergyCurveShape.LINEAR_DOWN, 8, 0.2, 1.0)
        for i in range(len(c.values) - 1):
            assert c.values[i] >= c.values[i + 1] - 1e-9

    def test_peak_shape(self, designer):
        c = designer.design(EnergyCurveShape.PEAK, 10, 0.2, 1.0)
        peak_idx = np.argmax(c.values)
        assert 2 <= peak_idx <= 7

    def test_at_method(self, designer):
        c = designer.design(EnergyCurveShape.PLATEAU, 5, 0.3, 0.9)
        for i in range(5):
            assert c.at(i) == c.values[i]

    def test_tolerance_at(self, designer):
        c = designer.design(EnergyCurveShape.PLATEAU, 5, 0.3, 0.9)
        lo, hi = c.tolerance_at(2, 0.1)
        assert lo <= c.at(2) + 1e-6
        assert hi >= c.at(2) - 1e-6

    def test_fit_curve_to_tracks(self, designer):
        energies = [0.3, 0.5, 0.7, 0.9, 0.8, 0.6]
        best_shape, best_corr = designer.fit_curve_to_tracks(energies)
        assert isinstance(best_shape, EnergyCurveShape)
        assert -1.0 <= best_corr <= 1.0

    def test_single_slot(self, designer):
        c = designer.design(EnergyCurveShape.LINEAR_UP, 1, 0.2, 1.0)
        assert c.n_slots == 1
        assert len(c.values) == 1


# ---------------------------------------------------------------------------
# Playlist optimizer
# ---------------------------------------------------------------------------

class TestPlaylistOptimizer:
    @pytest.fixture
    def optimizer(self):
        return PlaylistOptimizer()

    def _candidates(self, n=20):
        rng = np.random.default_rng(42)
        keys = ["8A", "9A", "7A", "8B", "10A", "5A", "3B"]
        artists = ["Artist_A", "Artist_B", "Artist_C", "Artist_D", "Artist_E"]
        out = []
        for i in range(n):
            out.append({
                "bpm": float(rng.uniform(110, 140)),
                "key": keys[i % len(keys)],
                "energy": float(rng.uniform(0.2, 1.0)),
                "valence": float(rng.uniform(0.1, 0.9)),
                "arousal": float(rng.uniform(0.1, 0.9)),
                "artist": artists[i % len(artists)],
                "title": f"Track_{i}",
            })
        return out

    def test_basic_optimize(self, optimizer):
        cands = self._candidates()
        constraints = PlaylistConstraints(n_tracks=8)
        slots = optimizer.optimize(cands, constraints, seed=1)
        assert len(slots) == 8
        assert all(isinstance(s, TrackSlot) for s in slots)

    def test_no_duplicate_tracks(self, optimizer):
        cands = self._candidates()
        constraints = PlaylistConstraints(n_tracks=10)
        slots = optimizer.optimize(cands, constraints, seed=2)
        titles = [s.track["title"] for s in slots]
        assert len(titles) == len(set(titles))

    def test_respects_n_tracks(self, optimizer):
        cands = self._candidates(5)
        constraints = PlaylistConstraints(n_tracks=5)
        slots = optimizer.optimize(cands, constraints, seed=3)
        assert len(slots) == 5

    def test_more_requested_than_available(self, optimizer):
        cands = self._candidates(3)
        constraints = PlaylistConstraints(n_tracks=10)
        slots = optimizer.optimize(cands, constraints, seed=4)
        assert len(slots) == 3

    def test_empty_candidates(self, optimizer):
        slots = optimizer.optimize([], PlaylistConstraints(), seed=5)
        assert slots == []

    def test_evaluate_returns_metrics(self, optimizer):
        cands = self._candidates()
        constraints = PlaylistConstraints(n_tracks=8)
        slots = optimizer.optimize(cands, constraints, seed=6)
        metrics = optimizer.evaluate(slots)
        assert "overall" in metrics
        assert "energy_fit" in metrics
        assert "transition_avg" in metrics
        assert "flow" in metrics
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_evaluate_empty(self, optimizer):
        metrics = optimizer.evaluate([])
        assert metrics["overall"] == 0.0

    def test_energy_curve_followed(self, optimizer):
        cands = self._candidates(30)
        constraints = PlaylistConstraints(
            n_tracks=10,
            energy_shape=EnergyCurveShape.LINEAR_UP,
            min_energy=0.2,
            max_energy=1.0,
        )
        slots = optimizer.optimize(cands, constraints, seed=7)
        metrics = optimizer.evaluate(slots)
        assert metrics["energy_fit"] > 0.3

    def test_key_lock(self, optimizer):
        cands = self._candidates(20)
        constraints = PlaylistConstraints(n_tracks=5, key_lock="8A")
        slots = optimizer.optimize(cands, constraints, seed=8)
        if slots:
            assert slots[0].track.get("key") == "8A"

    def test_transitions_scored(self, optimizer):
        cands = self._candidates()
        constraints = PlaylistConstraints(n_tracks=6)
        slots = optimizer.optimize(cands, constraints, seed=9)
        assert slots[0].transition_from_prev is None
        for s in slots[1:]:
            assert s.transition_from_prev is not None
            assert 0.0 <= s.transition_from_prev.overall <= 1.0
