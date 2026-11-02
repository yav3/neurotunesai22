"""Tests for playlist transition model."""
import numpy as np
import pytest
from models.music_playlist.transition_model import (
    TransitionFeatures, transition_score, key_compatibility_score,
    tempo_compatibility, energy_curve_fitness, crossfade_duration,
    PlaylistOptimizer,
)


def test_key_same():
    assert key_compatibility_score(0, 0, 0, 0) == 1.0


def test_key_adjacent():
    s = key_compatibility_score(0, 7, 0, 0)
    assert 0.7 < s < 1.0


def test_key_distant():
    s = key_compatibility_score(0, 6, 0, 0)
    assert s < 0.7


def test_tempo_identical():
    assert tempo_compatibility(120, 120) > 0.99


def test_tempo_double():
    assert tempo_compatibility(120, 60) > 0.7


def test_tempo_far():
    assert tempo_compatibility(60, 180) < 0.5


def test_transition_score_range():
    feat = TransitionFeatures(0, 1.0, 0, 0, 0)
    s = transition_score(feat)
    assert 0 <= s <= 1


def test_energy_ramp_up():
    energies = [0.1, 0.3, 0.5, 0.7, 0.9]
    f = energy_curve_fitness(energies, "ramp_up")
    assert f > 0.9


def test_energy_chill_down():
    energies = [0.9, 0.7, 0.5, 0.3, 0.1]
    f = energy_curve_fitness(energies, "chill_down")
    assert f > 0.9


def test_energy_short():
    assert energy_curve_fitness([0.5], "wave") == 0.0


def test_crossfade_range():
    d = crossfade_duration(120, 120, 0.5, 0.5)
    assert 1.0 <= d <= 16.0


def test_optimizer_total():
    rng = np.random.default_rng(42)
    scores = rng.random((5, 5))
    np.fill_diagonal(scores, 0)
    opt = PlaylistOptimizer(scores)
    order = opt.optimize()
    assert len(order) == 5
    assert set(order) == set(range(5))


def test_optimizer_improves():
    rng = np.random.default_rng(42)
    scores = rng.random((8, 8))
    np.fill_diagonal(scores, 0)
    opt = PlaylistOptimizer(scores)
    nn = opt.nearest_neighbor_order()
    improved = opt.two_opt_improve(nn)
    assert opt.total_score(improved) >= opt.total_score(nn) - 1e-10


def test_tempo_zero():
    assert tempo_compatibility(0, 120) == 0.0
