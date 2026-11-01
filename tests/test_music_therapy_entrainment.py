"""Tests for rhythmic entrainment in music therapy."""
import numpy as np
import pytest

from models.music_therapy.rhythmic_entrainment import (
    kuramoto_phase_coupling, phase_locking_value,
    adaptive_tempo_prescription, rhythmic_complexity_for_therapy,
    entrainment_trajectory, iso_principle_trajectory, EntrainmentState,
)


class TestKuramoto:
    def test_basic_coupling(self):
        n = 200
        theta_stim = np.linspace(0, 4 * np.pi, n)
        theta_resp = np.linspace(0, 3.5 * np.pi, n)
        result = kuramoto_phase_coupling(theta_stim, theta_resp, coupling_k=2.0)
        assert len(result) == n

    def test_empty(self):
        assert len(kuramoto_phase_coupling(np.array([]), np.array([]))) == 0


class TestPLV:
    def test_locked_phases(self):
        phase = np.linspace(0, 10 * np.pi, 500)
        plv = phase_locking_value(phase, phase)
        assert plv == pytest.approx(1.0, abs=0.01)

    def test_random_phases(self):
        rng = np.random.default_rng(0)
        p1 = rng.uniform(0, 2 * np.pi, 500)
        p2 = rng.uniform(0, 2 * np.pi, 500)
        plv = phase_locking_value(p1, p2)
        assert plv < 0.3

    def test_empty(self):
        assert phase_locking_value(np.array([]), np.array([])) == 0.0


class TestAdaptiveTempo:
    def test_endpoints(self):
        tempos = adaptive_tempo_prescription(60.0, 80.0, steps=5)
        assert len(tempos) == 5
        assert tempos[0] == pytest.approx(60.0)
        assert tempos[-1] == pytest.approx(80.0)

    def test_monotonic_increase(self):
        tempos = adaptive_tempo_prescription(60.0, 80.0, steps=10)
        assert np.all(np.diff(tempos) > 0)


class TestRhythmicComplexity:
    def test_regular(self):
        ioi = np.ones(20) * 0.5
        result = rhythmic_complexity_for_therapy(ioi)
        assert result["regularity"] == pytest.approx(1.0, abs=0.01)

    def test_irregular(self):
        rng = np.random.default_rng(0)
        ioi = rng.uniform(0.1, 1.0, 20)
        result = rhythmic_complexity_for_therapy(ioi)
        assert result["regularity"] < 0.8

    def test_short(self):
        result = rhythmic_complexity_for_therapy(np.array([0.5]))
        assert result["regularity"] == 1.0


class TestEntrainmentTrajectory:
    def test_converges(self):
        state = entrainment_trajectory(2.0, 1.8, coupling=2.0, n_steps=500)
        assert isinstance(state, EntrainmentState)
        assert state.phase_coherence > 0.5

    def test_weak_coupling(self):
        state = entrainment_trajectory(2.0, 1.0, coupling=0.01, n_steps=50)
        assert state.coupling_strength == 0.01


class TestISOPrinciple:
    def test_endpoints(self):
        trajectory = iso_principle_trajectory(60.0, 80.0, session_minutes=10.0)
        assert trajectory[0] == pytest.approx(60.0)
        assert trajectory[-1] > trajectory[0]

    def test_monotonic(self):
        trajectory = iso_principle_trajectory(120.0, 80.0)
        assert np.all(np.diff(trajectory) < 0)
