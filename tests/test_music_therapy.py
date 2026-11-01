"""Tests for music_therapy module."""

import numpy as np
import pytest

from models.music_therapy.iso_principle import (
    iso_match_tempo, iso_match_mode, emotional_trajectory, iso_energy_curve,
)
from models.music_therapy.entrainment import (
    EntropicCoupling, phase_coupling_strength, entrainment_time,
    pulse_synchronization, kuramoto_order_parameter,
)
from models.music_therapy.receptive_methods import (
    relaxation_playlist_score, guided_imagery_tempo_profile,
    song_discussion_features, session_arc_quality,
)
from models.music_therapy.group_dynamics import (
    rhythmic_cohesion, turn_taking_balance, collective_tempo_drift,
    synchronization_index,
)
from models.music_therapy.session_protocol import (
    build_protocol, TherapyGoal, SessionPhase, arousal_regulation_protocol,
)


class TestIso:
    def test_tempo_trajectory_endpoints(self):
        traj = iso_match_tempo(120, 60, steps=10)
        assert len(traj) == 10
        assert abs(traj[0] - 120) < 1
        assert abs(traj[-1] - 60) < 1

    def test_tempo_monotone_decrease(self):
        traj = iso_match_tempo(120, 60, steps=20)
        for i in range(1, len(traj)):
            assert traj[i] <= traj[i - 1] + 0.01

    def test_mode_path(self):
        path = iso_match_mode("aeolian", "ionian")
        assert path[0] == "aeolian"
        assert path[-1] == "ionian"
        assert len(path) >= 2

    def test_emotional_trajectory_shape(self):
        vals = emotional_trajectory(-0.8, 0.6, 15)
        assert len(vals) == 15
        assert abs(vals[0] - (-0.8)) < 0.05
        assert abs(vals[-1] - 0.6) < 0.05

    def test_energy_curve(self):
        curve = iso_energy_curve(0.8, 0.2, duration_min=10.0, step_min=2.0)
        assert len(curve) >= 2
        assert curve[0][0] == 0.0
        assert abs(curve[-1][0] - 10.0) < 0.1


class TestEntrainment:
    def test_coupling_step(self):
        ec = EntropicCoupling(natural_freq=2.0, coupling_k=1.0)
        p = ec.step(driver_phase=1.0, dt=0.01)
        assert isinstance(p, float)

    def test_coupling_run(self):
        ec = EntropicCoupling(natural_freq=1.0, coupling_k=5.0)
        phases = ec.run([float(i * 0.1) for i in range(100)], dt=0.01)
        assert len(phases) == 100

    def test_phase_coupling_exact(self):
        assert phase_coupling_strength(2.0, 2.0) == 1.0

    def test_phase_coupling_off(self):
        s = phase_coupling_strength(2.0, 2.5, tolerance=0.1)
        assert 0.0 <= s <= 1.0

    def test_entrainment_time_zero_diff(self):
        assert entrainment_time(0.0, 1.0) == 0.0

    def test_entrainment_time_no_coupling(self):
        assert entrainment_time(1.0, 0.0) == float("inf")

    def test_pulse_sync(self):
        beats = [0.0, 0.5, 1.0, 1.5, 2.0]
        taps = [0.01, 0.49, 1.02, 1.48, 2.01]
        ratio, offsets = pulse_synchronization(taps, beats)
        assert ratio > 0.8
        assert len(offsets) == len(taps)

    def test_kuramoto(self):
        assert kuramoto_order_parameter([0.0, 0.0, 0.0]) == pytest.approx(1.0)
        r = kuramoto_order_parameter([0, np.pi])
        assert r < 0.1


class TestReceptive:
    def test_relaxation_score(self):
        tempos = [120, 110, 100, 90, 80, 70]
        valences = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        score = relaxation_playlist_score(tempos, valences)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_gim_profile(self):
        profile = guided_imagery_tempo_profile(600.0, n_points=10)
        assert len(profile) == 10
        assert profile[0][0] == 0.0

    def test_song_features(self):
        f = song_discussion_features("I feel sad and lonely but hope remains")
        assert f["emotional_density"] > 0
        assert f["self_reference"] > 0
        assert f["word_count"] == 8.0

    def test_arc_quality(self):
        good = session_arc_quality([0.2, 0.3, 0.1, 0.4, 0.7])
        bad = session_arc_quality([0.7, 0.3, 0.1])
        assert good > bad


class TestGroup:
    def test_rhythmic_cohesion_perfect(self):
        taps_a = [0.0, 0.5, 1.0, 1.5]
        taps_b = [0.0, 0.5, 1.0, 1.5]
        assert rhythmic_cohesion([taps_a, taps_b]) == pytest.approx(1.0, abs=0.01)

    def test_rhythmic_cohesion_poor(self):
        taps_a = [0.0, 0.5, 1.0, 1.5]
        taps_b = [0.25, 0.75, 1.25, 1.75]
        c = rhythmic_cohesion([taps_a, taps_b])
        assert c < 0.8

    def test_turn_taking_equal(self):
        b = turn_taking_balance([10, 10, 10, 10])
        assert b > 0.9

    def test_turn_taking_unequal(self):
        b = turn_taking_balance([30, 5, 5, 5])
        assert b < 0.5

    def test_tempo_drift(self):
        metro = [0.0, 0.5, 1.0, 1.5, 2.0]
        group = [0.0, 0.48, 0.96, 1.44, 1.92]
        d = collective_tempo_drift(group, metro)
        assert "drift_bpm" in d
        assert "stability" in d

    def test_sync_index(self):
        s = synchronization_index([[0, 0.5, 1.0], [0, 0.5, 1.0]])
        assert s > 0.5


class TestProtocol:
    def test_build_protocol(self):
        p = build_protocol(TherapyGoal.AROUSAL_REGULATION, 45.0)
        assert p.total_duration_min == 45.0
        assert len(p.phases) == 5
        assert p.n_interventions == 5

    def test_arousal_protocol(self):
        p = arousal_regulation_protocol(30.0)
        assert p.goal == TherapyGoal.AROUSAL_REGULATION
        phases = [ph for ph, _ in p.phases]
        assert SessionPhase.OPENING in phases
        assert SessionPhase.CLOSING in phases


class TestPackageExports:
    def test_imports(self):
        import models.music_therapy as mt
        assert hasattr(mt, "iso_match_tempo")
        assert hasattr(mt, "EntropicCoupling")
        assert hasattr(mt, "build_protocol")
        assert hasattr(mt, "rhythmic_cohesion")
