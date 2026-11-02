"""Tests for biofeedback music therapy module."""
import numpy as np
import pytest
from models.music_therapy.biofeedback_music import (
    heart_rate_from_ecg, hrv_rmssd, hrv_sdnn, respiration_rate,
    skin_conductance_level, emg_activation, MappingRule,
    BiofeedbackState, apply_mappings, arousal_from_physiology,
    valence_from_physiology, relaxation_index, AdaptiveController,
    coherence_score, session_progress, MusicalOutput,
)


class TestHeartRate:
    def test_synthetic_ecg(self):
        sr = 500.0
        t = np.arange(5000) / sr
        ecg = np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm
        hr = heart_rate_from_ecg(ecg, sr)
        assert len(hr) >= 1
        assert 40 < np.mean(hr) < 200

    def test_short_signal(self):
        hr = heart_rate_from_ecg(np.zeros(10), 500.0)
        assert len(hr) >= 1


class TestHRV:
    def test_rmssd(self):
        rr = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.80])
        r = hrv_rmssd(rr)
        assert r > 0

    def test_sdnn(self):
        rr = np.array([0.8, 0.82, 0.79, 0.81, 0.83])
        s = hrv_sdnn(rr)
        assert s > 0

    def test_short(self):
        assert hrv_rmssd(np.array([0.8])) == 0.0


class TestRespiration:
    def test_normal(self):
        sr = 100.0
        t = np.arange(3000) / sr
        resp = np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
        rate = respiration_rate(resp, sr)
        assert 10 < rate < 25

    def test_short(self):
        rate = respiration_rate(np.zeros(10), 100.0)
        assert rate == 15.0


class TestEDA:
    def test_scl(self):
        sr = 100.0
        eda = np.random.randn(1000) * 0.1 + 5.0
        scl, phasic = skin_conductance_level(eda, sr)
        assert 3 < scl < 7
        assert len(phasic) == len(eda)


class TestEMG:
    def test_activation(self):
        sr = 1000.0
        emg = np.random.randn(5000) * 0.5
        act = emg_activation(emg, sr)
        assert len(act) == len(emg)
        assert np.all(act >= 0)


class TestMapping:
    def test_linear(self):
        rule = MappingRule("hr", "tempo", (50, 120), (60, 140))
        assert abs(rule.map(50) - 60) < 1e-6
        assert abs(rule.map(120) - 140) < 1e-6

    def test_inverted(self):
        rule = MappingRule("resp", "phrase", (8, 25), (8, 2), invert=True)
        assert rule.map(8) > rule.map(25)

    def test_exponential(self):
        rule = MappingRule("x", "y", (0, 1), (0, 1), curve="exponential")
        assert rule.map(0.5) < 0.5

    def test_sigmoid(self):
        rule = MappingRule("x", "y", (0, 1), (0, 1), curve="sigmoid")
        mid = rule.map(0.5)
        assert abs(mid - 0.5) < 0.1


class TestBiofeedbackState:
    def test_defaults(self):
        state = BiofeedbackState()
        output = apply_mappings(state)
        assert 40 < output.tempo_bpm < 180
        assert 0 < output.dynamics < 1

    def test_high_arousal(self):
        state = BiofeedbackState(heart_rate=110, skin_conductance=15.0)
        output = apply_mappings(state)
        assert output.tempo_bpm > 100


class TestArousalValence:
    def test_arousal_range(self):
        state = BiofeedbackState()
        a = arousal_from_physiology(state)
        assert 0 <= a <= 1

    def test_high_hr_high_arousal(self):
        low = BiofeedbackState(heart_rate=55)
        high = BiofeedbackState(heart_rate=115)
        assert arousal_from_physiology(high) > arousal_from_physiology(low)

    def test_valence_range(self):
        state = BiofeedbackState()
        v = valence_from_physiology(state)
        assert 0 <= v <= 1

    def test_relaxation(self):
        state = BiofeedbackState(heart_rate=55, skin_conductance=2.0)
        assert relaxation_index(state) > 0.5


class TestAdaptiveController:
    def test_converges(self):
        ctrl = AdaptiveController(target_arousal=0.3, learning_rate=0.2)
        state = BiofeedbackState(heart_rate=100, skin_conductance=12.0)
        output = MusicalOutput(tempo_bpm=120, dynamics=0.8)
        for _ in range(5):
            output = ctrl.step(state, output)
        assert output.tempo_bpm < 120

    def test_convergence_rate(self):
        ctrl = AdaptiveController(target_arousal=0.5)
        state = BiofeedbackState()
        output = MusicalOutput()
        for _ in range(10):
            output = ctrl.step(state, output)
        assert ctrl.convergence_rate() >= 0


class TestCoherence:
    def test_correlated(self):
        n = 500
        hr = np.sin(2 * np.pi * 0.1 * np.arange(n) / 10)
        resp = np.sin(2 * np.pi * 0.1 * np.arange(n) / 10)
        c = coherence_score(hr, resp, 10.0)
        assert c > 0.5

    def test_short(self):
        assert coherence_score(np.array([1, 2]), np.array([3, 4]), 10.0) == 0.0


class TestSessionProgress:
    def test_basic(self):
        states = [
            BiofeedbackState(heart_rate=100, skin_conductance=12.0),
            BiofeedbackState(heart_rate=90, skin_conductance=8.0),
            BiofeedbackState(heart_rate=75, skin_conductance=5.0),
            BiofeedbackState(heart_rate=68, skin_conductance=3.0),
        ]
        p = session_progress(states)
        assert p["relaxation_trend"] > 0
        assert p["arousal_change"] < 0
