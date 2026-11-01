"""Tests for embodied music cognition."""
import numpy as np
import pytest
from models.music_cognition.embodied_cognition import (
    BODY_PARTS, movement_periodicity, entrainment_strength,
    phase_coupling, kinetic_energy, movement_fluency,
    gesture_segmentation, body_resonance, optimal_tempo_for_movement,
    movement_audio_synchrony, groove_body_response, analyze_dance,
)


class TestMovementPeriodicity:
    def test_periodic(self):
        sr = 100.0
        t = np.arange(500) / sr
        pos = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz
        freq, strength = movement_periodicity(pos, sr)
        assert abs(freq - 2.0) < 0.5
        assert strength > 0.5

    def test_random(self):
        pos = np.random.randn(200)
        freq, strength = movement_periodicity(pos, 100.0)
        assert strength < 0.5


class TestEntrainment:
    def test_perfect_sync(self):
        beats = np.arange(0, 5, 0.5)
        onsets = beats.copy()
        s = entrainment_strength(onsets, beats)
        assert s > 0.9

    def test_random_weak(self):
        beats = np.arange(0, 5, 0.5)
        onsets = np.random.uniform(0, 5, 20)
        s = entrainment_strength(onsets, beats)
        assert s < 0.8

    def test_empty(self):
        assert entrainment_strength(np.array([]), np.array([1, 2])) == 0.0


class TestPhaseCoupling:
    def test_identical(self):
        sig = np.sin(2 * np.pi * 5 * np.arange(200) / 100)
        plv = phase_coupling(sig, sig)
        assert plv > 0.9

    def test_empty(self):
        assert phase_coupling(np.array([]), np.array([])) == 0.0


class TestKineticEnergy:
    def test_basic(self):
        v = np.array([1.0, 2.0, 3.0])
        ke = kinetic_energy(v, mass=1.0)
        np.testing.assert_allclose(ke, [0.5, 2.0, 4.5])


class TestFluency:
    def test_smooth(self):
        t = np.linspace(0, 1, 100)
        pos = np.sin(2 * np.pi * t)
        f = movement_fluency(pos, 100.0)
        assert f > 0.5

    def test_jerky(self):
        pos = np.random.randn(100)
        f = movement_fluency(pos, 100.0)
        assert f < 0.5


class TestGestureSegmentation:
    def test_segments(self):
        velocity = np.array([0, 0, 0.5, 0.8, 0.3, 0, 0, 0.6, 0.9, 0])
        segs = gesture_segmentation(velocity, threshold=0.1)
        assert len(segs) == 2

    def test_no_movement(self):
        velocity = np.zeros(10)
        assert gesture_segmentation(velocity) == []


class TestBodyResonance:
    def test_at_resonance(self):
        freq = BODY_PARTS["leg"].resonant_freq_hz
        r = body_resonance(freq, "leg")
        assert r > 1.0

    def test_off_resonance(self):
        r = body_resonance(100.0, "leg")
        assert r < body_resonance(BODY_PARTS["leg"].resonant_freq_hz, "leg")


class TestOptimalTempo:
    def test_walk(self):
        bpm = optimal_tempo_for_movement("leg", "walk")
        assert 40 < bpm < 60

    def test_tap(self):
        bpm = optimal_tempo_for_movement("hand", "tap")
        assert bpm > 100


class TestAudioSynchrony:
    def test_correlated(self):
        sr = 100.0
        t = np.arange(500) / sr
        movement = np.sin(2 * np.pi * 2 * t)
        audio = np.sin(2 * np.pi * 2 * t)
        s = movement_audio_synchrony(movement, audio, sr)
        assert s > 0.5


class TestGrooveResponse:
    def test_swing(self):
        beats = np.arange(0, 4, 0.5)
        microtiming = np.array([0.02, -0.01, 0.03, -0.02, 0.02, -0.01, 0.03, -0.02])
        r = groove_body_response(beats, microtiming)
        assert r > 0

    def test_empty(self):
        assert groove_body_response(np.array([]), np.array([])) == 0.0


class TestDanceAnalysis:
    def test_basic(self):
        sr = 100.0
        t = np.arange(500) / sr
        pos = np.sin(2 * np.pi * 2 * t)
        beats = np.arange(0, 5, 0.5)
        m = analyze_dance(pos, beats, sr, tempo_bpm=120)
        assert 0 <= m.synchrony <= 1
        assert 0 <= m.fluency <= 1
