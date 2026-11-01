"""Tests for algorithmic composition module."""
import numpy as np
import pytest
from models.music_generation.algorithmic_composition import (
    LSystem, LSystemRule, lsystem_to_melody,
    CellularAutomaton, ca_to_rhythm, ca_to_pitches,
    MarkovChainComposer, ConstraintSolver,
    stochastic_process, stochastic_to_pitches,
    euclidean_rhythm, serial_row, serial_transformations,
    all_transpositions, fractal_melody,
)


class TestLSystem:
    def test_basic(self):
        ls = LSystem(
            axiom="A",
            rules=[LSystemRule("A", "AB"), LSystemRule("B", "A")],
            iterations=3,
        )
        result = ls.generate()
        assert len(result) > 3
        assert result.startswith("A")

    def test_to_melody(self):
        melody = lsystem_to_melody("F+F-F+F", base_pitch=60)
        assert len(melody) > 1
        assert melody[0] == 60


class TestCellularAutomaton:
    def test_rule30(self):
        ca = CellularAutomaton(rule_number=30, width=32, generations=16)
        history = ca.run()
        assert history.shape == (16, 32)
        assert np.any(history > 0)

    def test_rule110(self):
        ca = CellularAutomaton(rule_number=110, width=64, generations=32)
        history = ca.run()
        assert history.shape == (32, 64)

    def test_to_rhythm(self):
        ca = CellularAutomaton(rule_number=90, width=16, generations=8)
        history = ca.run()
        rhythm = ca_to_rhythm(history, density=0.3)
        assert rhythm.shape == history.shape
        assert set(np.unique(rhythm)).issubset({0, 1})

    def test_to_pitches(self):
        ca = CellularAutomaton(rule_number=30, width=8, generations=4)
        history = ca.run()
        scale = np.array([0, 2, 4, 5, 7, 9, 11])
        pitches = ca_to_pitches(history, scale)
        assert pitches.shape == history.shape


class TestMarkovComposer:
    def test_train_generate(self):
        seq = np.array([60, 62, 64, 62, 60, 62, 64, 65, 64, 62])
        mc = MarkovChainComposer(order=1)
        mc.train(seq)
        gen = mc.generate(20)
        assert len(gen) == 20
        assert all(g in seq for g in gen)

    def test_order2(self):
        seq = np.array([60, 62, 64, 65, 67, 65, 64, 62, 60, 62, 64])
        mc = MarkovChainComposer(order=2)
        mc.train(seq)
        gen = mc.generate(15)
        assert len(gen) == 15


class TestConstraintSolver:
    def test_basic(self):
        solver = ConstraintSolver(pitch_range=(48, 84))
        melody = solver.solve(16)
        assert len(melody) == 16
        assert all(48 <= p <= 84 for p in melody)

    def test_with_scale(self):
        scale = np.array([0, 2, 4, 5, 7, 9, 11])
        solver = ConstraintSolver(pitch_range=(60, 84), scale_degrees=scale)
        melody = solver.solve(12)
        for p in melody:
            assert p % 12 in scale

    def test_fixed_points(self):
        solver = ConstraintSolver()
        melody = solver.solve(8, fixed={0: 60, 7: 72})
        assert melody[0] == 60
        assert melody[7] == 72


class TestStochastic:
    def test_process(self):
        vals = stochastic_process(100, drift=0.1, volatility=0.5, start=60)
        assert len(vals) == 100
        assert vals[0] == 60

    def test_mean_revert(self):
        vals = stochastic_process(500, mean_revert=0.5, volatility=0.3, start=60)
        assert abs(np.mean(vals) - 60) < 5

    def test_to_pitches(self):
        vals = stochastic_process(20, start=60)
        scale = np.array([0, 2, 4, 5, 7, 9, 11])
        pitches = stochastic_to_pitches(vals, scale)
        assert len(pitches) == 20
        assert all(36 <= p <= 96 for p in pitches)


class TestEuclidean:
    def test_3_8(self):
        pattern = euclidean_rhythm(3, 8)
        assert len(pattern) == 8
        assert np.sum(pattern) == 3

    def test_4_16(self):
        pattern = euclidean_rhythm(4, 16)
        assert np.sum(pattern) == 4

    def test_empty(self):
        assert len(euclidean_rhythm(0, 8)) == 8
        assert np.sum(euclidean_rhythm(0, 8)) == 0


class TestSerialRow:
    def test_all_12(self):
        row = serial_row(seed=42)
        assert len(row) == 12
        assert set(row) == set(range(12))

    def test_transformations(self):
        row = serial_row(seed=0)
        forms = serial_transformations(row)
        assert len(forms) == 4
        assert len(forms["R0"]) == 12
        np.testing.assert_array_equal(forms["R0"], row[::-1])

    def test_all_transpositions(self):
        row = serial_row(seed=1)
        all_forms = all_transpositions(row)
        assert len(all_forms) == 48


class TestFractalMelody:
    def test_basic(self):
        melody = fractal_melody(depth=3)
        assert len(melody) > 10
        assert all(36 <= p <= 96 for p in melody)

    def test_custom_intervals(self):
        melody = fractal_melody(depth=2, base_intervals=np.array([0, 3, -2]))
        assert len(melody) > 5
