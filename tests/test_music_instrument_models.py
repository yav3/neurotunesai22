"""Tests for formant and resonance instrument models."""
import math
import numpy as np
import pytest

from models.music_instruments.formant_models import FormantSynthesizer, ResonanceModel


class TestFormantSynthesizer:
    @pytest.fixture
    def fs(self):
        return FormantSynthesizer()

    def test_vowel_formants_known(self, fs):
        f = fs.vowel_formants("a")
        assert len(f) == 6
        assert f[0] > 600

    def test_vowel_formants_all_five(self, fs):
        for v in "aeiou":
            f = fs.vowel_formants(v)
            assert len(f) == 6
            assert all(x > 0 for x in f)

    def test_vowel_formants_unknown(self, fs):
        f = fs.vowel_formants("x")
        assert f == fs.vowel_formants("a")

    def test_formant_filter_shape(self, fs):
        sig = np.random.default_rng(0).standard_normal(1000)
        out = fs.formant_filter(sig, 16000, [500, 1500], [100, 100])
        assert out.shape == sig.shape

    def test_formant_filter_empty(self, fs):
        out = fs.formant_filter(np.array([]), 16000, [500], [100])
        assert len(out) == 0

    def test_singing_formant_returns_triple(self, fs):
        center, bw, gain = fs.singing_formant(440.0)
        assert center == 3000.0
        assert bw > 0
        assert gain > 1.0

    def test_formant_transition_shape(self, fs):
        traj = fs.formant_transition("a", "i", n_frames=10)
        assert traj.shape == (10, 3)

    def test_formant_transition_endpoints(self, fs):
        traj = fs.formant_transition("a", "i", n_frames=20)
        a_f = fs.vowel_formants("a")[:3]
        i_f = fs.vowel_formants("i")[:3]
        np.testing.assert_allclose(traj[0], a_f, atol=1)
        np.testing.assert_allclose(traj[-1], i_f, atol=1)

    def test_estimate_formants_count(self, fs):
        spec = np.abs(np.random.default_rng(0).standard_normal(256))
        freqs = np.linspace(0, 8000, 256)
        result = fs.estimate_formants(spec, freqs, n_formants=3)
        assert len(result) == 3

    def test_estimate_formants_empty(self, fs):
        result = fs.estimate_formants(np.array([1.0]), np.array([100.0]), n_formants=4)
        assert len(result) == 4


class TestResonanceModel:
    @pytest.fixture
    def rm(self):
        return ResonanceModel()

    def test_string_modes_harmonic(self, rm):
        modes = rm.string_modes(0.65, 70.0, 0.001)
        assert len(modes) == 16
        f1 = modes[0]
        for n in range(1, 16):
            np.testing.assert_allclose(modes[n], f1 * (n + 1), rtol=1e-10)

    def test_string_modes_zero_length(self, rm):
        modes = rm.string_modes(0, 70, 0.001)
        assert np.all(modes == 0)

    def test_inharmonicity_positive(self, rm):
        B = rm.inharmonicity_coefficient(0.001, 0.5, 2e11)
        assert B > 0

    def test_inharmonicity_zero_length(self, rm):
        assert rm.inharmonicity_coefficient(0.001, 0, 2e11) == 0.0

    def test_tube_open_open_harmonics(self, rm):
        modes = rm.tube_resonance(1.0, end_correction=0, n_modes=4, open_open=True)
        assert len(modes) == 4
        for i in range(1, 4):
            np.testing.assert_allclose(modes[i] / modes[0], i + 1, rtol=0.01)

    def test_tube_open_closed_odd(self, rm):
        modes = rm.tube_resonance(1.0, end_correction=0, n_modes=3, open_open=False)
        ratios = modes / modes[0]
        np.testing.assert_allclose(ratios, [1, 3, 5], rtol=0.01)

    def test_tube_zero_length(self, rm):
        modes = rm.tube_resonance(0)
        assert np.all(modes == 0)

    def test_membrane_modes_count(self, rm):
        modes = rm.membrane_modes(0.3, 1000, 0.5, n_modes=8)
        assert len(modes) == 8

    def test_membrane_modes_sorted(self, rm):
        modes = rm.membrane_modes(0.3, 1000, 0.5)
        assert np.all(np.diff(modes) >= 0)

    def test_membrane_modes_zero_radius(self, rm):
        modes = rm.membrane_modes(0, 1000, 0.5)
        assert np.all(modes == 0)

    def test_coupled_oscillators_shape(self, rm):
        freqs = [100, 200]
        K = [[0, 0.1], [0.1, 0]]
        t, x = rm.coupled_oscillators(freqs, K, t_span=0.01, dt=0.001)
        assert len(t) == x.shape[0]
        assert x.shape[1] == 2

    def test_coupled_oscillators_empty(self, rm):
        t, x = rm.coupled_oscillators([], np.array([]).reshape(0, 0), 0.01)
        assert x.shape[1] == 0

    def test_decay_rates_formula(self, rm):
        freqs = np.array([100.0, 200.0])
        qs = np.array([50.0, 100.0])
        tau = rm.decay_rates(freqs, qs)
        np.testing.assert_allclose(tau[0], 50 / (np.pi * 100), rtol=1e-10)
        np.testing.assert_allclose(tau[1], 100 / (np.pi * 200), rtol=1e-10)

    def test_decay_rates_zero_freq(self, rm):
        tau = rm.decay_rates(np.array([0.0]), np.array([100.0]))
        assert tau[0] == 0.0

    def test_string_a440(self, rm):
        L = 0.326
        T = 73.0
        mu = 0.00065
        modes = rm.string_modes(L, T, mu, n_modes=1)
        assert 400 < modes[0] < 550
