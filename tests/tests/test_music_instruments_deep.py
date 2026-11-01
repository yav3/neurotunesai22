"""Tests for deep instrument models: resonance, classification."""
import numpy as np
import pytest

from models.music_instruments.resonance_models import (
    StringResonance, MembraneResonance, BarResonance,
    ResonatorBank, CoupledStringModel,
)
from models.music_instruments.instrument_classification import (
    InstrumentFeatureExtractor, InstrumentSimilarityMatrix,
    InstrumentFamilyClassifier,
)


SR = 16000


def _sine(freq, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


class TestStringResonance:
    def test_harmonic_series(self):
        s = StringResonance()
        freqs, amps = s.modes(0.65, 70.0, 0.001, n_modes=5)
        assert len(freqs) == 5
        for i in range(1, 5):
            ratio = freqs[i] / freqs[0]
            assert abs(ratio - (i + 1)) < 0.01

    def test_amplitude_decreasing(self):
        s = StringResonance()
        _, amps = s.modes(0.65, 70.0, 0.001, n_modes=8)
        for i in range(len(amps) - 1):
            assert amps[i] >= amps[i + 1]

    def test_pluck_at_center(self):
        s = StringResonance()
        w = s.pluck_spectrum(0.5, n_modes=6)
        assert w[0] > 0
        assert w[1] < 0.01

    def test_pluck_at_quarter(self):
        s = StringResonance()
        w = s.pluck_spectrum(0.25, n_modes=8)
        assert w[3] < 0.01

    def test_decay_envelope_shape(self):
        s = StringResonance()
        env = s.decay_envelope(440.0, 2.0, 0.5, SR)
        assert len(env) == int(SR * 0.5)
        assert abs(env[0]) > abs(env[-1])

    def test_synthesize_length(self):
        s = StringResonance()
        sig = s.synthesize(0.65, 70.0, 0.001, 0.3, 1.0, 0.5, SR, n_modes=5)
        assert len(sig) == int(SR * 0.5)
        assert np.max(np.abs(sig)) <= 1.0 + 1e-6


class TestMembraneResonance:
    def test_inharmonic_partials(self):
        m = MembraneResonance()
        freqs = m.modes(0.3, 100.0, 0.5, n_modes=5)
        ratios = freqs / freqs[0]
        assert not np.allclose(ratios, np.arange(1, 6))

    def test_positive_frequencies(self):
        m = MembraneResonance()
        freqs = m.modes(0.3, 100.0, 0.5, n_modes=8)
        assert np.all(freqs > 0)

    def test_mode_shape_dims(self):
        m = MembraneResonance()
        shape = m.mode_shapes(0.3, n_radial=0, n_angular=0, grid_size=32)
        assert shape.shape == (32, 32)

    def test_mode_shape_zero_outside(self):
        m = MembraneResonance()
        shape = m.mode_shapes(0.3, n_radial=0, n_angular=0, grid_size=64)
        assert shape[0, 0] == 0.0

    def test_inharmonicity_ratio_fundamental_one(self):
        m = MembraneResonance()
        r = m.inharmonicity_ratio(0.3, 100.0, 0.5, n_modes=4)
        assert abs(r[0] - 1.0) < 1e-6


class TestBarResonance:
    def test_free_free_modes(self):
        b = BarResonance()
        freqs = b.modes(0.4, 2e11, 7800, 1e-8, n_modes=4, boundary="free-free")
        assert len(freqs) == 4
        assert np.all(freqs > 0)

    def test_clamped_free_lower(self):
        b = BarResonance()
        ff = b.modes(0.4, 2e11, 7800, 1e-8, n_modes=1, boundary="free-free")
        cf = b.modes(0.4, 2e11, 7800, 1e-8, n_modes=1, boundary="clamped-free")
        assert cf[0] < ff[0]

    def test_frequency_ratio_first_is_one(self):
        b = BarResonance()
        r = b.frequency_ratio(0.4, 2e11, 7800, 1e-8, n_modes=4)
        assert abs(r[0] - 1.0) < 1e-6


class TestResonatorBank:
    def test_synthesize_nonzero(self):
        rb = ResonatorBank()
        rb.add(np.array([440.0, 880.0]), np.array([1.0, 0.5]), 2.0)
        exc = np.zeros(100)
        exc[0] = 1.0
        out = rb.synthesize(exc, SR, 0.2)
        assert len(out) == int(SR * 0.2)
        assert np.max(np.abs(out)) > 0

    def test_empty_bank(self):
        rb = ResonatorBank()
        out = rb.synthesize(np.array([1.0]), SR, 0.1)
        assert len(out) == int(SR * 0.1)


class TestCoupledStringModel:
    def test_energy_transfer(self):
        cs = CoupledStringModel()
        x1, x2 = cs.simulate(440.0, 445.0, 500.0, 0.5, SR)
        assert len(x1) == int(SR * 0.5)
        assert np.max(np.abs(x2)) > 0.01

    def test_no_coupling(self):
        cs = CoupledStringModel()
        x1, x2 = cs.simulate(440.0, 880.0, 0.0, 0.2, SR)
        assert np.max(np.abs(x2)) < 0.01

    def test_energy_curve_sums_to_one(self):
        cs = CoupledStringModel()
        e1, e2 = cs.energy_transfer_curve(440.0, 445.0, 500.0, 0.3, SR)
        total = e1 + e2
        assert np.allclose(total, 1.0, atol=0.01)


class TestInstrumentFeatureExtractor:
    @pytest.fixture
    def ext(self):
        return InstrumentFeatureExtractor()

    def test_spectral_envelope_shape(self, ext):
        spec = np.abs(np.fft.rfft(_sine(440.0)))
        env = ext.spectral_envelope(spec, n_coefficients=15)
        assert len(env) == 15

    def test_attack_characteristics_keys(self, ext):
        sig = _sine(440.0, dur=0.2)
        attack = ext.attack_characteristics(sig, SR)
        assert "attack_time" in attack
        assert "attack_slope" in attack
        assert "onset_strength" in attack

    def test_attack_time_nonneg(self, ext):
        sig = _sine(440.0, dur=0.3)
        attack = ext.attack_characteristics(sig, SR)
        assert attack["attack_time"] >= 0

    def test_steady_state_keys(self, ext):
        sig = _sine(440.0, dur=0.5)
        ss = ext.steady_state_features(sig, SR)
        assert "vibrato_rate" in ss
        assert "spectral_flux_mean" in ss

    def test_temporal_features_keys(self, ext):
        sig = _sine(440.0, dur=0.5)
        tf = ext.temporal_features(sig, SR)
        assert "attack_ms" in tf
        assert "sustain_level" in tf

    def test_extract_all_returns_dict(self, ext):
        sig = _sine(440.0, dur=0.3)
        feats = ext.extract_all(sig, SR)
        assert isinstance(feats, dict)
        assert "spectral_envelope" in feats

    def test_empty_signal(self, ext):
        result = ext.attack_characteristics(np.array([]), SR)
        assert result["attack_time"] == 0.0


class TestInstrumentSimilarityMatrix:
    def test_symmetric(self):
        ism = InstrumentSimilarityMatrix()
        vecs = np.random.default_rng(42).standard_normal((5, 10))
        sim = ism.compute(vecs)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)

    def test_diagonal_ones(self):
        ism = InstrumentSimilarityMatrix()
        vecs = np.random.default_rng(42).standard_normal((4, 8))
        sim = ism.compute(vecs)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_weighted(self):
        weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        ism = InstrumentSimilarityMatrix(feature_weights=weights)
        vecs = np.random.default_rng(42).standard_normal((3, 5))
        sim = ism.compute(vecs)
        assert sim.shape == (3, 3)

    def test_cluster_labels(self):
        ism = InstrumentSimilarityMatrix()
        vecs = np.random.default_rng(42).standard_normal((20, 5))
        labels = ism.cluster(vecs, n_clusters=3)
        assert len(labels) == 20
        assert len(set(labels)) <= 3


class TestInstrumentFamilyClassifier:
    def test_classify_returns_family(self):
        clf = InstrumentFamilyClassifier()
        feats = {"attack_time": 0.002, "vibrato_rate": 0.0, "spectral_flux_mean": 0.4,
                 "decay_ms": 5.0, "sustain_level": 0.0, "onset_strength": 0.8}
        family, conf = clf.classify(feats)
        assert family in ("strings", "woodwinds", "brass", "percussion", "keyboard")
        assert 0 <= conf <= 1.0

    def test_percussion_detection(self):
        clf = InstrumentFamilyClassifier()
        feats = {"attack_time": 0.001, "vibrato_rate": 0.0, "spectral_flux_mean": 0.6,
                 "decay_ms": 3.0, "sustain_level": 0.0, "onset_strength": 0.9}
        family, _ = clf.classify(feats)
        assert family == "percussion"

    def test_feature_importance_sums_to_one(self):
        clf = InstrumentFamilyClassifier()
        imp = clf.feature_importance()
        assert abs(sum(imp.values()) - 1.0) < 0.01
