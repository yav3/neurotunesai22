"""Tests for the music_instruments module."""

import numpy as np
import pytest
from models.music_instruments.taxonomy import (
    InstrumentFamily, instruments_by_family, pitch_range,
    transposition_interval, is_transposing,
)
from models.music_instruments.physical_models import (
    StringModel, string_frequency, string_harmonics,
    TubeModel, tube_resonance, open_pipe_harmonics, closed_pipe_harmonics,
    MembraneModel, membrane_modes,
    BarModel, bar_frequency,
)
from models.music_instruments.synthesis_params import (
    ADSREnvelope, OscillatorType, SynthPatch,
    default_patch, fm_ratio, detune_cents_to_ratio,
)
from models.music_instruments.tuning_systems import (
    equal_temperament, pythagorean, just_intonation,
    cents_deviation, frequency_ratio, compare_tunings,
)


class TestTaxonomy:
    def test_families_exist(self):
        for fam in InstrumentFamily:
            instruments = instruments_by_family(fam)
            assert isinstance(instruments, list)

    def test_pitch_range_piano(self):
        low, high = pitch_range("piano")
        assert low == 21
        assert high == 108

    def test_transposing_instrument(self):
        assert is_transposing("clarinet Bb")
        interval = transposition_interval("clarinet Bb")
        assert interval == -2


class TestPhysicalModels:
    def test_string_fundamental(self):
        f = string_frequency(length=0.65, tension=73.0, density=0.00058)
        assert f > 0

    def test_string_harmonics_count(self):
        model = StringModel(length_m=0.65, tension_n=73.0, linear_density_kg_m=0.00058)
        h = string_harmonics(model, n=8)
        assert len(h) == 8
        assert h[1] == pytest.approx(2 * h[0], rel=0.01)

    def test_open_pipe(self):
        freqs = open_pipe_harmonics(length=0.5, n=4)
        assert len(freqs) == 4

    def test_closed_pipe_odd_harmonics(self):
        freqs = closed_pipe_harmonics(length=0.5, n=3)
        assert len(freqs) == 3

    def test_tube_resonance(self):
        model = TubeModel(length_m=0.5, radius_m=0.01, is_open=True)
        f1 = tube_resonance(model, mode=1)
        assert f1 > 0

    def test_membrane_modes(self):
        model = MembraneModel(radius_m=0.3, tension_per_length=200.0, surface_density=0.5)
        modes = membrane_modes(model, n_modes=5)
        assert len(modes) == 5

    def test_bar_frequency(self):
        model = BarModel(length_m=0.3, thickness_m=0.01, youngs_modulus=2e11, density=7800.0)
        f = bar_frequency(model)
        assert f > 0


class TestSynthesis:
    def test_default_patch(self):
        p = default_patch()
        assert isinstance(p, SynthPatch)
        assert isinstance(p.envelope, ADSREnvelope)

    def test_fm_ratio(self):
        r = fm_ratio(carrier_hz=440.0, modulator_hz=880.0)
        assert r == pytest.approx(2.0)

    def test_detune_cents(self):
        ratio = detune_cents_to_ratio(100)
        assert ratio == pytest.approx(2 ** (100 / 1200))


class TestTuningSystems:
    def test_equal_temperament_octave(self):
        f12 = equal_temperament(12, reference=440.0)
        assert f12 == pytest.approx(880.0)

    def test_pythagorean_fifth(self):
        f = pythagorean(7, reference=440.0)
        assert f > 440.0

    def test_just_intonation_unison(self):
        f = just_intonation(0, reference=440.0)
        assert f == pytest.approx(440.0)

    def test_cents_deviation_unison(self):
        dev = cents_deviation(440.0, 440.0)
        assert dev == pytest.approx(0.0)

    def test_cents_deviation_semitone(self):
        dev = cents_deviation(440.0 * 2 ** (1 / 12), 440.0)
        assert abs(dev) == pytest.approx(100.0, abs=0.1)

    def test_frequency_ratio(self):
        r = frequency_ratio(880.0, 440.0)
        assert r == pytest.approx(2.0)

    def test_compare_tunings(self):
        result = compare_tunings(n_notes=7, reference=440.0)
        assert "equal" in result
        assert "pythagorean" in result


class TestExtendedTechniques:
    def test_natural_harmonics(self):
        from models.music_instruments.extended_techniques import natural_harmonic_series
        series = natural_harmonic_series(110.0, n_harmonics=8)
        assert len(series) == 8
        assert series[0] == pytest.approx(110.0)
        assert series[1] == pytest.approx(220.0)

    def test_harmonic_amplitude_model(self):
        from models.music_instruments.extended_techniques import harmonic_amplitude_model
        harmonics = np.arange(1, 9, dtype=float)
        amps = harmonic_amplitude_model(harmonics, decay_rate=1.0, model="inverse")
        assert len(amps) == 8
        assert amps[0] > amps[-1]

    def test_string_harmonic_nodes(self):
        from models.music_instruments.extended_techniques import string_harmonic_nodes
        nodes = string_harmonic_nodes(string_length_m=1.0, fundamental_hz=110.0, max_harmonic=4)
        assert len(nodes) > 0

    def test_multiphonic_roughness(self):
        from models.music_instruments.extended_techniques import multiphonic_roughness
        freqs = np.array([440.0, 445.0, 880.0])
        amps = np.array([1.0, 0.8, 0.5])
        r = multiphonic_roughness(freqs, amps)
        assert r > 0

    def test_technique_difficulty(self):
        from models.music_instruments.extended_techniques import technique_difficulty_score, COMMON_TECHNIQUES
        if COMMON_TECHNIQUES:
            score = technique_difficulty_score(COMMON_TECHNIQUES[0])
            assert 0.0 <= score <= 1.0

    def test_common_techniques_nonempty(self):
        from models.music_instruments.extended_techniques import COMMON_TECHNIQUES
        assert len(COMMON_TECHNIQUES) > 0


class TestInstrumentsPackageExports:
    def test_all_exports(self):
        import models.music_instruments as mi
        assert hasattr(mi, "InstrumentFamily")
        assert hasattr(mi, "StringModel")
        assert hasattr(mi, "ADSREnvelope")
        assert hasattr(mi, "TuningSystem")
        assert hasattr(mi, "ExtendedTechnique")
        assert hasattr(mi, "COMMON_TECHNIQUES")
