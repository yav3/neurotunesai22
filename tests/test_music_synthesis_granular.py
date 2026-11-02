"""Tests for granular synthesis engine."""
import numpy as np
import pytest

from models.music_synthesis.granular import (
    GrainEnvelope, EnvelopeShape, GranularSynthesizer,
)


def _sine(freq=440.0, sr=22050, dur=1.0):
    return np.sin(2 * np.pi * freq * np.arange(int(sr * dur)) / sr)


class TestGrainEnvelope:
    def test_hann_shape(self):
        env = GrainEnvelope(shape=EnvelopeShape.HANN)
        w = env.generate(100)
        assert len(w) == 100
        assert w[0] == pytest.approx(0.0, abs=1e-6)
        assert w[49] > 0.9

    def test_gaussian_peak(self):
        env = GrainEnvelope(shape=EnvelopeShape.GAUSSIAN)
        w = env.generate(101)
        assert w[50] == pytest.approx(1.0, abs=1e-3)

    def test_trapezoidal(self):
        env = GrainEnvelope(shape=EnvelopeShape.TRAPEZOIDAL, attack_ratio=0.2, release_ratio=0.2)
        w = env.generate(100)
        assert w[50] == pytest.approx(1.0)
        assert w[0] == pytest.approx(0.0, abs=0.02)

    def test_tukey(self):
        env = GrainEnvelope(shape=EnvelopeShape.TUKEY, attack_ratio=0.2, release_ratio=0.2)
        w = env.generate(200)
        assert w[100] == pytest.approx(1.0)

    def test_triangular(self):
        env = GrainEnvelope(shape=EnvelopeShape.TRIANGULAR)
        w = env.generate(101)
        assert w[50] == pytest.approx(1.0, abs=0.02)


class TestExtractGrain:
    def test_basic_extraction(self):
        src = _sine(dur=0.5)
        grain = GranularSynthesizer.extract_grain(src, 0, 512)
        assert len(grain) == 512

    def test_pitch_shift_up(self):
        src = _sine(dur=0.5)
        grain = GranularSynthesizer.extract_grain(src, 0, 512, pitch_shift=2.0)
        assert len(grain) == 512

    def test_pitch_shift_down(self):
        src = _sine(dur=0.5)
        grain = GranularSynthesizer.extract_grain(src, 0, 512, pitch_shift=0.5)
        assert len(grain) == 512

    def test_boundary(self):
        src = np.ones(100)
        grain = GranularSynthesizer.extract_grain(src, 90, 50)
        assert len(grain) == 50


class TestScatterGrains:
    def test_output_length(self):
        src = _sine(dur=1.0)
        out = GranularSynthesizer.scatter_grains(src, duration=22050)
        assert len(out) == 22050

    def test_normalized(self):
        src = _sine(dur=1.0)
        out = GranularSynthesizer.scatter_grains(src)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_pitch_spread(self):
        src = _sine(dur=1.0)
        out = GranularSynthesizer.scatter_grains(src, pitch_spread=3.0, duration=11025)
        assert len(out) == 11025


class TestTimeStretch:
    def test_slower(self):
        src = _sine(dur=0.5)
        out = GranularSynthesizer.time_stretch_granular(src, rate=0.5)
        assert len(out) > len(src) * 1.5

    def test_faster(self):
        src = _sine(dur=0.5)
        out = GranularSynthesizer.time_stretch_granular(src, rate=2.0)
        assert len(out) < len(src)

    def test_unity(self):
        src = _sine(dur=0.5)
        out = GranularSynthesizer.time_stretch_granular(src, rate=1.0)
        assert abs(len(out) - len(src)) < 4096

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            GranularSynthesizer.time_stretch_granular(np.zeros(1000), rate=0)


class TestFreeze:
    def test_output_length(self):
        src = _sine(dur=1.0)
        out = GranularSynthesizer.freeze(src, position=5000, n_grains=20)
        assert len(out) > 0

    def test_spread_zero(self):
        src = _sine(dur=1.0)
        out1 = GranularSynthesizer.freeze(src, position=5000, spread=0, seed=0)
        out2 = GranularSynthesizer.freeze(src, position=5000, spread=0, seed=0)
        np.testing.assert_array_equal(out1, out2)


class TestGranularDelay:
    def test_output_longer(self):
        src = _sine(dur=0.5)
        out = GranularSynthesizer.granular_delay(src, delay_samples=5000, n_repeats=3)
        assert len(out) > len(src)

    def test_normalized(self):
        src = _sine(dur=0.5)
        out = GranularSynthesizer.granular_delay(src, n_repeats=2)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6
