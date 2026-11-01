"""Tests for music_synthesis: oscillators, FM, additive, granular, wavetable, physical modeling."""
import numpy as np
import pytest

from models.music_synthesis import (
    SineOscillator, SawtoothOscillator, SquareOscillator, TriangleOscillator,
    NoiseGenerator, NoiseType, PulseWidthModulator,
    ADSREnvelope, FMOperator, FMAlgorithm, FMPatch, FMSynthesizer,
    AdditivePartial, AdditiveSpectrum, AdditiveSynthesizer, SpectralMorphing,
    Grain, GrainCloud, GranularSynthesizer, GranularFreeze,
    Wavetable, WavetableBank, WavetableOscillator, WavetableSynthesizer,
    KarplusStrong, WaveguideString, ModalSynthesizer, TubeSynthesizer,
)

SR = 44100
DUR = 0.1


# ── Oscillators ──────────────────────────────────────────────────────────────

class TestSineOscillator:
    def test_output_length(self):
        osc = SineOscillator(frequency=440.0, sample_rate=SR)
        out = osc.generate(DUR)
        assert len(out) == int(SR * DUR)

    def test_range(self):
        out = SineOscillator(440.0, SR).generate(DUR)
        assert np.max(out) <= 1.0 + 1e-6
        assert np.min(out) >= -1.0 - 1e-6

    def test_dtype(self):
        out = SineOscillator(440.0, SR).generate(DUR)
        assert out.dtype == np.float32


class TestSawtoothOscillator:
    def test_range(self):
        out = SawtoothOscillator(440.0, SR).generate(DUR)
        assert np.max(out) <= 1.0 + 1e-3
        assert np.min(out) >= -1.0 - 1e-3


class TestSquareOscillator:
    def test_binary_values(self):
        out = SquareOscillator(100.0, SR).generate(DUR)
        unique = np.unique(out)
        assert set(unique) == {-1.0, 1.0}


class TestTriangleOscillator:
    def test_range(self):
        out = TriangleOscillator(440.0, SR).generate(DUR)
        assert np.max(out) <= 1.0 + 1e-3
        assert np.min(out) >= -1.0 - 1e-3


class TestNoiseGenerator:
    def test_white_noise_stats(self):
        gen = NoiseGenerator(NoiseType.WHITE, SR)
        out = gen.generate(1.0, seed=42)
        assert abs(np.mean(out)) < 0.05
        assert np.std(out) > 0.5

    def test_pink_noise_length(self):
        gen = NoiseGenerator(NoiseType.PINK, SR)
        out = gen.generate(DUR, seed=42)
        assert len(out) == int(SR * DUR)

    def test_pink_noise_bounded(self):
        gen = NoiseGenerator(NoiseType.PINK, SR)
        out = gen.generate(0.5, seed=42)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


class TestPulseWidthModulator:
    def test_duty_cycle_50(self):
        pwm = PulseWidthModulator(frequency=100.0, sample_rate=SR, duty_cycle=0.5)
        out = pwm.generate(DUR)
        pos = np.sum(out > 0)
        neg = np.sum(out < 0)
        assert abs(pos - neg) < 20

    def test_modulated_output(self):
        pwm = PulseWidthModulator(frequency=100.0, sample_rate=SR, duty_cycle=0.5)
        out = pwm.generate_modulated(DUR, lfo_freq=2.0, lfo_depth=0.3)
        assert len(out) == int(SR * DUR)


# ── FM Synthesis ─────────────────────────────────────────────────────────────

class TestADSREnvelope:
    def test_starts_at_zero(self):
        env = ADSREnvelope(0.01, 0.1, 0.7, 0.2)
        out = env.generate(0.5, SR)
        assert out[0] == pytest.approx(0.0, abs=1e-6)

    def test_peak_at_one(self):
        env = ADSREnvelope(0.01, 0.1, 0.7, 0.2)
        out = env.generate(0.5, SR)
        assert np.max(out) == pytest.approx(1.0, abs=0.02)


class TestFMOperator:
    def test_render_length(self):
        op = FMOperator(ratio=1.0, index=1.0)
        out = op.render(440.0, DUR, SR)
        assert len(out) == int(SR * DUR)

    def test_modulated(self):
        mod = FMOperator(ratio=2.0).render(440.0, DUR, SR)
        carrier = FMOperator(ratio=1.0, index=3.0).render(440.0, DUR, SR, mod)
        assert len(carrier) == int(SR * DUR)

    def test_feedback(self):
        op = FMOperator(ratio=1.0, feedback=0.5)
        out = op.render(440.0, DUR, SR)
        assert len(out) == int(SR * DUR)


class TestFMAlgorithm:
    def test_simple_two_op(self):
        algo = FMAlgorithm.simple_two_op()
        assert algo.n_operators == 2
        assert algo.carriers == [0]

    def test_dx7_algorithm1(self):
        algo = FMAlgorithm.dx7_algorithm1()
        assert algo.n_operators == 6
        assert len(algo.carriers) == 2


class TestFMSynthesizer:
    def test_render_note(self):
        synth = FMSynthesizer(FMPatch.electric_piano(), SR)
        out = synth.render_note(440.0, DUR)
        assert len(out) == int(SR * DUR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_bell_patch(self):
        synth = FMSynthesizer(FMPatch.bell(), SR)
        out = synth.render_note(880.0, 0.2)
        assert len(out) == int(SR * 0.2)

    def test_velocity(self):
        synth = FMSynthesizer(FMPatch(), SR)
        loud = synth.render_note(440.0, DUR, velocity=1.0)
        soft = synth.render_note(440.0, DUR, velocity=0.3)
        assert np.max(np.abs(soft)) < np.max(np.abs(loud)) + 1e-6


# ── Additive ─────────────────────────────────────────────────────────────────

class TestAdditivePartial:
    def test_render_shape(self):
        p = AdditivePartial(440.0, 1.0, 0.0, 0.0)
        out = p.render(DUR, SR)
        assert len(out) == int(SR * DUR)

    def test_decay(self):
        p = AdditivePartial(440.0, 1.0, 0.0, 10.0)
        out = p.render(0.5, SR)
        assert abs(out[-1]) < abs(out[100])


class TestAdditiveSpectrum:
    def test_harmonic_series(self):
        s = AdditiveSpectrum.harmonic_series(220.0, 8)
        assert len(s.partials) == 8
        assert s.partials[0].frequency == 220.0

    def test_odd_harmonics(self):
        s = AdditiveSpectrum.odd_harmonics(100.0, 4)
        freqs = [p.frequency for p in s.partials]
        assert freqs == [100.0, 300.0, 500.0, 700.0]

    def test_render(self):
        s = AdditiveSpectrum.harmonic_series(440.0, 4)
        out = s.render(DUR, SR)
        assert len(out) == int(SR * DUR)


class TestAdditiveSynthesizer:
    def test_render_normalized(self):
        synth = AdditiveSynthesizer(SR)
        spec = AdditiveSpectrum.harmonic_series(440.0, 8)
        out = synth.render(spec, DUR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


class TestSpectralMorphing:
    def test_morph_endpoints(self):
        a = AdditiveSpectrum.harmonic_series(220.0, 4)
        b = AdditiveSpectrum.odd_harmonics(220.0, 4)
        out_a = SpectralMorphing.morph(a, b, 0.0, DUR, SR)
        out_b = SpectralMorphing.morph(a, b, 1.0, DUR, SR)
        assert len(out_a) == len(out_b) == int(SR * DUR)

    def test_crossfade(self):
        a = AdditiveSpectrum.harmonic_series(220.0, 4)
        b = AdditiveSpectrum.odd_harmonics(220.0, 4)
        out = SpectralMorphing.crossfade(a, b, DUR, SR)
        assert len(out) == int(SR * DUR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


# ── Granular ─────────────────────────────────────────────────────────────────

class TestGrain:
    def test_extract(self):
        source = np.sin(np.linspace(0, 20 * np.pi, 4410)).astype(np.float32)
        g = Grain(position=100, length=512)
        out = g.extract(source)
        assert len(out) == 512

    def test_pitch_shift(self):
        source = np.sin(np.linspace(0, 20 * np.pi, 4410)).astype(np.float32)
        g = Grain(position=100, length=512, pitch_shift=2.0)
        out = g.extract(source)
        assert len(out) == 512


class TestGrainCloud:
    def test_schedule(self):
        cloud = GrainCloud(density=20.0, grain_length_ms=50.0)
        grains = cloud.schedule_grains(1.0, SR, seed=42)
        assert len(grains) == 20


class TestGranularSynthesizer:
    def test_render(self):
        source = np.sin(np.linspace(0, 100 * np.pi, SR)).astype(np.float32)
        cloud = GrainCloud(density=30.0, grain_length_ms=40.0)
        synth = GranularSynthesizer(SR)
        out = synth.render(source, cloud, 0.5, seed=42)
        assert len(out) == int(SR * 0.5)

    def test_time_stretch(self):
        source = np.sin(np.linspace(0, 100 * np.pi, SR)).astype(np.float32)
        synth = GranularSynthesizer(SR)
        out = synth.time_stretch(source, 2.0)
        assert abs(len(out) - 2 * len(source)) < SR * 0.1

    def test_pitch_shift_length(self):
        source = np.sin(np.linspace(0, 100 * np.pi, SR)).astype(np.float32)
        synth = GranularSynthesizer(SR)
        out = synth.pitch_shift(source, 7.0)
        assert len(out) == len(source)


class TestGranularFreeze:
    def test_freeze(self):
        source = np.sin(np.linspace(0, 100 * np.pi, SR)).astype(np.float32)
        freeze = GranularFreeze(SR)
        out = freeze.freeze(source, 0.5, 1.0, seed=42)
        assert len(out) == SR


# ── Wavetable ────────────────────────────────────────────────────────────────

class TestWavetable:
    def test_sine(self):
        wt = Wavetable.sine(2048)
        assert len(wt.samples) == 2048
        assert np.max(np.abs(wt.samples)) <= 1.0 + 1e-6

    def test_saw_harmonics(self):
        wt = Wavetable.saw(32, 2048)
        assert len(wt.samples) == 2048

    def test_lookup_linear(self):
        wt = Wavetable.sine()
        assert wt.lookup(0.0) == pytest.approx(0.0, abs=0.01)
        assert wt.lookup(0.25) == pytest.approx(1.0, abs=0.01)

    def test_lookup_cubic(self):
        wt = Wavetable.sine()
        val = wt.lookup_cubic(0.25)
        assert val == pytest.approx(1.0, abs=0.02)


class TestWavetableBank:
    def test_basic_set(self):
        bank = WavetableBank.basic_set()
        assert len(bank.tables) == 4

    def test_morphing(self):
        bank = WavetableBank.basic_set()
        v0 = bank.lookup(0.0, 0.25)
        v1 = bank.lookup(1.0, 0.25)
        assert isinstance(v0, float)
        assert isinstance(v1, float)


class TestWavetableOscillator:
    def test_generate(self):
        osc = WavetableOscillator(sr=SR)
        out = osc.generate(440.0, DUR, position=0.0)
        assert len(out) == int(SR * DUR)

    def test_sweep(self):
        osc = WavetableOscillator(sr=SR)
        out = osc.generate_sweep(440.0, DUR, 0.0, 1.0)
        assert len(out) == int(SR * DUR)


class TestWavetableSynthesizer:
    def test_render_note(self):
        synth = WavetableSynthesizer(sr=SR)
        out = synth.render_note(440.0, DUR, position=0.5, velocity=0.8)
        assert len(out) == int(SR * DUR)
        assert out.dtype == np.float32


# ── Physical Modeling ────────────────────────────────────────────────────────

class TestKarplusStrong:
    def test_pluck_length(self):
        ks = KarplusStrong(SR)
        out = ks.pluck(440.0, DUR, seed=42)
        assert len(out) == int(SR * DUR)

    def test_pluck_decays(self):
        ks = KarplusStrong(SR)
        out = ks.pluck(220.0, 1.0, decay=0.990, seed=42)
        rms_start = np.sqrt(np.mean(out[:SR // 10] ** 2))
        rms_end = np.sqrt(np.mean(out[-SR // 10:] ** 2))
        assert rms_end < rms_start

    def test_drum(self):
        ks = KarplusStrong(SR)
        out = ks.drum(150.0, 0.2, seed=42)
        assert len(out) == int(SR * 0.2)


class TestWaveguideString:
    def test_synthesize_length(self):
        wg = WaveguideString(SR)
        out = wg.synthesize(220.0, DUR, seed=42)
        assert len(out) == int(SR * DUR)

    def test_normalized(self):
        wg = WaveguideString(SR)
        out = wg.synthesize(440.0, 0.5, seed=42)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


class TestModalSynthesizer:
    def test_string_modes(self):
        ms = ModalSynthesizer(SR)
        modes = ms.string_modes(220.0, 8)
        out = ms.synthesize(modes, DUR)
        assert len(out) == int(SR * DUR)

    def test_bar_modes(self):
        ms = ModalSynthesizer(SR)
        modes = ms.bar_modes(440.0, 6)
        out = ms.synthesize(modes, DUR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_plate_modes(self):
        ms = ModalSynthesizer(SR)
        modes = ms.plate_modes(200.0, 8)
        assert len(modes) == 8


class TestTubeSynthesizer:
    def test_synthesize_length(self):
        tube = TubeSynthesizer(SR)
        out = tube.synthesize(220.0, DUR, seed=42)
        assert len(out) == int(SR * DUR)

    def test_normalized(self):
        tube = TubeSynthesizer(SR)
        out = tube.synthesize(440.0, 0.2, n_sections=6, seed=42)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6
