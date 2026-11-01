"""Tests for reverb models and sound propagation."""
import numpy as np
import pytest
from models.music_acoustics.reverb_models import (
    ReverbParameters, estimate_rt60, generate_impulse_response,
    apply_reverb, clarity_c80, definition_d50, early_decay_time,
)
from models.music_acoustics.sound_propagation import (
    inverse_square_attenuation, air_absorption_coefficient,
    doppler_shift, interference_pattern, diffraction_coefficient,
    ground_reflection_coefficient,
)

SR = 16000


class TestReverbModels:
    def test_estimate_rt60_exponential(self):
        t = np.arange(SR * 2) / SR
        ir = np.exp(-6.91 / 1.5 * t) * np.random.default_rng(42).standard_normal(len(t)) * 0.01
        rt = estimate_rt60(ir, SR)
        assert 0.5 < rt < 5.0

    def test_generate_ir_length(self):
        p = ReverbParameters(rt60=1.0)
        ir = generate_impulse_response(p, SR, duration=0.5)
        assert len(ir) == int(SR * 0.5)

    def test_generate_ir_normalized(self):
        p = ReverbParameters(rt60=1.0)
        ir = generate_impulse_response(p, SR)
        assert np.max(np.abs(ir)) <= 1.01

    def test_apply_reverb_longer(self):
        sig = np.ones(1000)
        ir = np.array([1.0, 0.5, 0.25])
        out = apply_reverb(sig, ir)
        assert len(out) > len(sig)

    def test_apply_reverb_empty(self):
        out = apply_reverb(np.array([]), np.ones(10))
        assert len(out) == 0

    def test_clarity_c80_positive_for_early_heavy(self):
        ir = np.zeros(SR)
        ir[0] = 1.0
        c = clarity_c80(ir, SR)
        assert c > 0

    def test_definition_d50_range(self):
        ir = np.random.default_rng(42).standard_normal(SR)
        d = definition_d50(ir, SR)
        assert 0.0 <= d <= 1.0

    def test_edt_nonneg(self):
        t = np.arange(SR) / SR
        ir = np.exp(-6.91 * t)
        edt = early_decay_time(ir, SR)
        assert edt >= 0

    def test_estimate_rt60_silence(self):
        assert estimate_rt60(np.zeros(SR), SR) == 0.0


class TestSoundPropagation:
    def test_inverse_square_6db_per_doubling(self):
        att = inverse_square_attenuation(2.0, 1.0)
        assert abs(att - (-6.02)) < 0.1

    def test_inverse_square_zero_at_ref(self):
        att = inverse_square_attenuation(1.0, 1.0)
        assert abs(att) < 0.01

    def test_air_absorption_positive(self):
        a = air_absorption_coefficient(4000.0)
        assert a > 0

    def test_air_absorption_increases_with_freq(self):
        a1 = air_absorption_coefficient(1000.0)
        a2 = air_absorption_coefficient(8000.0)
        assert a2 > a1

    def test_doppler_approaching(self):
        f_shifted = doppler_shift(440.0, 30.0, 0.0)
        assert f_shifted > 440.0

    def test_doppler_receding(self):
        f_shifted = doppler_shift(440.0, -30.0, 0.0)
        assert f_shifted < 440.0

    def test_interference_constructive(self):
        amp = interference_pattern(343.0, 1.0, 2.0)
        assert amp > 0.9

    def test_interference_destructive(self):
        amp = interference_pattern(343.0, 1.0, 1.5)
        assert amp < 0.1

    def test_diffraction_range(self):
        d = diffraction_coefficient(1000.0, 2.0, 5.0, 5.0)
        assert 0.0 <= d <= 1.0

    def test_ground_reflection_magnitude(self):
        R = ground_reflection_coefficient(1000.0)
        assert abs(R) <= 1.01

    def test_ground_reflection_zero_freq(self):
        R = ground_reflection_coefficient(0.0)
        assert abs(R) <= 1.01
