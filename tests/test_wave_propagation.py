"""Tests for wave propagation module."""
import numpy as np
import pytest
from models.music_spatial.wave_propagation import (
    Medium, spherical_spreading, cylindrical_spreading,
    air_absorption_coefficient, apply_air_absorption,
    delay_samples, fractional_delay,
    WaveguideMesh2D, diffraction_edge,
    reflection_coefficient, absorption_to_impedance,
    inverse_square_law, sound_intensity_to_spl, spl_to_intensity,
    doppler_shift, fresnel_zone_radius, RayPath, trace_ray_2d,
)

SR = 44100


class TestSpreading:
    def test_spherical_unit(self):
        assert spherical_spreading(1.0) == 1.0

    def test_spherical_double(self):
        assert abs(spherical_spreading(2.0) - 0.5) < 1e-6

    def test_cylindrical_slower(self):
        assert cylindrical_spreading(4.0) > spherical_spreading(4.0)

    def test_zero_distance(self):
        assert spherical_spreading(0.0) == 0.0


class TestAirAbsorption:
    def test_positive(self):
        alpha = air_absorption_coefficient(1000.0)
        assert alpha > 0

    def test_higher_freq_more(self):
        a1 = air_absorption_coefficient(1000.0)
        a2 = air_absorption_coefficient(10000.0)
        assert a2 > a1

    def test_apply_reduces_energy(self):
        tone = np.sin(2 * np.pi * 4000 * np.arange(SR) / SR)
        out = apply_air_absorption(tone, 100.0, SR)
        assert np.sum(out ** 2) < np.sum(tone ** 2)


class TestDelay:
    def test_delay_samples(self):
        d = delay_samples(343.0, SR)
        assert abs(d - SR) < 1

    def test_fractional_delay_shape(self):
        sig = np.ones(100)
        out = fractional_delay(sig, 5.5)
        assert len(out) == 100


class TestWaveguide:
    def test_mesh_shape(self):
        mesh = WaveguideMesh2D(nx=16, ny=16)
        p = mesh.step()
        assert p.shape == (16, 16)

    def test_excite_propagates(self):
        mesh = WaveguideMesh2D(nx=20, ny=20, damping=0.0001)
        mesh.excite(10, 10, 1.0)
        for _ in range(5):
            mesh.step()
        p = mesh.step()
        assert np.max(np.abs(p)) > 0

    def test_run(self):
        mesh = WaveguideMesh2D(nx=16, ny=16)
        src = np.zeros(50)
        src[0] = 1.0
        out = mesh.run(50, 8, 8, src, 12, 12)
        assert len(out) == 50
        assert np.any(out != 0)


class TestDiffraction:
    def test_range(self):
        d = diffraction_edge(1000, 1.0, 1.0)
        assert 0 <= d <= 1

    def test_close_high(self):
        d = diffraction_edge(100, 0.1, 0.1)
        assert d > 0.3


class TestReflection:
    def test_same_impedance(self):
        assert reflection_coefficient(100, 100) == 0.0

    def test_rigid_wall(self):
        r = reflection_coefficient(100, 1e12)
        assert r > 0.99


class TestIntensity:
    def test_inverse_square(self):
        i1 = inverse_square_law(1.0, 1.0)
        i2 = inverse_square_law(1.0, 2.0)
        assert abs(i1 / i2 - 4.0) < 0.01

    def test_spl_roundtrip(self):
        spl = 80.0
        intensity = spl_to_intensity(spl)
        back = sound_intensity_to_spl(intensity)
        assert abs(back - spl) < 1e-6


class TestDoppler:
    def test_stationary(self):
        assert doppler_shift(440.0, 0.0) == 440.0

    def test_approaching(self):
        f = doppler_shift(440.0, -34.3)
        assert f > 440.0

    def test_receding(self):
        f = doppler_shift(440.0, 34.3)
        assert f < 440.0


class TestRayTracing:
    def test_trace(self):
        path = trace_ray_2d((2.0, 2.0), 0.3, 10.0, 8.0, max_reflections=5)
        assert path.reflections == 5
        assert path.total_distance > 0
        assert path.energy < 1.0

    def test_energy_decreases(self):
        path = trace_ray_2d((5.0, 4.0), 0.7, 10.0, 8.0, max_reflections=10)
        assert path.energy < 0.5


class TestMedium:
    def test_adjusted_speed(self):
        m = Medium(temperature_c=20.0)
        s = m.adjusted_speed()
        assert 340 < s < 350

    def test_cold_slower(self):
        cold = Medium(temperature_c=0.0).adjusted_speed()
        warm = Medium(temperature_c=30.0).adjusted_speed()
        assert cold < warm


class TestFresnel:
    def test_positive(self):
        r = fresnel_zone_radius(1000.0, 10.0)
        assert r > 0
