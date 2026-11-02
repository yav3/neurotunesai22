"""Tests for audio source separation primitives."""
import numpy as np
import pytest
from models.music_audio.source_separation import (
    ideal_binary_mask, ideal_ratio_mask, wiener_filter,
    harmonic_percussive_separation, griffin_lim, SDR, SIR, SAR,
)


def test_ibm_shape():
    mix = np.random.default_rng(0).standard_normal((128, 64)) + 1j * np.random.default_rng(1).standard_normal((128, 64))
    src = mix * 0.7
    mask = ideal_binary_mask(mix, src)
    assert mask.shape == mix.shape


def test_ibm_binary():
    mix = np.ones((4, 4)) + 0j
    src = np.ones((4, 4)) * 0.6 + 0j
    mask = ideal_binary_mask(mix, src)
    assert np.all((mask == 0) | (mask == 1))


def test_irm_range():
    rng = np.random.default_rng(42)
    mix = rng.random((10, 10)) + 0.01
    src = rng.random((10, 10)) * 0.5
    mask = ideal_ratio_mask(mix, src)
    assert np.all(mask >= 0)
    assert np.all(np.isfinite(mask))


def test_wiener_sums_to_mix():
    rng = np.random.default_rng(42)
    mix = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    s1 = mix * 0.6
    s2 = mix * 0.4
    filtered = wiener_filter(mix, [s1, s2])
    total = sum(filtered)
    np.testing.assert_allclose(np.abs(total), np.abs(mix), atol=1e-6)


def test_hpss_shape():
    spec = np.random.default_rng(42).random((64, 32))
    h, p = harmonic_percussive_separation(spec, kernel_h=5, kernel_p=5)
    assert h.shape == spec.shape
    assert p.shape == spec.shape


def test_griffin_lim_shape():
    mag = np.random.default_rng(42).random((65, 20))
    signal = griffin_lim(mag, n_iter=5, hop_length=256)
    assert len(signal) > 0


def test_sdr_identical():
    ref = np.random.default_rng(42).standard_normal(1000)
    assert SDR(ref, ref) == np.inf


def test_sdr_positive_for_good():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(1000)
    est = ref + 0.1 * rng.standard_normal(1000)
    assert SDR(ref, est) > 10


def test_sir_nonneg():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(500)
    intf = rng.standard_normal(500)
    est = ref + 0.1 * intf
    assert SIR(ref, est, intf) > 0


def test_sar_nonneg():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(500)
    est = ref + 0.01 * rng.standard_normal(500)
    assert SAR(ref, est) > 0


def test_sdr_zero_ref():
    ref = np.zeros(100)
    est = np.ones(100)
    sdr = SDR(ref, est)
    # Zero reference → s_target=0, e_noise=est → num=0, den>0 → 0.0 dB
    assert sdr == 0.0 or sdr == float('-inf')


def test_hpss_complex():
    rng = np.random.default_rng(42)
    spec = (rng.random((64, 32)) + 1j * rng.random((64, 32)))
    h, p = harmonic_percussive_separation(spec, kernel_h=5, kernel_p=5)
    assert np.iscomplexobj(h)
    assert np.iscomplexobj(p)
