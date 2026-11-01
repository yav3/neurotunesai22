"""Tests for music_psychoacoustics: critical bands, masking, loudness, pitch perception."""
import math
import numpy as np
import pytest

from models.music_psychoacoustics.critical_bands import (
    hz_to_bark, bark_to_hz, hz_to_erb, erb_to_hz, erb_bandwidth,
    CriticalBandAnalyzer, GammatoneFilterbank,
)
from models.music_psychoacoustics.masking import (
    SimultaneousMasking, TemporalMasking, MaskingModel,
)
from models.music_psychoacoustics.loudness import (
    EqualLoudness, SpecificLoudness,
)
from models.music_psychoacoustics.pitch_perception import (
    VirtualPitch, PitchSalience, estimate_pitch,
)


# --- Critical bands ---

class TestBarkScale:
    def test_hz_to_bark_1000(self):
        b = float(hz_to_bark(1000.0))
        assert 8.0 < b < 10.0

    def test_roundtrip(self):
        for f in [500, 1000]:
            recovered = float(bark_to_hz(hz_to_bark(float(f))))
            assert abs(recovered - f) / f < 0.15

    def test_monotonic(self):
        freqs = [100, 500, 1000, 2000, 4000, 8000]
        barks = [float(hz_to_bark(float(f))) for f in freqs]
        for i in range(len(barks) - 1):
            assert barks[i] < barks[i + 1]


class TestERBScale:
    def test_hz_to_erb_1000(self):
        e = float(hz_to_erb(1000.0))
        assert 10 < e < 20

    def test_roundtrip(self):
        for f in [200, 1000, 4000]:
            assert abs(float(erb_to_hz(hz_to_erb(float(f)))) - f) < 2.0

    def test_bandwidth_positive(self):
        for f in [100, 500, 1000, 4000]:
            assert float(erb_bandwidth(float(f))) > 0


class TestCriticalBandAnalyzer:
    def test_band_index_440(self):
        cba = CriticalBandAnalyzer()
        idx = cba.band_index(440.0)
        assert 0 <= idx < 24

    def test_band_overlap(self):
        cba = CriticalBandAnalyzer()
        ov = cba.band_overlap(440.0, 500.0)
        assert 0.0 <= ov <= 1.0

    def test_overlap_same_freq(self):
        cba = CriticalBandAnalyzer()
        assert cba.band_overlap(440.0, 440.0) == 1.0

    def test_spectral_spread(self):
        cba = CriticalBandAnalyzer()
        sp = cba.spectral_spread([440.0, 880.0, 1320.0])
        assert sp >= 0.0


class TestGammatoneFilterbank:
    def test_center_freqs_count(self):
        fb = GammatoneFilterbank(n_filters=32)
        assert len(fb.center_freqs) == 32

    def test_bandwidths_positive(self):
        fb = GammatoneFilterbank(n_filters=16)
        for bw in fb.bandwidths:
            assert bw > 0

    def test_filter_response_shape(self):
        fb = GammatoneFilterbank(n_filters=8)
        freqs = np.linspace(50, 8000, 256)
        resp = fb.filter_response(freqs, 0)
        assert len(resp) == 256

    def test_excitation_pattern(self):
        fb = GammatoneFilterbank(n_filters=8)
        freqs = np.linspace(50, 8000, 256)
        spectrum = np.random.rand(256)
        exc = fb.excitation_pattern(spectrum, freqs)
        assert len(exc) == 8
        assert all(e >= 0 for e in exc)


# --- Masking ---

class TestSimultaneousMasking:
    def test_threshold_positive(self):
        sm = SimultaneousMasking()
        t = sm.masking_threshold(1000.0, 80.0, 1200.0)
        assert t >= 0

    def test_masked_flag(self):
        sm = SimultaneousMasking()
        assert isinstance(sm.masked(1000.0, 80.0, 1200.0, 30.0), bool)

    def test_composite_threshold(self):
        sm = SimultaneousMasking()
        maskers = [(500.0, 70.0), (2000.0, 75.0)]
        t = sm.composite_threshold(maskers, 1000.0)
        assert t >= 0


class TestTemporalMasking:
    def test_forward_masking(self):
        tm = TemporalMasking()
        m = tm.forward_masking(80.0, 0.05)
        assert m >= 0

    def test_forward_masking_decays(self):
        tm = TemporalMasking()
        m1 = tm.forward_masking(80.0, 0.01)
        m2 = tm.forward_masking(80.0, 0.1)
        assert m1 >= m2

    def test_backward_masking(self):
        tm = TemporalMasking()
        m = tm.backward_masking(80.0, 0.01)
        assert m >= 0


class TestMaskingModel:
    def test_audibility(self):
        mm = MaskingModel()
        result = mm.audibility(
            probe_freq=1000.0, probe_level=40.0, probe_time=0.1,
            maskers=[(800.0, 70.0, 0.0)],
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# --- Loudness ---

class TestEqualLoudness:
    def test_threshold_at_1khz(self):
        el = EqualLoudness()
        t = el.threshold_at_freq(1000.0)
        assert -5 < t < 10

    def test_phon_to_sone(self):
        el = EqualLoudness()
        s = el.phon_to_sone(40.0)
        assert 0.5 < s < 2.0

    def test_sone_to_phon_roundtrip(self):
        el = EqualLoudness()
        for phon in [20, 40, 60, 80]:
            sone = el.phon_to_sone(float(phon))
            back = el.sone_to_phon(sone)
            assert abs(back - phon) < 1.0


class TestSpecificLoudness:
    def test_compute(self):
        sl = SpecificLoudness(n_bands=8)
        excitation = np.ones(8) * 1e6
        center_freqs = np.linspace(100, 4000, 8)
        vals = sl.compute(excitation, center_freqs)
        assert len(vals) == 8
        assert all(v >= 0 for v in vals)

    def test_total_loudness(self):
        sl = SpecificLoudness(n_bands=8)
        excitation = np.ones(8) * 1e6
        center_freqs = np.linspace(100, 4000, 8)
        t = sl.total_loudness(excitation, center_freqs)
        assert t >= 0

    def test_loudness_in_sone(self):
        sl = SpecificLoudness(n_bands=8)
        excitation = np.ones(8) * 1e6
        center_freqs = np.linspace(100, 4000, 8)
        s = sl.loudness_in_sone(excitation, center_freqs)
        assert s >= 0


# --- Pitch perception ---

class TestVirtualPitch:
    def test_missing_fundamental(self):
        vp = VirtualPitch()
        f0 = vp.missing_fundamental([200.0, 300.0, 400.0, 500.0])
        assert 90 < f0 < 110

    def test_harmonic_template(self):
        vp = VirtualPitch()
        freqs = np.array([440.0, 880.0, 1320.0, 1760.0])
        mags = np.array([1.0, 0.5, 0.3, 0.2])
        score = vp.harmonic_template_match(freqs, mags, 440.0)
        assert score > 0.0

    def test_estimate_f0(self):
        vp = VirtualPitch()
        freqs = np.array([200.0, 400.0, 600.0, 800.0])
        mags = np.array([1.0, 0.8, 0.5, 0.3])
        f0, score = vp.estimate_f0(freqs, mags)
        assert f0 > 0 and score > 0


class TestPitchSalience:
    def test_salience(self):
        ps = PitchSalience()
        t = np.arange(1024) / 16000.0
        sig = np.sin(2 * np.pi * 440 * t)
        autocorr = np.correlate(sig, sig, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        peaks = ps.salience(autocorr, 16000)
        assert isinstance(peaks, list)

    def test_pitch_strength(self):
        ps = PitchSalience()
        autocorr = np.array([1.0, 0.8, 0.5, 0.3, 0.1, 0.3, 0.5, 0.8])
        strength = ps.pitch_strength(autocorr)
        assert 0.0 <= strength <= 1.0


class TestEstimatePitch:
    def test_pure_tone(self):
        freqs = np.array([220.0, 440.0, 660.0, 880.0])
        mags = np.array([1.0, 0.8, 0.5, 0.3])
        f0, score = estimate_pitch(freqs, mags)
        assert f0 > 0
