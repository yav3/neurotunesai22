"""Tests for music_psychoacoustics: critical bands, masking, loudness, pitch, timbre, scene."""
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
from models.music_psychoacoustics.timbre import (
    TimbreDescriptors, TimbreSpace,
    spectral_centroid, spectral_spread, spectral_flux,
    spectral_rolloff, spectral_flatness, brightness,
    attack_time, log_attack_time, temporal_centroid,
    zero_crossing_rate, compute_timbre_descriptors,
)
from models.music_psychoacoustics.auditory_scene import (
    AuditoryStream, StreamSegregation, AuditoryGrouping,
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


# --- Timbre ---

class TestSpectralCentroid:
    def test_single_peak(self):
        freqs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        mag = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        assert spectral_centroid(mag, freqs) == 300.0

    def test_uniform_spectrum(self):
        freqs = np.array([100.0, 200.0, 300.0])
        mag = np.ones(3)
        sc = spectral_centroid(mag, freqs)
        assert abs(sc - 200.0) < 1e-6

    def test_silence(self):
        freqs = np.array([100.0, 200.0])
        mag = np.zeros(2)
        assert spectral_centroid(mag, freqs) == 0.0


class TestSpectralSpread:
    def test_single_bin(self):
        freqs = np.array([100.0, 200.0, 300.0])
        mag = np.array([0.0, 1.0, 0.0])
        assert spectral_spread(mag, freqs) == 0.0

    def test_wide_spread(self):
        freqs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        mag = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
        assert spectral_spread(mag, freqs) > 0


class TestSpectralFlux:
    def test_identical_frames(self):
        mag = np.ones(64)
        assert spectral_flux(mag, mag) == 0.0

    def test_different_frames(self):
        assert spectral_flux(np.ones(10), np.zeros(10)) > 0


class TestSpectralRolloff:
    def test_concentrated_low(self):
        freqs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        mag = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        assert spectral_rolloff(mag, freqs) == 100.0

    def test_silence(self):
        freqs = np.array([100.0, 200.0])
        assert spectral_rolloff(np.zeros(2), freqs) == 0.0


class TestSpectralFlatness:
    def test_white_noise(self):
        mag = np.ones(100)
        sf = spectral_flatness(mag)
        assert abs(sf - 1.0) < 0.01

    def test_pure_tone(self):
        mag = np.zeros(100)
        mag[25] = 10.0
        sf = spectral_flatness(mag)
        assert sf < 0.1


class TestBrightness:
    def test_all_above_cutoff(self):
        freqs = np.array([2000.0, 3000.0, 4000.0])
        mag = np.ones(3)
        assert brightness(mag, freqs, cutoff=1500.0) == 1.0

    def test_all_below_cutoff(self):
        freqs = np.array([100.0, 200.0, 300.0])
        mag = np.ones(3)
        assert brightness(mag, freqs, cutoff=1500.0) == 0.0


class TestAttackTime:
    def test_sharp_attack(self):
        env = np.zeros(1000)
        env[10:50] = np.linspace(0, 1, 40)
        env[50:] = np.linspace(1, 0.5, 950)
        at = attack_time(env, sr=1000.0)
        assert 0.0 < at < 0.1

    def test_empty_envelope(self):
        assert attack_time(np.array([])) == 0.0

    def test_log_attack_time_negative_for_fast(self):
        env = np.zeros(100)
        env[0] = 1.0
        env[1:] = 0.5
        lat = log_attack_time(env, sr=44100.0)
        assert lat < 0


class TestTemporalCentroid:
    def test_early_energy(self):
        env = np.zeros(100)
        env[:10] = 1.0
        tc = temporal_centroid(env, sr=100.0)
        assert tc < 0.15

    def test_late_energy(self):
        env = np.zeros(100)
        env[80:] = 1.0
        tc = temporal_centroid(env, sr=100.0)
        assert tc > 0.7


class TestZeroCrossingRate:
    def test_pure_sine(self):
        t = np.arange(1000) / 1000.0
        sig = np.sin(2 * np.pi * 10 * t)
        zcr = zero_crossing_rate(sig)
        assert 0.01 < zcr < 0.05

    def test_constant(self):
        assert zero_crossing_rate(np.ones(100)) == 0.0

    def test_short_signal(self):
        assert zero_crossing_rate(np.array([1.0])) == 0.0


class TestComputeTimbreDescriptors:
    def test_sine_tone(self):
        sr = 22050.0
        t = np.arange(4096) / sr
        sig = np.sin(2 * np.pi * 440 * t)
        d = compute_timbre_descriptors(sig, sr)
        assert isinstance(d, TimbreDescriptors)
        assert d.spectral_centroid > 0
        assert d.zero_crossing_rate > 0

    def test_short_signal_padded(self):
        d = compute_timbre_descriptors(np.array([1.0, -1.0, 0.5]))
        assert isinstance(d, TimbreDescriptors)

    def test_noise_higher_flatness(self):
        rng = np.random.RandomState(42)
        noise = rng.randn(4096)
        sr = 22050.0
        t = np.arange(4096) / sr
        tone = np.sin(2 * np.pi * 440 * t)
        d_noise = compute_timbre_descriptors(noise, sr)
        d_tone = compute_timbre_descriptors(tone, sr)
        assert d_noise.spectral_flatness > d_tone.spectral_flatness


class TestTimbreSpace:
    @pytest.fixture
    def sample_descriptors(self):
        rng = np.random.RandomState(42)
        descs = []
        for _ in range(20):
            sr = 22050.0
            freq = rng.uniform(200, 2000)
            t = np.arange(4096) / sr
            sig = np.sin(2 * np.pi * freq * t) + 0.1 * rng.randn(4096)
            descs.append(compute_timbre_descriptors(sig, sr))
        return descs

    def test_fit_transform(self, sample_descriptors):
        ts = TimbreSpace(n_dimensions=3)
        ts.fit(sample_descriptors)
        point = ts.transform(sample_descriptors[0])
        assert len(point) == 3

    def test_distance_self_zero(self, sample_descriptors):
        ts = TimbreSpace(n_dimensions=3)
        ts.fit(sample_descriptors)
        d = ts.distance(sample_descriptors[0], sample_descriptors[0])
        assert abs(d) < 1e-6

    def test_distance_triangle_inequality(self, sample_descriptors):
        ts = TimbreSpace(n_dimensions=3)
        ts.fit(sample_descriptors)
        a, b, c = sample_descriptors[:3]
        assert ts.distance(a, c) <= ts.distance(a, b) + ts.distance(b, c) + 1e-10

    def test_unfitted_fallback(self):
        ts = TimbreSpace(n_dimensions=2)
        d = compute_timbre_descriptors(np.sin(np.arange(4096) * 0.1))
        point = ts.transform(d)
        assert len(point) == 2


# --- Auditory scene analysis ---

class TestStreamSegregation:
    def test_freq_to_semitones_octave(self):
        ss = StreamSegregation()
        assert abs(ss.freq_to_semitones(220.0, 440.0) - 12.0) < 0.01

    def test_segregation_prob_small_sep(self):
        ss = StreamSegregation()
        p = ss.segregation_probability(2.0)
        assert p < 0.3

    def test_segregation_prob_large_sep(self):
        ss = StreamSegregation()
        p = ss.segregation_probability(15.0)
        assert p > 0.7

    def test_segregation_prob_increases_with_sep(self):
        ss = StreamSegregation()
        p_low = ss.segregation_probability(3.0)
        p_high = ss.segregation_probability(10.0)
        assert p_high > p_low

    def test_tempo_effect(self):
        ss = StreamSegregation()
        p_fast = ss.segregation_probability(8.0, ioi_seconds=0.05)
        p_slow = ss.segregation_probability(8.0, ioi_seconds=0.4)
        assert p_fast > p_slow

    def test_segregate_two_streams(self):
        ss = StreamSegregation()
        freqs = np.array([200.0, 800.0, 210.0, 790.0, 205.0, 810.0])
        streams = ss.segregate(freqs)
        assert len(streams) == 2
        assert sum(s.n_events for s in streams) == 6

    def test_segregate_single_stream(self):
        ss = StreamSegregation()
        freqs = np.array([440.0, 445.0, 438.0, 442.0])
        streams = ss.segregate(freqs)
        assert len(streams) == 1

    def test_segregate_empty(self):
        ss = StreamSegregation()
        assert ss.segregate(np.array([])) == []

    def test_buildup_rate_monotonic(self):
        ss = StreamSegregation()
        curve = ss.buildup_rate(8.0, n_repetitions=20)
        assert len(curve) == 20
        for i in range(len(curve) - 1):
            assert curve[i + 1] >= curve[i] - 1e-10

    def test_buildup_rate_zero_sep(self):
        ss = StreamSegregation()
        curve = ss.buildup_rate(0.0)
        assert np.allclose(curve, 0.0)


class TestAuditoryGrouping:
    def test_proximity_score_same_point(self):
        ag = AuditoryGrouping()
        assert abs(ag.proximity_score(0.0, 0.0) - 1.0) < 1e-6

    def test_proximity_score_far(self):
        ag = AuditoryGrouping()
        s = ag.proximity_score(10.0, 20.0)
        assert s < 0.01

    def test_similarity_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(AuditoryGrouping.similarity_score(v, v) - 1.0) < 1e-6

    def test_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(AuditoryGrouping.similarity_score(a, b)) < 1e-6

    def test_common_fate_correlated(self):
        t = np.arange(100) * 0.01
        a = np.sin(2 * np.pi * 5 * t)
        b = np.sin(2 * np.pi * 5 * t + 0.1)
        score = AuditoryGrouping.common_fate_score(a, b)
        assert score > 0.9

    def test_common_fate_uncorrelated(self):
        rng = np.random.RandomState(42)
        a = rng.randn(100)
        b = rng.randn(100)
        score = AuditoryGrouping.common_fate_score(a, b)
        assert abs(score) < 0.5

    def test_good_continuation_linear(self):
        freqs = np.linspace(200, 400, 10)
        times = np.linspace(0, 1, 10)
        score = AuditoryGrouping.good_continuation_score(freqs, times)
        assert score > 0.9

    def test_good_continuation_erratic(self):
        rng = np.random.RandomState(42)
        freqs = rng.uniform(100, 1000, 10)
        times = np.linspace(0, 1, 10)
        score = AuditoryGrouping.good_continuation_score(freqs, times)
        assert score < 0.5

    def test_group_events_cluster(self):
        ag = AuditoryGrouping()
        freqs = np.array([440.0, 445.0, 1200.0, 1210.0])
        times = np.array([0.0, 0.1, 0.0, 0.1])
        groups = ag.group_events(freqs, times, proximity_threshold=0.3)
        assert len(groups) == 2

    def test_group_events_empty(self):
        ag = AuditoryGrouping()
        assert ag.group_events(np.array([]), np.array([])) == []

    def test_closure_detect_loud_surround(self):
        mag = np.ones(100) * 10.0
        mag[40:50] = 5.0
        assert AuditoryGrouping.closure_detect(mag, 40, 50, threshold_db=10.0)

    def test_closure_detect_deep_gap(self):
        mag = np.ones(100) * 10.0
        mag[40:50] = 0.001
        assert not AuditoryGrouping.closure_detect(mag, 40, 50, threshold_db=6.0)
