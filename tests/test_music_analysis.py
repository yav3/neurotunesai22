"""Tests for music_analysis: key detection, beat tracking, structure, loudness, tonality."""
import numpy as np
import pytest
from models.music_analysis.key_detection import KeyDetector, detect_key
from models.music_analysis.beat_tracking import TempoEstimator, BeatTracker
from models.music_analysis.structure import StructureAnalyzer, SectionLabeler
from models.music_analysis.loudness import LoudnessAnalyzer, DynamicRangeAnalyzer
from models.music_analysis.tonality import TonalityAnalyzer, HarmonicRhythm


class TestKeyDetection:
    def test_c_major_chroma(self):
        chroma = np.zeros(12)
        chroma[[0, 2, 4, 5, 7, 9, 11]] = 1.0
        key, mode, corr = detect_key(chroma)
        assert key == "C"
        assert mode == "major"
        assert corr > 0.5

    def test_a_minor_chroma(self):
        chroma = np.zeros(12)
        chroma[[9, 11, 0, 2, 4, 5, 7]] = 1.0
        key, mode, corr = detect_key(chroma)
        assert mode in ("major", "minor")
        assert corr > 0.3

    def test_detector_class(self):
        det = KeyDetector()
        chroma = np.ones(12)
        key, mode, corr = det.detect(chroma)
        assert key in "C C# D D# E F F# G G# A A# B".split()
        assert mode in ("major", "minor")

    def test_all_zeros(self):
        chroma = np.zeros(12)
        key, mode, corr = detect_key(chroma)
        assert isinstance(key, str)

    def test_single_pitch_class(self):
        for pc in range(12):
            chroma = np.zeros(12)
            chroma[pc] = 1.0
            key, mode, corr = detect_key(chroma)
            assert isinstance(corr, float)

    def test_correlation_bounded(self):
        chroma = np.random.RandomState(42).rand(12)
        _, _, corr = detect_key(chroma)
        assert -1.0 <= corr <= 1.0


class TestBeatTracking:
    def test_tempo_estimator_basic(self):
        sr = 22050
        hop = 512
        bpm = 120.0
        beat_period = int(60.0 / bpm * sr / hop)
        onset_env = np.zeros(500)
        for i in range(0, 500, beat_period):
            onset_env[i] = 1.0
        est = TempoEstimator()
        tempo = est.estimate(onset_env, sr, hop)
        assert 80 < tempo < 180

    def test_beat_tracker_returns_array(self):
        onset_env = np.zeros(200)
        onset_env[::20] = 1.0
        bt = BeatTracker()
        beats = bt.track(onset_env, tempo=120.0)
        assert isinstance(beats, np.ndarray)

    def test_empty_signal(self):
        est = TempoEstimator()
        tempo = est.estimate(np.array([]), 22050)
        assert tempo >= 0

    def test_constant_signal(self):
        est = TempoEstimator()
        tempo = est.estimate(np.ones(100), 22050)
        assert isinstance(tempo, float)


class TestStructure:
    def test_ssm_shape(self):
        features = np.random.RandomState(0).rand(20, 12)
        sa = StructureAnalyzer()
        ssm = sa.compute_ssm(features)
        assert ssm.shape == (20, 20)
        assert np.allclose(np.diag(ssm), 1.0, atol=0.01)

    def test_ssm_symmetric(self):
        features = np.random.RandomState(1).rand(15, 8)
        sa = StructureAnalyzer()
        ssm = sa.compute_ssm(features)
        assert np.allclose(ssm, ssm.T, atol=1e-10)

    def test_novelty_curve_length(self):
        features = np.random.RandomState(2).rand(30, 12)
        sa = StructureAnalyzer()
        ssm = sa.compute_ssm(features)
        novelty = sa.novelty_curve(ssm)
        assert len(novelty) == 30

    def test_find_boundaries(self):
        features = np.random.RandomState(3).rand(50, 12)
        sa = StructureAnalyzer()
        ssm = sa.compute_ssm(features)
        bounds = sa.find_boundaries(ssm, threshold=0.3)
        assert isinstance(bounds, np.ndarray)
        for b in bounds:
            assert 0 <= b < 50

    def test_section_labeler(self):
        sl = SectionLabeler()
        features = np.random.RandomState(4).rand(40, 12)
        segments = sl.label_segments(features, boundaries=[0, 10, 20, 30])
        assert isinstance(segments, list)
        for start, end, label in segments:
            assert start < end
            assert isinstance(label, str)


class TestLoudness:
    def test_rms(self):
        signal = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
        la = LoudnessAnalyzer()
        rms = la.rms(signal)
        assert abs(rms - 1 / np.sqrt(2)) < 0.01

    def test_peak(self):
        signal = 0.5 * np.ones(100)
        la = LoudnessAnalyzer()
        assert la.peak(signal) == pytest.approx(0.5)

    def test_rms_db(self):
        la = LoudnessAnalyzer()
        signal = np.ones(100)
        assert la.rms_db(signal) == pytest.approx(0.0, abs=0.1)

    def test_peak_db(self):
        la = LoudnessAnalyzer()
        signal = np.ones(100) * 0.5
        assert la.peak_db(signal) < 0.0

    def test_lufs_approximate(self):
        sr = 44100
        signal = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(sr * 2) / sr)
        la = LoudnessAnalyzer()
        lufs = la.lufs_approximate(signal, sr)
        assert lufs < 0

    def test_crest_factor(self):
        dra = DynamicRangeAnalyzer()
        signal = np.sin(2 * np.pi * 440 * np.arange(1000) / 44100)
        cf = dra.crest_factor(signal)
        assert cf > 1.0

    def test_lra_estimate(self):
        dra = DynamicRangeAnalyzer()
        signal = np.random.RandomState(5).randn(44100)
        lra = dra.lra_estimate(signal, sr=44100)
        assert isinstance(lra, float)


class TestTonality:
    def test_chromatic_ratio(self):
        chroma = np.zeros(12)
        chroma[[0, 2, 4, 5, 7, 9, 11]] = 1.0
        ta = TonalityAnalyzer()
        ratio = ta.chromatic_ratio(chroma, key_root=0)
        assert 0.0 <= ratio <= 1.0

    def test_fully_chromatic(self):
        chroma = np.ones(12)
        ta = TonalityAnalyzer()
        ratio = ta.chromatic_ratio(chroma, key_root=0)
        assert ratio > 0.3

    def test_mode_mixture(self):
        chroma_seq = np.random.RandomState(6).rand(10, 12)
        ta = TonalityAnalyzer()
        score = ta.mode_mixture_score(chroma_seq)
        assert 0.0 <= score <= 1.0

    def test_harmonic_rhythm_change_rate(self):
        chroma_seq = np.random.RandomState(7).rand(20, 12)
        hr = HarmonicRhythm()
        rate = hr.change_rate(chroma_seq)
        assert 0.0 <= rate <= 1.0

    def test_avg_chord_duration(self):
        chroma_seq = np.random.RandomState(8).rand(30, 12)
        hr = HarmonicRhythm()
        dur = hr.average_chord_duration(chroma_seq)
        assert dur > 0.0


class TestRhythmAnalysis:
    def test_ioi_simple(self):
        from models.music_analysis.rhythm_analysis import inter_onset_intervals
        onsets = np.array([0.0, 0.5, 1.0, 1.5])
        iois = inter_onset_intervals(onsets)
        assert len(iois) == 3
        assert np.allclose(iois, 0.5)

    def test_ioi_statistics(self):
        from models.music_analysis.rhythm_analysis import ioi_statistics
        iois = np.array([0.5, 0.5, 0.5, 0.5])
        stats = ioi_statistics(iois)
        assert stats["mean"] == pytest.approx(0.5)
        assert stats["std"] == pytest.approx(0.0, abs=1e-10)

    def test_swing_ratio_straight(self):
        from models.music_analysis.rhythm_analysis import swing_ratio
        onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        ratio = swing_ratio(onsets, beat_duration=1.0)
        assert 0.45 < ratio < 0.55

    def test_microtiming_deviations(self):
        from models.music_analysis.rhythm_analysis import microtiming_deviations
        onsets = np.array([0.01, 0.49, 1.02, 1.48])
        grid = np.array([0.0, 0.5, 1.0, 1.5])
        devs = microtiming_deviations(onsets, grid)
        assert len(devs) == 4
        assert all(abs(d) < 0.05 for d in devs)

    def test_metric_weight_pattern_4_4(self):
        from models.music_analysis.rhythm_analysis import metric_weight_pattern
        pattern = metric_weight_pattern(4, subdivisions=2)
        assert pattern[0] > pattern[1]
        assert len(pattern) == 8

    def test_rhythmic_density(self):
        from models.music_analysis.rhythm_analysis import rhythmic_density
        onsets = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        density = rhythmic_density(onsets, duration=1.0)
        assert density == pytest.approx(5.0)

    def test_onset_entropy_uniform(self):
        from models.music_analysis.rhythm_analysis import onset_entropy
        positions = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        ent = onset_entropy(positions, n_bins=8)
        assert ent > 2.0

    def test_groove_profile_valid(self):
        from models.music_analysis.rhythm_analysis import compute_groove_profile
        onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        profile = compute_groove_profile(onsets, grid, 120.0)
        assert hasattr(profile, "swing")
        assert hasattr(profile, "mean_deviation")

    def test_syncopation_lhl_downbeats(self):
        from models.music_analysis.rhythm_analysis import syncopation_lhl
        onsets = np.array([0.0, 1.0, 2.0, 3.0])
        score = syncopation_lhl(onsets, beats_per_bar=4)
        assert score >= 0.0

    def test_metric_clarity(self):
        from models.music_analysis.rhythm_analysis import metric_clarity
        onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        clarity = metric_clarity(onsets, tempo_bpm=120.0)
        assert 0.0 <= clarity <= 1.0


class TestSpectralAnalysis:
    def test_spectral_centroid(self):
        from models.music_analysis.spectral_analysis import spectral_centroid
        freqs = np.linspace(0, 11025, 512)
        mags = np.zeros(512)
        mags[100] = 1.0
        c = spectral_centroid(mags, freqs)
        assert abs(c - freqs[100]) < 50.0

    def test_spectral_spread(self):
        from models.music_analysis.spectral_analysis import spectral_spread, spectral_centroid
        freqs = np.linspace(0, 11025, 512)
        mags = np.zeros(512)
        mags[100] = 1.0
        c = spectral_centroid(mags, freqs)
        s = spectral_spread(mags, freqs, c)
        assert s >= 0.0

    def test_spectral_flatness_pure_tone(self):
        from models.music_analysis.spectral_analysis import spectral_flatness
        mags = np.zeros(256)
        mags[50] = 1.0
        flat = spectral_flatness(mags)
        assert flat < 0.01

    def test_spectral_flatness_white_noise(self):
        from models.music_analysis.spectral_analysis import spectral_flatness
        rng = np.random.RandomState(42)
        mags = np.abs(rng.randn(256)) + 0.01
        flat = spectral_flatness(mags)
        assert flat > 0.5

    def test_spectral_crest(self):
        from models.music_analysis.spectral_analysis import spectral_crest
        mags = np.ones(256)
        crest = spectral_crest(mags)
        assert crest == pytest.approx(1.0)

    def test_spectral_rolloff(self):
        from models.music_analysis.spectral_analysis import spectral_rolloff
        freqs = np.linspace(0, 11025, 512)
        mags = np.zeros(512)
        mags[:100] = 1.0
        rolloff = spectral_rolloff(mags, freqs, threshold=0.85)
        assert rolloff < freqs[200]

    def test_spectral_slope(self):
        from models.music_analysis.spectral_analysis import spectral_slope
        freqs = np.linspace(100, 11025, 512)
        mags = np.linspace(1.0, 0.1, 512)
        slope = spectral_slope(mags, freqs)
        assert slope < 0

    def test_spectral_flux(self):
        from models.music_analysis.spectral_analysis import spectral_flux
        current = np.ones(256)
        previous = np.zeros(256)
        flux = spectral_flux(current, previous)
        assert flux > 0

    def test_compute_descriptors(self):
        from models.music_analysis.spectral_analysis import compute_spectral_descriptors
        freqs = np.linspace(0, 11025, 512)
        mags = np.abs(np.random.RandomState(0).randn(512)) + 0.01
        desc = compute_spectral_descriptors(mags, freqs)
        assert hasattr(desc, "centroid")
        assert hasattr(desc, "spread")
        assert hasattr(desc, "flatness")

    def test_formant_estimation(self):
        from models.music_analysis.spectral_analysis import estimate_formants
        freqs = np.linspace(0, 4000, 2048)
        mags = np.zeros(2048)
        mags[100] = 2.0
        mags[400] = 1.5
        mags[700] = 1.0
        formants = estimate_formants(mags, freqs, n_formants=3)
        assert len(formants) <= 3

    def test_hnr_harmonic_signal(self):
        from models.music_analysis.spectral_analysis import harmonic_to_noise_ratio
        freqs = np.linspace(0, 11025, 512)
        mags = np.zeros(512)
        f0 = 440.0
        for h in range(1, 6):
            idx = int(h * f0 / 11025 * 512)
            if idx < 512:
                mags[idx] = 1.0 / h
        hnr = harmonic_to_noise_ratio(mags, freqs, f0)
        assert hnr > 5.0


class TestAnalysisPackageExports:
    def test_new_exports(self):
        import models.music_analysis as ma
        assert hasattr(ma, "MetricLevel")
        assert hasattr(ma, "GrooveProfile")
        assert hasattr(ma, "SpectralDescriptors")
        assert hasattr(ma, "spectral_centroid")
