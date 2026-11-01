"""Tests for the music_rhythm module: tempo, beat, meter, groove, pattern."""
import numpy as np
import pytest

from models.music_rhythm.tempo import (
    bpm_to_period, period_to_bpm, tempo_from_onsets,
    tempo_from_autocorrelation, tempo_from_spectral_flux, TempoEstimator,
)
from models.music_rhythm.beat import (
    inter_beat_intervals, beat_strength, beat_phase_coherence,
    dynamic_programming_beats, BeatTracker,
)
from models.music_rhythm.meter import (
    metric_hierarchy, metric_weight_profile, downbeat_detection,
    time_signature_from_beats, MeterAnalyzer,
)
from models.music_rhythm.groove import (
    quantize_onsets, humanize_grid, microtiming_deviations,
    swing_ratio, syncopation_index, groove_template_distance,
    GrooveAnalyzer,
)
from models.music_rhythm.pattern import (
    RhythmPattern, euclidean_rhythm, pattern_similarity,
    necklace_equivalence, rhythm_complexity, pattern_from_ioi, pattern_to_ioi,
)


# ── Tempo ────────────────────────────────────────────────────────────


class TestBpmConversion:
    def test_roundtrip(self):
        for bpm in [60.0, 120.0, 180.0]:
            period = bpm_to_period(bpm)
            assert abs(period_to_bpm(period) - bpm) < 0.01

    def test_zero_bpm(self):
        assert bpm_to_period(0.0) == 0.0

    def test_zero_period(self):
        assert period_to_bpm(0.0) == 0.0

    def test_negative(self):
        assert bpm_to_period(-10.0) == 0.0
        assert period_to_bpm(-5.0) == 0.0


class TestTempoFromOnsets:
    def test_regular_120bpm(self):
        onsets = np.arange(0, 10, 0.5)  # 0.5s apart = 120 BPM
        bpm = tempo_from_onsets(onsets)
        assert 110 < bpm < 130

    def test_too_few_onsets(self):
        assert tempo_from_onsets(np.array([1.0])) == 0.0
        assert tempo_from_onsets(np.array([])) == 0.0

    def test_bpm_range_filtering(self):
        onsets = np.arange(0, 10, 0.5)
        bpm = tempo_from_onsets(onsets, min_bpm=60, max_bpm=200)
        assert 60 <= bpm <= 200


class TestTempoFromAutocorrelation:
    def test_sinusoidal_pulse(self):
        sr = 22050.0
        dur = 5.0
        bpm_target = 120.0
        t = np.arange(0, dur, 1.0 / sr)
        freq = bpm_target / 60.0
        signal = np.sin(2 * np.pi * freq * t)
        signal = np.maximum(signal, 0) ** 2
        bpm, acf = tempo_from_autocorrelation(signal, sr)
        assert bpm > 0
        assert len(acf) > 0

    def test_empty_signal(self):
        bpm, acf = tempo_from_autocorrelation(np.array([]))
        assert bpm == 0.0


class TestTempoFromSpectralFlux:
    def test_returns_positive_bpm(self):
        sr = 22050.0
        t = np.arange(0, 3.0, 1.0 / sr)
        signal = np.sin(2 * np.pi * 2.0 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        bpm = tempo_from_spectral_flux(signal, sr)
        assert bpm >= 0

    def test_silent_signal(self):
        assert tempo_from_spectral_flux(np.zeros(22050), 22050.0) == 0.0


class TestTempoEstimator:
    def test_estimate_returns_dict(self):
        sr = 22050.0
        t = np.arange(0, 3.0, 1.0 / sr)
        signal = np.sin(2 * np.pi * 2.0 * t)
        est = TempoEstimator(sr=sr)
        result = est.estimate(signal)
        assert "bpm" in result
        assert "confidence" in result

    def test_octave_correct(self):
        est = TempoEstimator()
        assert 60 <= est.octave_correct(30.0) <= 180
        assert 60 <= est.octave_correct(240.0) <= 180
        assert est.octave_correct(120.0) == 120.0
        assert est.octave_correct(0.0) == 0.0


# ── Beat ─────────────────────────────────────────────────────────────


class TestInterBeatIntervals:
    def test_regular(self):
        beats = np.array([0.0, 0.5, 1.0, 1.5])
        ibi = inter_beat_intervals(beats)
        assert len(ibi) == 3
        np.testing.assert_allclose(ibi, [0.5, 0.5, 0.5])

    def test_single_beat(self):
        assert len(inter_beat_intervals(np.array([1.0]))) == 0

    def test_empty(self):
        assert len(inter_beat_intervals(np.array([]))) == 0


class TestBeatStrength:
    def test_output_length(self):
        signal = np.random.randn(44100)
        frames = np.array([0, 10, 20, 30])
        strengths = beat_strength(signal, frames)
        assert len(strengths) == 4

    def test_nonnegative(self):
        signal = np.random.randn(22050)
        frames = np.array([0, 5, 10])
        strengths = beat_strength(signal, frames)
        assert np.all(strengths >= 0)


class TestBeatPhaseCoherence:
    def test_perfect_regularity(self):
        beats = np.arange(0, 5, 0.5)
        assert beat_phase_coherence(beats) > 0.95

    def test_irregular(self):
        beats = np.array([0.0, 0.3, 0.9, 1.1, 2.0])
        coh = beat_phase_coherence(beats)
        assert 0 <= coh <= 1

    def test_too_few(self):
        assert beat_phase_coherence(np.array([1.0])) == 1.0


class TestDynamicProgrammingBeats:
    def test_peaks_detected(self):
        env = np.zeros(100)
        for i in range(0, 100, 10):
            env[i] = 1.0
        beats = dynamic_programming_beats(env, period=10.0)
        assert len(beats) > 0
        assert all(0 <= b < 100 for b in beats)

    def test_empty_envelope(self):
        assert len(dynamic_programming_beats(np.array([]), 10.0)) == 0

    def test_period_less_than_one(self):
        env = np.ones(10)
        beats = dynamic_programming_beats(env, period=0.5)
        assert len(beats) > 0


class TestBeatTracker:
    def test_track_returns_dict(self):
        sr = 22050.0
        t = np.arange(0, 2.0, 1.0 / sr)
        signal = np.sin(2 * np.pi * 2.0 * t)
        tracker = BeatTracker(sr=sr)
        result = tracker.track(signal)
        assert "beat_frames" in result
        assert "beat_times" in result
        assert "bpm" in result
        assert "coherence" in result


# ── Meter ────────────────────────────────────────────────────────────


class TestMetricHierarchy:
    def test_4_4(self):
        w = metric_hierarchy(4)
        assert len(w) == 4
        assert w[0] == 1.0
        assert w[2] > w[1]

    def test_3_4(self):
        w = metric_hierarchy(3)
        assert len(w) == 3
        assert w[0] == 1.0

    def test_zero(self):
        assert metric_hierarchy(0) == []

    def test_one(self):
        w = metric_hierarchy(1)
        assert w == [1.0]


class TestMetricWeightProfile:
    def test_strong_4_correlation(self):
        beats = np.arange(16, dtype=float)
        strengths = np.array([1, 0.2, 0.5, 0.2] * 4, dtype=float)
        scores = metric_weight_profile(beats, strengths)
        assert scores[4] > scores[3]

    def test_empty(self):
        scores = metric_weight_profile(np.array([]), np.array([]))
        assert all(v == 0.0 for v in scores.values())

    def test_mismatched_lengths(self):
        scores = metric_weight_profile(np.arange(5), np.arange(3))
        assert all(v == 0.0 for v in scores.values())


class TestDownbeatDetection:
    def test_regular_4(self):
        beats = np.arange(16, dtype=float)
        strengths = np.array([1, 0.2, 0.5, 0.2] * 4, dtype=float)
        db = downbeat_detection(beats, strengths, meter=4)
        assert len(db) == 4
        assert db[0] == 0

    def test_empty(self):
        assert len(downbeat_detection(np.array([]), np.array([]), 4)) == 0


class TestTimeSignature:
    def test_strong_4_4(self):
        beats = np.arange(16, dtype=float)
        strengths = np.array([1, 0.2, 0.5, 0.2] * 4, dtype=float)
        ts = time_signature_from_beats(beats, strengths)
        assert ts[1] == 4
        assert ts[0] in (2, 3, 4, 6)

    def test_few_beats(self):
        assert time_signature_from_beats(np.array([0, 1]), np.array([1, 0.5])) == (4, 4)


class TestMeterAnalyzer:
    def test_analyze(self):
        beats = np.arange(16, dtype=float)
        strengths = np.array([1, 0.2, 0.5, 0.2] * 4, dtype=float)
        analyzer = MeterAnalyzer()
        result = analyzer.analyze(beats, strengths)
        assert "time_signature" in result
        assert "downbeat_indices" in result
        assert "metric_scores" in result


# ── Groove ───────────────────────────────────────────────────────────


class TestQuantizeOnsets:
    def test_full_quantization(self):
        onsets = np.array([0.12, 0.26, 0.48])
        q = quantize_onsets(onsets, grid_size=0.25, strength=1.0)
        np.testing.assert_allclose(q, [0.0, 0.25, 0.5], atol=1e-10)

    def test_no_quantization(self):
        onsets = np.array([0.12, 0.26, 0.48])
        q = quantize_onsets(onsets, grid_size=0.25, strength=0.0)
        np.testing.assert_allclose(q, onsets)

    def test_half_strength(self):
        onsets = np.array([0.12])
        q = quantize_onsets(onsets, grid_size=0.25, strength=0.5)
        expected = 0.12 + 0.5 * (0.0 - 0.12)
        np.testing.assert_allclose(q, [expected], atol=1e-10)

    def test_zero_grid(self):
        onsets = np.array([0.1, 0.2])
        np.testing.assert_array_equal(quantize_onsets(onsets, 0.0), onsets)

    def test_empty(self):
        assert len(quantize_onsets(np.array([]), 0.25)) == 0


class TestHumanizeGrid:
    def test_deviations_added(self):
        grid = np.arange(0, 2, 0.25)
        h = humanize_grid(grid, amount=0.02, seed=42)
        assert not np.allclose(h, grid)
        assert len(h) == len(grid)

    def test_reproducible(self):
        grid = np.arange(0, 1, 0.5)
        h1 = humanize_grid(grid, seed=123)
        h2 = humanize_grid(grid, seed=123)
        np.testing.assert_array_equal(h1, h2)


class TestMicrotimingDeviations:
    def test_on_grid(self):
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        dev = microtiming_deviations(onsets, 0.25)
        np.testing.assert_allclose(dev, 0.0, atol=1e-10)

    def test_ahead(self):
        onsets = np.array([0.26])
        dev = microtiming_deviations(onsets, 0.25)
        assert dev[0] > 0

    def test_behind(self):
        onsets = np.array([0.24])
        dev = microtiming_deviations(onsets, 0.25)
        assert dev[0] < 0

    def test_empty(self):
        assert len(microtiming_deviations(np.array([]), 0.25)) == 0


class TestSwingRatio:
    def test_straight(self):
        onsets = np.arange(0, 4, 0.25)
        sr = swing_ratio(onsets, beat_period=0.5)
        assert abs(sr - 1.0) < 0.3

    def test_too_few(self):
        assert swing_ratio(np.array([0.0, 0.5]), 0.5) == 1.0

    def test_zero_period(self):
        assert swing_ratio(np.arange(5, dtype=float), 0.0) == 1.0


class TestSyncopationIndex:
    def test_all_on_downbeats(self):
        onsets = np.array([0.0, 2.0])  # only strong beats in 4/4
        s = syncopation_index(onsets, grid_size=1.0, meter=4)
        assert s == 0.0

    def test_all_on_weak(self):
        onsets = np.array([0.25, 0.75, 1.25, 1.75])
        s = syncopation_index(onsets, grid_size=0.25, meter=4)
        assert s > 0.0

    def test_empty(self):
        assert syncopation_index(np.array([]), 0.25) == 0.0


class TestGrooveTemplateDistance:
    def test_identical(self):
        template = np.array([0.5, 0.2, 0.2, 0.1])
        onsets = np.array([0.0, 0.25, 0.5, 0.75])
        d = groove_template_distance(onsets, template, beat_period=1.0)
        assert d < 1.0

    def test_empty(self):
        assert groove_template_distance(np.array([]), np.array([1]), 1.0) == float("inf")


class TestGrooveAnalyzer:
    def test_analyze(self):
        onsets = np.arange(0, 4, 0.125)
        analyzer = GrooveAnalyzer(grid_size=0.125, meter=4)
        result = analyzer.analyze(onsets, beat_period=0.5)
        assert "microtiming_mean" in result
        assert "swing_ratio" in result
        assert "syncopation" in result
        assert "n_onsets" in result


# ── Pattern ──────────────────────────────────────────────────────────


class TestRhythmPattern:
    def test_density(self):
        p = RhythmPattern(steps=8, onsets=(0, 2, 4))
        assert p.density == 3 / 8

    def test_binary(self):
        p = RhythmPattern(steps=4, onsets=(0, 2))
        assert p.binary == (1, 0, 1, 0)

    def test_ioi(self):
        p = RhythmPattern(steps=8, onsets=(0, 3, 5))
        assert p.ioi == (3, 2, 3)

    def test_empty(self):
        p = RhythmPattern(steps=0, onsets=())
        assert p.density == 0.0
        assert p.binary == ()
        assert p.ioi == ()


class TestEuclideanRhythm:
    def test_tresillo(self):
        p = euclidean_rhythm(3, 8)
        assert len(p.onsets) == 3
        assert p.steps == 8

    def test_cinquillo(self):
        p = euclidean_rhythm(5, 8)
        assert len(p.onsets) == 5

    def test_full(self):
        p = euclidean_rhythm(4, 4)
        assert set(p.onsets) == {0, 1, 2, 3}

    def test_empty(self):
        p = euclidean_rhythm(0, 8)
        assert len(p.onsets) == 0

    def test_zero_steps(self):
        p = euclidean_rhythm(3, 0)
        assert p.steps == 0


class TestPatternSimilarity:
    def test_identical(self):
        p = RhythmPattern(steps=8, onsets=(0, 2, 4))
        assert pattern_similarity(p, p) == 1.0

    def test_complement(self):
        a = RhythmPattern(steps=4, onsets=(0, 2))
        b = RhythmPattern(steps=4, onsets=(1, 3))
        assert pattern_similarity(a, b) == 0.0


class TestNecklaceEquivalence:
    def test_rotation(self):
        a = RhythmPattern(steps=8, onsets=(0, 3, 5))
        b = RhythmPattern(steps=8, onsets=(1, 4, 6))
        assert necklace_equivalence(a, b)

    def test_not_equivalent(self):
        a = RhythmPattern(steps=8, onsets=(0, 1, 2))
        b = RhythmPattern(steps=8, onsets=(0, 3, 6))
        assert not necklace_equivalence(a, b)

    def test_different_lengths(self):
        a = RhythmPattern(steps=8, onsets=(0,))
        b = RhythmPattern(steps=6, onsets=(0,))
        assert not necklace_equivalence(a, b)


class TestRhythmComplexity:
    def test_simple(self):
        p = euclidean_rhythm(4, 4)
        c = rhythm_complexity(p)
        assert 0 <= c <= 1

    def test_more_complex(self):
        simple = euclidean_rhythm(4, 8)
        complex_p = RhythmPattern(steps=16, onsets=(0, 3, 7, 9, 14))
        c_simple = rhythm_complexity(simple)
        c_complex = rhythm_complexity(complex_p)
        assert c_simple >= 0 and c_complex >= 0

    def test_empty(self):
        p = RhythmPattern(steps=0, onsets=())
        assert rhythm_complexity(p) == 0.0


class TestPatternIOI:
    def test_roundtrip(self):
        ioi = (3, 2, 3)
        p = pattern_from_ioi(ioi)
        assert p.steps == 8
        assert len(p.onsets) == 3
        result_ioi = pattern_to_ioi(p)
        assert result_ioi == ioi

    def test_empty(self):
        p = pattern_from_ioi(())
        assert p.steps == 0
        assert p.ioi == ()


# ── Integration ──────────────────────────────────────────────────────


class TestRhythmIntegration:
    def test_tempo_beat_meter_pipeline(self):
        sr = 22050.0
        bpm_target = 100.0
        beat_period = 60.0 / bpm_target
        dur = 4.0
        t = np.arange(0, dur, 1.0 / sr)
        signal = np.zeros_like(t)
        for beat_t in np.arange(0, dur, beat_period):
            idx = int(beat_t * sr)
            if idx < len(signal):
                signal[idx:idx + 200] += np.hanning(min(200, len(signal) - idx))

        tracker = BeatTracker(sr=sr)
        result = tracker.track(signal)
        assert result["bpm"] > 0
        assert len(result["beat_times"]) > 0

    def test_groove_on_euclidean(self):
        p = euclidean_rhythm(5, 16)
        onsets = np.array(p.onsets, dtype=float) * 0.125
        analyzer = GrooveAnalyzer(grid_size=0.125, meter=4)
        result = analyzer.analyze(onsets, beat_period=0.5)
        assert result["n_onsets"] == 5
        assert abs(result["microtiming_mean"]) < 0.01

    def test_pattern_quantize_roundtrip(self):
        onsets = np.array([0.13, 0.37, 0.52, 0.76])
        quantized = quantize_onsets(onsets, grid_size=0.25, strength=1.0)
        dev = microtiming_deviations(quantized, 0.25)
        np.testing.assert_allclose(dev, 0.0, atol=1e-10)
