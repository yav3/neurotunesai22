"""Tests for models.music_performance — expression, tempo, intonation, ensemble, practice."""
import numpy as np
import pytest

from models.music_performance import (
    DynamicLevel,
    ExpressionCurve,
    DynamicsAnalyzer,
    VibratoAnalyzer,
    ArticulationDetector,
    RubatoDetector,
    AccelerandoDetector,
    AgogicAccent,
    MetricalStrength,
    PitchTracker,
    IntonationAnalyzer,
    TemperamentComparator,
    MicrotoneDetector,
    SynchronyAnalyzer,
    BlendAnalyzer,
    BalanceAnalyzer,
    TuningAnalyzer,
    PracticeSession,
    ProgressTracker,
    DifficultyEstimator,
    PracticeScheduler,
)


# ── ExpressionCurve ──────────────────────────────────────────────

class TestExpressionCurve:
    def test_duration(self):
        ec = ExpressionCurve(times=np.array([0.0, 0.5, 1.0]), values=np.array([0.1, 0.5, 0.3]))
        assert ec.duration() == pytest.approx(1.0)

    def test_mean_value(self):
        ec = ExpressionCurve(times=np.array([0, 1, 2.0]), values=np.array([0.2, 0.4, 0.6]))
        assert ec.mean_value() == pytest.approx(0.4)

    def test_range_value(self):
        ec = ExpressionCurve(times=np.array([0, 1.0]), values=np.array([0.1, 0.9]))
        assert ec.range_value() == pytest.approx(0.8)

    def test_interpolate(self):
        ec = ExpressionCurve(times=np.array([0.0, 1.0]), values=np.array([0.0, 1.0]))
        assert ec.interpolate(0.5) == pytest.approx(0.5)

    def test_smoothed(self):
        ec = ExpressionCurve(
            times=np.arange(20, dtype=np.float64),
            values=np.random.default_rng(42).random(20),
        )
        s = ec.smoothed(window=5)
        assert len(s.values) == len(ec.values)

    def test_empty(self):
        ec = ExpressionCurve()
        assert ec.duration() == 0.0
        assert ec.mean_value() == 0.0


# ── DynamicsAnalyzer ─────────────────────────────────────────────

class TestDynamicsAnalyzer:
    def test_classify_pp(self):
        assert DynamicsAnalyzer.classify_level(0.02) == DynamicLevel.PP

    def test_classify_ff(self):
        assert DynamicsAnalyzer.classify_level(0.9) == DynamicLevel.FF

    def test_amplitude_envelope(self):
        sr = 44100
        t = np.arange(sr, dtype=np.float64) / sr
        sig = 0.5 * np.sin(2 * np.pi * 440 * t)
        env = DynamicsAnalyzer.amplitude_envelope(sig, sr, hop_ms=50)
        assert len(env.values) > 0
        assert all(v >= 0 for v in env.values)

    def test_detect_crescendo(self):
        vals = np.concatenate([np.linspace(0.05, 0.8, 50), np.ones(10) * 0.8])
        times = np.arange(len(vals)) * 0.05
        env = ExpressionCurve(times=times, values=vals, label="dyn")
        cresc = DynamicsAnalyzer.detect_crescendo(env, min_duration_s=0.3, min_ratio=1.3)
        assert len(cresc) >= 1
        assert cresc[0]["type"] == "crescendo"

    def test_detect_sforzando(self):
        vals = np.array([0.1, 0.1, 0.1, 0.8, 0.1, 0.1])
        times = np.arange(len(vals)) * 0.1
        env = ExpressionCurve(times=times, values=vals, label="dyn")
        sf = DynamicsAnalyzer.detect_sforzando(env, threshold_ratio=3.0)
        assert len(sf) >= 1
        assert sf[0]["type"] == "sforzando"

    def test_dynamic_profile(self):
        vals = np.ones(30) * 0.4
        times = np.arange(30) * 0.05
        env = ExpressionCurve(times=times, values=vals)
        profile = DynamicsAnalyzer.dynamic_profile(env, window_frames=10)
        assert len(profile) >= 1
        assert profile[0]["level"] == "mf"


# ── VibratoAnalyzer ──────────────────────────────────────────────

class TestVibratoAnalyzer:
    def test_detect_vibrato(self):
        sr_pitch = 200.0
        t = np.arange(400) / sr_pitch
        base = 440.0
        vibrato_rate = 5.5
        vibrato_extent_cents = 30.0
        pitch = base * 2 ** (vibrato_extent_cents / 1200.0 * np.sin(2 * np.pi * vibrato_rate * t))
        result = VibratoAnalyzer.detect(pitch, sr_pitch)
        assert result["detected"] is True
        assert 4.0 <= result["rate_hz"] <= 8.0
        assert result["extent_cents"] > 5

    def test_no_vibrato(self):
        pitch = np.ones(100) * 440.0
        result = VibratoAnalyzer.detect(pitch, sr_pitch=100.0)
        assert result["detected"] is False

    def test_vibrato_curve(self):
        pitch = np.ones(50) * 440.0
        curve = VibratoAnalyzer.vibrato_curve(pitch, sr_pitch=100)
        assert len(curve.times) == 50


# ── ArticulationDetector ─────────────────────────────────────────

class TestArticulationDetector:
    def test_classify_staccato(self):
        assert ArticulationDetector.classify(0.9, 0.2) == "staccato"

    def test_classify_legato(self):
        assert ArticulationDetector.classify(0.3, 0.95, gap_ratio=0.01) == "legato"

    def test_classify_marcato(self):
        assert ArticulationDetector.classify(0.9, 0.5) == "marcato"

    def test_analyze_notes(self):
        onsets = np.array([0.0, 0.5, 1.0])
        offsets = np.array([0.45, 0.95, 1.45])
        amps = np.array([0.5, 0.6, 0.5])
        result = ArticulationDetector.analyze_notes(onsets, offsets, amps, nominal_duration=0.5)
        assert len(result) == 3
        assert all("articulation" in r for r in result)


# ── RubatoDetector ───────────────────────────────────────────────

class TestRubatoDetector:
    def test_strict(self):
        beats = np.arange(20) * 0.5
        idx = RubatoDetector.rubato_index(beats)
        assert idx < 0.01
        assert RubatoDetector.classify(idx) == "strict"

    def test_free(self):
        rng = np.random.default_rng(7)
        beats = np.cumsum(0.5 + rng.normal(0, 0.1, 20))
        idx = RubatoDetector.rubato_index(beats)
        assert idx > 0.05

    def test_analyze_sections(self):
        beats = np.arange(30) * 0.5
        sections = RubatoDetector.analyze_sections(beats, section_beats=8)
        assert len(sections) >= 1
        assert "rubato_index" in sections[0]


# ── AccelerandoDetector ──────────────────────────────────────────

class TestAccelerandoDetector:
    def test_detect_accel(self):
        ioi = np.linspace(0.6, 0.35, 10)
        beats = np.concatenate([[0], np.cumsum(ioi)])
        result = AccelerandoDetector.detect(beats, min_beats=4, threshold=0.05)
        types = [r["type"] for r in result]
        assert "accelerando" in types

    def test_detect_rit(self):
        ioi = np.linspace(0.35, 0.65, 10)
        beats = np.concatenate([[0], np.cumsum(ioi)])
        result = AccelerandoDetector.detect(beats, min_beats=4, threshold=0.05)
        types = [r["type"] for r in result]
        assert "ritardando" in types

    def test_tempo_curve(self):
        beats = np.arange(10) * 0.5
        times, bpm = AccelerandoDetector.tempo_curve(beats)
        assert len(times) == 9
        assert all(bpm > 0)


# ── AgogicAccent ─────────────────────────────────────────────────

class TestAgogicAccent:
    def test_detect(self):
        durations = np.array([0.5, 0.5, 0.62, 0.5, 0.5])
        accents = AgogicAccent.detect(durations, nominal_duration=0.5, threshold_ratio=1.15)
        assert accents[2] == True
        assert accents[0] == False

    def test_strength(self):
        durations = np.array([0.5, 0.7, 0.5])
        strength = AgogicAccent.accent_strength(durations, 0.5)
        assert strength[1] == pytest.approx(0.4)
        assert strength[0] == pytest.approx(0.0)


# ── MetricalStrength ─────────────────────────────────────────────

class TestMetricalStrength:
    def test_emphasis_ratio(self):
        amps = np.array([0.8, 0.3, 0.4, 0.3, 0.9, 0.3, 0.4, 0.3])
        result = MetricalStrength.beat_emphasis(amps, beats_per_bar=4)
        assert result["emphasis_ratio"] > 1.5

    def test_downbeat_pattern(self):
        amps = np.array([0.9, 0.2, 0.5, 0.2] * 4)
        pattern = MetricalStrength.downbeat_pattern(amps, beats_per_bar=4)
        assert len(pattern) == 4
        assert sum(pattern) == pytest.approx(1.0)
        assert pattern[0] > pattern[1]


# ── PitchTracker ─────────────────────────────────────────────────

class TestPitchTracker:
    def test_autocorrelation_sine(self):
        sr = 44100
        t = np.arange(sr) / sr
        sig = np.sin(2 * np.pi * 440 * t)
        pitch = PitchTracker.autocorrelation_pitch(sig[:4096], sr)
        assert 420 < pitch < 460

    def test_silent_signal(self):
        pitch = PitchTracker.autocorrelation_pitch(np.zeros(4096), 44100)
        assert pitch == 0.0

    def test_track(self):
        sr = 44100
        t = np.arange(sr // 2) / sr
        sig = np.sin(2 * np.pi * 440 * t)
        times, pitches = PitchTracker.track(sig, sr, hop_ms=20, frame_ms=40)
        assert len(times) == len(pitches)
        assert len(times) > 0


# ── IntonationAnalyzer ───────────────────────────────────────────

class TestIntonationAnalyzer:
    def test_nearest_et(self):
        result = IntonationAnalyzer.nearest_equal_temperament(440.0)
        assert result["note"] == "A4"
        assert abs(result["deviation_cents"]) < 1

    def test_sharp_note(self):
        result = IntonationAnalyzer.nearest_equal_temperament(445.0)
        assert result["deviation_cents"] > 0

    def test_analyze_pitches(self):
        pitches = np.array([440.0, 442.0, 438.0, 0.0, 440.5])
        result = IntonationAnalyzer.analyze_pitches(pitches, "equal")
        assert result["n_notes"] == 4
        assert result["mean_deviation"] >= 0


# ── TemperamentComparator ────────────────────────────────────────

class TestTemperamentComparator:
    def test_compare_et_just(self):
        rows = TemperamentComparator.compare("equal", "just")
        assert len(rows) == 12
        assert rows[0]["difference_cents"] == pytest.approx(0.0, abs=0.01)

    def test_max_difference(self):
        diff = TemperamentComparator.max_difference("equal", "just")
        assert diff > 0


# ── MicrotoneDetector ────────────────────────────────────────────

class TestMicrotoneDetector:
    def test_detect_quarter_tones(self):
        a4 = 440.0
        quarter_sharp = a4 * 2 ** (50 / 1200)
        pitches = np.array([a4, quarter_sharp, a4, quarter_sharp, a4])
        qts = MicrotoneDetector.detect_quarter_tones(pitches, a4=a4, tolerance_cents=15)
        assert len(qts) >= 2

    def test_classify_intentional(self):
        detections = [
            {"index": i, "freq_hz": 452.0, "nearest_note": "A4", "deviation_cents": 47, "type": "quarter_tone"}
            for i in range(5)
        ]
        result = MicrotoneDetector.classify_microtones(detections, consistency_threshold=3)
        assert len(result["intentional"]) == 5
        assert len(result["errors"]) == 0


# ── SynchronyAnalyzer ────────────────────────────────────────────

class TestSynchronyAnalyzer:
    def test_perfect_sync(self):
        a = np.arange(10) * 0.5
        result = SynchronyAnalyzer.pairwise_asynchrony(a, a)
        assert result["mean_ms"] == pytest.approx(0.0)
        assert result["matched_pairs"] == 10

    def test_offset(self):
        a = np.arange(10) * 0.5
        b = a + 0.01
        result = SynchronyAnalyzer.pairwise_asynchrony(a, b)
        assert abs(result["mean_ms"] - 10.0) < 2

    def test_group(self):
        a = np.arange(10) * 0.5
        b = a + 0.005
        c = a - 0.003
        result = SynchronyAnalyzer.group_synchrony([a, b, c])
        assert result["n_pairs"] == 3


# ── BlendAnalyzer ────────────────────────────────────────────────

class TestBlendAnalyzer:
    def test_identical_blend(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100)
        score = BlendAnalyzer.blend_score([sig, sig], sr=44100)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_different_blend(self):
        sr = 44100
        t = np.arange(sr) / sr
        a = np.sin(2 * np.pi * 200 * t)
        b = np.sin(2 * np.pi * 8000 * t)
        score = BlendAnalyzer.blend_score([a, b], sr=sr)
        assert score < 0.8


# ── BalanceAnalyzer ──────────────────────────────────────────────

class TestBalanceAnalyzer:
    def test_loudness_bands(self):
        sr = 44100
        t = np.arange(sr) / sr
        sig = np.sin(2 * np.pi * 1000 * t)
        bands = BalanceAnalyzer.loudness_per_band(sig, sr)
        assert len(bands) == 8
        assert any(b > 0 for b in bands)

    def test_masking_detection(self):
        sr = 44100
        t = np.arange(sr) / sr
        loud = 10 * np.sin(2 * np.pi * 500 * t)
        quiet = 0.01 * np.sin(2 * np.pi * 500 * t)
        masking = BalanceAnalyzer.detect_masking([loud, quiet], sr)
        assert len(masking) > 0


# ── TuningAnalyzer ───────────────────────────────────────────────

class TestTuningAnalyzer:
    def test_beat_frequency(self):
        bf = TuningAnalyzer.beat_frequency(440.0, 442.0)
        assert bf == pytest.approx(2.0)

    def test_unison_drift(self):
        a = np.ones(100) * 440.0
        b = np.ones(100) * 442.0
        result = TuningAnalyzer.unison_drift(a, b)
        assert result["mean_cents"] > 5
        assert result["n_frames"] == 100

    def test_drift_over_time(self):
        a = np.ones(200) * 440.0
        b = np.linspace(440, 445, 200)
        indices, drift = TuningAnalyzer.drift_over_time(a, b, window=20)
        assert len(indices) == len(drift)


# ── PracticeSession ──────────────────────────────────────────────

class TestPracticeSession:
    def test_duration(self):
        s = PracticeSession(start_time=0, end_time=1800)
        assert s.duration_minutes() == pytest.approx(30.0)

    def test_error_rate(self):
        s = PracticeSession(start_time=0, end_time=600, error_count=10)
        assert s.error_rate() == pytest.approx(1.0)


# ── ProgressTracker ──────────────────────────────────────────────

class TestProgressTracker:
    def test_tempo_improvement(self):
        pt = ProgressTracker()
        pt.add_session(PracticeSession(start_time=0, end_time=600, tempo_range=(80, 100)))
        pt.add_session(PracticeSession(start_time=700, end_time=1300, tempo_range=(90, 120)))
        assert pt.tempo_improvement() > 0

    def test_error_reduction(self):
        pt = ProgressTracker()
        pt.add_session(PracticeSession(start_time=0, end_time=600, error_count=20))
        pt.add_session(PracticeSession(start_time=700, end_time=1300, error_count=5))
        assert pt.error_reduction() > 0

    def test_section_coverage(self):
        pt = ProgressTracker()
        pt.add_session(PracticeSession(sections_practiced=["A", "B"]))
        pt.add_session(PracticeSession(sections_practiced=["B", "C"]))
        cov = pt.section_coverage()
        assert cov["B"] == 2
        assert cov["A"] == 1


# ── DifficultyEstimator ─────────────────────────────────────────

class TestDifficultyEstimator:
    def test_easy(self):
        d = DifficultyEstimator.estimate(2.0, 2.0, 0.1, 60)
        assert d < 3.0
        assert DifficultyEstimator.classify(d) in ("beginner", "intermediate")

    def test_hard(self):
        d = DifficultyEstimator.estimate(12.0, 12.0, 1.0, 200)
        assert d >= 7.0
        assert DifficultyEstimator.classify(d) in ("expert", "virtuoso")


# ── PracticeScheduler ───────────────────────────────────────────

class TestPracticeScheduler:
    def test_generate_plan(self):
        sched = PracticeScheduler(
            sections=["A", "B", "C", "D"],
            difficulties=[3.0, 7.0, 5.0, 9.0],
        )
        plan = sched.generate_plan(n_sections=3)
        assert len(plan) == 3
        assert plan[0]["priority"] >= plan[1]["priority"]

    def test_mastered_deprioritized(self):
        sched = PracticeScheduler(sections=["A", "B"], difficulties=[5.0, 5.0])
        for _ in range(10):
            sched.record_practice("A", quality=1.0)
        plan = sched.generate_plan(n_sections=2)
        b_entry = next(p for p in plan if p["section"] == "B")
        a_entry = next(p for p in plan if p["section"] == "A")
        assert b_entry["priority"] > a_entry["priority"]

    def test_staleness_matters(self):
        sched = PracticeScheduler(sections=["X", "Y"], difficulties=[5.0, 5.0])
        sched.record_practice("X", quality=0.5)
        for _ in range(10):
            sched.record_practice("Y", quality=0.1)
        plan = sched.generate_plan(n_sections=2)
        x_entry = next(p for p in plan if p["section"] == "X")
        assert x_entry["sessions_since_last"] > 0
