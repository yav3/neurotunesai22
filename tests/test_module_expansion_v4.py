"""Tests for expansion round 4: cognition, production, training, generation."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_cognition.rhythm_perception import (
    AccentType, PerceptualAccent, MetricalGrid,
    compute_syncopation, compute_groove_complexity,
    beat_induction, phase_coherence, PerceptualGrouping,
)
from models.music_production.eq_design import (
    EQBandType, EQBand, biquad_coefficients, apply_biquad,
    frequency_response, spectral_tilt, ParametricEQ, auto_eq_match,
)
from models.music_training.curriculum import (
    DifficultyMetric, CurriculumStrategy, SampleDifficulty,
    score_by_loss, score_by_entropy, score_by_complexity,
    CurriculumScheduler, SelfPacedWeight,
)
from models.music_generation.counterpoint import (
    MotionType, classify_motion, is_consonant, is_perfect,
    check_parallel_fifths, check_parallel_octaves,
    check_voice_crossing, check_hidden_fifths,
    first_species, CounterpointAnalyzer, CounterpointRules,
)


class TestRhythmPerception:
    def test_metrical_grid_from_tempo(self):
        grid = MetricalGrid.from_tempo(120.0, (4, 4), 4.0)
        assert len(grid.accents) > 0
        assert grid.accents[0].strength == 1.0

    def test_strength_at_downbeat(self):
        grid = MetricalGrid.from_tempo(120.0, (4, 4), 2.0)
        assert grid.strength_at(0.0) == 1.0

    def test_compute_syncopation_on_beat(self):
        grid = MetricalGrid.from_tempo(120.0, (4, 4), 4.0)
        onsets = [0.0, 0.5, 1.0, 1.5]
        score = compute_syncopation(onsets, grid)
        assert score >= 0.0

    def test_compute_syncopation_empty(self):
        grid = MetricalGrid.from_tempo(120.0)
        assert compute_syncopation([], grid) == 0.0

    def test_groove_complexity_range(self):
        grid = MetricalGrid.from_tempo(120.0, (4, 4), 4.0)
        onsets = [0.1, 0.6, 1.1, 1.6]
        c = compute_groove_complexity(onsets, grid)
        assert 0.0 <= c <= 1.0

    def test_beat_induction_120bpm(self):
        onsets = [i * 0.5 for i in range(20)]
        bpm = beat_induction(onsets)
        assert abs(bpm - 120.0) < 15 or abs(bpm - 240.0) < 15

    def test_beat_induction_few_onsets(self):
        assert beat_induction([0.0, 0.5]) == 120.0

    def test_phase_coherence_perfect(self):
        onsets = [i * 0.5 for i in range(20)]
        r = phase_coherence(onsets, 120.0)
        assert r > 0.9

    def test_phase_coherence_empty(self):
        assert phase_coherence([], 120.0) == 0.0

    def test_grouping_by_proximity(self):
        onsets = [0.0, 0.1, 0.2, 1.0, 1.1, 1.2]
        groups = PerceptualGrouping.group_by_proximity(onsets, threshold_ms=300)
        assert len(groups) == 2

    def test_grouping_by_similarity(self):
        onsets = [0.0, 0.1, 0.2]
        features = np.array([[1, 0], [1, 0.1], [5, 5]])
        groups = PerceptualGrouping.group_by_similarity(onsets, features, threshold=2.0)
        assert len(groups) >= 2


class TestEQDesign:
    def test_biquad_peak(self):
        band = EQBand(EQBandType.PEAK, 1000.0, 6.0, 1.0)
        b, a = biquad_coefficients(band, 44100.0)
        assert len(b) == 3 and len(a) == 3
        assert abs(a[0] - 1.0) < 1e-10

    def test_biquad_lowpass(self):
        band = EQBand(EQBandType.LOW_PASS, 5000.0, 0.0, 0.707)
        b, a = biquad_coefficients(band, 44100.0)
        assert abs(a[0] - 1.0) < 1e-10

    def test_apply_biquad_length(self):
        sig = np.random.randn(1000)
        band = EQBand(EQBandType.PEAK, 1000.0, 0.0, 1.0)
        b, a = biquad_coefficients(band, 44100.0)
        out = apply_biquad(sig, b, a)
        assert len(out) == len(sig)

    def test_frequency_response_peak_boost(self):
        band = EQBand(EQBandType.PEAK, 1000.0, 12.0, 2.0)
        freqs = np.array([100.0, 1000.0, 10000.0])
        resp = frequency_response(band, freqs, 44100.0)
        assert resp[1] > resp[0]
        assert resp[1] > resp[2]

    def test_spectral_tilt_flat(self):
        freqs = np.logspace(1, 4, 100)
        spec = np.zeros(100)
        tilt = spectral_tilt(spec, freqs)
        assert abs(tilt) < 0.1

    def test_parametric_eq_process(self):
        eq = ParametricEQ([EQBand(EQBandType.PEAK, 1000.0, 6.0)], sr=44100.0)
        sig = np.random.randn(4410)
        out = eq.process(sig)
        assert out.shape == sig.shape

    def test_parametric_eq_response(self):
        eq = ParametricEQ([EQBand(EQBandType.PEAK, 1000.0, 6.0)])
        freqs = np.linspace(20, 20000, 100)
        resp = eq.response(freqs)
        assert len(resp) == 100

    def test_auto_eq_match(self):
        freqs = np.linspace(20, 20000, 256)
        source = np.zeros(256)
        target = np.zeros(256)
        target[128] = 10.0
        bands = auto_eq_match(source, target, freqs, n_bands=3)
        assert len(bands) >= 1
        assert bands[0].gain_db > 0


class TestCurriculum:
    def test_score_by_loss_sorted(self):
        losses = {"a": 0.5, "b": 0.1, "c": 0.9}
        scored = score_by_loss(losses)
        assert scored[0].sample_id == "b"
        assert scored[-1].sample_id == "c"

    def test_score_by_entropy(self):
        preds = {"a": np.array([0.5, 0.5]), "b": np.array([0.99, 0.01])}
        scored = score_by_entropy(preds)
        assert scored[0].sample_id == "b"

    def test_score_by_complexity(self):
        feats = {"flat": np.ones(100), "varied": np.random.randn(100)}
        scored = score_by_complexity(feats)
        assert all(isinstance(s, SampleDifficulty) for s in scored)

    def test_competence_warmup(self):
        sched = CurriculumScheduler(CurriculumStrategy.EASY_TO_HARD, 100, 0.2)
        assert sched.competence(0) == 0.0
        assert sched.competence(10) == 0.5
        assert sched.competence(20) == 1.0
        assert sched.competence(50) == 1.0

    def test_select_batch_size(self):
        diffs = [SampleDifficulty(f"s{i}", float(i), DifficultyMetric.LOSS)
                 for i in range(50)]
        sched = CurriculumScheduler(CurriculumStrategy.EASY_TO_HARD, 100, 0.2)
        batch = sched.select_batch(diffs, 10, 50)
        assert len(batch) == 10

    def test_self_paced_hard_threshold(self):
        spw = SelfPacedWeight(lambda_sp=0.5)
        losses = np.array([0.1, 0.3, 0.6, 0.9])
        w = spw.compute_weights(losses)
        assert w[0] == 1.0
        assert w[3] == 0.0

    def test_self_paced_soft(self):
        spw = SelfPacedWeight(lambda_sp=1.0, soft=True, gamma=1.0)
        losses = np.array([0.0, 0.5, 1.0, 1.5])
        w = spw.compute_weights(losses)
        assert w[0] == 1.0
        assert abs(w[1] - 0.5) < 1e-10
        assert w[2] == 0.0

    def test_update_lambda(self):
        spw = SelfPacedWeight(lambda_sp=1.0)
        spw.update_lambda(2.0)
        assert spw.lambda_sp == 2.0

    def test_pacing_easy_to_hard(self):
        sched = CurriculumScheduler(CurriculumStrategy.EASY_TO_HARD, 100, 0.5)
        assert sched.pacing_function(0) == 0.0
        assert sched.pacing_function(50) == 1.0


class TestCounterpoint:
    def test_classify_contrary(self):
        assert classify_motion(2, -3) == MotionType.CONTRARY

    def test_classify_parallel(self):
        assert classify_motion(3, 3) == MotionType.PARALLEL

    def test_classify_similar(self):
        assert classify_motion(2, 5) == MotionType.SIMILAR

    def test_classify_oblique(self):
        assert classify_motion(0, 3) == MotionType.OBLIQUE

    def test_is_consonant(self):
        assert is_consonant(0)
        assert is_consonant(7)
        assert is_consonant(4)
        assert not is_consonant(1)
        assert not is_consonant(6)

    def test_is_perfect(self):
        assert is_perfect(0)
        assert is_perfect(7)
        assert is_perfect(12)
        assert not is_perfect(4)

    def test_parallel_fifths_detected(self):
        v1 = np.array([60, 62, 64, 66])
        v2 = np.array([67, 69, 71, 73])
        violations = check_parallel_fifths(v1, v2)
        assert len(violations) >= 1

    def test_no_parallel_fifths(self):
        v1 = np.array([60, 62, 64])
        v2 = np.array([67, 68, 71])
        violations = check_parallel_fifths(v1, v2)
        assert len(violations) == 0

    def test_voice_crossing(self):
        v1 = np.array([60, 70, 60])
        v2 = np.array([67, 65, 67])
        crossings = check_voice_crossing(v1, v2)
        assert 1 in crossings

    def test_first_species_length(self):
        cf = np.array([60, 62, 64, 65, 67, 65, 64, 62, 60])
        cp = first_species(cf)
        assert len(cp) == len(cf)

    def test_first_species_consonance(self):
        cf = np.array([60, 62, 64, 65, 67, 65, 64, 62, 60])
        cp = first_species(cf)
        analyzer = CounterpointAnalyzer()
        report = analyzer.analyze(cf, cp)
        assert report.consonance_ratio > 0.7

    def test_first_species_empty(self):
        cp = first_species(np.array([]))
        assert len(cp) == 0

    def test_analyzer_report(self):
        v1 = np.array([60, 62, 64, 65, 67])
        v2 = np.array([72, 71, 69, 68, 67])
        analyzer = CounterpointAnalyzer()
        report = analyzer.analyze(v1, v2)
        assert "contrary" in report.motion_distribution
        assert report.consonance_ratio >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
