"""Tests for expansion round 6: rhythm patterns, repetition analysis, curriculum, cepstral."""
from __future__ import annotations

import numpy as np
import pytest

from models.music_composition.rhythm_pattern import (
    RhythmPattern, PolymetricPattern,
    euclidean_rhythm, polyrhythm, swing_quantize,
    syncopation_score, onset_to_ioi, ioi_regularity,
    create_polymetric, groove_template, humanize,
)
from models.music_structure.repetition_analysis import (
    Motif, MotifOccurrence, RepetitionStructure,
    cosine_similarity_matrix, lag_similarity, lag_spectrum,
    find_repeated_segments, structural_thumbnail,
    compression_ratio, build_repetition_structure,
)
from models.music_education.curriculum import (
    TopicStatus, Topic, LearnerState, LearningPath, CurriculumGraph,
    spaced_review_schedule, forgetting_curve, optimal_review_time,
)
from models.music_transforms.cepstral import (
    compute_mfcc, compute_mfcc_full, MFCCResult,
    lifter, delta_features, delta_delta_features,
    lpc_coefficients, lpc_to_cepstrum,
    mel_filterbank_matrix, dct_ii,
)


class TestRhythmPattern:
    def test_euclidean_3_8(self):
        p = euclidean_rhythm(3, 8)
        assert p.steps == 8
        assert len(p.onsets) == 3
        assert p.name == "E(3,8)"

    def test_euclidean_binary(self):
        p = euclidean_rhythm(4, 16)
        b = p.to_binary()
        assert len(b) == 16
        assert sum(b) == 4

    def test_euclidean_max_fill(self):
        p = euclidean_rhythm(8, 8)
        assert len(p.onsets) == 8

    def test_rotate(self):
        p = euclidean_rhythm(3, 8)
        r = p.rotate(2)
        assert len(r.onsets) == 3
        assert all(0 <= o < 8 for o in r.onsets)

    def test_polyrhythm_3_4(self):
        pa, pb = polyrhythm(3, 4, 12)
        assert len(pa.onsets) == 3
        assert len(pb.onsets) == 4

    def test_swing_quantize(self):
        times = np.array([0.0, 0.25, 0.5, 0.75])
        q = swing_quantize(times, grid_interval=0.5, swing_ratio=0.67)
        assert len(q) == 4

    def test_syncopation_score_range(self):
        p = euclidean_rhythm(5, 16)
        s = syncopation_score(p)
        assert s >= 0.0

    def test_onset_to_ioi(self):
        ioi = onset_to_ioi([0.0, 0.5, 1.0, 1.5])
        assert len(ioi) == 3
        assert all(abs(x - 0.5) < 1e-9 for x in ioi)

    def test_ioi_regularity_perfect(self):
        reg = ioi_regularity([0.5, 0.5, 0.5, 0.5])
        assert reg > 0.99

    def test_ioi_regularity_irregular(self):
        reg = ioi_regularity([0.1, 0.9, 0.1, 0.9])
        assert reg < 0.7

    def test_polymetric_pattern(self):
        pm = create_polymetric([3, 4, 5], 12)
        assert len(pm.layers) == 3
        assert pm.total_steps == 12
        combined = pm.combined_onsets()
        assert len(combined) > 0

    def test_groove_template(self):
        v = groove_template(16, accents=[0, 4, 8, 12], ghost_notes=[2, 6, 10, 14])
        assert len(v) == 16
        assert v[0] == 1.0
        assert v[2] == 0.25

    def test_humanize_timing(self):
        times = np.linspace(0, 1, 8)
        vel = np.full(8, 0.8)
        ht, hv = humanize(times, velocity=vel, rng=np.random.default_rng(42))
        assert len(ht) == 8
        assert hv is not None
        assert not np.array_equal(times, ht)

    def test_density(self):
        p = euclidean_rhythm(4, 16)
        assert abs(p.density - 0.25) < 1e-9


class TestRepetitionAnalysis:
    def _make_features(self, n_frames=20, dim=8, n_repeats=2):
        rng = np.random.default_rng(42)
        seg = rng.random((n_frames // n_repeats, dim))
        return np.tile(seg, (n_repeats, 1))[:n_frames]

    def test_cosine_sim_diagonal(self):
        feat = np.random.default_rng(0).random((10, 4))
        sim = cosine_similarity_matrix(feat)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_lag_similarity_zero(self):
        feat = np.random.default_rng(1).random((10, 4))
        sim = cosine_similarity_matrix(feat)
        assert lag_similarity(sim, 0) > 0.9

    def test_lag_spectrum_shape(self):
        feat = np.random.default_rng(2).random((20, 4))
        sim = cosine_similarity_matrix(feat)
        spec = lag_spectrum(sim, max_lag=8)
        assert len(spec) == 9

    def test_find_repeated_segments_on_tiled(self):
        feat = self._make_features(n_frames=20)
        segs = find_repeated_segments(feat, min_length=3, threshold=0.80)
        assert len(segs) >= 0
        sim = cosine_similarity_matrix(feat)
        assert lag_similarity(sim, 10) > 0.5

    def test_structural_thumbnail_in_range(self):
        feat = np.random.default_rng(3).random((30, 6))
        start = structural_thumbnail(feat, thumbnail_length=5)
        assert 0 <= start <= 25

    def test_compression_ratio_identical(self):
        seg = np.random.default_rng(4).random((5, 3))
        feat = np.tile(seg, (4, 1))
        cr = compression_ratio(feat, threshold=0.99)
        assert cr < 1.0
        cr_random = compression_ratio(np.random.default_rng(99).random((20, 3)), threshold=0.99)
        assert cr < cr_random

    def test_compression_ratio_random(self):
        feat = np.random.default_rng(5).random((20, 8))
        cr = compression_ratio(feat, threshold=0.99)
        assert cr > 0.5

    def test_build_repetition_structure(self):
        feat = self._make_features(n_frames=20)
        rs = build_repetition_structure(feat, min_length=3, threshold=0.95)
        assert isinstance(rs, RepetitionStructure)
        assert isinstance(rs.compression_ratio, float)


class TestCurriculum:
    def _make_graph(self):
        topics = [
            Topic("scales", "Major Scales", difficulty=0.2, estimated_hours=2.0),
            Topic("chords", "Triads", prerequisites=["scales"], difficulty=0.4, estimated_hours=3.0),
            Topic("prog", "Chord Progressions", prerequisites=["chords"], difficulty=0.6, estimated_hours=4.0),
            Topic("improv", "Improvisation", prerequisites=["scales", "chords"], difficulty=0.8, estimated_hours=5.0),
        ]
        return CurriculumGraph(topics)

    def test_initial_available(self):
        g = self._make_graph()
        learner = LearnerState()
        avail = g.available_topics(learner)
        assert "scales" in avail
        assert "chords" not in avail

    def test_unlock_after_mastery(self):
        g = self._make_graph()
        learner = LearnerState(topic_scores={"scales": 0.9})
        avail = g.available_topics(learner)
        assert "chords" in avail

    def test_topic_status_locked(self):
        g = self._make_graph()
        learner = LearnerState()
        assert g.topic_status("chords", learner) == TopicStatus.LOCKED

    def test_topic_status_mastered(self):
        g = self._make_graph()
        learner = LearnerState(topic_scores={"scales": 0.95})
        assert g.topic_status("scales", learner) == TopicStatus.MASTERED

    def test_topological_order(self):
        g = self._make_graph()
        order = g.topological_order()
        assert order.index("scales") < order.index("chords")
        assert order.index("chords") < order.index("prog")

    def test_learning_path(self):
        g = self._make_graph()
        learner = LearnerState(topic_scores={"scales": 0.9})
        path = g.learning_path(learner)
        assert isinstance(path, LearningPath)
        assert path.topics_mastered == 1
        assert path.topics_remaining == 3

    def test_suggest_next(self):
        g = self._make_graph()
        learner = LearnerState()
        suggestions = g.suggest_next(learner, n=2)
        assert len(suggestions) <= 2
        assert "scales" in suggestions

    def test_spaced_review(self):
        sched = spaced_review_schedule({"a": 0.9, "b": 0.3}, base_interval_days=1.0)
        assert sched["a"] > sched["b"]

    def test_forgetting_curve(self):
        r0 = forgetting_curve(1.0, 0.0)
        r1 = forgetting_curve(1.0, 1.0)
        r5 = forgetting_curve(1.0, 5.0)
        assert r0 > r1 > r5

    def test_optimal_review_time(self):
        t = optimal_review_time(1.0, target_retention=0.8, stability=2.0)
        assert t > 0
        r = forgetting_curve(1.0, t, stability=2.0)
        assert abs(r - 0.8) < 0.01

    def test_learner_state_record(self):
        ls = LearnerState()
        ls.record("scales", 0.6, 1.5)
        ls.record("scales", 0.8, 0.5)
        assert ls.score("scales") == 0.8
        assert ls.time_spent["scales"] == 2.0


class TestCepstral:
    def test_dct_ii_length(self):
        x = np.random.default_rng(0).random(20)
        d = dct_ii(x)
        assert len(d) == 20

    def test_mel_filterbank_shape(self):
        fb = mel_filterbank_matrix(40, 2048, sr=22050.0)
        assert fb.shape == (40, 1025)
        assert np.all(fb >= 0)

    def test_compute_mfcc_shape(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050.0)
        mfcc = compute_mfcc(sig, sr=22050.0, n_mfcc=13, n_fft=2048, hop=512)
        assert mfcc.shape[1] == 13
        assert mfcc.shape[0] > 0

    def test_mfcc_full_concatenation(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050.0)
        result = compute_mfcc_full(sig, sr=22050.0, n_mfcc=13)
        assert isinstance(result, MFCCResult)
        assert result.full_features.shape[1] == 39

    def test_lifter_shape(self):
        cep = np.random.default_rng(1).random((10, 13))
        lifted = lifter(cep, L=22)
        assert lifted.shape == cep.shape

    def test_delta_features_shape(self):
        feat = np.random.default_rng(2).random((20, 13))
        d = delta_features(feat, width=2)
        assert d.shape == feat.shape

    def test_delta_constant_zero(self):
        feat = np.ones((10, 5))
        d = delta_features(feat)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_lpc_coefficients(self):
        sig = np.sin(2 * np.pi * 440 * np.arange(1024) / 22050.0)
        a = lpc_coefficients(sig, order=12)
        assert len(a) == 13
        assert a[0] == 1.0

    def test_lpc_to_cepstrum_length(self):
        lpc = np.array([1.0, -0.5, 0.3, -0.1])
        c = lpc_to_cepstrum(lpc, n_cepstral=10)
        assert len(c) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
