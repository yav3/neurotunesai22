"""Tests for set theory, music theory tutor, performance assessment, cross-validation."""
import numpy as np
import pytest
from models.music_harmony.set_theory import (
    PCSet, to_pc_set, transpose, invert, complement,
    normal_form, prime_form, interval_vector,
    is_z_related, is_subset, is_superset,
    common_tones_under_transposition, similarity_index,
    FORTE_CATALOG_3, FORTE_CATALOG_4, lookup_forte, all_subsets,
)
from models.music_education.music_theory_tutor import (
    MasteryLevel, BloomLevel, Concept,
    MUSIC_THEORY_GRAPH, build_concept_graph, topological_order,
    prerequisites_met, knowledge_gaps, lesson_sequence,
    update_mastery, bloom_level_for_exercise, recommend_next_concepts,
)
from models.music_education.performance_assessment import (
    ErrorType, PerformanceError, RubricScore,
    dtw_alignment, classify_errors, score_performance,
    difficulty_adjustment, ProgressRecord,
)
from models.music_pipeline.cross_validation import (
    FoldResult, CVResult,
    k_fold_split, stratified_k_fold, leave_one_out, group_k_fold,
    accuracy, mean_absolute_error, f1_score_binary, confusion_matrix,
    learning_curve_points,
)


class TestSetTheory:
    def test_to_pc_set(self):
        pcs = to_pc_set([0, 4, 7, 12, 16])
        assert pcs.pcs == (0, 4, 7)

    def test_transpose(self):
        pcs = PCSet((0, 4, 7))
        t = transpose(pcs, 3)
        assert t.pcs == (3, 7, 10)

    def test_invert(self):
        pcs = PCSet((0, 4, 7))
        inv = invert(pcs, 0)
        assert 0 in inv.pcs
        assert 8 in inv.pcs
        assert 5 in inv.pcs

    def test_complement(self):
        pcs = PCSet((0, 4, 7))
        comp = complement(pcs)
        assert comp.cardinality == 9
        assert 0 not in comp.pcs

    def test_normal_form(self):
        pcs = PCSet((0, 4, 7))
        nf = normal_form(pcs)
        assert nf.pcs[0] == 0

    def test_prime_form_major_triad(self):
        pcs = PCSet((0, 4, 7))
        pf = prime_form(pcs)
        assert pf.pcs == (0, 3, 7)

    def test_interval_vector_major_triad(self):
        pcs = PCSet((0, 4, 7))
        iv = interval_vector(pcs)
        assert iv == (0, 0, 1, 1, 1, 0)

    def test_z_related(self):
        a = PCSet(FORTE_CATALOG_4["4-z15"])
        b = PCSet(FORTE_CATALOG_4["4-z29"])
        assert is_z_related(a, b) is True

    def test_not_z_related_same_prime(self):
        a = PCSet((0, 1, 2, 3))
        b = PCSet((3, 4, 5, 6))
        assert is_z_related(a, b) is False

    def test_subset(self):
        a = PCSet((0, 4))
        b = PCSet((0, 4, 7))
        assert is_subset(a, b) is True
        assert is_superset(b, a) is True

    def test_common_tones(self):
        pcs = PCSet((0, 4, 7))
        ct = common_tones_under_transposition(pcs, 5)
        assert ct == 1

    def test_similarity_index(self):
        a = PCSet((0, 4, 7))
        b = PCSet((0, 3, 7))
        s = similarity_index(a, b)
        assert -0.01 <= s <= 1.01

    def test_lookup_forte_major(self):
        pcs = PCSet((0, 3, 7))
        name = lookup_forte(pcs)
        assert name == "3-11"

    def test_all_subsets(self):
        pcs = PCSet((0, 4, 7))
        subs = all_subsets(pcs, 2)
        assert len(subs) == 3


class TestMusicTheoryTutor:
    def test_build_concept_graph(self):
        g = build_concept_graph()
        assert len(g) == len(MUSIC_THEORY_GRAPH)
        assert "pitch" in g
        assert "modulation" in g

    def test_topological_order(self):
        g = build_concept_graph()
        order = topological_order(g)
        pitch_idx = order.index("pitch")
        intervals_idx = order.index("intervals")
        assert pitch_idx < intervals_idx

    def test_prerequisites_met(self):
        g = build_concept_graph()
        assert prerequisites_met("pitch", g) is True
        assert prerequisites_met("intervals", g) is False

    def test_knowledge_gaps(self):
        g = build_concept_graph()
        gaps = knowledge_gaps(g, "chords")
        assert "pitch" in gaps or "intervals" in gaps

    def test_lesson_sequence(self):
        g = build_concept_graph()
        seq = lesson_sequence(g, "triads")
        assert seq[-1] == "triads"

    def test_update_mastery(self):
        c = Concept("test", "Test")
        for _ in range(10):
            update_mastery(c, True)
        assert c.mastery >= MasteryLevel.INTERMEDIATE

    def test_bloom_level(self):
        assert bloom_level_for_exercise("identify") == BloomLevel.REMEMBER
        assert bloom_level_for_exercise("compose") == BloomLevel.CREATE
        assert bloom_level_for_exercise("analyze") == BloomLevel.ANALYZE

    def test_recommend_next(self):
        g = build_concept_graph()
        recs = recommend_next_concepts(g, n=3)
        assert len(recs) <= 3
        assert all(isinstance(r, str) for r in recs)


class TestPerformanceAssessment:
    def test_dtw_identical(self):
        ref = np.array([60.0, 62.0, 64.0, 65.0])
        cost, path = dtw_alignment(ref, ref)
        for i, j in path:
            assert i == j

    def test_dtw_empty(self):
        cost, path = dtw_alignment(np.array([]), np.array([]))
        assert len(path) == 0

    def test_classify_errors_perfect(self):
        ref_p = np.array([60.0, 62.0, 64.0])
        ref_o = np.array([0.0, 0.5, 1.0])
        errors = classify_errors(ref_p, ref_p, ref_o, ref_o)
        pitch_errors = [e for e in errors if e.error_type == ErrorType.WRONG_NOTE]
        assert len(pitch_errors) == 0

    def test_classify_errors_wrong_note(self):
        ref_p = np.array([60.0, 62.0, 64.0])
        perf_p = np.array([60.0, 68.0, 64.0])
        ref_o = np.array([0.0, 0.5, 1.0])
        perf_o = np.array([0.0, 0.5, 1.0])
        errors = classify_errors(ref_p, perf_p, ref_o, perf_o)
        wrong = [e for e in errors if e.error_type == ErrorType.WRONG_NOTE]
        assert len(wrong) >= 1

    def test_score_performance(self):
        ref_p = np.array([60.0, 62.0, 64.0, 65.0])
        ref_o = np.array([0.0, 0.5, 1.0, 1.5])
        score = score_performance(ref_p, ref_p, ref_o, ref_o)
        assert score.pitch_accuracy >= 90.0
        assert score.overall > 0.0

    def test_difficulty_adjustment_up(self):
        s = RubricScore(pitch_accuracy=95, rhythm_accuracy=95, dynamics=90, articulation=90, expression=90)
        new_d = difficulty_adjustment(s, 0.5)
        assert new_d > 0.5

    def test_progress_record_trend(self):
        pr = ProgressRecord()
        for v in [60.0, 65.0, 70.0, 75.0, 80.0]:
            pr.add(v)
        assert pr.trend > 0.0
        assert pr.best == 80.0
        assert pr.recent_average == pytest.approx(70.0)


class TestCrossValidation:
    def test_k_fold_split_coverage(self):
        folds = k_fold_split(100, k=5)
        assert len(folds) == 5
        all_val = np.concatenate([v for _, v in folds])
        assert len(np.unique(all_val)) == 100

    def test_k_fold_no_overlap(self):
        folds = k_fold_split(50, k=5)
        for train, val in folds:
            assert len(np.intersect1d(train, val)) == 0

    def test_stratified_k_fold(self):
        labels = np.array([0]*20 + [1]*20 + [2]*20)
        folds = stratified_k_fold(labels, k=5)
        assert len(folds) == 5

    def test_leave_one_out(self):
        folds = leave_one_out(10)
        assert len(folds) == 10
        for train, val in folds:
            assert len(val) == 1
            assert len(train) == 9

    def test_group_k_fold(self):
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        folds = group_k_fold(groups, k=5)
        assert len(folds) == 5

    def test_accuracy(self):
        y = np.array([0, 1, 1, 0, 1])
        assert accuracy(y, y) == 1.0
        assert accuracy(y, 1 - y) == 0.0

    def test_f1_binary(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 1, 0])
        f1 = f1_score_binary(y_true, y_pred)
        assert 0.0 < f1 < 1.0

    def test_confusion_matrix(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2

    def test_cv_result_properties(self):
        folds = [
            FoldResult(0, np.array([]), np.array([]), train_score=0.9, val_score=0.8),
            FoldResult(1, np.array([]), np.array([]), train_score=0.92, val_score=0.82),
        ]
        cv = CVResult(folds)
        assert cv.mean_train == pytest.approx(0.91)
        assert cv.mean_val == pytest.approx(0.81)
        assert cv.overfitting_gap == pytest.approx(0.10)
        assert cv.std_val >= 0.0

    def test_learning_curve_points(self):
        pts = learning_curve_points([0.2, 0.5, 1.0], 100, k=5)
        assert len(pts) == 3
        assert pts[0][0] <= pts[1][0] <= pts[2][0]
