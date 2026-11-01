"""Tests for music_education: ear training, sight-reading, practice, assessment."""
import pytest

from models.music_education.ear_training import (
    IntervalTrainer, ChordTrainer, RhythmDictation,
    INTERVAL_NAMES, CHORD_TYPES, ExerciseResult,
)
from models.music_education.sight_reading import (
    SightReadingGenerator, DifficultyEstimator, SightReadingExercise,
)
from models.music_education.practice import (
    SpacedRepetition, SkillTracker, SkillRecord, PracticePlanner,
)
from models.music_education.assessment import (
    RubricCriterion, PerformanceRubric, SkillAssessment,
)


# --- Ear training ---

class TestIntervalTrainer:
    def test_difficulty_1_intervals(self):
        t = IntervalTrainer(difficulty=1)
        assert set(t.available_intervals) == {0, 5, 7, 12}

    def test_difficulty_3_all_chromatic(self):
        t = IntervalTrainer(difficulty=3)
        assert t.available_intervals == list(range(0, 13))

    def test_generate_exercise(self):
        t = IntervalTrainer(difficulty=2)
        ex = t.generate_exercise(root=60)
        assert "root_midi" in ex and ex["root_midi"] == 60
        assert "target_midi" in ex
        assert "interval_name" in ex

    def test_check_answer_correct(self):
        t = IntervalTrainer(difficulty=1)
        ex = t.generate_exercise()
        result = t.check_answer(ex, ex["interval_name"])
        assert result.correct is True

    def test_check_answer_wrong(self):
        t = IntervalTrainer(difficulty=1)
        ex = t.generate_exercise()
        result = t.check_answer(ex, "wrong answer")
        assert result.correct is False

    def test_accuracy(self):
        t = IntervalTrainer(difficulty=1)
        ex = t.generate_exercise()
        t.check_answer(ex, ex["interval_name"])
        t.check_answer(ex, "wrong")
        assert t.accuracy() == 0.5


class TestChordTrainer:
    def test_default_types(self):
        ct = ChordTrainer()
        assert "major" in ct._types and "minor" in ct._types

    def test_generate_exercise(self):
        ct = ChordTrainer(["major", "minor"])
        ex = ct.generate_exercise(root=60)
        assert ex["chord_type"] in ["major", "minor"]
        assert ex["root_midi"] == 60
        assert len(ex["midi_notes"]) >= 3

    def test_check_answer(self):
        ct = ChordTrainer(["major"])
        ex = ct.generate_exercise()
        r = ct.check_answer(ex, "major")
        assert r.correct is True

    def test_accuracy_empty(self):
        ct = ChordTrainer()
        assert ct.accuracy() == 0.0


class TestRhythmDictation:
    def test_generate_pattern_d1(self):
        rd = RhythmDictation(difficulty=1)
        p = rd.generate_pattern()
        assert sum(p) == pytest.approx(4.0, abs=0.01)

    def test_check_pattern_exact(self):
        expected = [1.0, 1.0, 1.0, 1.0]
        score = RhythmDictation.check_pattern(expected, expected)
        assert score == 1.0

    def test_check_pattern_wrong_length(self):
        score = RhythmDictation.check_pattern([1, 1, 1, 1], [1, 1])
        assert score == 0.0

    def test_check_pattern_tolerance(self):
        score = RhythmDictation.check_pattern([1.0, 1.0], [1.05, 0.95], tolerance=0.1)
        assert score == 1.0


# --- Sight-reading ---

class TestSightReadingGenerator:
    def test_deterministic_seed(self):
        g1 = SightReadingGenerator(seed=42)
        g2 = SightReadingGenerator(seed=42)
        e1 = g1.generate(n_bars=2, difficulty=2)
        e2 = g2.generate(n_bars=2, difficulty=2)
        assert e1.notes == e2.notes

    def test_exercise_fields(self):
        g = SightReadingGenerator(seed=1)
        ex = g.generate(n_bars=4, difficulty=1, time_sig=(4, 4))
        assert isinstance(ex, SightReadingExercise)
        assert len(ex.notes) == len(ex.durations)
        assert ex.key_signature == 0
        assert ex.time_signature == (4, 4)

    def test_total_beats(self):
        g = SightReadingGenerator(seed=1)
        ex = g.generate(n_bars=2, difficulty=1, time_sig=(3, 4))
        assert abs(sum(ex.durations) - 6.0) < 0.01

    def test_difficulty_score_range(self):
        g = SightReadingGenerator(seed=1)
        ex = g.generate(n_bars=4, difficulty=3)
        assert 0.0 <= ex.difficulty <= 1.0


class TestDifficultyEstimator:
    def test_empty(self):
        de = DifficultyEstimator()
        assert de.estimate([], [], 0, (4, 4)) == 0.0

    def test_monotonic_range(self):
        de = DifficultyEstimator()
        narrow = de.estimate([60, 62, 64], [1, 1, 1])
        wide = de.estimate([36, 60, 96], [1, 1, 1])
        assert wide > narrow

    def test_accidental_contribution(self):
        de = DifficultyEstimator()
        no_acc = de.estimate([60, 62, 64], [1, 1, 1], key_sig=0)
        with_acc = de.estimate([60, 62, 64], [1, 1, 1], key_sig=5)
        assert with_acc > no_acc


# --- Practice / spaced repetition ---

class TestSpacedRepetition:
    def test_first_interval(self):
        sr = SpacedRepetition()
        rec = SkillRecord(name="scales", attempts=0)
        interval = sr.next_interval(rec, quality=4)
        assert interval == 1.0

    def test_second_interval(self):
        sr = SpacedRepetition()
        rec = SkillRecord(name="scales", attempts=2)
        interval = sr.next_interval(rec, quality=4)
        assert interval == 6.0

    def test_low_quality_resets(self):
        sr = SpacedRepetition()
        rec = SkillRecord(name="scales", attempts=5, interval_days=10.0)
        interval = sr.next_interval(rec, quality=1)
        assert interval == 1.0

    def test_update_increments_attempts(self):
        sr = SpacedRepetition()
        rec = SkillRecord(name="scales")
        updated = sr.update(rec, quality=4, current_day=1.0)
        assert updated.attempts == 1
        assert updated.last_practiced == 1.0

    def test_ease_factor_minimum(self):
        sr = SpacedRepetition()
        rec = SkillRecord(name="scales", ease_factor=1.3)
        updated = sr.update(rec, quality=0)
        assert updated.ease_factor >= 1.3


class TestSkillTracker:
    def test_add_skill(self):
        st = SkillTracker()
        rec = st.add_skill("arpeggios")
        assert rec.name == "arpeggios"
        assert rec.level == 0.0

    def test_record_practice(self):
        st = SkillTracker()
        rec = st.record_practice("arpeggios", quality=4, current_day=1.0)
        assert rec.attempts == 1
        assert rec.level > 0

    def test_due_skills(self):
        st = SkillTracker()
        st.record_practice("arpeggios", quality=4, current_day=0.0)
        st.record_practice("scales", quality=4, current_day=0.0)
        due = st.due_skills(current_day=2.0)
        assert len(due) == 2

    def test_overall_level(self):
        st = SkillTracker()
        st.record_practice("a", quality=5, current_day=0)
        st.record_practice("b", quality=5, current_day=0)
        assert st.overall_level() > 0

    def test_get_skill(self):
        st = SkillTracker()
        assert st.get_skill("nonexistent") is None
        st.add_skill("test")
        assert st.get_skill("test") is not None


class TestPracticePlanner:
    def test_plan_empty(self):
        pp = PracticePlanner(session_minutes=30)
        st = SkillTracker()
        plan = pp.plan(st)
        assert plan == []

    def test_plan_allocates_minutes(self):
        pp = PracticePlanner(session_minutes=30)
        st = SkillTracker()
        st.record_practice("scales", quality=3, current_day=0)
        st.record_practice("arpeggios", quality=2, current_day=0)
        plan = pp.plan(st, current_day=5.0)
        assert len(plan) >= 1
        for item in plan:
            assert "skill" in item and "minutes" in item
            assert item["minutes"] >= 5


# --- Assessment ---

class TestRubricCriterion:
    def test_max_score(self):
        rc = RubricCriterion("test", weight=2.0)
        assert rc.max_score() == 8.0

    def test_default_levels(self):
        rc = RubricCriterion("test")
        assert 4 in rc.levels
        assert rc.levels[1] == "Beginning"


class TestPerformanceRubric:
    def test_default_criteria_count(self):
        pr = PerformanceRubric()
        assert len(pr.criteria) == 5

    def test_max_total(self):
        pr = PerformanceRubric()
        assert pr.max_total() == (1.5 + 1.5 + 1.0 + 1.0 + 1.0) * 4

    def test_score_all_max(self):
        pr = PerformanceRubric()
        ratings = {c.name: 4 for c in pr.criteria}
        assert pr.score(ratings) == pr.max_total()

    def test_normalized_score_range(self):
        pr = PerformanceRubric()
        ratings = {c.name: 2 for c in pr.criteria}
        ns = pr.normalized_score(ratings)
        assert 0.0 <= ns <= 1.0

    def test_feedback_length(self):
        pr = PerformanceRubric()
        ratings = {"pitch_accuracy": 3, "rhythm_accuracy": 2}
        fb = pr.feedback(ratings)
        assert len(fb) == 5
        for item in fb:
            assert "criterion" in item and "weighted_score" in item


class TestSkillAssessment:
    def test_add_assessment(self):
        sa = SkillAssessment()
        score = sa.add_assessment({"pitch_accuracy": 4, "rhythm_accuracy": 4,
                                    "dynamics": 4, "articulation": 4, "expression": 4})
        assert 0.0 <= score <= 1.0

    def test_average(self):
        sa = SkillAssessment()
        sa.add_assessment({"pitch_accuracy": 4, "rhythm_accuracy": 4,
                           "dynamics": 4, "articulation": 4, "expression": 4})
        sa.add_assessment({"pitch_accuracy": 1, "rhythm_accuracy": 1,
                           "dynamics": 1, "articulation": 1, "expression": 1})
        avg = sa.average()
        assert 0.0 < avg < 1.0

    def test_trend_needs_two(self):
        sa = SkillAssessment()
        assert sa.trend() == 0.0

    def test_mastery_level(self):
        sa = SkillAssessment()
        sa.add_assessment({"pitch_accuracy": 4, "rhythm_accuracy": 4,
                           "dynamics": 4, "articulation": 4, "expression": 4})
        assert sa.mastery_level() in ("master", "advanced", "proficient", "developing", "beginning")
