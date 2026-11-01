"""Deep tests for spaced repetition in music learning."""
import numpy as np
import pytest

from models.music_education.spaced_repetition import (
    ReviewItem, ReviewResult, sm2_update,
    retention_probability, optimal_interval,
    LeitnerBox, forgetting_curve,
    schedule_session, difficulty_estimate,
)


class TestSM2Update:
    def test_good_review_increases_interval(self):
        item = ReviewItem("a", interval_days=1.0, repetitions=1)
        result = sm2_update(item, quality=4, current_day=1.0)
        assert result.new_interval > 1.0

    def test_bad_review_resets(self):
        item = ReviewItem("a", interval_days=10.0, repetitions=5)
        result = sm2_update(item, quality=1, current_day=10.0)
        assert result.new_interval == 1.0

    def test_ease_bounded_below(self):
        item = ReviewItem("a", ease_factor=1.3)
        result = sm2_update(item, quality=0, current_day=0.0)
        assert result.new_ease >= 1.3

    def test_returns_review_result(self):
        item = ReviewItem("x")
        result = sm2_update(item, quality=3, current_day=0.0)
        assert isinstance(result, ReviewResult)
        assert result.item_id == "x"

    def test_next_review_day(self):
        item = ReviewItem("a")
        result = sm2_update(item, quality=4, current_day=5.0)
        assert result.next_review_day > 5.0

    def test_lapse_count_increments(self):
        item = ReviewItem("a", lapse_count=2)
        sm2_update(item, quality=1, current_day=0.0)
        assert item.lapse_count == 3


class TestRetention:
    def test_immediate_perfect(self):
        assert retention_probability(0.0) == 1.0

    def test_decays(self):
        r1 = retention_probability(1.0)
        r10 = retention_probability(10.0)
        assert r1 > r10

    def test_higher_stability_slower_decay(self):
        r_low = retention_probability(5.0, stability=1.0)
        r_high = retention_probability(5.0, stability=5.0)
        assert r_high > r_low


class TestOptimalInterval:
    def test_high_retention_short(self):
        i = optimal_interval(0.95)
        assert i > 0

    def test_low_retention_long(self):
        i_high = optimal_interval(0.95)
        i_low = optimal_interval(0.5)
        assert i_low > i_high

    def test_perfect_zero(self):
        assert optimal_interval(1.0) == 0.0


class TestLeitnerBox:
    def test_add_item(self):
        lb = LeitnerBox()
        lb.add_item("chord-I")
        assert "chord-I" in lb.item_box
        assert lb.item_box["chord-I"] == 0

    def test_correct_promotes(self):
        lb = LeitnerBox()
        lb.add_item("a")
        new_box = lb.review("a", correct=True)
        assert new_box == 1

    def test_incorrect_demotes(self):
        lb = LeitnerBox()
        lb.add_item("a")
        lb.review("a", correct=True)
        new_box = lb.review("a", correct=False)
        assert new_box == 0

    def test_max_box(self):
        lb = LeitnerBox(n_boxes=3)
        lb.add_item("a")
        for _ in range(10):
            lb.review("a", correct=True)
        assert lb.item_box["a"] == 2

    def test_due_items_day_0(self):
        lb = LeitnerBox()
        lb.add_item("a")
        lb.add_item("b")
        due = lb.due_items(0)
        assert len(due) >= 2

    def test_mastery_ratio(self):
        lb = LeitnerBox(n_boxes=3)
        lb.add_item("a")
        lb.add_item("b")
        for _ in range(5):
            lb.review("a", correct=True)
        ratio = lb.mastery_ratio()
        assert 0.0 <= ratio <= 1.0


class TestForgettingCurve:
    def test_shape(self):
        days = np.array([0, 1, 5, 10, 30])
        curve = forgetting_curve(days)
        assert len(curve) == 5
        assert curve[0] == pytest.approx(1.0)
        assert all(curve[i] >= curve[i + 1] for i in range(4))


class TestScheduleSession:
    def test_overdue_first(self):
        items = [
            ReviewItem("a", ease_factor=2.5, last_review_day=0.0),
            ReviewItem("b", ease_factor=2.5, last_review_day=9.0),
        ]
        due = schedule_session(items, current_day=10.0, target_retention=0.9)
        if len(due) >= 2:
            assert due[0] == "a"

    def test_max_reviews(self):
        items = [ReviewItem(str(i)) for i in range(50)]
        due = schedule_session(items, current_day=100.0, max_reviews=5)
        assert len(due) <= 5


class TestDifficultyEstimate:
    def test_empty(self):
        assert difficulty_estimate([]) == 0.5

    def test_perfect_easy(self):
        d = difficulty_estimate([5, 5, 5])
        assert d < 0.2

    def test_poor_hard(self):
        d = difficulty_estimate([0, 1, 0])
        assert d > 0.7
