"""Tests for curriculum learning, augmentation, and LR schedules."""
import numpy as np
import pytest
from models.music_training.curriculum_learning import (
    DifficultyScorer, CurriculumScheduler, AntiCurriculumScheduler,
    self_paced_weight, MixupAugmentor, LearningRateSchedule,
)


def test_difficulty_flat():
    ds = DifficultyScorer()
    assert ds.score(np.array([1.0, 1.0, 1.0])) < 0.01


def test_difficulty_decreasing():
    ds = DifficultyScorer()
    s = ds.score(np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
    assert s > 0


def test_curriculum_competence_start():
    cs = CurriculumScheduler(100, "linear")
    assert cs.competence(0, 100) < 0.2


def test_curriculum_competence_end():
    cs = CurriculumScheduler(100, "linear")
    assert cs.competence(100, 100) == 1.0


def test_curriculum_select():
    cs = CurriculumScheduler(20, "linear")
    diffs = np.arange(20, dtype=float)
    batch = cs.select_batch(diffs, 0.5, 5)
    assert len(batch) == 5
    assert all(diffs[i] <= 10 for i in batch)


def test_anti_curriculum_hard_first():
    acs = AntiCurriculumScheduler(20, "linear")
    diffs = np.arange(20, dtype=float)
    batch = acs.select_batch(diffs, 0.3, 5)
    assert all(diffs[i] >= 5 for i in batch)


def test_self_paced_binary():
    assert self_paced_weight(0.5, 1.0) == 1.0
    assert self_paced_weight(1.5, 1.0) == 0.0


def test_self_paced_soft():
    w = self_paced_weight(1.0, 1.0, soft=True)
    assert 0.4 < w < 0.6


def test_mixup_shape():
    aug = MixupAugmentor(seed=42)
    x1 = np.ones(10)
    x2 = np.zeros(10)
    mixed, lam = aug.mixup(x1, x2, alpha=0.2)
    assert mixed.shape == (10,)
    assert 0 <= lam <= 1


def test_cutmix_1d():
    aug = MixupAugmentor(seed=42)
    x1 = np.ones(20)
    x2 = np.zeros(20)
    result, lam = aug.cutmix_1d(x1, x2)
    assert 0.0 in result or 1.0 in result


def test_specaugment_masks():
    aug = MixupAugmentor(seed=42)
    spec = np.ones((64, 32))
    masked = aug.specaugment(spec, n_freq_masks=2, n_time_masks=2, freq_width=5, time_width=5)
    assert np.sum(masked == 0) > 0


def test_warmup_cosine_start():
    lr = LearningRateSchedule.warmup_cosine(0, 100, 1000, 0.001)
    assert lr == 0.0


def test_warmup_cosine_peak():
    lr = LearningRateSchedule.warmup_cosine(100, 100, 1000, 0.001)
    assert abs(lr - 0.001) < 1e-6


def test_cyclic_period():
    lrs = [LearningRateSchedule.cyclic(s, 20, 0.001, 0.01) for s in range(40)]
    assert abs(lrs[0] - lrs[20]) < 1e-6


def test_one_cycle_ends_low():
    lr_start = LearningRateSchedule.one_cycle(0, 1000, 0.01)
    lr_end = LearningRateSchedule.one_cycle(1000, 1000, 0.01)
    assert lr_end < lr_start
