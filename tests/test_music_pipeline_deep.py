"""Deep tests for dataset building: splits, caching, statistics, balancing."""
import numpy as np
import pytest

from models.music_pipeline.dataset_builder import (
    DatasetSplit, stratified_split, FeatureCache,
    compute_dataset_statistics, DatasetStats,
    balance_classes, normalize_features,
)


class TestStratifiedSplit:
    def test_all_items_assigned(self):
        ids = [f"item_{i}" for i in range(100)]
        labels = [i % 3 for i in range(100)]
        split = stratified_split(ids, labels)
        all_assigned = set(split.train_ids) | set(split.val_ids) | set(split.test_ids)
        assert all_assigned == set(ids)

    def test_no_overlap(self):
        ids = [f"x{i}" for i in range(50)]
        labels = [0] * 25 + [1] * 25
        split = stratified_split(ids, labels)
        assert len(set(split.train_ids) & set(split.val_ids)) == 0
        assert len(set(split.train_ids) & set(split.test_ids)) == 0
        assert len(set(split.val_ids) & set(split.test_ids)) == 0

    def test_approximate_ratios(self):
        ids = [str(i) for i in range(200)]
        labels = [i % 2 for i in range(200)]
        split = stratified_split(ids, labels, train_ratio=0.7, val_ratio=0.15)
        total = len(ids)
        assert abs(len(split.train_ids) / total - 0.7) < 0.1
        assert abs(len(split.val_ids) / total - 0.15) < 0.1

    def test_deterministic(self):
        ids = [str(i) for i in range(50)]
        labels = [i % 3 for i in range(50)]
        a = stratified_split(ids, labels, seed=0)
        b = stratified_split(ids, labels, seed=0)
        assert a.train_ids == b.train_ids

    def test_sizes_property(self):
        split = stratified_split(["a", "b", "c"], [0, 0, 1])
        assert sum(split.sizes) == 3


class TestFeatureCache:
    def test_put_get(self):
        cache = FeatureCache()
        arr = np.array([1.0, 2.0])
        cache.put("k1", arr)
        result = cache.get("k1")
        assert result is not None
        assert np.array_equal(result, arr)

    def test_miss(self):
        cache = FeatureCache()
        assert cache.get("missing") is None

    def test_hit_rate(self):
        cache = FeatureCache()
        cache.put("a", np.array([1.0]))
        cache.get("a")
        cache.get("a")
        cache.get("b")
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_eviction(self):
        cache = FeatureCache(max_entries=2)
        cache.put("a", np.array([1]))
        cache.put("b", np.array([2]))
        cache.put("c", np.array([3]))
        assert cache.get("a") is None
        assert cache.get("c") is not None

    def test_clear(self):
        cache = FeatureCache()
        cache.put("a", np.array([1]))
        cache.clear()
        assert cache.get("a") is None
        assert cache.hit_rate == 0.0

    def test_content_key(self):
        k = FeatureCache.content_key(np.array([1.0, 2.0]))
        assert isinstance(k, str)
        assert len(k) == 16


class TestDatasetStats:
    def test_basic(self):
        feats = np.random.default_rng(0).random((50, 5))
        labels = np.array([0] * 25 + [1] * 25)
        stats = compute_dataset_statistics(feats, labels)
        assert isinstance(stats, DatasetStats)
        assert stats.n_samples == 50
        assert stats.n_features == 5
        assert stats.n_classes == 2

    def test_imbalance(self):
        feats = np.random.default_rng(0).random((30, 3))
        labels = np.array([0] * 5 + [1] * 25)
        stats = compute_dataset_statistics(feats, labels)
        assert stats.imbalance_ratio == 5.0

    def test_no_labels(self):
        feats = np.random.default_rng(0).random((10, 2))
        stats = compute_dataset_statistics(feats)
        assert stats.n_classes == 0

    def test_outlier_fraction_bounded(self):
        feats = np.random.default_rng(0).random((100, 3))
        stats = compute_dataset_statistics(feats)
        assert 0.0 <= stats.outlier_fraction <= 1.0


class TestBalanceClasses:
    def test_oversample(self):
        feats = np.random.default_rng(0).random((30, 2))
        labels = np.array([0] * 5 + [1] * 25)
        new_f, new_l = balance_classes(feats, labels, method="oversample")
        _, counts = np.unique(new_l, return_counts=True)
        assert counts[0] == counts[1]

    def test_undersample(self):
        feats = np.random.default_rng(0).random((30, 2))
        labels = np.array([0] * 5 + [1] * 25)
        new_f, new_l = balance_classes(feats, labels, method="undersample")
        _, counts = np.unique(new_l, return_counts=True)
        assert counts[0] == counts[1]


class TestNormalize:
    def test_standard(self):
        feats = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
        normed = normalize_features(feats, method="standard")
        assert np.abs(normed.mean(axis=0)).max() < 1e-10
        assert np.abs(normed.std(axis=0) - 1.0).max() < 1e-10

    def test_minmax(self):
        feats = np.array([[1.0], [5.0], [10.0]])
        normed = normalize_features(feats, method="minmax")
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)
