"""Tests for hierarchical structure segmentation."""
import numpy as np
import pytest
from models.music_structure.hierarchical_segmentation import (
    checkerboard_kernel, compute_novelty, pick_peaks,
    laplacian_segmentation, build_hierarchy, segment_homogeneity,
    boundary_precision_recall, form_string, SegmentNode,
)


def test_checkerboard_shape():
    k = checkerboard_kernel(8)
    assert k.shape == (8, 8)


def test_checkerboard_signs():
    k = checkerboard_kernel(4)
    assert k[0, 0] == -1
    assert k[0, 3] == 1
    assert k[3, 0] == 1
    assert k[3, 3] == -1


def test_novelty_nonneg():
    ssm = np.eye(20)
    n = compute_novelty(ssm, kernel_size=4)
    assert np.all(n >= 0)


def test_novelty_shape():
    ssm = np.random.default_rng(42).random((30, 30))
    n = compute_novelty(ssm, kernel_size=6)
    assert len(n) == 30


def test_pick_peaks_min_distance():
    novelty = np.zeros(20)
    novelty[5] = 1.0
    novelty[6] = 0.9
    novelty[15] = 0.8
    peaks = pick_peaks(novelty, threshold=0.5, min_distance=5)
    assert 5 in peaks
    assert 6 not in peaks


def test_laplacian_labels_count():
    rng = np.random.default_rng(42)
    features = np.vstack([rng.standard_normal((10, 5)) + i * 3 for i in range(3)])
    labels = laplacian_segmentation(features, k_clusters=3)
    assert len(labels) == 30
    assert len(set(labels)) <= 3


def test_build_hierarchy_root():
    features = np.random.default_rng(42).standard_normal((20, 4))
    root = build_hierarchy(features, levels=[2, 4])
    assert root.start_time == 0
    assert root.end_time == 20
    assert len(root.children) > 0


def test_segment_homogeneity_perfect():
    features = np.tile(np.array([1, 0, 0, 0, 0], dtype=float), (10, 1))
    h = segment_homogeneity(features, np.array([]))
    assert h > 0.99


def test_segment_homogeneity_mixed():
    rng = np.random.default_rng(42)
    f = rng.standard_normal((20, 5))
    h = segment_homogeneity(f, np.array([10]))
    assert 0 <= h <= 1


def test_boundary_pr_perfect():
    ref = np.array([5, 10, 15])
    p, r, f1 = boundary_precision_recall(ref, ref, tolerance=0.5)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_boundary_pr_partial():
    est = np.array([5, 11])
    ref = np.array([5, 10, 15])
    p, r, f1 = boundary_precision_recall(est, ref, tolerance=1.5)
    assert 0 < p <= 1 and 0 < r <= 1


def test_boundary_pr_empty():
    p, r, f1 = boundary_precision_recall(np.array([]), np.array([5, 10]))
    assert p == 0 and r == 0 and f1 == 0


def test_form_string_basic():
    labels = np.array([0, 0, 1, 1, 0, 0, 2, 2])
    assert form_string(labels) == "ABA" or form_string(labels) == "ABAC"


def test_form_string_repeat():
    labels = np.array([0, 0, 0])
    assert form_string(labels) == "A"
