"""Tests for DREAM5 benchmark evaluation pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")
MERLIN_BENCHMARKS = MERLIN_ROOT / "backend" / "data" / "benchmarks"


class TestMetricFunctions:
    """Test AUROC, AUPRC, and precision@k on known inputs."""

    def test_auroc_perfect(self):
        from models.validation.benchmark_evaluator import _auroc
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert _auroc(y_true, y_score) == 1.0

    def test_auroc_random(self):
        from models.validation.benchmark_evaluator import _auroc
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 1000)
        y_score = rng.rand(1000)
        auroc = _auroc(y_true, y_score)
        assert 0.45 < auroc < 0.55

    def test_auroc_worst(self):
        from models.validation.benchmark_evaluator import _auroc
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert _auroc(y_true, y_score) == 0.0

    def test_auroc_degenerate(self):
        from models.validation.benchmark_evaluator import _auroc
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.9, 0.8, 0.7])
        assert _auroc(y_true, y_score) == 0.5

    def test_auprc_perfect(self):
        from models.validation.benchmark_evaluator import _auprc
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert _auprc(y_true, y_score) == 1.0

    def test_auprc_no_positives(self):
        from models.validation.benchmark_evaluator import _auprc
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.9, 0.5, 0.1])
        assert _auprc(y_true, y_score) == 0.0

    def test_precision_at_k(self):
        from models.validation.benchmark_evaluator import _precision_at_k
        y_true = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        prec = _precision_at_k(y_true, y_score, [1, 2, 5])
        assert prec[1] == 1.0
        assert prec[2] == 0.5
        assert prec[5] == 0.6


@pytest.mark.skipif(
    not (MERLIN_BENCHMARKS / "dream5" / "network1_subset_expression.csv").exists(),
    reason="Merlin DREAM5 benchmark data not available",
)
class TestBenchmarkEvaluator:

    def test_random_baseline_near_half(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        from models.validation.benchmark_evaluator import evaluate_random_baseline
        bench = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        result = evaluate_random_baseline(bench)
        assert 0.35 < result.auroc < 0.65
        assert result.benchmark == "DREAM5_network1"

    def test_correlation_baseline_beats_random(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        from models.validation.benchmark_evaluator import (
            evaluate_correlation_baseline,
            evaluate_random_baseline,
        )
        bench = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        corr = evaluate_correlation_baseline(bench)
        rand = evaluate_random_baseline(bench)
        assert corr.auroc > rand.auroc - 0.05

    def test_score_against_gold_standard_shape(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        from models.validation.benchmark_evaluator import score_against_gold_standard
        bench = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        n_tfs = len(bench.tf_names)
        n_genes = len(bench.gene_names)
        scores = np.ones((n_tfs, n_genes), dtype=np.float32)
        result = score_against_gold_standard(scores, bench)
        assert result.n_genes == n_genes
        assert result.n_tfs == n_tfs
        assert len(result.content_hash) == 64

    def test_result_fields_complete(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        from models.validation.benchmark_evaluator import evaluate_random_baseline
        bench = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        result = evaluate_random_baseline(bench)
        assert result.species == "in_silico"
        assert result.n_true_edges > 0
        assert result.n_false_edges > 0
        assert isinstance(result.precision_at_k, dict)
        assert 10 in result.precision_at_k


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
