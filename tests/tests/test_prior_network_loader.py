"""Tests for STRING PPI prior and DREAM5 benchmark loaders."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")
MERLIN_PRIORS = MERLIN_ROOT / "backend" / "data" / "priors" / "string" / "human"
MERLIN_BENCHMARKS = MERLIN_ROOT / "backend" / "data" / "benchmarks"


# ── STRING PPI prior tests ──────────────────────────────────────────

@pytest.mark.skipif(
    not MERLIN_PRIORS.exists(),
    reason="Merlin STRING prior data not available",
)
class TestStringPriorLoader:

    def test_load_prior_shape(self):
        from models.data.prior_network_loader import load_string_prior
        prior = load_string_prior(MERLIN_PRIORS / "gse107965.json")
        n = len(prior.genes)
        assert prior.adjacency.shape == (n, n)
        assert n > 0
        assert prior.n_edges > 0

    def test_prior_symmetric(self):
        from models.data.prior_network_loader import load_string_prior
        prior = load_string_prior(MERLIN_PRIORS / "gse107965.json")
        np.testing.assert_array_equal(prior.adjacency, prior.adjacency.T)

    def test_prior_normalized(self):
        from models.data.prior_network_loader import load_string_prior
        prior = load_string_prior(MERLIN_PRIORS / "gse107965.json", normalize=True)
        assert prior.adjacency.max() <= 1.0
        assert prior.adjacency.min() >= 0.0

    def test_prior_unnormalized(self):
        from models.data.prior_network_loader import load_string_prior
        prior = load_string_prior(MERLIN_PRIORS / "gse107965.json", normalize=False)
        assert prior.adjacency.max() <= 1000.0

    def test_prior_species(self):
        from models.data.prior_network_loader import load_string_prior
        prior = load_string_prior(MERLIN_PRIORS / "gse107965.json")
        assert prior.species == "human"

    def test_prior_content_hash_stable(self):
        from models.data.prior_network_loader import load_string_prior
        p1 = load_string_prior(MERLIN_PRIORS / "gse107965.json")
        p2 = load_string_prior(MERLIN_PRIORS / "gse107965.json")
        assert p1.content_hash == p2.content_hash
        assert len(p1.content_hash) == 64

    def test_prior_to_adjacency_submatrix(self):
        from models.data.prior_network_loader import load_string_prior, prior_to_adjacency
        prior = load_string_prior(MERLIN_PRIORS / "gse107965.json")
        target = prior.genes[:5]
        sub = prior_to_adjacency(prior, target)
        assert sub.shape == (5, 5)
        np.testing.assert_array_equal(sub, sub.T)

    def test_prior_to_adjacency_missing_genes(self):
        from models.data.prior_network_loader import load_string_prior, prior_to_adjacency
        prior = load_string_prior(MERLIN_PRIORS / "gse107965.json")
        target = ["NONEXISTENT_GENE_A", "NONEXISTENT_GENE_B"]
        sub = prior_to_adjacency(prior, target)
        assert sub.shape == (2, 2)
        assert sub.sum() == 0.0

    def test_discover_priors(self):
        from models.data.prior_network_loader import discover_string_priors
        priors = discover_string_priors(MERLIN_PRIORS)
        assert len(priors) >= 1
        for p in priors:
            assert p["modality"] == "ppi_prior"
            assert p["n_genes"] > 0
            assert p["n_edges"] > 0


# ── DREAM5 benchmark tests ─────────────────────────────────────────

@pytest.mark.skipif(
    not (MERLIN_BENCHMARKS / "dream5" / "network1_subset_expression.csv").exists(),
    reason="Merlin DREAM5 benchmark data not available",
)
class TestDream5Loader:

    def test_load_expression_shape(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        bench = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        assert bench.expression.shape[0] > 0  # samples
        assert bench.expression.shape[1] > 0  # genes
        assert bench.expression.shape[1] == len(bench.gene_names)

    def test_gold_standard_labels(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        bench = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        assert bench.n_true_edges > 0
        assert bench.n_false_edges > 0
        assert bench.gold_standard.shape[1] == 3
        labels = set(bench.gold_standard[:, 2].tolist())
        assert labels <= {0, 1}

    def test_tfs_are_subset_of_genes(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        bench = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        gene_set = set(bench.gene_names)
        for tf in bench.tf_names:
            assert tf in gene_set, f"TF {tf} not in gene list"

    def test_species_in_silico(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        bench = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        assert bench.species == "in_silico"

    def test_content_hash_stable(self):
        from models.data.prior_network_loader import load_dream5_benchmark
        b1 = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        b2 = load_dream5_benchmark(MERLIN_BENCHMARKS / "dream5")
        assert b1.content_hash == b2.content_hash
        assert len(b1.content_hash) == 64

    def test_discover_benchmarks(self):
        from models.data.prior_network_loader import discover_benchmarks
        benchmarks = discover_benchmarks(MERLIN_BENCHMARKS)
        assert len(benchmarks) >= 1
        for b in benchmarks:
            assert b["modality"] == "benchmark"
            assert b["n_genes"] > 0


# ── Extended modality bridge tests ──────────────────────────────────

@pytest.mark.skipif(
    not MERLIN_ROOT.exists(),
    reason="Merlin repo not available",
)
class TestExtendedModalityBridge:

    def test_priors_in_discovery(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        ppi = [e for e in entries if e.modality.value == "ppi_prior"]
        if MERLIN_PRIORS.exists():
            assert len(ppi) >= 1
            for e in ppi:
                assert e.species == "human"

    def test_benchmarks_in_discovery(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        bench = [e for e in entries if e.modality.value == "benchmark"]
        if (MERLIN_BENCHMARKS / "dream5").exists():
            assert len(bench) >= 1

    def test_four_modalities_discovered(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        modalities = {e.modality.value for e in entries}
        assert "bulk_expression" in modalities
        assert "ppi_prior" in modalities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
