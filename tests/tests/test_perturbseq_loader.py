"""Tests for the Perturb-seq data loader and modality bridge."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MERLIN_PERTURBSEQ = Path("/home/user/merlinaineuralplus2/backend/data/perturbseq/human")
MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")


# ── Perturb-seq loader tests ─────────────────────────────────────────

@pytest.mark.skipif(
    not (MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet").exists(),
    reason="Merlin Perturb-seq data not available",
)
class TestPerturbSeqLoader:

    def test_load_parquet_shape(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet
        expr, targets, genes, batches, meta = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        assert expr.shape[0] == meta.n_cells
        assert expr.shape[1] == meta.n_genes
        assert len(targets) == meta.n_cells
        assert len(genes) == meta.n_genes
        assert meta.n_cells > 0
        assert meta.n_genes > 0

    def test_load_parquet_species(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet
        _, _, _, _, meta = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        assert meta.species == "human"

    def test_control_cells_exist(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet, CONTROL_LABEL
        _, targets, _, _, meta = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        assert meta.n_control_cells > 0
        assert (targets == CONTROL_LABEL).sum() == meta.n_control_cells

    def test_content_hash_stable(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet
        _, _, _, _, meta1 = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        _, _, _, _, meta2 = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        assert meta1.content_hash == meta2.content_hash
        assert len(meta1.content_hash) == 64

    def test_pseudobulk_aggregation(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet, pseudobulk_by_target
        expr, targets, genes, _, meta = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        geo_data = pseudobulk_by_target(expr, targets, genes)
        assert geo_data.expression.shape[0] > 0
        assert geo_data.expression.shape[1] == len(genes)
        assert "control" in geo_data.conditions
        assert "treated" in geo_data.conditions
        assert geo_data.species == "human"

    def test_pseudobulk_conditions(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet, pseudobulk_by_target
        expr, targets, genes, _, _ = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        geo_data = pseudobulk_by_target(expr, targets, genes)
        n_control = sum(1 for c in geo_data.conditions if c == "control")
        n_treated = sum(1 for c in geo_data.conditions if c == "treated")
        assert n_control == 1  # Non-Targeting
        assert n_treated >= 1  # at least one knockout target

    def test_single_cell_pairs(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet, single_cell_pairs
        expr, targets, genes, _, _ = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        pairs = single_cell_pairs(expr, targets, genes, max_pairs_per_target=5)
        assert len(pairs) > 0
        for p in pairs:
            assert p["baseline"].shape == p["perturbed"].shape
            assert p["gene_target"] != "Non-Targeting"
            assert "gene_idx" in p
            assert "ptype" in p

    def test_single_cell_pairs_deterministic(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet, single_cell_pairs
        expr, targets, genes, _, _ = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
        )
        p1 = single_cell_pairs(expr, targets, genes, max_pairs_per_target=3, seed=42)
        p2 = single_cell_pairs(expr, targets, genes, max_pairs_per_target=3, seed=42)
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2):
            np.testing.assert_array_equal(a["baseline"], b["baseline"])

    def test_discover_perturbseq_datasets(self):
        from models.data.perturbseq_loader import discover_perturbseq_datasets
        datasets = discover_perturbseq_datasets(MERLIN_PERTURBSEQ)
        assert len(datasets) >= 1
        for d in datasets:
            assert d["modality"] == "perturbseq"
            assert d["species"] == "human"
            assert d["n_cells"] > 0
            assert d["in_regime"] is True  # 62k cells is in-regime

    def test_min_cells_filter(self):
        from models.data.perturbseq_loader import load_perturbseq_parquet
        _, _, _, _, meta_low = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
            min_cells_per_target=5,
        )
        _, _, _, _, meta_high = load_perturbseq_parquet(
            MERLIN_PERTURBSEQ / "xaira_panel_cells_subset.parquet",
            min_cells_per_target=200,
        )
        assert meta_low.n_targets >= meta_high.n_targets


# ── Modality bridge tests ────────────────────────────────────────────

@pytest.mark.skipif(
    not MERLIN_ROOT.exists(),
    reason="Merlin repo not available",
)
class TestModalityBridge:

    def test_discover_all_datasets(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        assert len(entries) > 0
        modalities = {e.modality.value for e in entries}
        assert "bulk_expression" in modalities

    def test_regime_summary(self):
        from models.data.modality_bridge import discover_all_datasets, regime_summary
        entries = discover_all_datasets(MERLIN_ROOT)
        summary = regime_summary(entries)
        assert "_total" in summary
        assert summary["_total"]["datasets"] == len(entries)
        assert summary["_total"]["total_samples"] > 0

    def test_dataset_entry_to_dict(self):
        from models.data.modality_bridge import DatasetEntry, DataModality
        entry = DatasetEntry(
            modality=DataModality.BULK_EXPRESSION,
            path="/test/path.csv",
            species="human",
            n_features=100,
            n_samples=50,
            content_hash="abc123",
            regime="data-driven",
            in_regime=True,
            accession="GSE123456",
        )
        d = entry.to_dict()
        assert d["modality"] == "bulk_expression"
        assert d["species"] == "human"
        assert d["n_features"] == 100

    def test_perturbseq_in_discovery(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        ps_entries = [e for e in entries if e.modality.value == "perturbseq"]
        if (MERLIN_ROOT / "backend/data/perturbseq/human").exists():
            assert len(ps_entries) >= 1
            for e in ps_entries:
                assert e.in_regime is True
                assert e.regime == "data-driven"

    def test_bulk_expression_regime_classification(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        bulk = [e for e in entries if e.modality.value == "bulk_expression"]
        if bulk:
            regimes = {e.regime for e in bulk}
            assert len(regimes) > 0
            for e in bulk:
                if e.n_samples >= 30:
                    assert e.regime == "data-driven"
                else:
                    assert e.regime in ("prior-contrast", "prior-mean")

    def test_species_valid(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        valid_species = {"human", "in_silico", "in_vitro", "mouse", "synthetic", "not_applicable"}
        for e in entries:
            assert e.species in valid_species, f"{e.path}: unexpected species {e.species}"
            if e.modality.value in ("bulk_expression", "perturbseq", "ppi_prior"):
                assert e.species == "human"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
