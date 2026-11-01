"""Tests for crosscheck data loaders (AlphaFold DB + STRING PPI)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")
MERLIN_CROSSCHECKS = MERLIN_ROOT / "backend" / "data" / "crosschecks" / "human"


class TestAlphaFoldCrosscheckUnit:
    """Unit tests on synthetic AlphaFold data."""

    @pytest.fixture
    def af_path(self, tmp_path):
        data = {
            "description": "Test crosscheck",
            "fetched_date": "2026-10-31",
            "n_genes": 3,
            "source_apis": ["alphafold.ebi.ac.uk"],
            "genes": {
                "GENE_A": {"mean_plddt": 85.0, "grn_composite_score": 0.9},
                "GENE_B": {"mean_plddt": 55.0, "grn_composite_score": 0.3},
                "GENE_C": {"mean_plddt": 72.0, "grn_composite_score": 0.6},
            },
        }
        p = tmp_path / "alphafold_test_crosscheck.json"
        p.write_text(json.dumps(data))
        return p

    def test_load_basic(self, af_path):
        from models.data.crosscheck_loader import load_alphafold_crosscheck
        af = load_alphafold_crosscheck(af_path)
        assert af.n_genes == 3
        assert len(af.content_hash) == 64

    def test_gene_names_sorted(self, af_path):
        from models.data.crosscheck_loader import load_alphafold_crosscheck
        af = load_alphafold_crosscheck(af_path)
        assert af.gene_names() == ["GENE_A", "GENE_B", "GENE_C"]

    def test_structurally_actionable(self, af_path):
        from models.data.crosscheck_loader import load_alphafold_crosscheck
        af = load_alphafold_crosscheck(af_path)
        actionable = af.structurally_actionable(threshold=70.0)
        assert set(actionable) == {"GENE_A", "GENE_C"}


class TestStringPPICrosscheckUnit:
    """Unit tests on synthetic STRING PPI data."""

    @pytest.fixture
    def ppi_path(self, tmp_path):
        data = {
            "genes": ["GENE_A", "GENE_B", "GENE_C"],
            "rows": [
                {"preferredName_A": "GENE_A", "preferredName_B": "GENE_B",
                 "score": "0.85", "escore": "0.7", "tscore": "0.4"},
                {"preferredName_A": "GENE_A", "preferredName_B": "GENE_C",
                 "score": "0.45", "escore": "0.2", "tscore": "0.3"},
                {"preferredName_A": "GENE_B", "preferredName_B": "GENE_C",
                 "score": "0.92", "escore": "0.9", "tscore": "0.5"},
            ],
            "source": "STRING v12.0 test",
            "fetched_for": "test",
        }
        p = tmp_path / "string_ppi_test.json"
        p.write_text(json.dumps(data))
        return p

    def test_load_basic(self, ppi_path):
        from models.data.crosscheck_loader import load_string_ppi_crosscheck
        sp = load_string_ppi_crosscheck(ppi_path)
        assert sp.n_genes == 3
        assert sp.n_interactions == 3

    def test_adjacency_matrix(self, ppi_path):
        from models.data.crosscheck_loader import load_string_ppi_crosscheck
        sp = load_string_ppi_crosscheck(ppi_path)
        adj, genes = sp.adjacency_matrix()
        assert adj.shape == (3, 3)
        np.testing.assert_array_equal(adj, adj.T)
        # A-B score = 0.85
        ia = genes.index("GENE_A")
        ib = genes.index("GENE_B")
        assert adj[ia, ib] == pytest.approx(0.85)

    def test_adjacency_subscore(self, ppi_path):
        from models.data.crosscheck_loader import load_string_ppi_crosscheck
        sp = load_string_ppi_crosscheck(ppi_path)
        adj, genes = sp.adjacency_matrix(score_col="escore")
        ia = genes.index("GENE_B")
        ib = genes.index("GENE_C")
        assert adj[ia, ib] == pytest.approx(0.9)

    def test_high_confidence_edges(self, ppi_path):
        from models.data.crosscheck_loader import load_string_ppi_crosscheck
        sp = load_string_ppi_crosscheck(ppi_path)
        hc = sp.high_confidence_edges(threshold=0.7)
        assert len(hc) == 2  # A-B (0.85) and B-C (0.92)
        scores = [s for _, _, s in hc]
        assert all(s >= 0.7 for s in scores)

    def test_content_hash_stable(self, ppi_path):
        from models.data.crosscheck_loader import load_string_ppi_crosscheck
        sp1 = load_string_ppi_crosscheck(ppi_path)
        sp2 = load_string_ppi_crosscheck(ppi_path)
        assert sp1.content_hash == sp2.content_hash


class TestCrosscheckDiscovery:

    def test_discover_finds_both_types(self, tmp_path):
        from models.data.crosscheck_loader import discover_crosschecks
        af = {"description": "test", "genes": {"G1": {"mean_plddt": 80}}, "n_genes": 1}
        sp = {"genes": ["G1"], "rows": [], "source": "test"}
        (tmp_path / "alphafold_test_crosscheck.json").write_text(json.dumps(af))
        (tmp_path / "string_ppi_test.json").write_text(json.dumps(sp))
        results = discover_crosschecks(tmp_path)
        assert len(results) == 2
        types = {r["crosscheck_type"] for r in results}
        assert types == {"alphafold_structure", "string_ppi"}

    def test_discover_empty(self, tmp_path):
        from models.data.crosscheck_loader import discover_crosschecks
        assert discover_crosschecks(tmp_path) == []


@pytest.mark.skipif(
    not MERLIN_CROSSCHECKS.exists(),
    reason="Merlin crosscheck data not available",
)
class TestRealCrosschecks:

    def test_load_alphafold(self):
        from models.data.crosscheck_loader import load_alphafold_crosscheck
        af = load_alphafold_crosscheck(
            MERLIN_CROSSCHECKS / "alphafold_db_isomorphic_crosscheck.json"
        )
        assert af.n_genes == 14
        assert len(af.content_hash) == 64

    def test_load_string_ppi(self):
        from models.data.crosscheck_loader import load_string_ppi_crosscheck
        sp = load_string_ppi_crosscheck(
            MERLIN_CROSSCHECKS / "string_ppi_reprogramming.json"
        )
        assert sp.n_genes == 56
        assert sp.n_interactions == 402

    def test_string_adjacency_shape(self):
        from models.data.crosscheck_loader import load_string_ppi_crosscheck
        sp = load_string_ppi_crosscheck(
            MERLIN_CROSSCHECKS / "string_ppi_reprogramming.json"
        )
        adj, genes = sp.adjacency_matrix()
        assert adj.shape == (56, 56)
        np.testing.assert_array_equal(adj, adj.T)

    def test_modality_bridge_includes_crosschecks(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        cc = [e for e in entries if e.modality.value == "crosscheck"]
        assert len(cc) >= 2
        types = {e.extra["crosscheck_type"] for e in cc}
        assert "alphafold_structure" in types
        assert "string_ppi" in types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
