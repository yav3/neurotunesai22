"""Tests for literature evidence loader."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")
MERLIN_LITERATURE = MERLIN_ROOT / "backend" / "data" / "literature" / "human"


class TestLiteratureDigestUnit:
    """Unit tests on synthetic literature evidence data."""

    @pytest.fixture
    def digest_path(self, tmp_path):
        data = {
            "_about": {"built": "2026-10-31"},
            "clinvar": {
                "source": "ClinVar test",
                "genes": {
                    "COL5A1": {
                        "pathogenic": 12,
                        "likely pathogenic": 5,
                        "benign": 3,
                        "uncertain significance": 8,
                        "total_submitted": 28,
                    },
                    "COL3A1": {
                        "pathogenic": 25,
                        "likely pathogenic": 10,
                        "benign": 1,
                        "uncertain significance": 4,
                        "total_submitted": 40,
                    },
                    "TNXB": {
                        "pathogenic": 0,
                        "likely pathogenic": 0,
                        "benign": 2,
                        "uncertain significance": 3,
                        "total_submitted": 5,
                    },
                },
            },
            "pubmed": {
                "query": "test query",
                "source": "PubMed test",
                "total_records": 100,
                "top_n_by_relevance": 3,
                "records": [
                    {"pmid": "12345", "title": "Study A", "journal": "J1", "year": 2023},
                    {"pmid": "12346", "title": "Study B", "journal": "J2", "year": 2024},
                    {"pmid": "12347", "title": "Study C", "journal": "J1", "year": 2024},
                ],
            },
            "clinicaltrials": {
                "condition": "Test Condition",
                "source": "CT.gov test",
                "total_studies": 50,
                "top_n_most_recently_updated": 4,
                "studies": [
                    {"nct_id": "NCT001", "title": "Trial 1", "status": "RECRUITING", "phases": ["PHASE2"]},
                    {"nct_id": "NCT002", "title": "Trial 2", "status": "COMPLETED", "phases": ["PHASE3"]},
                    {"nct_id": "NCT003", "title": "Trial 3", "status": "NOT_YET_RECRUITING", "phases": []},
                    {"nct_id": "NCT004", "title": "Trial 4", "status": "ACTIVE_NOT_RECRUITING", "phases": ["PHASE1", "PHASE2"]},
                ],
            },
        }
        p = tmp_path / "test_evidence_digest.json"
        p.write_text(json.dumps(data))
        return p

    def test_load_basic(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(digest_path)
        assert digest.n_clinvar_genes == 3
        assert digest.n_pubmed_records == 3
        assert digest.total_pubmed == 100
        assert digest.n_clinical_trials == 4
        assert digest.total_trials == 50
        assert len(digest.content_hash) == 64

    def test_clinvar_gene_names(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(digest_path)
        assert digest.clinvar_gene_names() == ["COL3A1", "COL5A1", "TNXB"]

    def test_pathogenic_genes(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(digest_path)
        patho = digest.pathogenic_genes(min_pathogenic=1)
        assert set(patho) == {"COL3A1", "COL5A1"}
        patho_strict = digest.pathogenic_genes(min_pathogenic=20)
        assert patho_strict == ["COL3A1"]

    def test_pathogenicity_vector(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(digest_path)
        vec, genes = digest.pathogenicity_vector()
        assert len(vec) == 3
        assert genes == ["COL3A1", "COL5A1", "TNXB"]
        assert vec[0] == pytest.approx(35.0)  # COL3A1: 25+10
        assert vec[1] == pytest.approx(17.0)  # COL5A1: 12+5
        assert vec[2] == pytest.approx(0.0)   # TNXB: 0+0

    def test_publication_years(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(digest_path)
        years = digest.publication_years()
        assert len(years) == 3
        assert set(years) == {2023, 2024}

    def test_trials_by_phase(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(digest_path)
        phases = digest.trials_by_phase()
        assert phases["PHASE2"] == 2  # NCT001 + NCT004
        assert phases["PHASE3"] == 1
        assert phases["PHASE1"] == 1
        assert phases["UNPHASED"] == 1  # NCT003

    def test_active_trials(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(digest_path)
        active = digest.active_trials()
        assert len(active) == 3  # RECRUITING + NOT_YET_RECRUITING + ACTIVE_NOT_RECRUITING
        nct_ids = {t.nct_id for t in active}
        assert nct_ids == {"NCT001", "NCT003", "NCT004"}

    def test_content_hash_stable(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        d1 = load_literature_digest(digest_path)
        d2 = load_literature_digest(digest_path)
        assert d1.content_hash == d2.content_hash

    def test_built_date(self, digest_path):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(digest_path)
        assert digest.built_date == "2026-10-31"


class TestLiteratureDiscovery:

    def test_discover_finds_digest(self, tmp_path):
        from models.data.literature_loader import discover_literature_digests
        data = {
            "_about": {"built": "2026-10-31"},
            "clinvar": {"genes": {"G1": {"pathogenic": 1}}},
            "pubmed": {"records": [], "total_records": 0},
            "clinicaltrials": {"studies": [], "total_studies": 0},
        }
        (tmp_path / "test_evidence_digest.json").write_text(json.dumps(data))
        results = discover_literature_digests(tmp_path)
        assert len(results) == 1
        assert results[0]["modality"] == "literature_evidence"
        assert results[0]["n_clinvar_genes"] == 1

    def test_discover_empty_dir(self, tmp_path):
        from models.data.literature_loader import discover_literature_digests
        assert discover_literature_digests(tmp_path) == []

    def test_discover_nonexistent(self):
        from models.data.literature_loader import discover_literature_digests
        assert discover_literature_digests("/nonexistent/path") == []


@pytest.mark.skipif(
    not MERLIN_LITERATURE.exists(),
    reason="Merlin literature data not available",
)
class TestRealLiteratureDigest:

    def test_load_eds_digest(self):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(MERLIN_LITERATURE / "eds_evidence_digest.json")
        assert digest.n_clinvar_genes == 15
        assert digest.total_pubmed == 5858
        assert digest.total_trials == 97
        assert len(digest.content_hash) == 64

    def test_pathogenic_genes_present(self):
        from models.data.literature_loader import load_literature_digest
        digest = load_literature_digest(MERLIN_LITERATURE / "eds_evidence_digest.json")
        patho = digest.pathogenic_genes(min_pathogenic=1)
        assert len(patho) >= 5

    def test_modality_bridge_includes_literature(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        lit = [e for e in entries if e.modality.value == "literature_evidence"]
        assert len(lit) >= 1
        assert lit[0].species == "human"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
