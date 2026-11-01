"""Tests for governed study runner.

Validates:
  - Data sufficiency checking against Merlin's committed datasets
  - Pre-registration enforcement (must declare before run)
  - Verdict immutability (cannot change after first recording)
  - Lineage chain integrity (hash-chained evidence records)
  - End-to-end governed study on real Merlin GEO data
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from models.training.study_runner import (
    DataSufficiency,
    StudyRunner,
    check_data_sufficiency,
)

MERLIN_DATA_DIR = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")


# ── Data sufficiency ──────────────────────────────────────────────────

@pytest.mark.skipif(not MERLIN_DATA_DIR.exists(), reason="Merlin data not available")
class TestDataSufficiency:

    def test_all_datasets_sufficient(self):
        result = check_data_sufficiency(MERLIN_DATA_DIR)
        assert result.datasets_found >= 8
        assert result.datasets_qualified >= 2  # GSE107965 + GSE119262 at min

    def test_gse119262_is_data_driven(self):
        result = check_data_sufficiency(MERLIN_DATA_DIR, accessions=["GSE119262"])
        assert result.sufficient
        assert result.regime == "data-driven"
        assert result.total_samples >= 30

    def test_gse107965_is_prior_contrast(self):
        result = check_data_sufficiency(
            MERLIN_DATA_DIR, accessions=["GSE107965"], min_samples=5,
        )
        assert result.sufficient
        assert result.regime == "prior-contrast"
        assert result.total_control == 6
        assert result.total_treated == 12

    def test_nonexistent_accession(self):
        result = check_data_sufficiency(MERLIN_DATA_DIR, accessions=["GSE999999"])
        assert not result.sufficient
        assert result.datasets_qualified == 0

    def test_high_min_samples_filters(self):
        result = check_data_sufficiency(
            MERLIN_DATA_DIR, accessions=["GSE107965"], min_samples=100,
        )
        assert not result.sufficient

    def test_sufficiency_dict_roundtrip(self):
        result = check_data_sufficiency(MERLIN_DATA_DIR, accessions=["GSE119262"])
        d = result.to_dict()
        assert d["sufficient"] is True
        assert isinstance(d["accessions"], list)


# ── Study runner governance ───────────────────────────────────────────

@pytest.mark.skipif(not MERLIN_DATA_DIR.exists(), reason="Merlin data not available")
class TestStudyRunnerGovernance:

    def test_preregistration_enforced(self):
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        # cannot record verdict for unregistered study
        with pytest.raises(KeyError, match="not registered"):
            runner.registry.record_verdict(
                "NONEXISTENT", {}, "test", "met",
            )

    def test_duplicate_registration_refused(self):
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        runner.registry.register("TEST-001", "Test", "val_loss < 2.0")
        with pytest.raises(ValueError, match="already registered"):
            runner.registry.register("TEST-001", "Test2", "val_loss < 1.0")

    def test_verdict_immutability(self):
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        runner.registry.register("TEST-002", "Test", "val_loss < 2.0")
        runner.registry.record_verdict("TEST-002", {"val_loss": 1.5}, "ok", "met")
        with pytest.raises(ValueError, match="already recorded"):
            runner.registry.record_verdict("TEST-002", {"val_loss": 0.5}, "better", "met")

    def test_insufficient_data_records_not_met(self):
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        result = runner.run_study(
            study_id="TEST-INSUFF",
            title="Test insufficient data",
            accessions=["GSE999999"],
        )
        assert result.verdict == "not_met"
        assert "insufficient" in result.provenance.get("reason", "").lower() or \
               result.provenance.get("stage") == "data_validation"

    def test_lineage_chain_integrity(self):
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        # run a study that will fail due to impossible accession
        runner.run_study("LIN-001", "Lineage test 1", accessions=["GSE999999"])
        runner.run_study("LIN-002", "Lineage test 2", accessions=["GSE888888"])
        assert runner.lineage.verify_chain()


# ── End-to-end governed study ──────────────────────────────────────────

@pytest.mark.skipif(not MERLIN_DATA_DIR.exists(), reason="Merlin data not available")
class TestEndToEndStudy:

    def test_governed_study_gse107965(self):
        """Run a governed study on GSE107965 (prior-contrast regime, 18 samples)."""
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        result = runner.run_study(
            study_id="NT-GEO-TEST-001",
            title="NT-001 cVAE on GSE107965 autophagy subset",
            criterion="val_loss < 2.0",
            accessions=["GSE107965"],
            epochs=3,
            batch_size=4,
            latent_dim=16,
            min_samples=5,
        )
        assert result.verdict in ("met", "not_met")
        assert result.data_sufficiency.sufficient
        assert result.data_sufficiency.regime == "prior-contrast"
        assert result.training_result.get("n_train", 0) > 0
        assert result.training_result.get("final_val_loss") is not None
        assert result.lineage_hash != ""

        # verify the study was recorded
        study = runner.registry.get("NT-GEO-TEST-001")
        assert study is not None
        assert study.verdict != "pending"

        # verify lineage integrity
        assert runner.lineage.verify_chain()
        assert len(runner.lineage) == 1

    def test_governed_study_provenance_has_data_hashes(self):
        """Provenance must contain SHA-256 hashes of input data files."""
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        result = runner.run_study(
            study_id="NT-GEO-TEST-002",
            title="Provenance test",
            accessions=["GSE107965"],
            epochs=2,
            batch_size=4,
            latent_dim=16,
            min_samples=5,
        )
        assert "data_hashes" in result.provenance
        hashes = result.provenance["data_hashes"]
        assert "GSE107965" in hashes
        assert len(hashes["GSE107965"]) == 64  # SHA-256 hex

    def test_result_dict_serializable(self):
        """StudyResult.to_dict() must be JSON-serializable."""
        import json
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        result = runner.run_study(
            study_id="NT-GEO-TEST-003",
            title="Serialization test",
            accessions=["GSE107965"],
            epochs=2,
            batch_size=4,
            latent_dim=16,
            min_samples=5,
        )
        d = result.to_dict()
        serialized = json.dumps(d, default=str)
        assert len(serialized) > 100

    def test_multiple_studies_chain(self):
        """Multiple studies produce a valid lineage chain."""
        runner = StudyRunner(merlin_data_dir=MERLIN_DATA_DIR)
        r1 = runner.run_study(
            study_id="CHAIN-001", title="Chain 1",
            accessions=["GSE107965"], epochs=2, batch_size=4,
            latent_dim=16, min_samples=5,
        )
        r2 = runner.run_study(
            study_id="CHAIN-002", title="Chain 2",
            accessions=["GSE107965"], epochs=2, batch_size=4,
            latent_dim=16, min_samples=5, seed=42,
        )
        assert len(runner.lineage) == 2
        assert runner.lineage.verify_chain()
        assert r1.lineage_hash != r2.lineage_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
