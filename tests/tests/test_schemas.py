"""Tests for the shared evidence protocol."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from evidence_protocol.schemas import (
    DatasetManifest,
    EvidenceRecord,
    PerturbationSpec,
    PredictionEnvelope,
    StudyDeclaration,
    Verdict,
)
from evidence_protocol.firewall import classify_regime, validate_species
from evidence_protocol.hashing import chain_hash, content_hash, prediction_hash


class TestHashing:

    def test_content_hash_deterministic(self):
        h1 = content_hash(b"hello")
        h2 = content_hash(b"hello")
        assert h1 == h2
        assert len(h1) == 64

    def test_content_hash_differs_for_different_data(self):
        assert content_hash(b"a") != content_hash(b"b")

    def test_chain_hash_deterministic(self):
        h1 = chain_hash("prev", "rec-1", {"x": 1}, {"y": 2}, 1000.0)
        h2 = chain_hash("prev", "rec-1", {"x": 1}, {"y": 2}, 1000.0)
        assert h1 == h2

    def test_chain_hash_changes_with_prev(self):
        h1 = chain_hash("aaa", "rec-1", {}, {}, 0.0)
        h2 = chain_hash("bbb", "rec-1", {}, {}, 0.0)
        assert h1 != h2

    def test_prediction_hash_truncates_to_16(self):
        h = prediction_hash("a" * 64, "b" * 64, "drug", "spec123")
        assert len(h) == 64


class TestFirewall:

    def test_validate_species_human_ok(self):
        validate_species("human")

    def test_validate_species_mouse_rejected(self):
        with pytest.raises(ValueError, match="mouse"):
            validate_species("mouse")

    def test_classify_regime_data_driven(self):
        assert classify_regime(44, 23, 21) == "data_driven"

    def test_classify_regime_prior_contrast(self):
        assert classify_regime(18, 6, 12) == "prior_contrast"

    def test_classify_regime_prior_mean_no_control(self):
        assert classify_regime(9, 0, 0) == "prior_mean"

    def test_classify_regime_prior_mean_few_samples_no_conditions(self):
        assert classify_regime(28, 0, 0) == "prior_mean"

    def test_classify_regime_boundary_30(self):
        assert classify_regime(30, 15, 15) == "data_driven"
        assert classify_regime(29, 15, 14) == "prior_contrast"


class TestEvidenceRecord:

    def test_round_trip(self):
        rec = EvidenceRecord(
            record_id="rec-001",
            prev_hash="0" * 64,
            problem_spec={"type": "pk"},
            answer_values={"c_max": 1.5},
            timestamp=1000.0,
        )
        d = rec.to_dict()
        rec2 = EvidenceRecord.from_dict(d)
        assert rec2.record_id == rec.record_id
        assert rec2.hash == rec.hash
        assert rec2.problem_spec == rec.problem_spec

    def test_hash_auto_computed(self):
        rec = EvidenceRecord(
            record_id="rec-002",
            prev_hash="abc",
            problem_spec={},
            answer_values={},
            timestamp=0.0,
        )
        assert len(rec.hash) == 64

    def test_content_hash_deterministic(self):
        rec = EvidenceRecord(
            record_id="rec-003",
            prev_hash="x",
            problem_spec={"a": 1},
            answer_values={"b": 2},
            timestamp=99.0,
        )
        assert rec.content_hash() == rec.content_hash()


class TestStudyDeclaration:

    def test_round_trip(self):
        decl = StudyDeclaration(
            study_id="ST-001",
            title="Autophagy response",
            criterion="val_loss < 1.0",
            data_kind="geo_expression",
            species="human",
            regime="prior_contrast",
            declared_at=1000.0,
        )
        d = decl.to_dict()
        decl2 = StudyDeclaration.from_dict(d)
        assert decl2.study_id == "ST-001"
        assert decl2.regime == "prior_contrast"


class TestVerdict:

    def test_round_trip(self):
        v = Verdict(study_id="ST-001", verdict="met", evidence_hash="abc123")
        d = v.to_dict()
        v2 = Verdict.from_dict(d)
        assert v2.verdict == "met"
        assert v2.evidence_hash == "abc123"


class TestPerturbationSpec:

    def test_round_trip(self):
        spec = PerturbationSpec(ptype="drug", drug_idx=3, dose=1.5, duration=24.0)
        d = spec.to_dict()
        spec2 = PerturbationSpec.from_dict(d)
        assert spec2.ptype == "drug"
        assert spec2.dose == 1.5


class TestPredictionEnvelope:

    def test_round_trip(self):
        env = PredictionEnvelope(
            prediction_id="pred-000001",
            spec_hash="abc",
            regime="data_driven",
            species="human",
            perturbation=PerturbationSpec(ptype="knockout", gene_idx=5),
            n_genes=20,
            latent_variance=0.05,
            data_hash="def456",
        )
        d = env.to_dict()
        env2 = PredictionEnvelope.from_dict(d)
        assert env2.prediction_id == "pred-000001"
        assert env2.perturbation.ptype == "knockout"
        assert env2.perturbation.gene_idx == 5
        assert env2.n_genes == 20

    def test_content_hash_deterministic(self):
        env = PredictionEnvelope(
            prediction_id="pred-000002",
            spec_hash="xyz",
            regime="prior_contrast",
            species="human",
            perturbation=PerturbationSpec(ptype="drug"),
            n_genes=29,
        )
        assert env.content_hash() == env.content_hash()


class TestDatasetManifest:

    def test_round_trip(self):
        m = DatasetManifest(
            accession="GSE119262",
            n_samples=44,
            n_genes=17183,
            n_control=23,
            n_treated=21,
            species="human",
            regime="data_driven",
            content_hash_hex="c8ee578b" + "0" * 56,
            path="/data/expression/human/gse119262.csv",
        )
        d = m.to_dict()
        m2 = DatasetManifest.from_dict(d)
        assert m2.accession == "GSE119262"
        assert m2.n_samples == 44

    def test_from_dict_content_hash_alias(self):
        d = {
            "accession": "GSE107965",
            "n_samples": 18,
            "n_genes": 29,
            "n_control": 6,
            "n_treated": 12,
            "species": "human",
            "regime": "prior_contrast",
            "content_hash": "807db67d" + "0" * 56,
            "path": "/data/human/gse107965.csv",
        }
        m = DatasetManifest.from_dict(d)
        assert m.content_hash_hex == "807db67d" + "0" * 56


class TestChainIntegrity:

    def test_three_record_chain(self):
        r1 = EvidenceRecord(
            record_id="r1", prev_hash="genesis", problem_spec={"a": 1},
            answer_values={"b": 2}, timestamp=1.0,
        )
        r2 = EvidenceRecord(
            record_id="r2", prev_hash=r1.hash, problem_spec={"c": 3},
            answer_values={"d": 4}, timestamp=2.0,
        )
        r3 = EvidenceRecord(
            record_id="r3", prev_hash=r2.hash, problem_spec={"e": 5},
            answer_values={"f": 6}, timestamp=3.0,
        )
        expected_r2_hash = chain_hash(r1.hash, "r2", {"c": 3}, {"d": 4}, 2.0)
        assert r2.hash == expected_r2_hash
        expected_r3_hash = chain_hash(r2.hash, "r3", {"e": 5}, {"f": 6}, 3.0)
        assert r3.hash == expected_r3_hash

    def test_tampered_chain_detected(self):
        r1 = EvidenceRecord(
            record_id="r1", prev_hash="genesis", problem_spec={},
            answer_values={}, timestamp=0.0,
        )
        r2 = EvidenceRecord(
            record_id="r2", prev_hash=r1.hash, problem_spec={},
            answer_values={}, timestamp=1.0,
        )
        tampered_hash = chain_hash("WRONG", "r2", {}, {}, 1.0)
        assert tampered_hash != r2.hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
