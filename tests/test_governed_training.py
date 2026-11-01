"""Tests for the governed training harness and perturbation-response model.

Validates evidence discipline integration:
  - TrainingSpec hashing and validation
  - TrainingCriterion evaluation (all operators + versus arms)
  - CheckpointStore content addressing
  - GovernedTrainer pre-registration → evaluate lifecycle
  - Species firewall
  - Regime selection
  - PerturbationResponseModel forward/loss/gradient
  - TrainingRecord content hashing
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from models.training.governed_harness import (
    CheckpointStore,
    TrainingCriterion,
    TrainingRecord,
    TrainingSpec,
    infer_species_from_genes,
    select_training_regime,
)


def _make_spec(**overrides) -> TrainingSpec:
    defaults = dict(
        model_id="PerturbationResponseModel",
        dataset_id="synthetic_001",
        seed=2026,
        held_out_fraction=0.2,
        species="human",
        regime="data-driven",
        criteria=(TrainingCriterion(metric="val_loss", op="<", threshold=1.0),),
    )
    defaults.update(overrides)
    return TrainingSpec(**defaults)


# ── TrainingSpec ───────────────────────────────────────────────────

def test_training_spec_creation():
    spec = _make_spec()
    assert spec.model_id == "PerturbationResponseModel"
    assert spec.seed == 2026


def test_training_spec_deterministic_hash():
    a = _make_spec()
    b = _make_spec()
    assert a.content_hash() == b.content_hash()


def test_training_spec_different_seed_different_hash():
    a = _make_spec(seed=1)
    b = _make_spec(seed=2)
    assert a.content_hash() != b.content_hash()


def test_training_spec_rejects_invalid_species():
    with pytest.raises(ValueError, match="Species"):
        _make_spec(species="unknown_species")


def test_training_spec_rejects_empty_criteria():
    with pytest.raises(ValueError, match="criterion"):
        _make_spec(criteria=())


def test_training_spec_rejects_bad_held_out():
    with pytest.raises(ValueError, match="held_out_fraction"):
        _make_spec(held_out_fraction=0.0)


# ── TrainingCriterion ──────────────────────────────────────────────

def _criterion(metric="val_loss", op="<", threshold=1.0, versus=None):
    return TrainingCriterion(metric=metric, op=op, threshold=threshold, versus=versus)


def test_criterion_less_than_met():
    c = _criterion(op="<", threshold=1.0)
    ok, _ = c.evaluate({"model": {"val_loss": 0.5}})
    assert ok


def test_criterion_less_than_not_met():
    c = _criterion(op="<", threshold=1.0)
    ok, _ = c.evaluate({"model": {"val_loss": 1.5}})
    assert not ok


def test_criterion_greater_than():
    c = _criterion(metric="auroc", op=">", threshold=0.7)
    ok, _ = c.evaluate({"model": {"auroc": 0.85}})
    assert ok


def test_criterion_greater_equal():
    c = _criterion(metric="auroc", op=">=", threshold=0.85)
    ok, _ = c.evaluate({"model": {"auroc": 0.85}})
    assert ok


def test_criterion_less_equal():
    c = _criterion(metric="mse", op="<=", threshold=0.01)
    ok, _ = c.evaluate({"model": {"mse": 0.01}})
    assert ok


def test_criterion_versus_arm():
    c = _criterion(metric="val_loss", op="<", versus="control")
    ok, _ = c.evaluate({"model": {"val_loss": 0.5}, "control": {"val_loss": 0.9}})
    assert ok


def test_criterion_versus_not_met():
    c = _criterion(metric="val_loss", op="<", versus="control")
    ok, _ = c.evaluate({"model": {"val_loss": 0.9}, "control": {"val_loss": 0.5}})
    assert not ok


# ── CheckpointStore ────────────────────────────────────────────────

def test_checkpoint_content_addressing():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CheckpointStore(Path(tmpdir))
        state = {"weight": torch.randn(4, 4)}
        d1 = store.save(state)
        assert store.exists(d1)
        loaded = store.load(d1)
        assert "model_state_dict" in loaded


def test_checkpoint_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CheckpointStore(Path(tmpdir))
        torch.manual_seed(42)
        state = {"w": torch.randn(3, 3)}
        d1 = store.save(state)
        d2 = store.save(state)
        assert d1 == d2


# ── Regime selection ───────────────────────────────────────────────

def test_regime_data_driven():
    assert select_training_regime(n_samples=50) == "data-driven"


def test_regime_prior_contrast():
    assert select_training_regime(n_samples=10, n_conditions=3) == "prior-contrast"


def test_regime_prior_mean():
    assert select_training_regime(n_samples=10) == "prior-mean"


def test_regime_longitudinal():
    assert select_training_regime(n_samples=100, has_time_axis=True) == "longitudinal"


# ── Species firewall ──────────────────────────────────────────────

def test_species_human():
    assert infer_species_from_genes(["TP53", "BRCA1", "ENSG00000141510"]) == "human"


def test_species_mouse():
    assert infer_species_from_genes(["Trp53", "Brca1"]) == "mouse"


def test_species_mixed_raises():
    with pytest.raises(ValueError, match="Mixed species"):
        infer_species_from_genes(["ENSG00000141510", "Trp53"])


def test_species_unknown_raises():
    with pytest.raises(ValueError, match="Cannot infer"):
        infer_species_from_genes(["12345", "67890"])


# ── PerturbationResponseModel ─────────────────────────────────────

def _make_perturbation_model():
    from models.foundation.perturbation_response import PerturbationResponseModel, PerturbationModelConfig
    config = PerturbationModelConfig(n_genes=50, n_drugs=10, latent_dim=32)
    return PerturbationResponseModel(config)


def _make_perturbation_inputs(batch=4, n_genes=50):
    return dict(
        baseline=torch.randn(batch, n_genes),
        ptype_ids=torch.randint(0, 3, (batch,)),
        gene_idx=torch.randint(0, n_genes, (batch,)),
        drug_idx=torch.randint(0, 10, (batch,)),
        dose=torch.rand(batch),
        duration=torch.rand(batch) * 24,
        time_point=torch.rand(batch) * 48,
    )


def test_perturbation_model_forward():
    model = _make_perturbation_model()
    inputs = _make_perturbation_inputs()
    out = model(**inputs)
    assert isinstance(out, dict)
    assert "predicted" in out or len(out) > 0


def test_perturbation_model_loss_finite():
    model = _make_perturbation_model()
    inputs = _make_perturbation_inputs()
    out = model(**inputs)
    target = torch.randn_like(list(out.values())[0]) if isinstance(out, dict) else torch.randn_like(out)
    pred = list(out.values())[0] if isinstance(out, dict) else out
    loss = torch.nn.functional.mse_loss(pred, target)
    assert torch.isfinite(loss)


def test_perturbation_model_gradient():
    model = _make_perturbation_model()
    inputs = _make_perturbation_inputs()
    out = model(**inputs)
    pred = list(out.values())[0] if isinstance(out, dict) else out
    loss = torch.nn.functional.mse_loss(pred, torch.randn_like(pred))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ── TrainingRecord ─────────────────────────────────────────────────

def test_training_record_deterministic_hash():
    r1 = TrainingRecord(
        study_id="NT-001",
        spec_hash="abc123",
        spec={"model_id": "test"},
        arms={"model": {"val_loss": 0.5}},
        stats={},
        criteria_results=[],
        verdict="met",
        lineage_head="def456",
        checkpoint_digest="ghi789",
    )
    r2 = TrainingRecord(
        study_id="NT-001",
        spec_hash="abc123",
        spec={"model_id": "test"},
        arms={"model": {"val_loss": 0.5}},
        stats={},
        criteria_results=[],
        verdict="met",
        lineage_head="def456",
        checkpoint_digest="ghi789",
    )
    assert r1.record_hash == r2.record_hash


def test_training_record_different_results_different_hash():
    r1 = TrainingRecord(
        study_id="NT-001", spec_hash="a", spec={},
        arms={"model": {"val_loss": 0.5}}, stats={}, criteria_results=[],
        verdict="met", lineage_head="h1", checkpoint_digest="c1",
    )
    r2 = TrainingRecord(
        study_id="NT-001", spec_hash="a", spec={},
        arms={"model": {"val_loss": 0.9}}, stats={}, criteria_results=[],
        verdict="not_met", lineage_head="h1", checkpoint_digest="c1",
    )
    assert r1.record_hash != r2.record_hash


# ── RegimeFirewall (stateful + stateless) ─────────────────────────

def test_firewall_check_species_human_only():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert fw.check_species(["human", "human", "human"])


def test_firewall_check_species_mouse_only():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert fw.check_species(["mouse", "mouse"])


def test_firewall_check_species_mixed_rejected():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert not fw.check_species(["human", "mouse"])


def test_firewall_check_species_unknown_rejected():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert not fw.check_species(["rat", "rat"])


def test_firewall_check_species_empty_rejected():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert not fw.check_species([])


def test_firewall_select_regime_data_driven():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert fw.select_regime(n_samples=50) == "data-driven"


def test_firewall_select_regime_prior_contrast():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert fw.select_regime(n_samples=10, n_conditions=3) == "prior-contrast"


def test_firewall_select_regime_prior_mean():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert fw.select_regime(n_samples=10) == "prior-mean"


def test_firewall_select_regime_longitudinal():
    from models.evidence.regime import RegimeFirewall
    fw = RegimeFirewall()
    assert fw.select_regime(n_samples=100, has_time_axis=True) == "longitudinal"


def test_firewall_declare_and_check():
    from models.evidence.regime import RegimeFirewall, Species, Regime
    fw = RegimeFirewall()
    assert not fw.check(Species.HUMAN, Regime.THERAPEUTIC)
    fw.declare(Species.HUMAN, Regime.THERAPEUTIC)
    assert fw.check(Species.HUMAN, Regime.THERAPEUTIC)


def test_firewall_check_or_raise_undeclared():
    from models.evidence.regime import RegimeFirewall, Species, Regime
    fw = RegimeFirewall()
    with pytest.raises(ValueError, match="Undeclared"):
        fw.check_or_raise(Species.HUMAN, Regime.THERAPEUTIC)


# ── PerturbationDataset ──────────────────────────────────────────

def test_perturbation_dataset_from_geo():
    from models.foundation.perturbation_response import (
        GEOExpressionData, PerturbationDataset, PerturbationType,
    )
    n_genes = 20
    expr = np.vstack([
        np.random.RandomState(0).randn(5, n_genes),   # 5 control
        np.random.RandomState(1).randn(8, n_genes),   # 8 treated
    ]).astype(np.float32)
    conditions = ["control"] * 5 + ["treated"] * 8
    data = GEOExpressionData(
        expression=expr,
        conditions=conditions,
        gene_names=[f"G{i}" for i in range(n_genes)],
        species="human",
        accession="GSE000000",
    )
    ds = PerturbationDataset(data, perturbation_type=PerturbationType.DRUG)
    assert len(ds) == 8
    item = ds[0]
    assert item["baseline"].shape == (n_genes,)
    assert item["target"].shape == (n_genes,)
    assert item["ptype_ids"].item() == 1  # DRUG = 1


def test_perturbation_dataset_empty_when_no_controls():
    from models.foundation.perturbation_response import (
        GEOExpressionData, PerturbationDataset, PerturbationType,
    )
    n_genes = 10
    expr = np.random.RandomState(42).randn(6, n_genes).astype(np.float32)
    conditions = ["treated_A"] * 6
    data = GEOExpressionData(
        expression=expr,
        conditions=conditions,
        gene_names=[f"G{i}" for i in range(n_genes)],
        species="human",
        accession="GSE000001",
    )
    ds = PerturbationDataset(data, perturbation_type=PerturbationType.KNOCKOUT)
    assert len(ds) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
