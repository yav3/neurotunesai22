"""Tests for the frontier foundational AI bridge.

Validates the governed evidence layer, cross-repo GEO bridge, run manifests,
species firewall, validation discipline, and governed training harness.
These are the connective tissue between the math kernel (NeuroTunesAI) and
the evidence system (merlinaineuralplus2).
"""
import sys
import os
import hashlib
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Evidence discipline: lineage chain ---

def test_lineage_chain_integrity():
    from models.evidence.lineage import LineageChain

    chain = LineageChain()

    r1 = chain.append("rec-001", {"type": "pk", "drug": "donepezil"}, {"concentration": 42.0})
    r2 = chain.append("rec-002", {"type": "pk", "drug": "donepezil"}, {"concentration": 43.0})

    assert len(chain) == 2
    assert chain.verify_chain(), "Chain should verify after honest appends"
    assert r1.record_hash != r2.record_hash, "Different records -> different hashes"
    assert r2.prev_hash == r1.record_hash, "Second record chains from first"


def test_lineage_chain_tamper_detection():
    from models.evidence.lineage import LineageChain, EvidenceRecord

    chain = LineageChain()
    chain.append("rec-001", {"type": "test"}, {"y": 1})
    chain.append("rec-002", {"type": "test"}, {"y": 2})

    assert chain.verify_chain()

    tampered = chain._records[0]
    chain._records[0] = EvidenceRecord(
        record_id=tampered.record_id,
        problem_spec=tampered.problem_spec,
        answer_values={"y": 999},
        timestamp=tampered.timestamp,
        prev_hash=tampered.prev_hash,
        record_hash=tampered.record_hash,
    )
    assert not chain.verify_chain(), "Tampered chain should fail verification"


# --- Evidence discipline: species firewall ---

def test_species_firewall():
    from models.evidence.regime import RegimeFirewall, Species, Regime

    fw = RegimeFirewall()
    fw.declare(Species.HUMAN, Regime.THERAPEUTIC, source="GSE107965")

    assert fw.check(Species.HUMAN, Regime.THERAPEUTIC)
    assert not fw.check(Species.MOUSE, Regime.THERAPEUTIC)

    try:
        fw.check_or_raise(Species.MOUSE, Regime.THERAPEUTIC)
        assert False, "Should have raised"
    except ValueError as e:
        assert "Undeclared" in str(e)


def test_species_bridge():
    from models.evidence.regime import RegimeFirewall, Species, Regime

    fw = RegimeFirewall()
    fw.declare(Species.HUMAN, Regime.VALIDATION, source="test")
    fw.declare_bridge(Species.MOUSE, Species.HUMAN, justification="Conserved pathway")

    assert fw.can_bridge(Species.MOUSE, Species.HUMAN)
    assert not fw.can_bridge(Species.HUMAN, Species.MOUSE)


# --- Evidence discipline: study registry ---

def test_study_registry():
    from models.evidence.registry import StudyRegistry

    reg = StudyRegistry()
    study = reg.register(
        study_id="NF-001",
        title="Narrow foundation model test",
        criterion="val_mse < 0.5",
    )

    assert study.study_id == "NF-001"
    assert study.verdict == "pending"
    assert len(reg.pending()) == 1

    updated = reg.record_verdict(
        study_id="NF-001",
        measured={"val_mse": 0.3},
        summary="Met criterion",
        verdict="met",
    )
    assert updated.verdict == "met"
    assert len(reg.pending()) == 0


# --- Run manifest ---

def test_run_manifest_dual_identity():
    from models.evidence.run_manifest import build_manifest, fingerprint

    m = build_manifest(
        analysis="test_analysis",
        request_payload={"input": [1, 2, 3]},
        parameters={"lr": 0.001},
        seed=42,
        input_fingerprint=fingerprint({"data": "test"}),
        preprocess_steps=[{"step": "zscore"}],
        output_payload={"result": 0.95},
        species="human",
        regime="validation",
    )

    assert m.manifest_id.startswith("run_")
    assert m.input_identity.startswith("in_")
    assert m.manifest_id != m.input_identity, "Full manifest and input identity must differ"
    assert len(m.manifest_id) > 40
    assert m.species == "human"
    assert m.regime == "validation"


def test_run_manifest_determinism():
    from models.evidence.run_manifest import build_manifest, fingerprint

    kwargs = dict(
        analysis="determinism_test",
        request_payload={"x": 1},
        parameters={"seed": 0},
        seed=0,
        input_fingerprint=fingerprint({"data": "fixed"}),
        preprocess_steps=[],
        output_payload={"y": 2},
    )

    m1 = build_manifest(**kwargs)
    m2 = build_manifest(**kwargs)

    assert m1.input_identity == m2.input_identity, "Same inputs -> same input_identity"


# --- Validation discipline ---

def test_degeneracy_detection():
    from models.evidence.validation_discipline import is_degenerate

    zeros = np.zeros(10)
    degen, reason = is_degenerate(zeros)
    assert degen
    assert "zero" in reason

    constant = np.full(10, 3.14)
    degen, reason = is_degenerate(constant)
    assert degen
    assert "discriminative" in reason

    good = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    degen, reason = is_degenerate(good)
    assert not degen
    assert reason == "ok"

    nan_arr = np.array([1.0, float("nan"), 3.0])
    degen, reason = is_degenerate(nan_arr)
    assert degen
    assert "non-finite" in reason


def test_signal_survives_transform():
    from models.evidence.validation_discipline import assert_signal_survives_transform

    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 10))
    data = np.abs(data) + 0.1

    def scoring_fn(x):
        return np.max(x, axis=0)

    checks = assert_signal_survives_transform(scoring_fn, data)
    assert len(checks) >= 4

    n_degenerate = sum(1 for c in checks if c.is_degenerate)
    assert n_degenerate == 0, (
        f"Healthy scoring should survive all transforms, got {n_degenerate} degenerate: "
        f"{[(c.transform_name, c.reason) for c in checks if c.is_degenerate]}"
    )


def test_signal_collapse_under_zscore():
    """The bug that validation_discipline was born to catch:
    a scoring function that returns row-means collapses under z-score."""
    from models.evidence.validation_discipline import assert_signal_survives_transform

    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 10))

    def row_mean_scorer(x):
        return x.mean(axis=1)

    checks = assert_signal_survives_transform(row_mean_scorer, data)
    zscore_check = next((c for c in checks if c.transform_name == "zscore_row"), None)
    assert zscore_check is not None
    assert zscore_check.is_degenerate, "Row means should collapse to zero under row z-score"


# --- GEO bridge ---

def test_geo_bridge_catalog():
    from models.data.geo_bridge import GEOBridge

    bridge = GEOBridge()
    assert len(bridge.list_all()) >= 6

    gse107965 = bridge.get("GSE107965")
    assert gse107965 is not None
    assert gse107965.species == "human"
    assert gse107965.n_samples == 18
    assert gse107965.validation_verdict == "met"

    gse252062 = bridge.get("GSE252062")
    assert gse252062 is not None
    assert gse252062.species == "mouse"


def test_geo_bridge_species_filter():
    from models.data.geo_bridge import GEOBridge

    bridge = GEOBridge()
    human = bridge.filter_by_species("human")
    mouse = bridge.filter_by_species("mouse")

    assert len(human) >= 5
    assert len(mouse) >= 1
    assert all(r.species == "human" for r in human)
    assert all(r.species == "mouse" for r in mouse)


def test_geo_bridge_validated():
    from models.data.geo_bridge import GEOBridge

    bridge = GEOBridge()
    validated = bridge.filter_validated()
    assert len(validated) >= 3

    met = bridge.filter_validated("met")
    assert len(met) >= 1
    assert all(r.validation_verdict == "met" for r in met)


def test_geo_bridge_in_regime():
    from models.data.geo_bridge import GEOBridge

    bridge = GEOBridge()
    in_regime = bridge.filter_in_regime(min_samples=30)

    assert len(in_regime) >= 1
    for r in in_regime:
        assert r.n_samples >= 30
        assert not r.has_time_axis


def test_geo_bridge_summary():
    from models.data.geo_bridge import GEOBridge

    bridge = GEOBridge()
    s = bridge.summary()
    assert s["total"] >= 6
    assert s["human"] >= 5
    assert s["mouse"] >= 1
    assert s["validated"] >= 3


# --- Governed trainer preconditions ---

def test_governed_trainer_species_check():
    """Governed trainer refuses undeclared species."""
    import torch
    from models.evidence.regime import RegimeFirewall, Species, Regime
    from models.evidence.registry import StudyRegistry
    from models.training.governed_trainer import GovernedTrainer, TrainingSpec

    model = torch.nn.Linear(10, 10)
    dummy = torch.utils.data.TensorDataset(torch.randn(50, 10), torch.randn(50, 10))
    loader = torch.utils.data.DataLoader(dummy, batch_size=10)

    spec = TrainingSpec(
        study_id="test",
        model_name="test_model",
        dataset_ids=["GSE107965"],
        species="mouse",
        regime="validation",
        criterion="test",
        min_samples=10,
        max_epochs=1,
        early_stopping_patience=1,
        seed=42,
        eval_metric="mse",
        eval_threshold=0.5,
    )

    registry = StudyRegistry()
    firewall = RegimeFirewall()
    firewall.declare(Species.HUMAN, Regime.VALIDATION, source="test")

    trainer = GovernedTrainer(model, loader, loader, spec, registry, firewall)

    try:
        trainer.run()
        assert False, "Should refuse undeclared species"
    except ValueError as e:
        assert "Undeclared" in str(e)


def test_governed_trainer_min_samples():
    """Governed trainer refuses insufficient data."""
    import torch
    from models.evidence.regime import RegimeFirewall, Species, Regime
    from models.evidence.registry import StudyRegistry
    from models.training.governed_trainer import GovernedTrainer, TrainingSpec

    model = torch.nn.Linear(10, 10)
    dummy = torch.utils.data.TensorDataset(torch.randn(5, 10), torch.randn(5, 10))
    loader = torch.utils.data.DataLoader(dummy, batch_size=5)

    spec = TrainingSpec(
        study_id="test",
        model_name="test_model",
        dataset_ids=["test"],
        species="human",
        regime="validation",
        criterion="test",
        min_samples=30,
        max_epochs=1,
        early_stopping_patience=1,
        seed=42,
        eval_metric="mse",
        eval_threshold=0.5,
    )

    registry = StudyRegistry()
    firewall = RegimeFirewall()
    firewall.declare(Species.HUMAN, Regime.VALIDATION, source="test")

    trainer = GovernedTrainer(model, loader, loader, spec, registry, firewall)

    try:
        trainer.run()
        assert False, "Should refuse insufficient samples"
    except ValueError as e:
        assert "below declared minimum" in str(e)


def test_governed_trainer_runs():
    """Governed trainer completes a tiny training run with manifest."""
    import torch
    from models.evidence.regime import RegimeFirewall, Species, Regime
    from models.evidence.registry import StudyRegistry
    from models.training.governed_trainer import GovernedTrainer, TrainingSpec

    model = torch.nn.Linear(10, 10)
    x = torch.randn(50, 10)
    dummy = torch.utils.data.TensorDataset(x, x)
    train_loader = torch.utils.data.DataLoader(dummy, batch_size=10)
    val_loader = torch.utils.data.DataLoader(dummy, batch_size=10)

    spec = TrainingSpec(
        study_id="NF-test",
        model_name="identity_test",
        dataset_ids=["synthetic"],
        species="human",
        regime="validation",
        criterion="val_mse < 10.0",
        min_samples=10,
        max_epochs=3,
        early_stopping_patience=5,
        seed=42,
        eval_metric="mse",
        eval_threshold=10.0,
    )

    registry = StudyRegistry()
    firewall = RegimeFirewall()
    firewall.declare(Species.HUMAN, Regime.VALIDATION, source="synthetic")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = GovernedTrainer(
            model, train_loader, val_loader, spec, registry, firewall,
            checkpoint_dir=tmpdir, device="cpu",
        )
        result = trainer.run()

    assert result.manifest_id.startswith("run_")
    assert result.input_identity.startswith("in_")
    assert result.epochs_completed >= 1
    assert result.checkpoint_hash
    assert result.wall_time_seconds > 0

    studies = registry.list_studies()
    assert len(studies) >= 1


# --- Cross-repo integration: math kernel + evidence ---

def test_math_kernel_with_evidence():
    """Full integration: solve a problem through the math kernel, deposit evidence."""
    from models.evidence.lineage import LineageChain
    from models.evidence.run_manifest import build_manifest, fingerprint
    from models.math_core.pk_solver import PKParams, PKSolver

    pk = PKSolver()
    params = PKParams(
        ka_per_hr=1.0, ke_per_hr=0.1, vd_l=50.0,
        bioavailability=0.8, dose_mg=10.0, route="oral",
    )
    t = np.array([4.0])
    conc = float(pk.single_dose_oral(params, t)[0])

    chain = LineageChain()
    record = chain.append(
        record_id="PK-001",
        problem_spec={"type": "pk_concentration", "drug": "donepezil", "t_hr": 4.0},
        answer_values={"concentration": conc},
    )

    manifest = build_manifest(
        analysis="pk_single_dose",
        request_payload={"drug": "donepezil", "t_hr": 4.0},
        parameters={"ka": 1.0, "ke": 0.1, "vd": 50.0},
        seed=None,
        input_fingerprint=fingerprint({"dose_mg": 10.0, "route": "oral"}),
        preprocess_steps=[],
        output_payload={"concentration": conc},
        species="human",
        regime="therapeutic",
        study_id="PK-001",
    )

    assert chain.verify_chain()
    assert record.record_hash != ""
    assert manifest.manifest_id.startswith("run_")
    assert manifest.species == "human"
    assert isinstance(conc, float)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
