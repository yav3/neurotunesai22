"""Tests for the evidence discipline bridge — lineage, registry, regime,
eval harness, governed solver, RTP bridge, and cross-repo manifest.
"""
import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_lineage_chain():
    from models.evidence.lineage import LineageChain, EvidenceRecord

    chain = LineageChain()
    assert len(chain) == 0
    assert chain.head_hash == ""

    r1 = chain.append("R001", {"type": "pk"}, {"conc": 5.0})
    assert r1.record_id == "R001"
    assert len(r1.record_hash) == 64
    assert r1.prev_hash == ""
    assert r1.verify()

    r2 = chain.append("R002", {"type": "pd"}, {"effect": 0.8})
    assert r2.prev_hash == r1.record_hash
    assert r2.verify()

    assert len(chain) == 2
    assert chain.verify_chain()
    assert chain.head_hash == r2.record_hash

    exported = chain.export()
    assert len(exported) == 2
    assert exported[0]["record_id"] == "R001"
    assert exported[1]["prev_hash"] == exported[0]["record_hash"]


def test_lineage_tamper_detection():
    from models.evidence.lineage import LineageChain

    chain = LineageChain()
    chain.append("R001", {"x": 1}, {"y": 2})
    chain.append("R002", {"x": 3}, {"y": 4})
    assert chain.verify_chain()

    chain._records[0].answer_values["y"] = 999
    chain._records[0].record_hash = chain._records[0].compute_hash()
    assert not chain.verify_chain()


def test_lineage_numpy_serialization():
    from models.evidence.lineage import LineageChain

    chain = LineageChain()
    arr = np.array([1.0, 2.0, 3.0])
    r = chain.append("R001", {"data": "test"}, {"values": arr})
    exported = r.to_dict()
    assert exported["answer_values"]["values"] == [1.0, 2.0, 3.0]


def test_study_registry():
    from models.evidence.registry import StudyRegistry

    reg = StudyRegistry()
    assert reg.list_studies() == []

    rec = reg.register("TEST-001", "Test study", "metric > 0.5")
    assert rec.study_id == "TEST-001"
    assert rec.verdict == "pending"
    assert "TEST-001" in reg.pending()

    reg.record_verdict("TEST-001", {"metric": 0.7}, "passed", "met")
    updated = reg.get("TEST-001")
    assert updated.verdict == "met"
    assert updated.measured["metric"] == 0.7
    assert reg.pending() == []


def test_registry_immutability():
    from models.evidence.registry import StudyRegistry

    reg = StudyRegistry()
    reg.register("IMM-001", "Immutability test", "x > 1")
    reg.record_verdict("IMM-001", {"x": 2}, "ok", "met")

    try:
        reg.record_verdict("IMM-001", {"x": 3}, "changed", "not met")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "already recorded" in str(e)


def test_registry_no_run_before_register():
    from models.evidence.registry import StudyRegistry

    reg = StudyRegistry()
    try:
        reg.record_verdict("UNREGISTERED", {}, "nope", "met")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "not registered" in str(e)


def test_registry_no_duplicate():
    from models.evidence.registry import StudyRegistry

    reg = StudyRegistry()
    reg.register("DUP-001", "First", "x > 0")
    try:
        reg.register("DUP-001", "Duplicate", "y > 0")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_regime_firewall():
    from models.evidence.regime import (
        RegimeFirewall, SpeciesDeclaration, Species, Regime,
    )

    ok, _ = RegimeFirewall.validate(
        SpeciesDeclaration(Species.SYNTHETIC, Regime.BENCHMARK)
    )
    assert ok

    ok, msg = RegimeFirewall.validate(
        SpeciesDeclaration(Species.HUMAN, Regime.EXPERIMENTAL)
    )
    assert not ok
    assert "justification" in msg

    ok, _ = RegimeFirewall.validate(
        SpeciesDeclaration(Species.HUMAN, Regime.EXPERIMENTAL, justification="IRB approved")
    )
    assert ok


def test_regime_compatibility():
    from models.evidence.regime import (
        RegimeFirewall, SpeciesDeclaration, Species, Regime,
    )

    human = SpeciesDeclaration(Species.HUMAN, Regime.THERAPEUTIC)
    mouse = SpeciesDeclaration(Species.MOUSE, Regime.THERAPEUTIC)

    ok, msg = RegimeFirewall.check_compatibility(human, mouse)
    assert not ok
    assert "Species mismatch" in msg

    human_cross = SpeciesDeclaration(Species.HUMAN, Regime.THERAPEUTIC, cross_species=True, justification="ok")
    mouse_cross = SpeciesDeclaration(Species.MOUSE, Regime.THERAPEUTIC, cross_species=True, justification="ok")
    ok, _ = RegimeFirewall.check_compatibility(human_cross, mouse_cross)
    assert ok


def test_calibration_report():
    from models.evidence.eval_harness import CalibrationEntry, calibration_report

    entries = [
        CalibrationEntry("Q1", 0.95, "correct"),
        CalibrationEntry("Q2", 0.85, "correct"),
        CalibrationEntry("Q3", 0.75, "incorrect"),
        CalibrationEntry("Q4", 0.55, "partial"),
        CalibrationEntry("Q5", 0.30, "incorrect"),
        CalibrationEntry("Q6", 0.90, "correct"),
        CalibrationEntry("Q7", 0.10, "unanswerable"),
    ]

    report = calibration_report(entries)
    assert report["n_total"] == 7
    assert report["n_scored"] == 6
    assert report["n_excluded"] == 1
    assert 0.0 <= report["expected_calibration_error"] <= 1.0
    assert 0.0 <= report["brier_score"] <= 1.0
    assert report["log_loss"] > 0
    assert len(report["bins"]) == 10


def test_calibration_empty():
    from models.evidence.eval_harness import calibration_report

    report = calibration_report([])
    assert report["n_scored"] == 0


def test_truth_eval():
    from models.evidence.eval_harness import TruthEntry, truth_eval

    entries = [
        TruthEntry(
            entry_id="T1",
            query={"q": "Does drug X treat Y?"},
            expected_claims=[{"predicate": "treats", "subject": "drug x", "object": "y"}],
            forbidden_substrings=["cures"],
        ),
        TruthEntry(
            entry_id="T2",
            query={"q": "Effect of Z?"},
            expected_claims=[{"predicate": "modulates", "subject": "z"}],
        ),
    ]

    produced = {
        "T1": {
            "claims": [{"predicate": "treats", "subject": "Drug X compound", "object": "condition Y"}],
            "confidence": 0.9,
            "text": "Drug X treats condition Y with high efficacy.",
        },
        "T2": {
            "claims": [],
            "confidence": 0.3,
            "text": "No data available for Z.",
        },
    }

    report = truth_eval(entries, produced)
    assert report["n"] == 2
    assert report["pass"] == 1
    assert report["fail"] == 1
    assert report["macro_recall"] > 0


def test_governed_solver():
    from models.evidence.governed_solver import GovernedSolver
    from models.evidence.regime import SpeciesDeclaration, Species, Regime
    from models.interface.problem import Problem, ProblemType

    solver = GovernedSolver()
    decl = SpeciesDeclaration(Species.SYNTHETIC, Regime.BENCHMARK)

    problem = Problem(
        type=ProblemType.PK_CONCENTRATION,
        params={"drug": "donepezil", "weight_kg": 70.0},
        query={"time_hr": 4.0},
    )

    result = solver.solve(problem, declaration=decl)
    assert result.answer is not None
    assert result.evidence.record_id == "RUN-000001"
    assert result.chain_position == 1
    assert len(result.chain_head_hash) == 64
    assert result.species_declaration.species == Species.SYNTHETIC

    result2 = solver.solve(problem, declaration=decl)
    assert result2.chain_position == 2
    assert result2.evidence.prev_hash == result.evidence.record_hash

    assert solver.verify_lineage()
    lineage = solver.export_lineage()
    assert len(lineage) == 2


def test_governed_solver_regime_rejection():
    from models.evidence.governed_solver import GovernedSolver
    from models.evidence.regime import SpeciesDeclaration, Species, Regime
    from models.interface.problem import Problem, ProblemType

    solver = GovernedSolver()
    bad_decl = SpeciesDeclaration(Species.HUMAN, Regime.EXPERIMENTAL)

    problem = Problem(type=ProblemType.PK_CONCENTRATION, params={}, query={})

    try:
        solver.solve(problem, declaration=bad_decl)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Regime firewall" in str(e)


def test_entity_series_contract():
    from models.evidence.rtp_bridge import EntitySeries, fnv1a64

    vals = np.random.randn(50, 5)
    series = EntitySeries(
        entity_ids=["A", "B", "C", "D", "E"],
        values=vals,
        domain="test",
    )
    assert series.n_entities == 5
    assert series.n_steps == 50
    h = series.content_hash()
    assert len(h) == 64
    assert series.content_hash() == h

    assert fnv1a64("test") == fnv1a64("test")
    assert fnv1a64("a") != fnv1a64("b")


def test_rtp_263_bridge():
    from models.evidence.rtp_bridge import EntitySeries, run_263_bridge

    np.random.seed(42)
    vals = np.random.randn(100, 8)
    series = EntitySeries(
        entity_ids=[f"entity_{i}" for i in range(8)],
        values=vals,
    )

    result, manifest = run_263_bridge(series)

    assert "d_results" in result
    assert "sim_matrix" in result
    assert "target" in result

    assert len(manifest.traces) == 6
    elements = [t.element for t in manifest.traces]
    assert elements == ["dominance", "evolution", "simulate", "eliminate", "test", "identify"]
    assert "US12361263B1/1" in manifest.claim_coverage
    assert manifest.input_hash == series.content_hash()


def test_cross_repo_manifest():
    from models.evidence.cross_repo_manifest import (
        PATENT_ESTATE, MERLIN_STUDY_SERIES, MERLIN_ENGINE_CATALOG,
        MERLIN_HARVEST_SOURCES, MATH_KERNEL_COVERAGE, IntegrationStatus,
    )

    assert len(PATENT_ESTATE["granted"]) == 3
    assert len(PATENT_ESTATE["pending_international"]) == 2

    geo_total = len(MERLIN_STUDY_SERIES["geo_replications"]) + len(MERLIN_STUDY_SERIES["geo_controls"])
    assert geo_total == 17

    assert len(MERLIN_ENGINE_CATALOG) == 12
    assert len(MERLIN_HARVEST_SOURCES) == 14

    assert "patent_263_claim_1" in MATH_KERNEL_COVERAGE
    assert len(MATH_KERNEL_COVERAGE["patent_263_claim_1"]["elements"]) == 6

    status = IntegrationStatus()
    assert "52 problem types" in status.summary
    assert "3 granted" in status.summary


if __name__ == "__main__":
    tests = [
        test_lineage_chain,
        test_lineage_tamper_detection,
        test_lineage_numpy_serialization,
        test_study_registry,
        test_registry_immutability,
        test_registry_no_run_before_register,
        test_registry_no_duplicate,
        test_regime_firewall,
        test_regime_compatibility,
        test_calibration_report,
        test_calibration_empty,
        test_truth_eval,
        test_governed_solver,
        test_governed_solver_regime_rejection,
        test_entity_series_contract,
        test_rtp_263_bridge,
        test_cross_repo_manifest,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(1 if failed else 0)
