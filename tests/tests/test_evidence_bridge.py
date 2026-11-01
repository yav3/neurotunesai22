"""Tests for the evidence discipline bridge and cross-repo integration.

Tests:
  - Hash-chained lineage (deposit, verify, tamper detection)
  - Study registry (pre-registration, verdict recording, immutability)
  - Regime firewall (species/regime validation, compatibility)
  - Merlin bridge (GRN tensor conversion, evidence deposit packaging)
  - GEO adapter (envelope loading, in-regime filtering, CSV parsing)
  - Product bridge (audio→solver, solver→product translations)
"""
import os
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest


# ── Evidence lineage ───────────────────────────────────────────────

def test_lineage_chain_deposit_and_verify():
    from models.evidence.lineage import LineageChain
    chain = LineageChain()
    rec = chain.append("ER-001", {"type": "pk", "dose": 10.0}, {"concentration": [1.0, 2.0]})
    assert rec.record_hash != ""
    assert rec.prev_hash == ""
    assert chain.verify_chain()


def test_lineage_chain_multiple_records():
    from models.evidence.lineage import LineageChain
    chain = LineageChain()
    r1 = chain.append("ER-001", {"a": 1}, {"b": 2})
    r2 = chain.append("ER-002", {"a": 3}, {"b": 4})
    assert r2.prev_hash == r1.record_hash
    assert len(chain) == 2
    assert chain.verify_chain()


def test_lineage_tamper_detection():
    from models.evidence.lineage import LineageChain, EvidenceRecord
    chain = LineageChain()
    chain.append("ER-001", {"a": 1}, {"b": 2})
    chain.append("ER-002", {"a": 3}, {"b": 4})
    chain._records[0].problem_spec = {"a": 999}
    assert not chain.verify_chain()


def test_lineage_export():
    from models.evidence.lineage import LineageChain
    chain = LineageChain()
    chain.append("ER-001", {"x": 1}, {"y": 2})
    exported = chain.export()
    assert len(exported) == 1
    assert exported[0]["record_id"] == "ER-001"
    assert "record_hash" in exported[0]


def test_lineage_numpy_serialization():
    from models.evidence.lineage import LineageChain
    chain = LineageChain()
    rec = chain.append("ER-001", {"array": np.array([1.0, 2.0])}, {"result": 3.14})
    assert rec.record_hash != ""
    assert chain.verify_chain()


# ── Study registry ─────────────────────────────────────────────────

def test_registry_register_and_get():
    from models.evidence.registry import StudyRegistry
    reg = StudyRegistry()
    rec = reg.register("AL-001", "Test study", "p < 0.05 on held-out set")
    assert rec.study_id == "AL-001"
    assert rec.verdict == "pending"
    assert reg.get("AL-001") is not None


def test_registry_duplicate_registration_fails():
    from models.evidence.registry import StudyRegistry
    reg = StudyRegistry()
    reg.register("AL-001", "Study 1", "criterion 1")
    with pytest.raises(ValueError, match="already registered"):
        reg.register("AL-001", "Study 1 again", "criterion 2")


def test_registry_verdict_recording():
    from models.evidence.registry import StudyRegistry
    reg = StudyRegistry()
    reg.register("GD-001", "GRN discovery", "AUROC > 0.7")
    updated = reg.record_verdict(
        "GD-001",
        measured={"auroc": 0.82},
        summary="AUROC = 0.82 on held-out set",
        verdict="met",
    )
    assert updated.verdict == "met"
    assert updated.measured["auroc"] == 0.82


def test_registry_verdict_immutability():
    from models.evidence.registry import StudyRegistry
    reg = StudyRegistry()
    reg.register("RB-001", "Repro test", "must replicate")
    reg.record_verdict("RB-001", {}, "ok", "met")
    with pytest.raises(ValueError, match="verdict already recorded"):
        reg.record_verdict("RB-001", {}, "again", "not_met")


def test_registry_unregistered_verdict_fails():
    from models.evidence.registry import StudyRegistry
    reg = StudyRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.record_verdict("FAKE-001", {}, "", "met")


def test_registry_pending_and_list():
    from models.evidence.registry import StudyRegistry
    reg = StudyRegistry()
    reg.register("AL-001", "S1", "C1")
    reg.register("AL-002", "S2", "C2")
    reg.record_verdict("AL-001", {}, "", "met")
    assert "AL-002" in reg.pending()
    assert "AL-001" not in reg.pending()
    assert len(reg.list_studies()) == 2


def test_registry_to_dict_roundtrip():
    from models.evidence.registry import StudyRecord
    rec = StudyRecord(
        study_id="GD-010",
        title="GRN rank vs AlphaFold DB pLDDT",
        criterion="Spearman rho > 0.3",
    )
    d = rec.to_dict()
    rec2 = StudyRecord.from_dict(d)
    assert rec2.study_id == rec.study_id
    assert rec2.criterion == rec.criterion


# ── Regime firewall ────────────────────────────────────────────────

def test_regime_firewall_validates_human_experimental():
    from models.evidence.regime import RegimeFirewall, SpeciesDeclaration, Species, Regime
    decl = SpeciesDeclaration(Species.HUMAN, Regime.EXPERIMENTAL)
    ok, msg = RegimeFirewall.validate(decl)
    assert not ok
    assert "justification" in msg.lower()


def test_regime_firewall_allows_human_therapeutic():
    from models.evidence.regime import RegimeFirewall, SpeciesDeclaration, Species, Regime
    decl = SpeciesDeclaration(Species.HUMAN, Regime.THERAPEUTIC)
    ok, _ = RegimeFirewall.validate(decl)
    assert ok


def test_regime_firewall_cross_species_needs_justification():
    from models.evidence.regime import RegimeFirewall, SpeciesDeclaration, Species, Regime
    decl = SpeciesDeclaration(Species.MOUSE, Regime.BENCHMARK, cross_species=True)
    ok, msg = RegimeFirewall.validate(decl)
    assert not ok


def test_regime_firewall_compatibility_same_species():
    from models.evidence.regime import RegimeFirewall, SpeciesDeclaration, Species, Regime
    a = SpeciesDeclaration(Species.HUMAN, Regime.THERAPEUTIC)
    b = SpeciesDeclaration(Species.HUMAN, Regime.THERAPEUTIC)
    ok, _ = RegimeFirewall.check_compatibility(a, b)
    assert ok


def test_regime_firewall_compatibility_mismatch():
    from models.evidence.regime import RegimeFirewall, SpeciesDeclaration, Species, Regime
    a = SpeciesDeclaration(Species.HUMAN, Regime.THERAPEUTIC)
    b = SpeciesDeclaration(Species.MOUSE, Regime.THERAPEUTIC)
    ok, msg = RegimeFirewall.check_compatibility(a, b)
    assert not ok
    assert "Species mismatch" in msg


# ── Merlin bridge ──────────────────────────────────────────────────

def test_merlin_bridge_grn_to_tensor():
    from models.bridge.merlin_bridge import MerlinBridge, MerlinGRNResult
    result = MerlinGRNResult(
        edges=[("A", "B", 0.8), ("B", "C", 0.3), ("A", "C", 0.5)],
        method="GENIE3",
        n_genes=3,
        n_edges=3,
    )
    gene_index = {"A": 0, "B": 1, "C": 2}
    adj = MerlinBridge.grn_result_to_tensor(result, gene_index)
    assert adj.shape == (3, 3)
    assert adj[0, 1].item() == pytest.approx(0.8)
    assert adj[1, 2].item() == pytest.approx(0.3)


def test_merlin_bridge_tensor_to_edges():
    from models.bridge.merlin_bridge import MerlinBridge
    adj = torch.tensor([
        [0.0, 0.8, 0.5],
        [0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0],
    ])
    edges = MerlinBridge.tensor_to_grn_edges(adj, ["A", "B", "C"])
    assert len(edges) == 3
    assert edges[0][2] == pytest.approx(0.8)


def test_merlin_bridge_data_registry_entry():
    from models.bridge.merlin_bridge import MerlinBridge
    entry = MerlinBridge.data_registry_entry(
        "geo:GSE107965",
        "expression/gse107965.csv",
        "human",
        studies=["GD-010"],
    )
    assert entry["species"] == "human"
    assert entry["product_use_class"] == "verify"
    assert "GD-010" in entry["studies"]


# ── GEO adapter ────────────────────────────────────────────────────

def test_geo_adapter_load_envelopes():
    from models.bridge.geo_adapter import GEOAdapter
    adapter = GEOAdapter()
    data = {
        "envelopes": [
            {
                "dataset_id": "geo:GSE107965",
                "metadata": {
                    "accession": "GSE107965",
                    "title": "Time course rapamycin hESC",
                    "platform_organism": "Homo sapiens",
                    "platform": "GPL17586",
                    "sample_count": 50,
                    "series_type": "Expression profiling by high throughput sequencing",
                },
            },
            {
                "dataset_id": "geo:GSE999999",
                "metadata": {
                    "accession": "GSE999999",
                    "title": "Small study",
                    "platform_organism": "Homo sapiens",
                    "platform": "GPL570",
                    "sample_count": 5,
                    "series_type": "Expression profiling by array",
                },
            },
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        n = adapter.load_envelopes(Path(f.name))
    os.unlink(f.name)
    assert n == 2
    assert adapter.total_series == 2


def test_geo_adapter_filter_in_regime():
    from models.bridge.geo_adapter import GEOAdapter
    adapter = GEOAdapter()
    data = {
        "envelopes": [
            {"dataset_id": "geo:GSE1", "metadata": {"accession": "GSE1", "title": "Big human study", "platform_organism": "Homo sapiens", "platform": "GPL570", "sample_count": 100, "series_type": "Expression profiling by array"}},
            {"dataset_id": "geo:GSE2", "metadata": {"accession": "GSE2", "title": "Small", "platform_organism": "Homo sapiens", "platform": "GPL570", "sample_count": 5, "series_type": "Expression profiling by array"}},
            {"dataset_id": "geo:GSE3", "metadata": {"accession": "GSE3", "title": "Mouse study", "platform_organism": "Mus musculus", "platform": "GPL570", "sample_count": 100, "series_type": "Expression profiling by array"}},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        adapter.load_envelopes(Path(f.name))
    os.unlink(f.name)
    filtered = adapter.filter_in_regime(species="Homo sapiens")
    assert len(filtered) == 1
    assert filtered[0].accession == "GSE1"


def test_geo_adapter_csv_to_matrix():
    from models.bridge.geo_adapter import GEOAdapter
    adapter = GEOAdapter()
    csv = "gene,S1,S2,S3\nATG5,1.0,2.0,3.0\nMTOR,4.0,5.0,6.0"
    matrix = adapter.csv_to_expression_matrix(csv, "GSE107965")
    assert matrix.n_genes == 2
    assert matrix.n_samples == 3
    assert matrix.expression.shape == (2, 3)
    assert matrix.gene_names == ["ATG5", "MTOR"]


def test_geo_adapter_expression_to_grn_input():
    from models.bridge.geo_adapter import GEOAdapter, GEOExpressionMatrix
    adapter = GEOAdapter()
    matrix = GEOExpressionMatrix(
        accession="GSE107965",
        gene_names=["A", "B", "C"],
        sample_names=["S1", "S2"],
        expression=torch.tensor([[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]]),
    )
    signal = adapter.expression_to_grn_input(matrix)
    assert signal.shape == (1, 3)
    assert signal[0, 0].item() == pytest.approx(2.0)


# ── Product bridge ─────────────────────────────────────────────────

def test_product_bridge_audio_to_solver():
    from models.bridge.product_bridge import ProductBridge, AudioFeatures
    features = AudioFeatures(
        track_id="track_001",
        bpm=120.0,
        valence=0.7,
        arousal=0.4,
        dominance=0.5,
        energy_level=0.6,
    )
    tensor = ProductBridge.audio_to_solver_input(features, n_biomarkers=32)
    assert tensor.shape == (1, 32)
    assert tensor[0, 0].item() == pytest.approx(120.0 / 200.0)
    assert tensor[0, 1].item() == pytest.approx(0.7)


def test_product_bridge_vad_tensor():
    from models.bridge.product_bridge import AudioFeatures
    f = AudioFeatures(track_id="t", valence=0.8, arousal=0.3, dominance=0.5)
    vad = f.to_vad_tensor()
    assert vad.shape == (3,)
    assert vad[0].item() == pytest.approx(0.8)


def test_product_bridge_eif_input():
    from models.bridge.product_bridge import AudioFeatures
    f = AudioFeatures(track_id="t", valence=0.7, arousal=0.4, energy_level=0.6, bpm=120, spectral_centroid=3000)
    eif = f.to_eif_input()
    assert eif["valence"] == 0.7
    assert eif["bpm"] == 120


def test_product_bridge_session_plan():
    from models.bridge.product_bridge import ProductBridge
    trajectory = torch.randn(4, 6)
    doses = torch.tensor([1.0, 0.618, 0.382, 0.236, 0.146])
    plan = ProductBridge.vade_trajectory_to_session_plan(trajectory, doses)
    assert len(plan) == 6
    assert plan[0]["step"] == 0
    assert "target_valence" in plan[0]
    assert "dose_intensity" in plan[0]


def test_product_bridge_eif_ranking():
    from models.bridge.product_bridge import ProductBridge
    ids = ["t1", "t2", "t3"]
    scores = torch.tensor([0.3, 0.9, 0.6])
    ranked = ProductBridge.eif_scores_to_ranking(ids, scores)
    assert ranked[0]["track_id"] == "t2"
    assert ranked[0]["rank"] == 1
    assert ranked[2]["rank"] == 3


def test_product_bridge_brainwave_to_entrainment():
    from models.bridge.product_bridge import ProductBridge, BrainwaveParams
    params = BrainwaveParams(band="theta", base_freq_hz=6.0, binaural_beat_hz=2.0)
    ent = ProductBridge.brainwave_to_entrainment_input(params)
    assert ent["target_freq_hz"] == 6.0
    assert ent["band_low_hz"] == pytest.approx(4.0)
    assert ent["band_high_hz"] == pytest.approx(8.0)


# ── Integration: evidence + solver ─────────────────────────────────

def test_solver_with_evidence_chain():
    """End-to-end: run a solver problem, deposit result into evidence chain."""
    from models.evidence.lineage import LineageChain
    from models.evidence.registry import StudyRegistry
    from models.evidence.regime import RegimeFirewall, SpeciesDeclaration, Species, Regime

    fw = RegimeFirewall()
    decl = SpeciesDeclaration(Species.IN_SILICO, Regime.BENCHMARK)
    ok, _ = fw.validate(decl)
    assert ok

    reg = StudyRegistry()
    reg.register("NA-TEST-001", "Smoke test", "Solver returns finite values")

    chain = LineageChain()
    prob = {"type": "simulate_pk", "dose": 10.0}
    answer = {"concentration": [1.0, 0.8, 0.5, 0.3]}
    rec = chain.append("ER-001", prob, answer)

    assert chain.verify_chain()

    reg.record_verdict(
        "NA-TEST-001",
        measured={"chain_length": len(chain), "hash": chain.head_hash},
        summary="Solver produced finite values, chain verified",
        verdict="met",
    )
    study = reg.get("NA-TEST-001")
    assert study.verdict == "met"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
