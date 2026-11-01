"""Tests for the frontier caller and end-to-end pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from models.foundation.gene_foundation import GeneFoundationConfig
from models.foundation.hybrid_model import HybridFoundationModel, HybridModelConfig
from models.foundation.frontier_bridge import QueryType
from models.foundation.frontier_caller import (
    ClaudeFrontierCaller,
    MockFrontierCaller,
    FrontierCallerConfig,
)
from models.training.hybrid_trainer import (
    HybridTrainConfig,
    HybridTrainingPipeline,
    TrialRecord,
)
from models.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_trial_record,
)


class TestMockFrontierCaller:

    def test_returns_structured_json(self):
        caller = MockFrontierCaller()
        result = caller("Test prompt with FOXO3: delta=+0.5")
        assert "reasoning" in result
        assert "confidence" in result
        assert "evidence_citations" in result
        assert "suggested_experiments" in result
        assert "caveats" in result

    def test_extracts_gene_names(self):
        caller = MockFrontierCaller()
        prompt = "FOXO3: delta=+0.5\nMTOR: delta=-0.3\nBECN1: delta=+0.2"
        result = caller(prompt)
        assert "FOXO3" in result["reasoning"]

    def test_confidence_configurable(self):
        caller = MockFrontierCaller(confidence=0.95)
        result = caller("test")
        assert result["confidence"] == 0.95

    def test_call_count_tracked(self):
        caller = MockFrontierCaller()
        caller("first")
        caller("second")
        assert caller.call_count == 2

    def test_extracts_regulation_counts(self):
        caller = MockFrontierCaller()
        prompt = "12 upregulated genes, 7 downregulated genes"
        result = caller(prompt)
        assert "12 upregulated" in result["reasoning"]
        assert "7 downregulated" in result["reasoning"]


class TestClaudeFrontierCaller:

    def test_json_extraction_clean(self):
        raw = json.dumps({
            "reasoning": "test",
            "confidence": 0.8,
            "evidence_citations": [],
            "suggested_experiments": [],
            "caveats": [],
        })
        parsed = ClaudeFrontierCaller._extract_json(raw)
        assert parsed["confidence"] == 0.8

    def test_json_extraction_with_fences(self):
        raw = "Here's my analysis:\n```json\n{\"reasoning\": \"test\", \"confidence\": 0.7}\n```"
        parsed = ClaudeFrontierCaller._extract_json(raw)
        assert parsed["reasoning"] == "test"
        assert parsed["confidence"] == 0.7

    def test_json_extraction_invalid_graceful(self):
        raw = "This is not JSON at all"
        parsed = ClaudeFrontierCaller._extract_json(raw)
        assert parsed["confidence"] == 0.1
        assert "not valid JSON" in parsed["caveats"][0]

    def test_no_api_key_raises(self):
        caller = ClaudeFrontierCaller(FrontierCallerConfig(api_key_env="_NONEXISTENT_KEY_"))
        with pytest.raises(ValueError, match="Set _NONEXISTENT_KEY_"):
            caller("test")


class TestCheckpoint:

    def test_save_and_load(self, tmp_path):
        gc = GeneFoundationConfig(max_genes=50, hidden_dim=64,
                                  num_layers=2, num_heads=4)
        config = HybridModelConfig(gene_config=gc, use_frontier=True,
                                   prediction_hidden=64)
        model = HybridFoundationModel(config)

        record = TrialRecord(
            study_id="TEST-001", species="human", regime="prior-contrast",
            n_samples=10, n_genes=50, n_series=1, model_params=100,
            train_mse=0.5, val_mse=0.6, test_mse=0.7,
            train_r=0.8, val_r=0.7, test_r=0.6,
            baseline_mse=1.0, baseline_r=0.3,
            criteria_met=4, verdict="MET",
        )
        record.compute_hash()

        ckpt_path = save_checkpoint(
            model, record, checkpoint_dir=tmp_path, tag="test",
        )
        assert ckpt_path.exists()
        assert (ckpt_path.parent / ckpt_path.name.replace(".pt", "_record.json")).exists()

        model2 = HybridFoundationModel(config)
        model2, meta = load_checkpoint(ckpt_path, model2)
        assert meta["study_id"] == "TEST-001"
        assert meta["content_hash"] == record.content_hash

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_load_trial_record(self, tmp_path):
        record = TrialRecord(
            study_id="TEST-002", species="human", regime="data-driven",
            n_samples=100, n_genes=200, n_series=3, model_params=5000,
            train_mse=0.1, val_mse=0.2, test_mse=0.3,
            train_r=0.9, val_r=0.85, test_r=0.8,
            baseline_mse=0.5, baseline_r=0.5,
            criteria_met=4, verdict="MET",
        )
        record.compute_hash()

        path = tmp_path / "test_record.json"
        from dataclasses import asdict
        with open(path, "w") as f:
            json.dump(asdict(record), f, default=str)

        loaded = load_trial_record(path)
        assert loaded.study_id == "TEST-002"
        assert loaded.criteria_met == 4

    def test_hash_integrity_verification(self, tmp_path):
        gc = GeneFoundationConfig(max_genes=20, hidden_dim=32,
                                  num_layers=1, num_heads=4)
        config = HybridModelConfig(gene_config=gc, use_frontier=False,
                                   prediction_hidden=32)
        model = HybridFoundationModel(config)

        record = TrialRecord(
            study_id="TEST-003", species="human", regime="prior-contrast",
            n_samples=5, n_genes=20, n_series=1, model_params=50,
            train_mse=1.0, val_mse=1.0, test_mse=1.0,
            train_r=0.0, val_r=0.0, test_r=0.0,
            baseline_mse=2.0, baseline_r=0.0,
            criteria_met=2, verdict="NOT_MET",
        )
        record.compute_hash()

        ckpt_path = save_checkpoint(model, record, checkpoint_dir=tmp_path)

        payload = torch.load(ckpt_path, weights_only=False)
        key = list(payload["model_state_dict"].keys())[0]
        payload["model_state_dict"][key] += 1.0
        torch.save(payload, ckpt_path)

        model2 = HybridFoundationModel(config)
        with pytest.raises(ValueError, match="Weights hash mismatch"):
            load_checkpoint(ckpt_path, model2, verify=True)


class TestEndToEndPipeline:

    def test_governed_trial_runs(self):
        """Full governed trial with real GEO data — the core integration test."""
        config = HybridTrainConfig(
            pretrain_epochs=3,
            finetune_epochs=3,
            patience=2,
        )
        pipeline = HybridTrainingPipeline(config=config)
        record = pipeline.run_governed_trial(
            study_id="TEST-E2E",
            max_genes=200,
            species="human",
            pretrain_first=True,
        )

        assert record.study_id == "TEST-E2E"
        assert record.species == "human"
        assert record.n_genes > 0
        assert record.n_samples > 0
        assert record.criteria_total == 4
        assert record.verdict in ("MET", "NOT_MET")
        assert np.isfinite(record.val_mse)
        assert np.isfinite(record.test_mse)
        assert record.content_hash

    def test_governed_trial_includes_frontier_evaluation(self):
        """Governed trial exercises the full two-tower pipeline."""
        caller = MockFrontierCaller(confidence=0.9)
        config = HybridTrainConfig(
            pretrain_epochs=2,
            finetune_epochs=2,
            patience=2,
        )
        pipeline = HybridTrainingPipeline(config=config, frontier_caller=caller)
        record = pipeline.run_governed_trial(
            study_id="TEST-FRONTIER-EVAL",
            max_genes=200,
            species="human",
            pretrain_first=False,
        )

        assert record.extra["frontier_used"] is True
        assert record.extra["frontier_confidence"] == 0.9
        assert record.extra["frontier_content_hash"]
        assert caller.call_count == 1

    def test_governed_inference_with_mock_frontier(self):
        """Full inference with mock frontier call."""
        gc = GeneFoundationConfig(max_genes=50, hidden_dim=64,
                                  num_layers=2, num_heads=4)
        config = HybridModelConfig(gene_config=gc, use_frontier=True,
                                   prediction_hidden=64)
        model = HybridFoundationModel(config)
        model.eval()

        expression = torch.randn(1, 50)
        gene_ids = torch.arange(50).unsqueeze(0)
        gene_names = [f"GENE{i}" for i in range(50)]
        caller = MockFrontierCaller(confidence=0.85)

        answer = model.predict_and_explain(
            expression=expression,
            gene_indices=gene_ids,
            gene_names=gene_names,
            species="human",
            regime="data-driven",
            hypothesis="Test hypothesis",
            study_id="TEST-INF",
            frontier_call=caller,
            query_type=QueryType.PERTURBATION_HYPOTHESIS,
        )

        assert answer.species == "human"
        assert answer.is_governed
        assert answer.content_hash
        assert len(answer.lineage_chain) >= 2
        assert answer.frontier_reasoning is not None
        assert answer.frontier_reasoning.confidence == 0.85
        assert caller.call_count == 1

    def test_train_save_load_infer(self, tmp_path):
        """Full cycle: train → save → load → infer."""
        gc = GeneFoundationConfig(max_genes=50, hidden_dim=64,
                                  num_layers=2, num_heads=4)
        config = HybridModelConfig(gene_config=gc, use_frontier=True,
                                   prediction_hidden=64)

        model = HybridFoundationModel(config)
        record = TrialRecord(
            study_id="TEST-CYCLE", species="human", regime="prior-contrast",
            n_samples=10, n_genes=50, n_series=1, model_params=100,
            train_mse=0.5, val_mse=0.6, test_mse=0.7,
            train_r=0.8, val_r=0.7, test_r=0.6,
            baseline_mse=1.0, baseline_r=0.3,
            criteria_met=4, verdict="MET",
        )
        record.compute_hash()

        ckpt_path = save_checkpoint(model, record, checkpoint_dir=tmp_path)

        model2 = HybridFoundationModel(config)
        model2, meta = load_checkpoint(ckpt_path, model2)

        expression = torch.randn(1, 50)
        gene_ids = torch.arange(50).unsqueeze(0)
        gene_names = [f"GENE{i}" for i in range(50)]
        caller = MockFrontierCaller()

        answer = model2.predict_and_explain(
            expression=expression,
            gene_indices=gene_ids,
            gene_names=gene_names,
            species="human",
            regime="prior-contrast",
            hypothesis="Loaded model test",
            study_id="TEST-CYCLE",
            frontier_call=caller,
        )

        assert answer.content_hash
        assert answer.frontier_reasoning is not None


class TestLineageIntegration:
    """Verify that training trials produce hash-chained lineage records."""

    def test_trial_record_chains_into_lineage(self):
        from models.evidence.lineage import LineageChain
        from models.evidence.registry import StudyRegistry

        chain = LineageChain()
        registry = StudyRegistry()

        from models.training.hybrid_trainer import HybridTrainingPipeline, HybridTrainConfig
        config = HybridTrainConfig(
            pretrain_epochs=1, finetune_epochs=1,
            pretrain_batch_size=4, finetune_batch_size=4,
        )
        pipeline = HybridTrainingPipeline(
            config=config, registry=registry, lineage=chain,
        )

        # Build a minimal synthetic dataset to avoid Merlin dependency
        from models.training.hybrid_trainer import GEOExpressionDataset
        rng = np.random.default_rng(2026)
        n_samples, n_genes = 20, 50
        expr = rng.standard_normal((n_samples, n_genes)).astype(np.float32)
        conditions = ["control"] * 10 + ["treated"] * 10
        gene_names = [f"G{i}" for i in range(n_genes)]
        dataset = GEOExpressionDataset(expr, conditions, gene_names, "human")

        # Monkey-patch load_pooled_data to return our synthetic dataset
        pipeline.load_pooled_data = lambda **_kw: dataset

        record = pipeline.run_governed_trial(study_id="NT-TEST-001")

        # Lineage chain should have exactly one record
        assert len(chain) == 1
        assert chain.verify_chain()
        assert record.extra.get("lineage_hash")

        # Registry should have the verdict
        study = registry.get("NT-TEST-001")
        assert study is not None
        assert study.verdict in ("met", "not_met")

        # Second trial chains
        pipeline2 = HybridTrainingPipeline(
            config=config, registry=None, lineage=chain,
        )
        pipeline2.load_pooled_data = lambda **_kw: dataset
        record2 = pipeline2.run_governed_trial(study_id="NT-TEST-002")

        assert len(chain) == 2
        assert chain.verify_chain()
        assert record2.extra["lineage_hash"] != record.extra["lineage_hash"]

    def test_chain_hash_cross_repo_compatibility(self):
        """Verify our chain_hash matches the canonical formula."""
        from models.evidence.lineage import chain_hash
        import hashlib, json

        # Canonical formula (inline from neurotunes3)
        payload = json.dumps({
            "prev_hash": "abc", "record_id": "r1",
            "problem_spec": {"x": 1}, "answer_values": {"y": 2},
            "timestamp": 999.0,
        }, sort_keys=True)
        expected = hashlib.sha256(payload.encode()).hexdigest()
        actual = chain_hash("abc", "r1", {"x": 1}, {"y": 2}, 999.0)
        assert actual == expected


class TestHybridInferenceService:
    """Verify the inference service provides clean checkpoint → governed answer."""

    def test_from_checkpoint(self, tmp_path):
        """Load checkpoint and run governed inference via the service."""
        from models.inference.hybrid_inference import HybridInferenceService

        gc = GeneFoundationConfig(max_genes=50, hidden_dim=64,
                                  num_layers=2, num_heads=4)
        config = HybridModelConfig(gene_config=gc, use_frontier=True,
                                   prediction_hidden=64)
        model = HybridFoundationModel(config)
        record = TrialRecord(
            study_id="TEST-SVC", species="human", regime="prior-contrast",
            n_samples=10, n_genes=50, n_series=1, model_params=100,
            train_mse=0.5, val_mse=0.6, test_mse=0.7,
            train_r=0.8, val_r=0.7, test_r=0.6,
            baseline_mse=1.0, baseline_r=0.3,
            criteria_met=4, verdict="MET",
        )
        record.compute_hash()

        ckpt_path = save_checkpoint(model, record, checkpoint_dir=tmp_path)
        gene_names = [f"GENE{i}" for i in range(50)]

        service = HybridInferenceService.from_checkpoint(
            ckpt_path, gene_names=gene_names, model_config=config,
        )

        expression = np.random.randn(50).astype(np.float32)
        answer = service.infer(expression, hypothesis="Test via service")

        assert answer.species == "human"
        assert answer.is_governed
        assert answer.content_hash
        assert answer.frontier_reasoning is not None
        assert answer.frontier_reasoning.confidence == 0.7  # MockFrontierCaller default

    def test_batch_infer(self, tmp_path):
        """Batch inference returns one answer per sample."""
        from models.inference.hybrid_inference import HybridInferenceService

        gc = GeneFoundationConfig(max_genes=30, hidden_dim=32,
                                  num_layers=1, num_heads=4)
        config = HybridModelConfig(gene_config=gc, use_frontier=True,
                                   prediction_hidden=32)
        model = HybridFoundationModel(config)
        record = TrialRecord(
            study_id="TEST-BATCH", species="human", regime="data-driven",
            n_samples=5, n_genes=30, n_series=1, model_params=50,
            train_mse=0.5, val_mse=0.6, test_mse=0.7,
            train_r=0.8, val_r=0.7, test_r=0.6,
            baseline_mse=1.0, baseline_r=0.3,
            criteria_met=4, verdict="MET",
        )
        record.compute_hash()
        ckpt_path = save_checkpoint(model, record, checkpoint_dir=tmp_path)

        service = HybridInferenceService.from_checkpoint(
            ckpt_path, gene_names=[f"G{i}" for i in range(30)],
            model_config=config,
        )

        batch = np.random.randn(3, 30).astype(np.float32)
        answers = service.batch_infer(batch, study_id="BATCH-001")

        assert len(answers) == 3
        for i, ans in enumerate(answers):
            assert ans.is_governed
            assert ans.content_hash
            assert ans.study_id == f"BATCH-001_sample_{i}"

    def test_full_train_save_load_service_infer(self, tmp_path):
        """Complete pipeline: train → save → load via service → infer."""
        from models.inference.hybrid_inference import HybridInferenceService

        config = HybridTrainConfig(
            pretrain_epochs=2,
            finetune_epochs=2,
            patience=2,
        )
        pipeline = HybridTrainingPipeline(config=config)
        record = pipeline.run_governed_trial(
            study_id="TEST-FULL-PIPE",
            max_genes=200,
            species="human",
            pretrain_first=False,
        )

        gc = GeneFoundationConfig(max_genes=record.n_genes, hidden_dim=256,
                                  num_layers=6, num_heads=8)
        model_config = HybridModelConfig(gene_config=gc, use_frontier=True)
        model = HybridFoundationModel(model_config)

        ckpt_path = save_checkpoint(model, record, checkpoint_dir=tmp_path)
        gene_names = [f"GENE{i}" for i in range(record.n_genes)]

        service = HybridInferenceService.from_checkpoint(
            ckpt_path, gene_names=gene_names, gene_config=gc,
        )

        expression = np.random.randn(record.n_genes).astype(np.float32)
        answer = service.infer(
            expression,
            hypothesis="Full pipeline governed inference",
            study_id="TEST-FULL-PIPE-INF",
        )

        assert answer.is_governed
        assert answer.content_hash
        assert answer.frontier_reasoning is not None
        assert len(answer.lineage_chain) >= 2

    def test_auto_checkpoint(self, tmp_path):
        """run_governed_trial with checkpoint_dir saves a loadable .pt file."""
        config = HybridTrainConfig(
            pretrain_epochs=2,
            finetune_epochs=2,
            patience=2,
        )
        pipeline = HybridTrainingPipeline(config=config)
        record = pipeline.run_governed_trial(
            study_id="TEST-AUTO-CKPT",
            max_genes=200,
            species="human",
            pretrain_first=False,
            checkpoint_dir=tmp_path,
        )

        assert "checkpoint_path" in record.extra
        ckpt_path = Path(record.extra["checkpoint_path"])
        assert ckpt_path.exists()

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in payload
        assert payload.get("config", {}).get("use_frontier") is True
        assert payload["study_id"] == "TEST-AUTO-CKPT"

    def test_run_and_serve(self, tmp_path):
        """run_and_serve returns (record, service) with working inference."""
        config = HybridTrainConfig(
            pretrain_epochs=2,
            finetune_epochs=2,
            patience=2,
        )
        pipeline = HybridTrainingPipeline(config=config)
        record, service = pipeline.run_and_serve(
            study_id="TEST-RUN-SERVE",
            max_genes=200,
            species="human",
            pretrain_first=False,
            checkpoint_dir=tmp_path,
        )

        assert record.verdict in ("MET", "NOT_MET")
        assert record.extra["frontier_used"] is True

        expression = np.random.randn(record.n_genes).astype(np.float32)
        answer = service.infer(
            expression,
            hypothesis="run_and_serve end-to-end test",
            study_id="TEST-RUN-SERVE-INF",
        )
        assert answer.is_governed
        assert answer.content_hash
        assert answer.frontier_reasoning is not None
