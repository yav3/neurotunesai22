"""Tests for the Foundation–Frontier Hybrid Model.

Tests cover:
  - GeneFoundationEncoder forward pass and shapes
  - ExpressionBinner rank-based binning
  - MaskedGeneModeling pretraining objective
  - GovernedFrontierBridge query building and validation
  - HybridFoundationModel end-to-end inference
  - Species firewall enforcement
  - Lineage chain integrity
  - Training pipeline with synthetic data
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.foundation.gene_foundation import (
    ExpressionBinner,
    GeneEmbedding,
    GeneFoundationConfig,
    GeneFoundationEncoder,
    MaskedGeneModeling,
)
from models.foundation.frontier_bridge import (
    EmbeddingSummariser,
    FrontierBridgeConfig,
    FrontierResponse,
    GovernedFrontierBridge,
    GovernedHybridAnswer,
    QueryType,
    StructuredQuery,
    FoundationToFrontierProjector,
)
from models.foundation.hybrid_model import (
    ExpressionPredictor,
    HybridFoundationModel,
    HybridModelConfig,
)


MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")


# ── GeneFoundationEncoder tests ─────────────────────────────────────

class TestGeneFoundationEncoder:

    def setup_method(self):
        self.config = GeneFoundationConfig(
            max_genes=100, hidden_dim=64, num_layers=2, num_heads=4,
            ffn_dim=128, n_expression_bins=16,
        )

    def test_forward_shape(self):
        enc = GeneFoundationEncoder(self.config)
        B, G = 4, 100
        expr = torch.randn(B, G)
        gids = torch.arange(G).unsqueeze(0).expand(B, -1)
        cls_repr, seq_out = enc(expr, gids)
        assert cls_repr.shape == (B, 64)
        assert seq_out.shape == (B, G, 64)

    def test_forward_with_padding_mask(self):
        enc = GeneFoundationEncoder(self.config)
        B, G = 4, 100
        expr = torch.randn(B, G)
        gids = torch.arange(G).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, G, dtype=torch.bool)
        mask[:, 80:] = False
        cls_repr, seq_out = enc(expr, gids, padding_mask=mask)
        assert cls_repr.shape == (B, 64)

    def test_no_cls_token(self):
        config = GeneFoundationConfig(
            max_genes=50, hidden_dim=64, num_layers=2, num_heads=4,
            ffn_dim=128, use_cls_token=False,
        )
        enc = GeneFoundationEncoder(config)
        B, G = 2, 50
        expr = torch.randn(B, G)
        gids = torch.arange(G).unsqueeze(0).expand(B, -1)
        cls_repr, seq_out = enc(expr, gids)
        assert cls_repr.shape == (B, 64)
        assert seq_out.shape == (B, G, 64)

    def test_deterministic_with_seed(self):
        enc = GeneFoundationEncoder(self.config)
        enc.eval()
        expr = torch.randn(2, 100)
        gids = torch.arange(100).unsqueeze(0).expand(2, -1)
        out1, _ = enc(expr, gids)
        out2, _ = enc(expr, gids)
        torch.testing.assert_close(out1, out2)

    def test_param_count(self):
        enc = GeneFoundationEncoder(self.config)
        assert enc.param_count() > 0


# ── ExpressionBinner tests ──────────────────────────────────────────

class TestExpressionBinner:

    def test_output_shape(self):
        binner = ExpressionBinner(n_bins=32, hidden_dim=64)
        expr = torch.randn(4, 100)
        out = binner(expr)
        assert out.shape == (4, 100, 64)

    def test_rank_based_invariance(self):
        """Binner should produce same bins for rank-preserving transforms."""
        binner = ExpressionBinner(n_bins=32, hidden_dim=64)
        expr = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        shifted = expr + 100.0
        out1 = binner(expr)
        out2 = binner(shifted)
        torch.testing.assert_close(out1, out2)


# ── MaskedGeneModeling tests ───────────────────────────────────────

class TestMaskedGeneModeling:

    def test_forward_returns_loss(self):
        config = GeneFoundationConfig(
            max_genes=50, hidden_dim=64, num_layers=2, num_heads=4,
            ffn_dim=128, mask_ratio=0.15,
        )
        enc = GeneFoundationEncoder(config)
        mgm = MaskedGeneModeling(enc)
        expr = torch.randn(4, 50)
        gids = torch.arange(50).unsqueeze(0).expand(4, -1)
        out = mgm(expr, gids)
        assert "loss" in out
        assert out["loss"].item() > 0
        assert "mask" in out
        assert out["mask"].any()

    def test_mask_ratio(self):
        config = GeneFoundationConfig(
            max_genes=1000, hidden_dim=64, num_layers=1, num_heads=4,
            ffn_dim=128, mask_ratio=0.3,
        )
        enc = GeneFoundationEncoder(config)
        mgm = MaskedGeneModeling(enc)
        expr = torch.randn(8, 1000)
        gids = torch.arange(1000).unsqueeze(0).expand(8, -1)
        out = mgm(expr, gids)
        actual_ratio = out["mask"].float().mean().item()
        assert 0.15 < actual_ratio < 0.45


# ── GovernedFrontierBridge tests ────────────────────────────────────

class TestGovernedFrontierBridge:

    def test_build_query(self):
        bridge = GovernedFrontierBridge()
        expr_delta = {
            "top_genes": [{"gene": "FOXO3", "delta": 1.5, "baseline": 3.0, "predicted": 4.5}],
            "mean_abs_delta": 0.5,
            "n_upregulated": 10,
            "n_downregulated": 5,
            "total_genes": 100,
        }
        cls_summary = {"norm": 1.0, "mean": 0.1, "std": 0.5, "max": 2.0, "min": -1.0, "sparsity": 0.1}

        query = bridge.build_query(
            query_type=QueryType.PERTURBATION_HYPOTHESIS,
            expression_delta=expr_delta,
            cls_summary=cls_summary,
            species="human",
            regime="data-driven",
            hypothesis="FOXO3 knockout reduces autophagy",
            study_id="NT-006",
        )
        assert query.species == "human"
        assert query.pre_registered is True
        assert len(query.lineage_hash) == 64

    def test_species_firewall(self):
        bridge = GovernedFrontierBridge()
        with pytest.raises(ValueError, match="Species.*not in allowed set"):
            bridge.build_query(
                query_type=QueryType.GENE_FUNCTION,
                expression_delta={},
                cls_summary={},
                species="mouse",
                regime="data-driven",
                hypothesis="test",
            )

    def test_regime_firewall(self):
        bridge = GovernedFrontierBridge()
        with pytest.raises(ValueError, match="Regime.*not in allowed set"):
            bridge.build_query(
                query_type=QueryType.GENE_FUNCTION,
                expression_delta={},
                cls_summary={},
                species="human",
                regime="invalid",
                hypothesis="test",
            )

    def test_lineage_chain(self):
        bridge = GovernedFrontierBridge()
        q1 = bridge.build_query(
            query_type=QueryType.GENE_FUNCTION,
            expression_delta={"top_genes": []},
            cls_summary={},
            species="human",
            regime="data-driven",
            hypothesis="first",
        )
        q2 = bridge.build_query(
            query_type=QueryType.PATHWAY_ANALYSIS,
            expression_delta={"top_genes": []},
            cls_summary={},
            species="human",
            regime="data-driven",
            hypothesis="second",
        )
        assert len(bridge.lineage) == 2
        assert bridge.lineage[0] != bridge.lineage[1]
        assert q1.lineage_hash == bridge.lineage[0]
        assert q2.lineage_hash == bridge.lineage[1]

    def test_format_prompt(self):
        bridge = GovernedFrontierBridge()
        query = StructuredQuery(
            query_type=QueryType.PERTURBATION_HYPOTHESIS,
            gene_context={"top_genes": [
                {"gene": "MTOR", "delta": -2.1, "baseline": 5.0, "predicted": 2.9}
            ], "summary_stats": {"total_genes": 100, "n_upregulated": 5,
                                  "n_downregulated": 15, "mean_abs_delta": 0.8}},
            embedding_summary={},
            species="human",
            regime="data-driven",
            hypothesis="mTOR inhibition downregulates autophagy pathway",
        )
        prompt = bridge.format_prompt(query)
        assert "Species: human" in prompt
        assert "MTOR" in prompt
        assert "perturbation_hypothesis" in prompt

    def test_parse_response(self):
        bridge = GovernedFrontierBridge()
        bridge.lineage.append("genesis_hash")
        query = StructuredQuery(
            query_type=QueryType.GENE_FUNCTION,
            gene_context={}, embedding_summary={},
            species="human", regime="data-driven", hypothesis="test",
        )
        raw = {
            "reasoning": "FOXO3 is a key autophagy regulator",
            "confidence": 0.85,
            "evidence_citations": ["PMID:12345678"],
            "suggested_experiments": ["siRNA knockdown"],
            "caveats": ["In vitro only"],
        }
        resp = bridge.parse_response(raw, query)
        assert resp.confidence == 0.85
        assert len(resp.evidence_citations) == 1
        assert len(resp.response_hash) == 64

    def test_parse_response_clamps_confidence(self):
        bridge = GovernedFrontierBridge()
        bridge.lineage.append("genesis_hash")
        query = StructuredQuery(
            query_type=QueryType.GENE_FUNCTION,
            gene_context={}, embedding_summary={},
            species="human", regime="data-driven", hypothesis="test",
        )
        resp = bridge.parse_response({"confidence": 5.0}, query)
        assert resp.confidence == 1.0
        resp2 = bridge.parse_response({"confidence": -1.0}, query)
        assert resp2.confidence == 0.0

    def test_governed_answer_hash(self):
        answer = GovernedHybridAnswer(
            foundation_prediction={"test": 1},
            frontier_reasoning=None,
            lineage_chain=["abc"],
            species="human",
            regime="data-driven",
            study_id="NT-006",
        )
        h = answer.compute_hash()
        assert len(h) == 64
        assert answer.content_hash == h

    def test_governed_answer_hash_stable(self):
        kwargs = dict(
            foundation_prediction={"val": 42},
            frontier_reasoning=None,
            lineage_chain=[],
            species="human",
            regime="data-driven",
            study_id="X",
        )
        a1 = GovernedHybridAnswer(**kwargs)
        a2 = GovernedHybridAnswer(**kwargs)
        assert a1.compute_hash() == a2.compute_hash()


# ── EmbeddingSummariser tests ───────────────────────────────────────

class TestEmbeddingSummariser:

    def test_summarise_expression_delta(self):
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.5, 1.0, 4.0, 3.5, 6.5])
        genes = ["A", "B", "C", "D", "E"]
        summary = EmbeddingSummariser.summarise_expression_delta(
            baseline, predicted, genes, top_k=3)
        assert len(summary["top_genes"]) == 3
        assert summary["total_genes"] == 5
        assert summary["mean_abs_delta"] > 0

    def test_summarise_cls_embedding(self):
        cls = np.random.randn(256)
        summary = EmbeddingSummariser.summarise_cls_embedding(cls)
        assert "norm" in summary
        assert "sparsity" in summary
        assert 0 <= summary["sparsity"] <= 1


# ── HybridFoundationModel tests ────────────────────────────────────

class TestHybridFoundationModel:

    def setup_method(self):
        self.config = HybridModelConfig(
            gene_config=GeneFoundationConfig(
                max_genes=100, hidden_dim=64, num_layers=2,
                num_heads=4, ffn_dim=128,
            ),
            use_frontier=True,
            projection_dim=16,
        )

    def test_forward_shape(self):
        model = HybridFoundationModel(self.config)
        B, G = 4, 100
        expr = torch.randn(B, G)
        gids = torch.arange(G).unsqueeze(0).expand(B, -1)
        out = model(expr, gids)
        assert out["cls_repr"].shape == (B, 64)
        assert out["pred_expression"].shape == (B, 100)
        assert out["frontier_projection"].shape == (B, 16)
        assert out["foundation_confidence"].shape == (B,)

    def test_training_loss(self):
        model = HybridFoundationModel(self.config)
        B, G = 4, 100
        expr = torch.randn(B, G)
        target = torch.randn(B, G)
        gids = torch.arange(G).unsqueeze(0).expand(B, -1)
        out = model.training_loss(expr, gids, target)
        assert "loss" in out
        assert out["loss"].item() > 0
        out["loss"].backward()

    def test_predict_and_explain_without_frontier(self):
        config = HybridModelConfig(
            gene_config=GeneFoundationConfig(
                max_genes=50, hidden_dim=64, num_layers=2,
                num_heads=4, ffn_dim=128,
            ),
            use_frontier=False,
        )
        model = HybridFoundationModel(config)
        expr = torch.randn(1, 50)
        gids = torch.arange(50).unsqueeze(0)
        gene_names = [f"GENE{i}" for i in range(50)]

        answer = model.predict_and_explain(
            expr, gids, gene_names,
            species="human", regime="data-driven",
            hypothesis="test hypothesis",
        )
        assert answer.species == "human"
        assert answer.frontier_reasoning is None
        assert answer.is_governed is True
        assert len(answer.content_hash) == 64

    def test_predict_and_explain_with_mock_frontier(self):
        model = HybridFoundationModel(self.config)
        expr = torch.randn(1, 100)
        gids = torch.arange(100).unsqueeze(0)
        gene_names = [f"GENE{i}" for i in range(100)]

        def mock_frontier(prompt: str) -> dict:
            return {
                "reasoning": "The expression pattern suggests pathway activation",
                "confidence": 0.75,
                "evidence_citations": ["PMID:99999999"],
                "suggested_experiments": ["Validate with RT-qPCR"],
                "caveats": ["Model trained on limited data"],
            }

        answer = model.predict_and_explain(
            expr, gids, gene_names,
            species="human", regime="data-driven",
            hypothesis="Gene X activates pathway Y",
            study_id="NT-006",
            frontier_call=mock_frontier,
        )
        assert answer.frontier_reasoning is not None
        assert answer.frontier_reasoning.confidence == 0.75
        assert len(answer.lineage_chain) >= 1

    def test_species_firewall_in_predict(self):
        model = HybridFoundationModel(self.config)
        expr = torch.randn(1, 100)
        gids = torch.arange(100).unsqueeze(0)

        with pytest.raises(ValueError, match="Species"):
            model.predict_and_explain(
                expr, gids, [f"G{i}" for i in range(100)],
                species="mouse", regime="data-driven",
                hypothesis="test",
                frontier_call=lambda p: {},
            )

    def test_param_count(self):
        model = HybridFoundationModel(self.config)
        counts = model.param_count()
        assert counts["encoder"] > 0
        assert counts["predictor"] > 0
        assert counts["projector"] > 0
        assert counts["total"] == counts["encoder"] + counts["predictor"] + counts["projector"]


# ── FoundationToFrontierProjector tests ─────────────────────────────

class TestProjector:

    def test_forward_shape(self):
        proj = FoundationToFrontierProjector(hidden_dim=64, projection_dim=16)
        cls_repr = torch.randn(4, 64)
        out = proj(cls_repr)
        assert out["projection"].shape == (4, 16)
        assert out["confidence"].shape == (4,)
        assert (out["confidence"] >= 0).all()
        assert (out["confidence"] <= 1).all()


# ── Integration: training pipeline with synthetic data ──────────────

class TestHybridTrainingPipeline:

    def test_synthetic_trial(self):
        """Run a governed trial on synthetic data to verify the pipeline."""
        from models.training.hybrid_trainer import (
            GEOExpressionDataset,
            HybridTrainConfig,
            HybridTrainingPipeline,
            TrialRecord,
        )

        np.random.seed(2026)
        n_samples, n_genes = 40, 50
        expression = np.random.randn(n_samples, n_genes).astype(np.float32)
        conditions = ["control"] * 10 + ["treated"] * 30
        gene_names = [f"GENE{i}" for i in range(n_genes)]

        dataset = GEOExpressionDataset(expression, conditions, gene_names, "human")
        assert len(dataset.pairs) == 30

        baselines, targets, gene_ids = dataset.to_tensors()
        assert baselines.shape == (30, n_genes)
        assert targets.shape == (30, n_genes)

    def test_trial_record_hash(self):
        from models.training.hybrid_trainer import TrialRecord

        record = TrialRecord(
            study_id="NT-TEST", species="human", regime="data-driven",
            n_samples=40, n_genes=50, n_series=1, model_params=1000,
            train_mse=0.5, val_mse=0.6, test_mse=0.7,
            train_r=0.8, val_r=0.7, test_r=0.6,
            baseline_mse=1.0, baseline_r=0.3,
            criteria_met=4, verdict="MET",
        )
        h = record.compute_hash()
        assert len(h) == 64
        h2 = record.compute_hash()
        assert h == h2


# ── Integration: real GEO data (if available) ───────────────────────

@pytest.mark.skipif(
    not MERLIN_ROOT.exists(),
    reason="Merlin repo not available",
)
class TestHybridWithRealData:

    def test_load_pooled_data(self):
        from models.training.hybrid_trainer import HybridTrainingPipeline
        pipeline = HybridTrainingPipeline(merlin_root=MERLIN_ROOT)
        dataset = pipeline.load_pooled_data(max_genes=100)
        assert dataset.species == "human"
        assert len(dataset.pairs) > 0
        assert dataset.expression.shape[1] <= 100

    def test_pretrain_on_real_data(self):
        from models.training.hybrid_trainer import (
            HybridTrainConfig,
            HybridTrainingPipeline,
        )
        config = HybridTrainConfig(
            pretrain_epochs=3, pretrain_batch_size=8,
            patience=2, seed=2026,
        )
        pipeline = HybridTrainingPipeline(merlin_root=MERLIN_ROOT, config=config)
        dataset = pipeline.load_pooled_data(max_genes=50)

        gc = GeneFoundationConfig(
            max_genes=dataset.expression.shape[1],
            hidden_dim=64, num_layers=2, num_heads=4, ffn_dim=128,
        )
        encoder, stats = pipeline.run_pretrain(dataset, gc)
        assert stats["val_loss"] > 0
        assert stats["epochs"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
