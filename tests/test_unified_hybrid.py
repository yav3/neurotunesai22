"""Tests for NeuroHybrid and UnifiedFoundationFrontierModel.

Covers:
  - NeuroHybrid standalone (encode, forward, confidence gate, perturbation head)
  - UnifiedFoundationFrontierModel (forward shapes, training_loss, governed_inference)
  - Species firewall enforcement through the full stack
  - Confidence-gated frontier deferral in governed mode
  - Hash-chained lineage on governed answers
"""
from __future__ import annotations

import json
import pytest
import torch
import numpy as np

from models.config import MODALITY_DIMS
from models.foundation.hybrid import (
    HybridConfig,
    HybridMode,
    NeuroHybrid,
    GeneExpressionEncoder,
    GeneBottleneckBlock,
    DiseaseClassificationHead,
    ExpressionReconstructionHead,
)
from models.foundation.unified_hybrid import (
    UnifiedFoundationFrontierModel,
    UnifiedHybridConfig,
)
from models.foundation.frontier_bridge import (
    GovernedHybridAnswer,
    QueryType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return HybridConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        n_genes=100,
        gene_latent_tokens=4,
        gene_bottleneck_heads=2,
        gene_bottleneck_layers=1,
        latent_dim=32,
        perturbation_dim=16,
        n_drugs=8,
        max_gene_idx=200,
        head_hidden_dim=32,
    )


@pytest.fixture
def small_unified_config(small_config):
    return UnifiedHybridConfig(
        hybrid_config=small_config,
        projection_dim=16,
        auto_defer_threshold=0.8,
        mode=HybridMode.GOVERNED,
    )


@pytest.fixture
def clinical_batch():
    """Synthetic clinical data for batch_size=2."""
    B = 2
    return {
        name: torch.randn(B, dim)
        for name, dim in MODALITY_DIMS.items()
    }


@pytest.fixture
def expression_batch():
    return torch.randn(2, 100)


@pytest.fixture
def gene_names_100():
    return [f"GENE{i}" for i in range(100)]


# ---------------------------------------------------------------------------
# NeuroHybrid standalone tests
# ---------------------------------------------------------------------------

class TestGeneExpressionEncoder:
    def test_output_shape(self):
        enc = GeneExpressionEncoder(n_genes=100, hidden_dim=64, n_latent_tokens=4)
        x = torch.randn(3, 100)
        out = enc(x)
        assert out.shape == (3, 4, 64)

    def test_different_batch_sizes(self):
        enc = GeneExpressionEncoder(n_genes=50, hidden_dim=32, n_latent_tokens=2)
        for B in [1, 4, 16]:
            out = enc(torch.randn(B, 50))
            assert out.shape == (B, 2, 32)


class TestGeneBottleneckBlock:
    def test_residual_connection(self):
        block = GeneBottleneckBlock(hidden_dim=32, num_heads=4)
        x = torch.randn(2, 4, 32)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)


class TestNeuroHybridEncode:
    def test_encode_expression_only(self, small_config, expression_batch):
        model = NeuroHybrid(small_config)
        cls = model.encode(expression=expression_batch)
        assert cls.shape == (2, small_config.hidden_dim)

    def test_encode_clinical_only(self, small_config, clinical_batch):
        model = NeuroHybrid(small_config)
        cls = model.encode(clinical_dict=clinical_batch)
        assert cls.shape == (2, small_config.hidden_dim)

    def test_encode_both(self, small_config, clinical_batch, expression_batch):
        model = NeuroHybrid(small_config)
        cls = model.encode(clinical_dict=clinical_batch, expression=expression_batch)
        assert cls.shape == (2, small_config.hidden_dim)

    def test_encode_requires_at_least_one_modality(self, small_config):
        model = NeuroHybrid(small_config)
        with pytest.raises(ValueError, match="At least one input"):
            model.encode()


class TestNeuroHybridForward:
    def test_forward_expression_only(self, small_config, expression_batch):
        model = NeuroHybrid(small_config)
        result = model(expression=expression_batch)

        assert result["cls_repr"].shape == (2, small_config.hidden_dim)
        assert result["confidence"].shape == (2,)
        assert result["disease_logits"].shape == (2, 14)
        assert result["expression_recon"].shape == (2, small_config.n_genes)
        assert "projection" in result

    def test_forward_clinical_only(self, small_config, clinical_batch):
        model = NeuroHybrid(small_config)
        result = model(clinical_dict=clinical_batch)
        assert "expression_recon" not in result
        assert "clinical_recon" in result
        assert result["disease_logits"].shape == (2, 14)

    def test_forward_both_modalities(self, small_config, clinical_batch, expression_batch):
        model = NeuroHybrid(small_config)
        result = model(clinical_dict=clinical_batch, expression=expression_batch)
        assert result["cls_repr"].shape == (2, small_config.hidden_dim)
        assert result["expression_recon"].shape == (2, small_config.n_genes)

    def test_confidence_bounded(self, small_config, expression_batch):
        model = NeuroHybrid(small_config)
        result = model(expression=expression_batch)
        conf = result["confidence"]
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_projection_normalized(self, small_config, expression_batch):
        model = NeuroHybrid(small_config)
        result = model(expression=expression_batch)
        proj = result["projection"]
        norms = proj.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestNeuroHybridDeferral:
    def test_should_defer_returns_bool_mask(self, small_config, expression_batch):
        model = NeuroHybrid(small_config)
        defer, diag = model.should_defer_to_frontier(expression=expression_batch)
        assert defer.shape == (2,)
        assert defer.dtype == torch.bool
        assert "confidence" in diag

    def test_high_threshold_always_defers(self, small_config, expression_batch):
        small_config.confidence_threshold = 1.0
        model = NeuroHybrid(small_config)
        defer, _ = model.should_defer_to_frontier(expression=expression_batch)
        assert defer.all()

    def test_zero_threshold_never_defers(self, small_config, expression_batch):
        small_config.confidence_threshold = 0.0
        model = NeuroHybrid(small_config)
        defer, _ = model.should_defer_to_frontier(expression=expression_batch)
        assert not defer.any()


class TestNeuroHybridPerturbation:
    def test_perturbation_loss(self, small_config, expression_batch):
        model = NeuroHybrid(small_config)
        B = 2
        target = torch.randn(B, small_config.n_genes)
        ptype_ids = torch.zeros(B, dtype=torch.long)
        gene_idx = torch.zeros(B, dtype=torch.long)
        drug_idx = torch.zeros(B, dtype=torch.long)
        dose = torch.ones(B)
        duration = torch.ones(B)
        time_point = torch.ones(B)

        loss = model.compute_perturbation_loss(
            expression_batch, target, ptype_ids, gene_idx,
            drug_idx, dose, duration, time_point,
        )
        assert "total" in loss
        assert "reconstruction" in loss
        assert "kl" in loss
        assert loss["total"].shape == ()
        assert loss["total"].item() > 0


class TestNeuroHybridProperties:
    def test_param_count(self, small_config):
        model = NeuroHybrid(small_config)
        assert model.num_parameters > 0
        assert model.num_trainable == model.num_parameters


# ---------------------------------------------------------------------------
# UnifiedFoundationFrontierModel tests
# ---------------------------------------------------------------------------

class TestUnifiedForward:
    def test_forward_expression_only(self, small_unified_config, expression_batch):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        result = model(expression=expression_batch)
        d = small_unified_config.hybrid_config.hidden_dim
        assert result["cls_repr"].shape == (2, d)
        assert result["frontier_projection"].shape == (2, small_unified_config.projection_dim)
        assert result["frontier_confidence"].shape == (2,)

    def test_forward_clinical_only(self, small_unified_config, clinical_batch):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        result = model(clinical_dict=clinical_batch)
        assert "frontier_projection" in result
        assert "frontier_confidence" in result

    def test_forward_both(self, small_unified_config, clinical_batch, expression_batch):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        result = model(clinical_dict=clinical_batch, expression=expression_batch)
        assert result["cls_repr"].shape[0] == 2
        assert result["expression_recon"].shape == (2, small_unified_config.hybrid_config.n_genes)


class TestUnifiedTrainingLoss:
    def test_training_loss_basic(self, small_unified_config, expression_batch):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        target = torch.randn_like(expression_batch)
        out = model.training_loss(expression_batch, target)
        assert "loss" in out
        assert "mse" in out
        assert out["loss"].shape == ()
        assert out["loss"].item() > 0

    def test_training_loss_with_clinical(self, small_unified_config, clinical_batch, expression_batch):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        target = torch.randn_like(expression_batch)
        out = model.training_loss(expression_batch, target, clinical_dict=clinical_batch)
        assert out["loss"].item() > 0

    def test_training_loss_requires_expression(self, small_unified_config, clinical_batch):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        target = torch.randn(2, small_unified_config.hybrid_config.n_genes)
        with pytest.raises(ValueError, match="Expression input required"):
            model.training_loss(
                expression=None,
                target_expression=target,
                clinical_dict=clinical_batch,
            )

    def test_gradient_flows(self, small_unified_config, expression_batch):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        target = torch.randn_like(expression_batch)
        out = model.training_loss(expression_batch, target)
        out["loss"].backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


class TestUnifiedGovernedInference:
    def _mock_frontier(self, prompt: str) -> dict:
        return {
            "reasoning": "Test reasoning from frontier model",
            "confidence": 0.85,
            "evidence_citations": ["DOI:10.1234/test"],
            "suggested_experiments": ["Validate in cell line"],
            "caveats": ["Limited to in silico prediction"],
        }

    def test_governed_inference_no_frontier(self, small_unified_config, gene_names_100):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        expr = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        answer = model.governed_inference(
            expression=expr,
            gene_names=gene_names_100,
            species="human",
            regime="data-driven",
            hypothesis="Test hypothesis",
        )
        assert isinstance(answer, GovernedHybridAnswer)
        assert answer.species == "human"
        assert answer.foundation_prediction["deferred_to_frontier"] is False

    def test_governed_inference_with_frontier(self, small_unified_config, gene_names_100):
        small_unified_config.mode = HybridMode.AUGMENTED
        model = UnifiedFoundationFrontierModel(small_unified_config)
        expr = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        answer = model.governed_inference(
            expression=expr,
            gene_names=gene_names_100,
            species="human",
            regime="data-driven",
            hypothesis="Test hypothesis",
            frontier_call=self._mock_frontier,
        )
        assert isinstance(answer, GovernedHybridAnswer)
        assert answer.foundation_prediction["deferred_to_frontier"] is True
        assert answer.frontier_reasoning is not None

    def test_governed_mode_blocks_confident_frontier(self, small_unified_config, gene_names_100):
        """In GOVERNED mode, high-confidence predictions should NOT defer."""
        small_unified_config.mode = HybridMode.GOVERNED
        small_unified_config.auto_defer_threshold = 0.0
        model = UnifiedFoundationFrontierModel(small_unified_config)
        expr = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        answer = model.governed_inference(
            expression=expr,
            gene_names=gene_names_100,
            species="human",
            regime="data-driven",
            hypothesis="Test hypothesis",
            frontier_call=self._mock_frontier,
        )
        assert answer.foundation_prediction["deferred_to_frontier"] is False

    def test_governed_inference_species_firewall(self, small_unified_config, gene_names_100):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        expr = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        answer = model.governed_inference(
            expression=expr,
            gene_names=gene_names_100,
            species="mouse",
            regime="data-driven",
            hypothesis="Test hypothesis",
        )
        assert answer.species == "mouse"

    def test_governed_inference_has_lineage(self, small_unified_config, gene_names_100):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        expr = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        answer = model.governed_inference(
            expression=expr,
            gene_names=gene_names_100,
            species="human",
            regime="data-driven",
            hypothesis="Test hypothesis",
        )
        assert answer.content_hash != ""
        assert len(answer.content_hash) == 64

    def test_governed_inference_with_clinical(self, small_unified_config, clinical_batch, gene_names_100):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        expr = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        clinical_single = {k: v[:1] for k, v in clinical_batch.items()}
        answer = model.governed_inference(
            expression=expr,
            gene_names=gene_names_100,
            species="human",
            regime="data-driven",
            hypothesis="Test hypothesis",
            clinical_dict=clinical_single,
        )
        assert isinstance(answer, GovernedHybridAnswer)
        assert "confidence" in answer.foundation_prediction

    def test_governed_inference_expression_delta(self, small_unified_config, gene_names_100):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        expr = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        answer = model.governed_inference(
            expression=expr,
            gene_names=gene_names_100,
            species="human",
            regime="data-driven",
            hypothesis="Test hypothesis",
        )
        pred = answer.foundation_prediction
        assert "expression_delta" in pred
        assert "predicted_expression" in pred
        assert "disease_logits" in pred


class TestUnifiedParamCount:
    def test_param_count_structure(self, small_unified_config):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        pc = model.param_count()
        assert pc["backbone"] > 0
        assert pc["projector"] > 0
        assert pc["total"] == pc["backbone"] + pc["projector"]


class TestUnifiedDeterminism:
    def test_deterministic_forward(self, small_unified_config, expression_batch):
        torch.manual_seed(42)
        model = UnifiedFoundationFrontierModel(small_unified_config)
        model.eval()
        with torch.no_grad():
            r1 = model(expression=expression_batch)
            r2 = model(expression=expression_batch)
        assert torch.allclose(r1["cls_repr"], r2["cls_repr"])
        assert torch.allclose(r1["frontier_confidence"], r2["frontier_confidence"])


class TestUnifiedLineageIntegrity:
    def test_two_runs_different_hashes(self, small_unified_config, gene_names_100):
        model = UnifiedFoundationFrontierModel(small_unified_config)
        e1 = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        e2 = torch.randn(1, small_unified_config.hybrid_config.n_genes)
        a1 = model.governed_inference(e1, gene_names_100, "human", "data-driven", "H1")
        a2 = model.governed_inference(e2, gene_names_100, "human", "data-driven", "H2")
        assert a1.content_hash != a2.content_hash
