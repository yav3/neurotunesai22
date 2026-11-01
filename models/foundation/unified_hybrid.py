"""Unified Foundational-Frontier Hybrid Model.

Composes the two existing architectures into a single governed pipeline:

    NeuroHybrid (Tower 1)
        clinical modalities + gene expression
        → CrossModalTransformer → CLS repr (d=256)
        → confidence gate, disease classifier, perturbation cVAE

    GovernedFrontierBridge (Tower 2 interface)
        CLS repr → EmbeddingSummariser → StructuredQuery
        → frontier LLM call → FrontierResponse
        → GovernedHybridAnswer (hash-chained, species-firewalled)

    UnifiedFoundationFrontierModel
        encode(clinical + expression) → CLS
        if confidence < threshold OR frontier_call provided:
            CLS → frontier bridge → governed answer
        else:
            foundation-only prediction

The unified model does NOT duplicate any module — it holds references
to NeuroHybrid and GovernedFrontierBridge, composing their interfaces.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.foundation.frontier_bridge import (
    EmbeddingSummariser,
    FrontierBridgeConfig,
    FrontierResponse,
    GovernedFrontierBridge,
    GovernedHybridAnswer,
    QueryType,
    FoundationToFrontierProjector,
)
from models.foundation.hybrid import (
    HybridConfig,
    HybridMode,
    NeuroHybrid,
)


@dataclass
class UnifiedHybridConfig:
    """Configuration for the unified foundational-frontier model."""
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)
    bridge_config: FrontierBridgeConfig = field(default_factory=FrontierBridgeConfig)
    projection_dim: int = 32
    auto_defer_threshold: float = 0.8
    mode: HybridMode = HybridMode.GOVERNED


class UnifiedFoundationFrontierModel(nn.Module):
    """Unified foundational-frontier hybrid model.

    Composes NeuroHybrid (multimodal backbone) with GovernedFrontierBridge
    (governed LLM reasoning) via a confidence-gated routing layer.

    Forward pass (foundation-only, differentiable):
        clinical_dict + expression → CLS + disease_logits + confidence

    Governed inference (non-differentiable frontier call):
        1. Encode → CLS, confidence, expression_recon
        2. Summarise expression delta + CLS embedding
        3. If confidence < threshold: build governed query → call frontier
        4. Return GovernedHybridAnswer with hash-chained lineage
    """

    def __init__(self, config: UnifiedHybridConfig | None = None):
        super().__init__()
        self.config = config or UnifiedHybridConfig()

        self.backbone = NeuroHybrid(self.config.hybrid_config)

        self.projector = FoundationToFrontierProjector(
            self.config.hybrid_config.hidden_dim,
            self.config.projection_dim,
        )

        self.bridge = GovernedFrontierBridge(self.config.bridge_config)
        self.summariser = EmbeddingSummariser()

    def forward(
        self,
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
        expression: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Foundation-only forward pass (differentiable, for training).

        Returns CLS repr, disease logits, confidence, and optional
        expression reconstruction + frontier projection.
        """
        result = self.backbone(clinical_dict, mask_dict, expression)

        proj = self.projector(result["cls_repr"])
        result["frontier_projection"] = proj["projection"]
        result["frontier_confidence"] = proj["confidence"]

        return result

    def training_loss(
        self,
        expression: torch.Tensor,
        target_expression: torch.Tensor,
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss: expression reconstruction MSE."""
        result = self.forward(clinical_dict, mask_dict, expression)

        if "expression_recon" not in result:
            raise ValueError("Expression input required for training loss")

        recon = result["expression_recon"]
        loss = F.mse_loss(recon, target_expression)

        return {"loss": loss, "mse": loss.detach(), **result}

    @torch.no_grad()
    def governed_inference(
        self,
        expression: torch.Tensor,
        gene_names: list[str],
        species: str,
        regime: str,
        hypothesis: str,
        study_id: str = "",
        clinical_dict: dict[str, torch.Tensor] | None = None,
        mask_dict: dict[str, torch.Tensor] | None = None,
        frontier_call: Callable[[str], str | dict] | None = None,
        query_type: QueryType = QueryType.PERTURBATION_HYPOTHESIS,
    ) -> GovernedHybridAnswer:
        """Full governed inference: encode → summarise → optionally query frontier.

        Args:
            expression: (1, n_genes) baseline expression
            gene_names: gene symbol list
            species: "human" (enforced by firewall)
            regime: "data-driven" or "prior-contrast"
            hypothesis: pre-registered hypothesis string
            study_id: study registry ID
            clinical_dict: optional clinical modality tensors
            mask_dict: optional modality masks
            frontier_call: optional callable(prompt) → JSON response
            query_type: type of frontier query

        Returns:
            GovernedHybridAnswer with prediction + optional frontier reasoning
        """
        self.eval()
        result = self.forward(clinical_dict, mask_dict, expression)

        cls_np = result["cls_repr"][0].cpu().numpy()
        confidence = float(result["confidence"][0].cpu())

        foundation_prediction: dict[str, Any] = {
            "cls_summary": self.summariser.summarise_cls_embedding(cls_np),
            "confidence": confidence,
            "disease_logits": result["disease_logits"][0].cpu().tolist(),
        }

        if "expression_recon" in result:
            baseline_np = expression[0].cpu().numpy()
            recon_np = result["expression_recon"][0].cpu().numpy()
            n_genes = min(len(gene_names), len(recon_np))
            expr_delta = self.summariser.summarise_expression_delta(
                baseline_np[:n_genes], recon_np[:n_genes],
                gene_names[:n_genes],
            )
            foundation_prediction["expression_delta"] = expr_delta
            foundation_prediction["predicted_expression"] = recon_np[:n_genes].tolist()

        if "frontier_confidence" in result:
            foundation_prediction["frontier_confidence"] = float(
                result["frontier_confidence"][0].cpu()
            )

        should_defer = confidence < self.config.auto_defer_threshold
        frontier_response = None

        if frontier_call is not None:
            if self.config.mode == HybridMode.GOVERNED and not should_defer:
                pass
            else:
                expr_delta = foundation_prediction.get("expression_delta", {
                    "top_genes": [], "mean_abs_delta": 0,
                    "n_upregulated": 0, "n_downregulated": 0,
                    "total_genes": len(gene_names),
                })
                cls_summary = foundation_prediction["cls_summary"]

                query = self.bridge.build_query(
                    query_type=query_type,
                    expression_delta=expr_delta,
                    cls_summary=cls_summary,
                    species=species,
                    regime=regime,
                    hypothesis=hypothesis,
                    study_id=study_id,
                )
                prompt = self.bridge.format_prompt(query)
                raw = frontier_call(prompt)
                frontier_response = self.bridge.parse_response(raw, query)

        foundation_prediction["deferred_to_frontier"] = frontier_response is not None

        return self.bridge.build_governed_answer(
            foundation_prediction=foundation_prediction,
            frontier_response=frontier_response,
            species=species,
            regime=regime,
            study_id=study_id,
        )

    def param_count(self) -> dict[str, int]:
        backbone = sum(p.numel() for p in self.backbone.parameters())
        proj = sum(p.numel() for p in self.projector.parameters())
        return {
            "backbone": backbone,
            "projector": proj,
            "total": backbone + proj,
        }
