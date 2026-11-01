"""Foundation–Frontier Hybrid Model.

Combines the gene-level foundation encoder (Tower 1) with the governed
frontier bridge (Tower 2 interface) into a single inference pipeline.

    expression + gene_ids
         │
         ▼
    GeneFoundationEncoder  ──→  CLS repr (d=256)
         │                           │
         ▼                           ▼
    ExpressionDecoder      FoundationToFrontierProjector
         │                           │
         ▼                           ▼
    predicted expression      StructuredQuery
         │                           │
         ▼                           ▼
    ┌─────────────────────────────────────┐
    │    GovernedFrontierBridge            │
    │    (species firewall, lineage,      │
    │     pre-registration, calibration)  │
    └─────────────────────────────────────┘
                    │
                    ▼
           GovernedHybridAnswer

The hybrid model owns the full pipeline from raw expression to governed answer.
The frontier LLM call is injected as a callable — the model builds the query
and validates the response, but does not hold network state.
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

from models.foundation.gene_foundation import (
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
    FoundationToFrontierProjector,
)


@dataclass
class HybridModelConfig:
    """Configuration for the full hybrid pipeline."""
    gene_config: GeneFoundationConfig = field(default_factory=GeneFoundationConfig)
    bridge_config: FrontierBridgeConfig = field(default_factory=FrontierBridgeConfig)
    prediction_hidden: int = 512
    prediction_dropout: float = 0.1
    use_frontier: bool = True
    projection_dim: int = 32


class ExpressionPredictor(nn.Module):
    """Predict post-perturbation expression from CLS + perturbation embedding.

    Takes the foundation encoder's CLS representation and a perturbation
    descriptor, outputs predicted expression for each gene.
    """

    def __init__(self, hidden_dim: int, n_genes: int,
                 pred_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, n_genes),
        )

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        return self.predictor(cls_repr)


class HybridFoundationModel(nn.Module):
    """The complete foundation–frontier hybrid model.

    Forward pass:
        1. Encode expression → CLS repr + per-gene repr
        2. Predict post-perturbation expression
        3. (Optional) Project CLS → frontier query features
        4. Build governed answer

    The model is trainable end-to-end on the foundation side.
    The frontier bridge is non-differentiable (LLM reasoning).
    """

    def __init__(self, config: HybridModelConfig | None = None):
        super().__init__()
        self.config = config or HybridModelConfig()
        c = self.config

        self.encoder = GeneFoundationEncoder(c.gene_config)

        self.predictor = ExpressionPredictor(
            c.gene_config.hidden_dim,
            c.gene_config.max_genes,
            c.prediction_hidden,
            c.prediction_dropout,
        )

        if c.use_frontier:
            self.projector = FoundationToFrontierProjector(
                c.gene_config.hidden_dim,
                c.projection_dim,
            )

        self.bridge = GovernedFrontierBridge(c.bridge_config)
        self.summariser = EmbeddingSummariser()

    def forward(
        self,
        expression: torch.Tensor,
        gene_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Foundation-only forward pass (differentiable, for training).

        Returns CLS repr, per-gene repr, and predicted expression.
        """
        cls_repr, seq_out = self.encoder(expression, gene_indices, padding_mask)
        pred_expr = self.predictor(cls_repr)

        result = {
            "cls_repr": cls_repr,
            "seq_out": seq_out,
            "pred_expression": pred_expr,
        }

        if self.config.use_frontier and hasattr(self, "projector"):
            proj = self.projector(cls_repr)
            result["frontier_projection"] = proj["projection"]
            result["foundation_confidence"] = proj["confidence"]

        return result

    def training_loss(
        self,
        expression: torch.Tensor,
        gene_indices: torch.Tensor,
        target_expression: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss: MSE on predicted expression."""
        result = self.forward(expression, gene_indices, padding_mask)
        pred = result["pred_expression"]

        if padding_mask is not None:
            mask_2d = padding_mask.float()
            loss = (F.mse_loss(pred, target_expression, reduction="none") * mask_2d).sum()
            loss = loss / mask_2d.sum().clamp(min=1)
        else:
            loss = F.mse_loss(pred, target_expression)

        return {"loss": loss, "mse": loss.detach(), **result}

    @torch.no_grad()
    def predict_and_explain(
        self,
        expression: torch.Tensor,
        gene_indices: torch.Tensor,
        gene_names: list[str],
        species: str,
        regime: str,
        hypothesis: str,
        study_id: str = "",
        frontier_call: Callable[[str], str | dict] | None = None,
        query_type: QueryType = QueryType.PERTURBATION_HYPOTHESIS,
        padding_mask: torch.Tensor | None = None,
    ) -> GovernedHybridAnswer:
        """Full hybrid inference: predict + explain.

        Args:
            expression: (1, G) baseline expression (single sample)
            gene_indices: (1, G) or (G,) gene IDs
            gene_names: list of gene symbols
            species: "human" (enforced by firewall)
            regime: "data-driven" or "prior-contrast"
            hypothesis: pre-registered hypothesis string
            study_id: study registry ID (empty = not pre-registered)
            frontier_call: optional callable(prompt) → JSON response
            query_type: type of frontier query
            padding_mask: (1, G) bool mask

        Returns:
            GovernedHybridAnswer with prediction + optional reasoning
        """
        self.eval()
        result = self.forward(expression, gene_indices, padding_mask)

        baseline_np = expression[0].cpu().numpy()
        pred_np = result["pred_expression"][0].cpu().numpy()
        cls_np = result["cls_repr"][0].cpu().numpy()

        n_genes = min(len(gene_names), len(pred_np))
        expr_delta = self.summariser.summarise_expression_delta(
            baseline_np[:n_genes], pred_np[:n_genes],
            gene_names[:n_genes],
        )
        cls_summary = self.summariser.summarise_cls_embedding(cls_np)

        foundation_prediction = {
            "predicted_expression": pred_np[:n_genes].tolist(),
            "expression_delta": expr_delta,
            "cls_summary": cls_summary,
        }
        if "foundation_confidence" in result:
            foundation_prediction["confidence"] = float(
                result["foundation_confidence"][0].cpu()
            )

        frontier_response = None
        if frontier_call is not None and self.config.use_frontier:
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

        return self.bridge.build_governed_answer(
            foundation_prediction=foundation_prediction,
            frontier_response=frontier_response,
            species=species,
            regime=regime,
            study_id=study_id,
        )

    def param_count(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        pred = sum(p.numel() for p in self.predictor.parameters())
        proj = sum(p.numel() for p in self.projector.parameters()) if hasattr(self, "projector") else 0
        return {
            "encoder": enc,
            "predictor": pred,
            "projector": proj,
            "total": enc + pred + proj,
        }
