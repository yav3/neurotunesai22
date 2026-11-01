"""Hybrid foundation–frontier inference service.

Provides the clean entry point for governed inference after training:
    1. Load a trained checkpoint
    2. Build a frontier-enabled model
    3. Run predict_and_explain with a real or mock frontier caller
    4. Return a GovernedHybridAnswer with full lineage

Usage:
    service = HybridInferenceService.from_checkpoint(
        ckpt_path, gene_names=gene_names,
    )
    answer = service.infer(
        expression=expression_vector,
        species="human",
        hypothesis="FOXO3 upregulation increases autophagy",
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from models.foundation.gene_foundation import GeneFoundationConfig
from models.foundation.hybrid_model import HybridFoundationModel, HybridModelConfig
from models.foundation.frontier_bridge import GovernedHybridAnswer, QueryType
from models.foundation.frontier_caller import (
    ClaudeFrontierCaller,
    FrontierCallerConfig,
    MockFrontierCaller,
)
from models.training.checkpoint import load_checkpoint


@dataclass
class InferenceConfig:
    """Configuration for hybrid inference."""
    use_real_frontier: bool = False
    frontier_model: str = "claude-sonnet-4-5-20250514"
    frontier_api_key_env: str = "ANTHROPIC_API_KEY"
    default_regime: str = "data-driven"
    default_query_type: QueryType = QueryType.PERTURBATION_HYPOTHESIS


class HybridInferenceService:
    """Governed inference from a trained hybrid model.

    Loads a checkpoint, rebuilds the frontier-enabled model,
    and provides predict_and_explain with full lineage tracking.
    """

    def __init__(
        self,
        model: HybridFoundationModel,
        gene_names: list[str],
        frontier_caller: Callable[[str], str | dict],
        config: InferenceConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.model = model
        self.model.eval()
        self.gene_names = gene_names
        self.frontier_caller = frontier_caller
        self.config = config or InferenceConfig()
        self.metadata = metadata or {}

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        gene_names: list[str],
        gene_config: GeneFoundationConfig | None = None,
        model_config: HybridModelConfig | None = None,
        inference_config: InferenceConfig | None = None,
    ) -> "HybridInferenceService":
        """Load a trained checkpoint and build the inference service."""
        cfg = inference_config or InferenceConfig()
        n_genes = len(gene_names)

        if model_config is not None:
            mc = HybridModelConfig(
                gene_config=model_config.gene_config,
                bridge_config=model_config.bridge_config,
                prediction_hidden=model_config.prediction_hidden,
                prediction_dropout=model_config.prediction_dropout,
                use_frontier=True,
                projection_dim=model_config.projection_dim,
            )
        else:
            gc = gene_config or GeneFoundationConfig(max_genes=n_genes)
            mc = HybridModelConfig(gene_config=gc, use_frontier=True)

        model = HybridFoundationModel(mc)
        model, metadata = load_checkpoint(ckpt_path, model, verify=True)

        if cfg.use_real_frontier:
            caller = ClaudeFrontierCaller(
                FrontierCallerConfig(
                    model=cfg.frontier_model,
                    api_key_env=cfg.frontier_api_key_env,
                )
            )
        else:
            caller = MockFrontierCaller()

        return cls(
            model=model,
            gene_names=gene_names,
            frontier_caller=caller,
            config=cfg,
            metadata=metadata,
        )

    def infer(
        self,
        expression: np.ndarray | torch.Tensor,
        species: str = "human",
        regime: str | None = None,
        hypothesis: str = "",
        study_id: str = "",
        query_type: QueryType | None = None,
    ) -> GovernedHybridAnswer:
        """Run governed inference on a single expression sample.

        Args:
            expression: (G,) or (1, G) expression vector
            species: must be "human" (enforced by firewall)
            regime: "data-driven" or "prior-contrast"
            hypothesis: pre-registered hypothesis
            study_id: study registry ID
            query_type: type of frontier query

        Returns:
            GovernedHybridAnswer with foundation prediction + frontier reasoning
        """
        if isinstance(expression, np.ndarray):
            expression = torch.tensor(expression, dtype=torch.float32)
        if expression.dim() == 1:
            expression = expression.unsqueeze(0)

        n_genes = expression.shape[1]
        gene_ids = torch.arange(n_genes).unsqueeze(0)

        return self.model.predict_and_explain(
            expression=expression,
            gene_indices=gene_ids,
            gene_names=self.gene_names[:n_genes],
            species=species,
            regime=regime or self.config.default_regime,
            hypothesis=hypothesis or f"Inference from checkpoint {self.metadata.get('study_id', 'unknown')}",
            study_id=study_id,
            frontier_call=self.frontier_caller,
            query_type=query_type or self.config.default_query_type,
        )

    def batch_infer(
        self,
        expressions: np.ndarray | torch.Tensor,
        species: str = "human",
        regime: str | None = None,
        hypothesis: str = "",
        study_id: str = "",
    ) -> list[GovernedHybridAnswer]:
        """Run governed inference on multiple samples."""
        if isinstance(expressions, np.ndarray):
            expressions = torch.tensor(expressions, dtype=torch.float32)
        return [
            self.infer(
                expressions[i],
                species=species,
                regime=regime,
                hypothesis=hypothesis,
                study_id=f"{study_id}_sample_{i}" if study_id else f"batch_sample_{i}",
            )
            for i in range(len(expressions))
        ]
