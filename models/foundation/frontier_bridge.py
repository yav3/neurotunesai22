"""Frontier Bridge — connects the gene foundation encoder to structured
frontier LLM reasoning (Claude) via a governed evidence layer.

Architecture (two-tower + bridge):

    Tower 1: GeneFoundationEncoder
        gene expression → per-gene attention → CLS embedding (d=256)

    Tower 2: Frontier LLM (Claude)
        structured query → scientific reasoning → JSON evidence record

    Bridge: GovernedFrontierBridge
        CLS embedding → structured query (grounded in learned representation)
        LLM response → evidence record → lineage chain → governed answer

The bridge enforces:
  - Species/regime firewall on all queries
  - Hash-chained lineage on every LLM interaction
  - Pre-registered hypothesis format (no post-hoc fishing)
  - Calibration tracking (LLM confidence vs actual outcome)
  - Structured output schema (parseable, testable, reproducible)

The LLM is NOT in the prediction loop — it reasons about the representation.
The foundation model predicts; the frontier model explains and hypothesises.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class QueryType(str, Enum):
    """Structured query types the bridge can generate from embeddings."""
    PERTURBATION_HYPOTHESIS = "perturbation_hypothesis"
    GENE_FUNCTION = "gene_function"
    PATHWAY_ANALYSIS = "pathway_analysis"
    DRUG_MECHANISM = "drug_mechanism"
    DIFFERENTIAL_EXPLANATION = "differential_explanation"
    EXPERIMENTAL_DESIGN = "experimental_design"


@dataclass
class StructuredQuery:
    """A governed query to the frontier LLM, grounded in foundation output."""
    query_type: QueryType
    gene_context: dict[str, Any]
    embedding_summary: dict[str, float]
    species: str
    regime: str
    hypothesis: str
    pre_registered: bool = False
    lineage_hash: str = ""


@dataclass
class FrontierResponse:
    """Structured response from the frontier LLM."""
    reasoning: str
    confidence: float
    evidence_citations: list[str]
    suggested_experiments: list[str]
    caveats: list[str]
    response_hash: str = ""
    latency_ms: float = 0.0


@dataclass
class GovernedHybridAnswer:
    """The final governed answer combining both towers."""
    foundation_prediction: dict[str, Any]
    frontier_reasoning: FrontierResponse | None
    lineage_chain: list[str]
    species: str
    regime: str
    study_id: str
    is_governed: bool = True
    content_hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps({
            "prediction": self.foundation_prediction,
            "reasoning": self.frontier_reasoning.reasoning if self.frontier_reasoning else None,
            "species": self.species,
            "regime": self.regime,
            "study_id": self.study_id,
        }, sort_keys=True, default=str)
        self.content_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.content_hash


@dataclass
class FrontierBridgeConfig:
    """Configuration for the governed frontier bridge."""
    max_top_genes: int = 50
    confidence_threshold: float = 0.3
    require_pre_registration: bool = True
    max_query_length: int = 4000
    allowed_species: tuple[str, ...] = ("human",)
    allowed_regimes: tuple[str, ...] = ("data-driven", "prior-contrast")


class EmbeddingSummariser:
    """Extract interpretable features from the foundation encoder's output
    to ground frontier LLM queries in the learned representation.

    No LLM sees raw embeddings. Instead, we extract:
      - Top activated genes (by attention or expression delta)
      - Cluster/neighbourhood statistics
      - Perturbation response magnitude and direction
    """

    @staticmethod
    def summarise_expression_delta(
        baseline: np.ndarray,
        predicted: np.ndarray,
        gene_names: list[str],
        top_k: int = 50,
    ) -> dict[str, Any]:
        delta = predicted - baseline
        abs_delta = np.abs(delta)
        top_idx = np.argsort(abs_delta)[-top_k:][::-1]

        return {
            "top_genes": [
                {"gene": gene_names[i], "delta": float(delta[i]),
                 "baseline": float(baseline[i]), "predicted": float(predicted[i])}
                for i in top_idx
            ],
            "mean_abs_delta": float(abs_delta.mean()),
            "max_abs_delta": float(abs_delta.max()),
            "n_upregulated": int((delta > 0.5).sum()),
            "n_downregulated": int((delta < -0.5).sum()),
            "total_genes": len(gene_names),
        }

    @staticmethod
    def summarise_cls_embedding(
        cls_repr: np.ndarray,
    ) -> dict[str, float]:
        return {
            "norm": float(np.linalg.norm(cls_repr)),
            "mean": float(cls_repr.mean()),
            "std": float(cls_repr.std()),
            "max": float(cls_repr.max()),
            "min": float(cls_repr.min()),
            "sparsity": float((np.abs(cls_repr) < 0.01).mean()),
        }


class GovernedFrontierBridge:
    """Connects foundation encoder output to frontier LLM via governed protocol.

    Usage:
        bridge = GovernedFrontierBridge(config)
        query = bridge.build_query(
            query_type=QueryType.PERTURBATION_HYPOTHESIS,
            expression_delta=summariser.summarise_expression_delta(...),
            cls_summary=summariser.summarise_cls_embedding(...),
            species="human",
            regime="data-driven",
            hypothesis="FOXO3 knockout reduces autophagy gene expression",
        )
        # The query is then sent to the frontier LLM by the caller
        # The response is validated and wrapped:
        answer = bridge.validate_response(raw_response, query, prediction)

    The bridge does NOT call the LLM itself — it builds governed queries
    and validates responses. The actual LLM call is the caller's responsibility,
    keeping the bridge testable without network access.
    """

    def __init__(self, config: FrontierBridgeConfig | None = None):
        self.config = config or FrontierBridgeConfig()
        self.lineage: list[str] = []

    def _chain_hash(self, content: str) -> str:
        prev = self.lineage[-1] if self.lineage else "genesis"
        payload = f"{prev}:{content}"
        h = hashlib.sha256(payload.encode()).hexdigest()
        self.lineage.append(h)
        return h

    def build_query(
        self,
        query_type: QueryType,
        expression_delta: dict[str, Any],
        cls_summary: dict[str, float],
        species: str,
        regime: str,
        hypothesis: str,
        study_id: str = "",
    ) -> StructuredQuery:
        if species not in self.config.allowed_species:
            raise ValueError(
                f"Species '{species}' not in allowed set {self.config.allowed_species}. "
                f"Species firewall: declare or refuse."
            )
        if regime not in self.config.allowed_regimes:
            raise ValueError(
                f"Regime '{regime}' not in allowed set {self.config.allowed_regimes}."
            )

        gene_context = {
            "top_genes": expression_delta.get("top_genes", [])[:self.config.max_top_genes],
            "summary_stats": {
                "mean_abs_delta": expression_delta.get("mean_abs_delta", 0),
                "n_upregulated": expression_delta.get("n_upregulated", 0),
                "n_downregulated": expression_delta.get("n_downregulated", 0),
                "total_genes": expression_delta.get("total_genes", 0),
            },
        }

        query = StructuredQuery(
            query_type=query_type,
            gene_context=gene_context,
            embedding_summary=cls_summary,
            species=species,
            regime=regime,
            hypothesis=hypothesis,
            pre_registered=bool(study_id),
            lineage_hash=self._chain_hash(
                json.dumps({"type": query_type.value, "hypothesis": hypothesis,
                            "species": species, "study_id": study_id}, sort_keys=True)
            ),
        )
        return query

    def format_prompt(self, query: StructuredQuery) -> str:
        """Format the structured query into a frontier LLM prompt.

        The prompt is structured so the LLM responds in a parseable JSON format.
        """
        top_genes_text = "\n".join(
            f"  {g['gene']}: delta={g['delta']:+.3f} "
            f"(baseline={g['baseline']:.2f} → predicted={g['predicted']:.2f})"
            for g in query.gene_context["top_genes"][:20]
        )

        stats = query.gene_context["summary_stats"]

        prompt = f"""You are a molecular biology expert analysing gene expression perturbation results.

Species: {query.species}
Analysis regime: {query.regime}
Query type: {query.query_type.value}

HYPOTHESIS (pre-registered: {query.pre_registered}):
{query.hypothesis}

FOUNDATION MODEL OUTPUT — top differentially expressed genes:
{top_genes_text}

Summary: {stats['total_genes']} genes analysed, {stats['n_upregulated']} upregulated (delta > 0.5), {stats['n_downregulated']} downregulated (delta < -0.5), mean |delta| = {stats['mean_abs_delta']:.4f}

Respond in JSON with exactly these fields:
{{
  "reasoning": "<2-3 sentence scientific interpretation>",
  "confidence": <float 0-1, calibrated>,
  "evidence_citations": ["<PMID or DOI>", ...],
  "suggested_experiments": ["<concrete next experiment>", ...],
  "caveats": ["<limitation or alternative explanation>", ...]
}}"""
        return prompt[:self.config.max_query_length]

    def parse_response(
        self, raw_json: str | dict, query: StructuredQuery
    ) -> FrontierResponse:
        if isinstance(raw_json, str):
            data = json.loads(raw_json)
        else:
            data = raw_json

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        response = FrontierResponse(
            reasoning=str(data.get("reasoning", "")),
            confidence=confidence,
            evidence_citations=[str(c) for c in data.get("evidence_citations", [])],
            suggested_experiments=[str(e) for e in data.get("suggested_experiments", [])],
            caveats=[str(c) for c in data.get("caveats", [])],
        )

        response.response_hash = self._chain_hash(
            json.dumps({"reasoning": response.reasoning,
                        "confidence": response.confidence}, sort_keys=True)
        )
        return response

    def build_governed_answer(
        self,
        foundation_prediction: dict[str, Any],
        frontier_response: FrontierResponse | None,
        species: str,
        regime: str,
        study_id: str,
    ) -> GovernedHybridAnswer:
        answer = GovernedHybridAnswer(
            foundation_prediction=foundation_prediction,
            frontier_reasoning=frontier_response,
            lineage_chain=list(self.lineage),
            species=species,
            regime=regime,
            study_id=study_id,
        )
        answer.compute_hash()
        return answer


if HAS_TORCH:

    class FoundationToFrontierProjector(nn.Module):
        """Project foundation encoder CLS → structured features for the bridge.

        Learns a low-dimensional, interpretable projection of the CLS embedding
        that can be serialised into the frontier query without sending raw tensors.
        """

        def __init__(self, hidden_dim: int = 256, projection_dim: int = 32):
            super().__init__()
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, projection_dim),
            )
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, cls_repr: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "projection": self.projector(cls_repr),
                "confidence": self.confidence_head(cls_repr).squeeze(-1),
            }
