"""Bridge to merlinaineuralplus2's evidence engine and GRN/GPCR modules.

Merlin's backend exposes:
  - 170 routes on FastAPI port 8100
  - GRNOntologyEpistemologyOS (sqlite-backed ontology)
  - GRN engines: ARACNE, GENIE3, generate_ontology
  - GPCR cascade, quiescence, bias computation
  - SOM regime classification
  - Evidence deposit/replay (grn_source_prose.py)
  - Data registry (backend/data/registry.json)

This bridge provides:
  1. Type translation between NeuroTunesAI Problem/Answer and Merlin's formats
  2. Evidence chain integration (deposit solver results into Merlin's lineage)
  3. GRN engine result consumption (feed Merlin's GRN outputs into our solvers)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MerlinGRNResult:
    """Merlin GRN engine output envelope.

    Maps from Merlin's backend/methods/grn.py GRNEngine result format
    to NeuroTunesAI's tensor-based representation.
    """
    edges: list[tuple[str, str, float]]
    method: str
    n_genes: int
    n_edges: int
    ontology_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MerlinEvidenceDeposit:
    """Deposit envelope for Merlin's evidence system.

    Maps from NeuroTunesAI Answer to the format expected by
    merlinaineuralplus2/backend/methods/grn_source_prose.py deposit().
    """
    study_id: str
    problem_type: str
    input_hash: str
    output_hash: str
    species: str
    regime: str
    engine_tag: str
    result_summary: dict[str, Any] = field(default_factory=dict)


class MerlinBridge:
    """Translates between NeuroTunesAI and Merlin's evidence engine.

    Merlin's evidence discipline (CONVENTIONS.md):
      - Run the control in the same run as the result
      - Declare the regime before choosing the method
      - Facts in JSON, pages from builders, gates on drift
      - Data by id (data_registry.path_for)
      - scripts/replicate_records.py must pass before a record is quoted
      - Pre-register the criterion, then run

    Engine portability contract (CONVENTIONS.md):
      1. Self-contained (numerics only)
      2. Pure & deterministic
      3. Frozen, hashable config
      4. Closed-form claims are testable
      5. Docstring = spec
      6. Self-describing results
    """

    @staticmethod
    def grn_result_to_tensor(result: MerlinGRNResult, gene_index: dict[str, int]) -> torch.Tensor:
        """Convert Merlin GRN edge list to adjacency matrix tensor."""
        n = len(gene_index)
        adj = torch.zeros(n, n)
        for src, tgt, weight in result.edges:
            if src in gene_index and tgt in gene_index:
                adj[gene_index[src], gene_index[tgt]] = weight
        return adj

    @staticmethod
    def tensor_to_grn_edges(
        adj: torch.Tensor,
        gene_names: list[str],
        threshold: float = 0.01,
    ) -> list[tuple[str, str, float]]:
        """Convert adjacency tensor back to Merlin-format edge list."""
        edges = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                w = adj[i, j].item()
                if abs(w) > threshold:
                    edges.append((gene_names[i], gene_names[j], w))
        return sorted(edges, key=lambda e: -abs(e[2]))

    @staticmethod
    def answer_to_deposit(
        answer: Any,
        study_id: str,
        input_hash: str,
        output_hash: str,
        species: str,
        regime: str,
    ) -> MerlinEvidenceDeposit:
        """Package a NeuroTunesAI Answer for deposit into Merlin's evidence system."""
        summary = {}
        if hasattr(answer, "result") and isinstance(answer.result, dict):
            for k, v in answer.result.items():
                if isinstance(v, torch.Tensor):
                    summary[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                else:
                    summary[k] = str(type(v).__name__)

        return MerlinEvidenceDeposit(
            study_id=study_id,
            problem_type=getattr(answer, "problem_type", "unknown"),
            input_hash=input_hash,
            output_hash=output_hash,
            species=species,
            regime=regime,
            engine_tag="neurotunes_solver",
            result_summary=summary,
        )

    @staticmethod
    def data_registry_entry(
        dataset_id: str,
        path: str,
        species: str,
        kind: str = "expression",
        licence: str = "",
        studies: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a Merlin-compatible data registry entry.

        Format matches merlinaineuralplus2/backend/data/registry.json schema.
        """
        return {
            "kind": kind,
            "path": path,
            "species": species,
            "species_source": {"source": f"declared for {dataset_id}"},
            "licence": licence,
            "product_use_class": "verify",
            "studies": studies or [],
            "consumers": [],
        }
