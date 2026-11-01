"""RTP bridge — maps NeuroTunesAI's RTP solver to Merlin's patent kernel contract.

Merlin's RTP kernel (lotwhitelabelnt/backend/app/patents/rtp/kernel.py) operates on:
  EntitySeries: entity_ids, values (T x N float), locations, domain, provenance
  RTPRunManifest: list of ElementTrace per claim element

This bridge translates between NeuroTunesAI's Problem→Answer interface and
the EntitySeries contract, enabling cross-repo identity verification via
golden.json vectors.

Patent claim coverage:
  US 12,361,263 B1 claim 1: inference (D/C/E → simulate → eliminate → test → identify)
  US 12,651,165 B2 claim 1: training (SOM → train GRN node models)
  US 11,694,786 B1: media selection (separate estate, in media_kernel.py)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.math_core.grn_solver import GRNSolver, GRNNode, GRNEdge
from models.math_core.rtp_solver import (
    dominance_factor, evolution_of_internal_component,
    simulate_relationships, eliminate_drivers, test_drivers, identify_target,
)


FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
FNV_MASK = (1 << 64) - 1


def fnv1a64(data: str) -> int:
    """FNV-1a 64-bit hash — must match Merlin's contract.py for cross-stack identity."""
    h = FNV_OFFSET
    for byte in data.encode("utf-8"):
        h ^= byte
        h = (h * FNV_PRIME) & FNV_MASK
    return h


@dataclass
class EntitySeries:
    """Cross-repo data contract for RTP kernel input.

    This matches merlinaineuralplus2/backend/app/patents/rtp/contract.py
    exactly for cross-stack identity verification.
    """
    entity_ids: list[str]
    values: np.ndarray  # shape (T, N)
    locations: list[str] | None = None
    domain: str = "neurotunes"
    provenance: str = ""

    @property
    def n_entities(self) -> int:
        return len(self.entity_ids)

    @property
    def n_steps(self) -> int:
        return self.values.shape[0]

    def content_hash(self) -> str:
        serialized = np.round(self.values, 6).tobytes()
        ids_str = ",".join(self.entity_ids)
        payload = ids_str.encode() + b"|" + serialized
        return hashlib.sha256(payload).hexdigest()


@dataclass
class ElementTrace:
    """Trace of one patent claim element execution."""
    claim: str
    element: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    module: str = "models.evidence.rtp_bridge"


@dataclass
class RTPRunManifest:
    """Complete manifest of an RTP kernel run."""
    traces: list[ElementTrace] = field(default_factory=list)
    input_hash: str = ""
    claim_coverage: list[str] = field(default_factory=list)

    def add(self, trace: ElementTrace):
        self.traces.append(trace)
        if trace.claim not in self.claim_coverage:
            self.claim_coverage.append(trace.claim)


def run_263_bridge(series: EntitySeries) -> tuple[dict[str, Any], RTPRunManifest]:
    """Execute US 12,361,263 B1 claim 1 via NeuroTunesAI's RTP solver.

    Steps: dominance → evolution → simulate → eliminate → test → identify.
    Returns the result dict and a manifest of claim element traces.
    """
    manifest = RTPRunManifest(input_hash=series.content_hash())
    claim = "US12361263B1/1"

    d_results = {}
    for j, eid in enumerate(series.entity_ids):
        col = series.values[:, j]
        d_results[eid] = dominance_factor(col)

    manifest.add(ElementTrace(
        claim=claim, element="dominance",
        inputs={"n_entities": series.n_entities, "n_steps": series.n_steps},
        outputs={"d_scores": {k: v["dominance"] for k, v in d_results.items()}},
    ))

    e_results = {}
    for j, eid in enumerate(series.entity_ids):
        col = series.values[:, j]
        e_results[eid] = evolution_of_internal_component(col)

    manifest.add(ElementTrace(
        claim=claim, element="evolution",
        inputs={"n_entities": series.n_entities},
        outputs={"e_final": {k: float(v["final"])
                             for k, v in e_results.items()}},
    ))

    sim_matrix = simulate_relationships(series.values)
    manifest.add(ElementTrace(
        claim=claim, element="simulate",
        inputs={"n_entities": series.n_entities},
        outputs={"sim_shape": list(sim_matrix.shape)},
    ))

    target_idx = 0
    survivors = eliminate_drivers(sim_matrix, target_idx=target_idx)
    n_total = series.n_entities
    eliminated = [i for i in range(n_total) if i != target_idx and i not in survivors]

    manifest.add(ElementTrace(
        claim=claim, element="eliminate",
        inputs={"target_idx": target_idx, "quantile": 0.67},
        outputs={"n_survivors": len(survivors), "n_eliminated": len(eliminated)},
    ))

    tested = test_drivers(series.values, target_idx=target_idx, kept_indices=survivors)
    manifest.add(ElementTrace(
        claim=claim, element="test",
        inputs={"n_survivors": len(survivors)},
        outputs={"n_passing": len(tested)},
    ))

    target = identify_target(tested, sim_matrix, target_idx)
    manifest.add(ElementTrace(
        claim=claim, element="identify",
        inputs={"n_tested": len(tested)},
        outputs={"target_entity_idx": target},
    ))

    return {
        "d_results": d_results,
        "e_results": e_results,
        "sim_matrix": sim_matrix,
        "survivors": survivors,
        "eliminated": eliminated,
        "tested": tested,
        "target": target,
    }, manifest
