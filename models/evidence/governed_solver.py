"""Governed solver — wraps UnifiedSolver with evidence discipline.

Every solve call is:
  1. Pre-checked against regime firewall
  2. Recorded in the lineage chain
  3. Optionally checked against a pre-registered study criterion

This is the "governed layer" the platform positions on: it makes solver
outputs admissible by attaching tamper-evident provenance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from models.interface.problem import Problem, ProblemType, Answer
from models.interface.solver import UnifiedSolver
from models.evidence.lineage import LineageChain, EvidenceRecord
from models.evidence.regime import (
    RegimeFirewall, SpeciesDeclaration, Species, Regime,
)
from models.evidence.registry import StudyRegistry


@dataclass
class GovernedAnswer:
    """Answer with full provenance chain."""
    answer: Answer
    evidence: EvidenceRecord
    species_declaration: SpeciesDeclaration
    chain_position: int
    chain_head_hash: str


class GovernedSolver:
    """Wraps UnifiedSolver with evidence discipline enforcement.

    Usage:
        solver = GovernedSolver()
        decl = SpeciesDeclaration(Species.SYNTHETIC, Regime.BENCHMARK)
        result = solver.solve(problem, declaration=decl)
        # result.evidence contains the tamper-evident record
        # result.chain_head_hash links to the full lineage
    """

    def __init__(
        self,
        solver: UnifiedSolver | None = None,
        registry: StudyRegistry | None = None,
    ):
        self._solver = solver or UnifiedSolver()
        self._chain = LineageChain()
        self._registry = registry or StudyRegistry()
        self._run_counter = 0

    def solve(
        self,
        problem: Problem,
        declaration: SpeciesDeclaration | None = None,
        study_id: str | None = None,
    ) -> GovernedAnswer:
        if declaration is None:
            declaration = SpeciesDeclaration(
                species=Species.SYNTHETIC,
                regime=Regime.SYNTHETIC,
            )

        ok, msg = RegimeFirewall.validate(declaration)
        if not ok:
            raise ValueError(f"Regime firewall rejected: {msg}")

        answer = self._solver.solve(problem)

        self._run_counter += 1
        record_id = f"RUN-{self._run_counter:06d}"

        answer_dict = {}
        for k, v in answer.values.items():
            if isinstance(v, np.ndarray):
                answer_dict[k] = float(np.mean(v)) if v.size > 10 else v.tolist()
            elif isinstance(v, (int, float)):
                answer_dict[k] = v

        evidence = self._chain.append(
            record_id=record_id,
            problem_spec={
                "type": problem.type.value,
                "params": {k: str(v) for k, v in problem.params.items()},
                "species": declaration.species.value,
                "regime": declaration.regime.value,
            },
            answer_values=answer_dict,
        )

        return GovernedAnswer(
            answer=answer,
            evidence=evidence,
            species_declaration=declaration,
            chain_position=len(self._chain),
            chain_head_hash=self._chain.head_hash,
        )

    def verify_lineage(self) -> bool:
        return self._chain.verify_chain()

    def export_lineage(self) -> list[dict]:
        return self._chain.export()

    @property
    def registry(self) -> StudyRegistry:
        return self._registry

    @property
    def chain_length(self) -> int:
        return len(self._chain)
