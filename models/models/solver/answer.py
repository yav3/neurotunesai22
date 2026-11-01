"""Structured answer types returned by the NeuroSolver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Answer:
    """Structured answer from the NeuroSolver.

    Every answer carries:
      - result: the primary computed output
      - problem_type: what was solved
      - confidence: how confident the answer is (when applicable)
      - intermediates: internal computation trace for inspection
    """
    result: dict[str, torch.Tensor]
    problem_type: str
    confidence: float | None = None
    intermediates: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        keys = list(self.result.keys())
        shapes = {k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
                  for k, v in self.result.items()}
        return f"Answer(type={self.problem_type}, keys={keys}, shapes={shapes})"
