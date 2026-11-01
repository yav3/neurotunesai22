"""Structured problem specifications for the NeuroSolver.

Every problem is a dataclass specifying:
  - problem_type: what kind of computation to perform
  - inputs: the known quantities
  - query: what to solve for
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class ProblemType(Enum):
    DIAGNOSE = "diagnose"
    PREDICT_PROGRESSION = "predict_progression"
    PREDICT_TREATMENT = "predict_treatment"
    IMPUTE = "impute"
    SIMULATE_PK = "simulate_pk"
    SIMULATE_PD = "simulate_pd"
    SIMULATE_PKPD = "simulate_pkpd"
    GRN_DYNAMICS = "grn_dynamics"
    VADE_NAVIGATE = "vade_navigate"
    SAMPLE_COHORT = "sample_cohort"


@dataclass
class PatientState:
    """Observed patient state — the known quantities."""
    biomarkers: dict[str, torch.Tensor] = field(default_factory=dict)
    demographics: dict[str, Any] = field(default_factory=dict)
    disease_label: int | None = None
    mask: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class DrugSpec:
    """Drug specification for PK/PD problems."""
    drug_idx: int = 0
    dose_mg: float = 0.0
    route: int = 0  # 0=oral, 1=iv
    tau_hours: float = 24.0


@dataclass
class Problem:
    """A fully specified problem for the NeuroSolver."""
    problem_type: ProblemType
    patient: PatientState | None = None
    drug: DrugSpec | None = None
    time_points: torch.Tensor | None = None
    n_samples: int = 1
    target_modalities: list[str] | None = None
    grn_signal: torch.Tensor | None = None
    grn_steps: int = 20
    vade_target: torch.Tensor | None = None
    vade_n_steps: int = 10
    disease_idx: int | None = None
    stochastic: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
