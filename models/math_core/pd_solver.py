"""Pharmacodynamic solvers — Emax sigmoid model.

    E(C) = E_max · C^γ / (EC50^γ + C^γ)

Where:
    E_max  = maximum drug effect (units depend on target biomarker)
    EC50   = concentration producing 50% of E_max
    γ      = Hill coefficient (steepness)
    C      = drug concentration (from PK solver)

Combined PK/PD pipeline:
    dose → PK → C(t) → PD → E(t) → biomarker_response(t)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from models.math_core.pk_solver import PKSolver, PKParams


@dataclass
class PDParams:
    emax: float          # maximum effect
    ec50_ng_ml: float    # half-maximal concentration
    hill: float          # Hill coefficient (γ)
    direction: str       # "increase" or "decrease"
    baseline: float      # baseline biomarker value


class PDSolver:
    """Emax sigmoid PD model."""

    @staticmethod
    def effect(C: np.ndarray, pd: PDParams) -> np.ndarray:
        """E(C) = Emax · C^γ / (EC50^γ + C^γ)"""
        C_safe = np.maximum(C, 0.0)
        num = pd.emax * np.power(C_safe, pd.hill)
        den = np.power(pd.ec50_ng_ml, pd.hill) + np.power(C_safe, pd.hill)
        return np.where(den > 0, num / den, 0.0)

    @staticmethod
    def biomarker_response(
        C: np.ndarray, pd: PDParams,
    ) -> np.ndarray:
        """Biomarker value = baseline + effect (direction-aware)."""
        E = PDSolver.effect(C, pd)
        if pd.direction == "decrease":
            return pd.baseline + E  # emax is negative for decrease
        return pd.baseline + E

    @staticmethod
    def pkpd_trajectory(
        pk: PKParams,
        pd: PDParams,
        t: np.ndarray,
        n_doses: int = 1,
    ) -> dict[str, np.ndarray]:
        """Full PK/PD pipeline: dose → concentration → effect → biomarker."""
        if pk.route == "iv":
            C = PKSolver.single_dose_iv(pk, t)
        else:
            C = PKSolver.multi_dose_oral(pk, t, n_doses)
        E = PDSolver.effect(C, pd)
        B = PDSolver.biomarker_response(C, pd)
        return {"time": t, "concentration": C, "effect": E, "biomarker": B}

    @staticmethod
    def inverse_effect(target_effect: float, pd: PDParams) -> float:
        """Concentration needed to achieve a target effect.
        C = EC50 · (E / (Emax - E))^{1/γ}
        """
        if abs(pd.emax) < 1e-12 or abs(target_effect) >= abs(pd.emax):
            return float("inf")
        ratio = abs(target_effect) / (abs(pd.emax) - abs(target_effect))
        return pd.ec50_ng_ml * np.power(ratio, 1.0 / pd.hill)
