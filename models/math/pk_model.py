"""Differentiable 1-compartment pharmacokinetic model.

Oral absorption:
    C(t) = (F·D·ka) / (Vd·(ka - ke)) · (exp(-ke·t) - exp(-ka·t))

IV bolus:
    C(t) = (D / Vd) · exp(-ke·t)

Steady-state (oral, repeated dosing interval τ):
    Css = (F·D) / (Vd·ke·τ)

Inter-individual variability (IIV):
    θ_i = θ_pop · exp(η_i),  η ~ N(0, ω²)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PKParameters(nn.Module):
    """Learnable population PK parameters with IIV."""

    def __init__(self, n_drugs: int = 11):
        super().__init__()
        self.log_ka = nn.Parameter(torch.randn(n_drugs) * 0.1)
        self.log_ke = nn.Parameter(torch.randn(n_drugs) * 0.1)
        self.log_vd = nn.Parameter(torch.randn(n_drugs) * 0.1)
        self.logit_f = nn.Parameter(torch.zeros(n_drugs))
        self.log_omega = nn.Parameter(torch.full((n_drugs, 4), -1.0))

    def forward(self, drug_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        ka = self.log_ka[drug_idx].exp()
        ke = self.log_ke[drug_idx].exp()
        vd = self.log_vd[drug_idx].exp()
        f = self.logit_f[drug_idx].sigmoid()
        omega = self.log_omega[drug_idx].exp()
        return {"ka": ka, "ke": ke, "vd": vd, "f": f, "omega": omega}


class OneCompartmentPK(nn.Module):
    """Differentiable 1-compartment PK solver.

    Given (drug_idx, dose, time, route), returns concentration C(t).
    Supports IIV via η samples.
    """

    def __init__(self, n_drugs: int = 11):
        super().__init__()
        self.params = PKParameters(n_drugs)

    def _apply_iiv(
        self,
        pop: dict[str, torch.Tensor],
        eta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if eta is None:
            return pop
        omega = pop["omega"]
        eta_scaled = eta * omega
        return {
            "ka": pop["ka"] * eta_scaled[:, 0].exp(),
            "ke": pop["ke"] * eta_scaled[:, 1].exp(),
            "vd": pop["vd"] * eta_scaled[:, 2].exp(),
            "f": (pop["f"].logit() + eta_scaled[:, 3]).sigmoid(),
            "omega": omega,
        }

    def oral(
        self,
        dose: torch.Tensor,
        t: torch.Tensor,
        drug_idx: torch.Tensor,
        eta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """C(t) = (F·D·ka)/(Vd·(ka-ke))·(exp(-ke·t) - exp(-ka·t))"""
        p = self._apply_iiv(self.params(drug_idx), eta)
        ka, ke, vd, f = p["ka"], p["ke"], p["vd"], p["f"]
        diff = ka - ke
        diff = torch.where(diff.abs() < 1e-8, torch.ones_like(diff) * 1e-8, diff)
        c = (f * dose * ka) / (vd * diff) * ((-ke * t).exp() - (-ka * t).exp())
        return c.clamp(min=0.0)

    def iv_bolus(
        self,
        dose: torch.Tensor,
        t: torch.Tensor,
        drug_idx: torch.Tensor,
        eta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """C(t) = (D/Vd)·exp(-ke·t)"""
        p = self._apply_iiv(self.params(drug_idx), eta)
        return (dose / p["vd"]) * (-p["ke"] * t).exp()

    def steady_state(
        self,
        dose: torch.Tensor,
        tau: torch.Tensor,
        drug_idx: torch.Tensor,
        eta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Css = (F·D)/(Vd·ke·τ)"""
        p = self._apply_iiv(self.params(drug_idx), eta)
        return (p["f"] * dose) / (p["vd"] * p["ke"] * tau)

    def forward(
        self,
        dose: torch.Tensor,
        t: torch.Tensor,
        drug_idx: torch.Tensor,
        route: torch.Tensor | None = None,
        eta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if route is None:
            return self.oral(dose, t, drug_idx, eta)
        oral_c = self.oral(dose, t, drug_idx, eta)
        iv_c = self.iv_bolus(dose, t, drug_idx, eta)
        is_oral = (route == 0).float()
        return is_oral * oral_c + (1 - is_oral) * iv_c
