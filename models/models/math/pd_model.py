"""Differentiable Emax pharmacodynamic model.

Sigmoidal Emax:
    E(C) = E0 + Emax · C^γ / (EC50^γ + C^γ)

where E0 = baseline effect, Emax = maximal effect, EC50 = half-maximal
concentration, γ = Hill coefficient (steepness).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EmaxPD(nn.Module):
    """Differentiable sigmoidal Emax model.

    Given drug concentration C (from PKSolver), returns effect E(C).
    Population parameters per drug; IIV optional.
    """

    def __init__(self, n_drugs: int = 11):
        super().__init__()
        self.log_emax = nn.Parameter(torch.randn(n_drugs) * 0.1)
        self.log_ec50 = nn.Parameter(torch.randn(n_drugs) * 0.1)
        self.log_gamma = nn.Parameter(torch.zeros(n_drugs))
        self.e0 = nn.Parameter(torch.zeros(n_drugs))
        self.log_omega = nn.Parameter(torch.full((n_drugs, 3), -1.0))

    def forward(
        self,
        concentration: torch.Tensor,
        drug_idx: torch.Tensor,
        eta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """E(C) = E0 + Emax · C^γ / (EC50^γ + C^γ)"""
        emax = self.log_emax[drug_idx].exp()
        ec50 = self.log_ec50[drug_idx].exp()
        gamma = self.log_gamma[drug_idx].exp().clamp(min=0.1, max=10.0)
        e0 = self.e0[drug_idx]

        if eta is not None:
            omega = self.log_omega[drug_idx].exp()
            emax = emax * (eta[:, 0] * omega[:, 0]).exp()
            ec50 = ec50 * (eta[:, 1] * omega[:, 1]).exp()
            gamma = (gamma.log() + eta[:, 2] * omega[:, 2]).exp().clamp(min=0.1, max=10.0)

        c_gamma = concentration.clamp(min=1e-12).pow(gamma)
        ec50_gamma = ec50.pow(gamma)
        effect = e0 + emax * c_gamma / (ec50_gamma + c_gamma)
        return effect


class PKPDCoupled(nn.Module):
    """Coupled PK/PD: dose,time → concentration → effect."""

    def __init__(self, pk: nn.Module, pd: EmaxPD):
        super().__init__()
        self.pk = pk
        self.pd = pd

    def forward(
        self,
        dose: torch.Tensor,
        t: torch.Tensor,
        drug_idx: torch.Tensor,
        route: torch.Tensor | None = None,
        eta_pk: torch.Tensor | None = None,
        eta_pd: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        conc = self.pk(dose, t, drug_idx, route, eta_pk)
        effect = self.pd(conc, drug_idx, eta_pd)
        return {"concentration": conc, "effect": effect}
