"""Differentiable disease progression models.

Linear decline:
    Y(t) = Y(0) + β·t + ε(t)

AR(1) within-subject variability:
    ε(t) = ρ·ε(t-1) + √(1-ρ²)·z,  z ~ N(0, σ²)

Episodic (MDD/BD):
    Phase-switching Markov model with state-dependent biomarkers.

Treatment-modified:
    Y(t) = Y(0) + β·t + δ_drug(t) + ε(t)
    where δ_drug(t) integrates the PD effect over time.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearProgression(nn.Module):
    """Y(t) = Y(0) + β·t, with learnable β per (disease, biomarker)."""

    def __init__(self, n_diseases: int = 14, n_biomarkers: int = 40):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(n_diseases, n_biomarkers))

    def forward(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        disease_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y0: (batch, n_biomarkers) baseline values
            t: (batch,) or (batch, n_timepoints) time in months
            disease_idx: (batch,) integer disease labels
        Returns:
            (batch, [n_timepoints,] n_biomarkers) predicted values
        """
        rates = self.beta[disease_idx]
        if t.dim() == 1 and t.shape[0] == y0.shape[0]:
            return y0 + rates * t.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(y0.shape[0], -1)
        return y0.unsqueeze(1) + rates.unsqueeze(1) * t.unsqueeze(-1)


class AR1Noise(nn.Module):
    """Autoregressive(1) within-subject variability.

    ε(t) = ρ·ε(t-1) + √(1-ρ²)·z,  z ~ N(0, σ²)
    """

    def __init__(self, n_biomarkers: int = 40, rho_init: float = 0.7):
        super().__init__()
        self.rho_logit = nn.Parameter(
            torch.full((n_biomarkers,), torch.tensor(rho_init).logit())
        )
        self.log_sigma = nn.Parameter(torch.zeros(n_biomarkers))

    @property
    def rho(self) -> torch.Tensor:
        return self.rho_logit.sigmoid()

    def forward(
        self, n_batch: int, n_timepoints: int, device: torch.device
    ) -> torch.Tensor:
        rho = self.rho
        sigma = self.log_sigma.exp()
        innovation_scale = (1 - rho ** 2).sqrt() * sigma

        eps = torch.zeros(n_batch, n_timepoints, rho.shape[0], device=device)
        z = torch.randn_like(eps)
        for t in range(1, n_timepoints):
            eps[:, t] = rho * eps[:, t - 1] + innovation_scale * z[:, t]
        return eps


class DiseaseTrajectory(nn.Module):
    """Full trajectory: linear trend + AR(1) noise + optional treatment effect.

    Y(t) = Y(0) + β·t + δ_drug(t) + ε(t)
    """

    def __init__(self, n_diseases: int = 14, n_biomarkers: int = 40):
        super().__init__()
        self.linear = LinearProgression(n_diseases, n_biomarkers)
        self.noise = AR1Noise(n_biomarkers)

    def forward(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        disease_idx: torch.Tensor,
        treatment_effect: torch.Tensor | None = None,
        stochastic: bool = True,
    ) -> torch.Tensor:
        trend = self.linear(y0, t, disease_idx)
        if treatment_effect is not None:
            trend = trend + treatment_effect
        if stochastic and self.training:
            if trend.dim() == 2:
                eps = self.noise(y0.shape[0], 1, y0.device).squeeze(1)
            else:
                n_tp = trend.shape[1]
                eps = self.noise(y0.shape[0], n_tp, y0.device)
            trend = trend + eps
        return trend
