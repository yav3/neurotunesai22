"""Differentiable multivariate correlation and distribution engine.

Covariance from correlation:
    Σ = D · R · D    where D = diag(σ), R = correlation matrix

Nearest positive-definite projection:
    Iterative Higham algorithm: X_{k+1} = 0.5·(S_k + Y_k),
    Y_{k+1} = PSD_project(X_{k+1}), S_{k+1} = unit_diag(Y_{k+1})

Lognormal conversion:
    μ_ln = ln(μ² / √(σ² + μ²))
    σ_ln = √(ln(1 + σ²/μ²))

Multivariate normal sampling:
    x ~ N(μ, Σ)  via Cholesky: x = μ + L·z, LL^T = Σ
"""
from __future__ import annotations

import torch
import torch.nn as nn


def nearest_positive_definite(mat: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
    """Higham (2002) nearest PD matrix. Operates on the last two dims."""
    n = mat.shape[-1]
    y = mat.clone()
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    if mat.dim() > 2:
        eye = eye.expand_as(mat)

    delta_s = torch.zeros_like(mat)
    for _ in range(max_iter):
        r = y - delta_s
        vals, vecs = torch.linalg.eigh(r)
        vals = vals.clamp(min=1e-8)
        x = vecs @ torch.diag_embed(vals) @ vecs.transpose(-2, -1)
        delta_s = x - r
        d_inv = 1.0 / x.diagonal(dim1=-2, dim2=-1).sqrt()
        y = torch.diag_embed(d_inv) @ x @ torch.diag_embed(d_inv)
        y = 0.5 * (y + y.transpose(-2, -1))
        diag_vals = torch.linalg.eigvalsh(y)
        if (diag_vals > 0).all():
            break
    return y


def lognormal_params(
    mu: torch.Tensor, sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (mean, std) of lognormal to (μ_ln, σ_ln) of underlying normal.

    μ_ln = ln(μ² / √(σ² + μ²))
    σ_ln = √(ln(1 + σ²/μ²))
    """
    mu2 = mu ** 2
    sigma2 = sigma ** 2
    sigma_ln = (1.0 + sigma2 / mu2.clamp(min=1e-12)).log().sqrt()
    mu_ln = (mu2 / (sigma2 + mu2).clamp(min=1e-12).sqrt()).log()
    return mu_ln, sigma_ln


class CorrelationEngine(nn.Module):
    """Learnable correlation structure for multimodal biomarkers.

    Maintains a set of correlation matrices R_k (one per modality group)
    and per-feature (μ, σ) parameters, supporting:
    - Correlated MVN sampling via Cholesky decomposition
    - Nearest-PD projection to ensure valid covariance
    - Lognormal conversion for positive-valued biomarkers
    """

    def __init__(self, group_dims: dict[str, int]):
        super().__init__()
        self.group_names = list(group_dims.keys())
        self.group_dims = group_dims

        for name, dim in group_dims.items():
            tril_size = dim * (dim - 1) // 2
            self.register_parameter(
                f"tril_{name}",
                nn.Parameter(torch.zeros(tril_size)),
            )
            self.register_parameter(
                f"mu_{name}",
                nn.Parameter(torch.zeros(dim)),
            )
            self.register_parameter(
                f"log_sigma_{name}",
                nn.Parameter(torch.zeros(dim)),
            )

    def _get_correlation(self, name: str) -> torch.Tensor:
        dim = self.group_dims[name]
        tril_params = getattr(self, f"tril_{name}")
        R = torch.eye(dim, device=tril_params.device)
        idx = torch.tril_indices(dim, dim, offset=-1)
        R[idx[0], idx[1]] = tril_params.tanh()
        R = R + R.T - torch.diag(R.diag())
        return nearest_positive_definite(R.unsqueeze(0)).squeeze(0)

    def _get_covariance(self, name: str) -> torch.Tensor:
        R = self._get_correlation(name)
        sigma = getattr(self, f"log_sigma_{name}").exp()
        D = torch.diag(sigma)
        return D @ R @ D

    def sample(
        self,
        name: str,
        n: int,
        lognormal: bool = False,
    ) -> torch.Tensor:
        """Sample n vectors from the modality group's distribution."""
        mu = getattr(self, f"mu_{name}")
        cov = self._get_covariance(name)
        L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
        z = torch.randn(n, mu.shape[0], device=mu.device)
        samples = mu + z @ L.T
        if lognormal:
            samples = samples.exp()
        return samples

    def log_prob(
        self,
        name: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Log-probability under the group's MVN."""
        mu = getattr(self, f"mu_{name}")
        cov = self._get_covariance(name)
        diff = x - mu
        prec = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
        maha = (diff @ prec * diff).sum(dim=-1)
        log_det = torch.linalg.slogdet(cov)[1]
        k = mu.shape[0]
        return -0.5 * (k * torch.log(torch.tensor(2 * 3.141592653589793)) + log_det + maha)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute log-probabilities for each modality group."""
        return {
            name: self.log_prob(name, x_dict[name])
            for name in self.group_names
            if name in x_dict
        }
