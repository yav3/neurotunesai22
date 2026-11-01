"""Differentiable Generative Relational Network (GRN) dynamics.

Patent US 12,651,165 — Training:
    SOM → dimensionality reduction → visualization
    Train ML models of a GRN on the visualization

Patent US 12,361,263 — Utilizing:
    Apply GRN to model relationships between entities
    Simulate → Eliminate → Test → Identify target interaction

GRN node math (D/C/E update rules):
    D(t) = α·I(t) + β·F(t)          dominance factor
    C(t) = C(t-1) + D(t)             cumulative change
    E(t) = E(t-1) + γ·C(t)           evolution of internal component

where I(t) = change intensity, F(t) = change frequency.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SelfOrganizingMap(nn.Module):
    """Differentiable SOM for dimensionality reduction (patent '165 §2a).

    Maps high-dimensional input to a 2D grid of prototypes, producing
    a lower-dimensionality visualization that gates GRN training.
    """

    def __init__(self, input_dim: int, grid_h: int = 10, grid_w: int = 10):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.weights = nn.Parameter(torch.randn(grid_h * grid_w, input_dim) * 0.1)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, dtype=torch.float),
            torch.arange(grid_w, dtype=torch.float),
            indexing="ij",
        )
        self.register_buffer("grid_coords", torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1))

    def find_bmu(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, self.weights)
        return dists.argmin(dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bmu_idx = self.find_bmu(x)
        bmu_coords = self.grid_coords[bmu_idx]
        quantized = self.weights[bmu_idx]
        quantization_error = ((x - quantized) ** 2).sum(dim=-1).mean()
        return {
            "bmu_idx": bmu_idx,
            "bmu_coords": bmu_coords,
            "quantized": quantized,
            "quantization_error": quantization_error,
        }


class GRNNode(nn.Module):
    """Single GRN entity node with D/C/E state variables.

    D(t) = α·I(t) + β·F(t)
    C(t) = C(t-1) + D(t)
    E(t) = E(t-1) + γ·C(t)
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def step(
        self,
        intensity: torch.Tensor,
        frequency: torch.Tensor,
        c_prev: torch.Tensor,
        e_prev: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        d = self.alpha * intensity + self.beta * frequency
        c = c_prev + d
        e = e_prev + self.gamma * c
        return {"d": d, "c": c, "e": e}


class GRNNetwork(nn.Module):
    """Network of interacting GRN nodes.

    Models relationships between n_entities (e.g., neurotransmitters,
    biomarkers, or brain regions) via coupled D/C/E dynamics.
    The interaction matrix W encodes how one entity's dominance
    factor influences another's intensity.
    """

    def __init__(self, n_entities: int):
        super().__init__()
        self.n_entities = n_entities
        self.nodes = nn.ModuleList([GRNNode() for _ in range(n_entities)])
        self.interaction = nn.Parameter(torch.randn(n_entities, n_entities) * 0.01)
        self.intensity_encoder = nn.Linear(1, 1, bias=False)
        self.frequency_encoder = nn.Linear(1, 1, bias=False)

    def forward(
        self,
        external_signal: torch.Tensor,
        n_steps: int,
    ) -> dict[str, torch.Tensor]:
        """Simulate GRN dynamics over n_steps.

        Args:
            external_signal: (batch, n_entities) external driving input
            n_steps: number of time steps to simulate
        Returns:
            d_history: (batch, n_steps, n_entities)
            c_history: (batch, n_steps, n_entities)
            e_history: (batch, n_steps, n_entities)
        """
        batch = external_signal.shape[0]
        device = external_signal.device

        c = torch.zeros(batch, self.n_entities, device=device)
        e = torch.zeros(batch, self.n_entities, device=device)

        d_hist = []
        c_hist = []
        e_hist = []

        for t in range(n_steps):
            coupled = torch.matmul(c, self.interaction.T)
            intensity = external_signal + coupled
            freq = (c.abs() > 0).float().cumsum(dim=0).clamp(max=t + 1) / max(t + 1, 1)

            d_all, c_all, e_all = [], [], []
            for i, node in enumerate(self.nodes):
                out = node.step(intensity[:, i], freq[:, i], c[:, i], e[:, i])
                d_all.append(out["d"])
                c_all.append(out["c"])
                e_all.append(out["e"])

            d = torch.stack(d_all, dim=-1)
            c = torch.stack(c_all, dim=-1)
            e = torch.stack(e_all, dim=-1)
            d_hist.append(d)
            c_hist.append(c)
            e_hist.append(e)

        return {
            "d": torch.stack(d_hist, dim=1),
            "c": torch.stack(c_hist, dim=1),
            "e": torch.stack(e_hist, dim=1),
            "final_state": e,
        }


class GRNSimulateEliminate(nn.Module):
    """Simulate → Eliminate → Test → Identify (patent '263).

    Runs GRN forward for candidate interactions, scores each,
    eliminates below-threshold, and returns the target interaction.
    """

    def __init__(self, grn: GRNNetwork, scorer: nn.Module | None = None):
        super().__init__()
        self.grn = grn
        self.scorer = scorer or nn.Sequential(
            nn.Linear(grn.n_entities, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        candidates: torch.Tensor,
        n_sim_steps: int = 20,
        threshold: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        result = self.grn(candidates, n_sim_steps)
        scores = self.scorer(result["final_state"]).squeeze(-1)
        mask = scores > threshold
        return {
            "scores": scores,
            "mask": mask,
            "best_idx": scores.argmax(dim=0),
            "trajectories": result,
        }
