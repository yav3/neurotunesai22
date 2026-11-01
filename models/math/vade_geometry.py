"""4-D Affective VADE Glome geometry (drafted patent, unfiled).

Affine projection to 4-D:
    P_VADE = W_proxy · X + b

Mapping to S³ (unit 3-sphere in R⁴):
    P_norm = P_VADE / ‖P_VADE‖

Geodesic distance on S³:
    d(p, q) = arccos(⟨p, q⟩)

Fibonacci dosing constraint:
    φ = (1+√5)/2,  dose_k = dose_0 · φ^{-k}

Valence-arousal constraint:
    V ≥ 1.6 · A  (therapeutic regime)

BPM bounds: [46, 186]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

PHI = (1 + math.sqrt(5)) / 2


class VADEProjection(nn.Module):
    """Affine projection: X → P_VADE = W·X + b, then normalize to S³."""

    def __init__(self, input_dim: int, vade_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, vade_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.linear(x)
        p_norm = nn.functional.normalize(p, dim=-1)
        return {"p_vade": p, "p_sphere": p_norm}


def geodesic_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Geodesic distance on S³: d(p,q) = arccos(⟨p,q⟩)."""
    dot = (p * q).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return dot.acos()


def fibonacci_doses(dose_0: float, n_steps: int) -> torch.Tensor:
    """Fibonacci-constrained dosing: dose_k = dose_0 · φ^{-k}."""
    k = torch.arange(n_steps, dtype=torch.float)
    return dose_0 * PHI ** (-k)


class VADENavigator(nn.Module):
    """Navigate affective state on S³ toward a target coordinate.

    Uses geodesic distance to score how close a candidate intervention
    moves the state toward the target, with Fibonacci dosing schedule
    and the V ≥ 1.6·A therapeutic constraint.
    """

    def __init__(self, input_dim: int, vade_dim: int = 4):
        super().__init__()
        self.projection = VADEProjection(input_dim, vade_dim)
        self.state_update = nn.GRUCell(vade_dim, vade_dim)

    def project(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.projection(x)

    def score_candidates(
        self,
        current_state: torch.Tensor,
        target: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score intervention candidates by how much they reduce geodesic distance."""
        current_dist = geodesic_distance(current_state, target)
        projected = self.state_update(candidates, current_state)
        projected = nn.functional.normalize(projected, dim=-1)
        new_dist = geodesic_distance(projected, target)
        improvement = current_dist.unsqueeze(-1) - new_dist
        return improvement

    def check_therapeutic_constraint(
        self,
        p_sphere: torch.Tensor,
        valence_idx: int = 0,
        arousal_idx: int = 1,
    ) -> torch.Tensor:
        """V ≥ 1.6·A constraint check. Returns True where satisfied."""
        v = p_sphere[..., valence_idx]
        a = p_sphere[..., arousal_idx]
        return v >= 1.6 * a

    def plan_session(
        self,
        initial_state: torch.Tensor,
        target: torch.Tensor,
        n_steps: int,
        dose_0: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Plan a Fibonacci-dosed session trajectory on S³."""
        doses = fibonacci_doses(dose_0, n_steps).to(initial_state.device)
        trajectory = [initial_state]
        state = initial_state

        for k in range(n_steps):
            direction = target - state
            direction = nn.functional.normalize(direction, dim=-1)
            step = doses[k] * direction
            state = nn.functional.normalize(state + step, dim=-1)
            trajectory.append(state)

        trajectory = torch.stack(trajectory, dim=1)
        distances = geodesic_distance(
            trajectory, target.unsqueeze(1).expand_as(trajectory)
        )
        return {
            "trajectory": trajectory,
            "distances": distances,
            "doses": doses,
        }
