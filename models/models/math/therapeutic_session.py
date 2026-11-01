"""Differentiable therapeutic session scoring and planning.

Track–goal compatibility:
    score = σ(−‖v_track − v_goal‖₂) · bpm_score

Emotional Impact Factor:
    eif = w_v·v + w_a·|a| + w_d·d + patient adjustment

Session VAD trajectory: GRU-based phase transition dynamics.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TherapeuticScorer(nn.Module):
    """Score track–goal compatibility in VAD + BPM space."""

    def __init__(self, vad_dim: int = 3, n_goals: int = 6):
        super().__init__()
        self.goal_embeddings = nn.Embedding(n_goals, vad_dim)
        self.bpm_targets = nn.Embedding(n_goals, 2)  # (bpm_min, bpm_max)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        track_vad: torch.Tensor,
        track_bpm: torch.Tensor,
        goal_idx: torch.Tensor,
    ) -> torch.Tensor:
        goal_vad = self.goal_embeddings(goal_idx)
        d = torch.norm(track_vad - goal_vad, dim=-1)
        vad_score = torch.sigmoid(-d * self.temperature)

        bpm_bounds = self.bpm_targets(goal_idx)
        bpm_lo = bpm_bounds[..., 0]
        bpm_hi = bpm_bounds[..., 1]
        in_range = ((track_bpm >= bpm_lo) & (track_bpm <= bpm_hi)).float()
        dist_lo = (bpm_lo - track_bpm).clamp(min=0)
        dist_hi = (track_bpm - bpm_hi).clamp(min=0)
        out_penalty = torch.exp(-(dist_lo + dist_hi) / 20.0)
        bpm_score = in_range + (1 - in_range) * out_penalty

        return vad_score * bpm_score


class EmotionalImpactFactor(nn.Module):
    """EIF = w_v·v + w_a·|a| + w_d·d, with patient adjustment."""

    def __init__(self):
        super().__init__()
        self.w_valence = nn.Parameter(torch.tensor(0.4))
        self.w_arousal = nn.Parameter(torch.tensor(0.3))
        self.w_dominance = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        dominance: torch.Tensor,
        patient_valence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        eif_valence = (valence + 1) / 2
        eif_arousal = arousal.abs()
        eif_total = (
            self.w_valence * eif_valence
            + self.w_arousal * eif_arousal
            + self.w_dominance * dominance
        )
        if patient_valence is not None:
            eif_patient = eif_total * (1 + (valence - patient_valence).abs())
        else:
            eif_patient = eif_total
        return {
            "eif_valence": eif_valence,
            "eif_arousal": eif_arousal,
            "eif_total": eif_total,
            "eif_patient": eif_patient,
        }


class SessionPlanner(nn.Module):
    """Plan a therapeutic VAD trajectory over n_phases using GRU dynamics."""

    def __init__(self, vad_dim: int = 3, n_goals: int = 6):
        super().__init__()
        self.goal_embedding = nn.Embedding(n_goals, vad_dim)
        self.transition = nn.GRUCell(vad_dim, vad_dim)
        self.phase_scorer = nn.Linear(vad_dim, 1)

    def forward(
        self,
        initial_vad: torch.Tensor,
        goal_idx: torch.Tensor,
        n_phases: int = 5,
    ) -> dict[str, torch.Tensor]:
        target = self.goal_embedding(goal_idx)
        state = initial_vad
        trajectory = [state]
        scores = []

        for _ in range(n_phases - 1):
            direction = nn.functional.normalize(target - state, dim=-1)
            state = self.transition(direction, state)
            state = nn.functional.normalize(state, dim=-1)
            trajectory.append(state)
            scores.append(self.phase_scorer(state).squeeze(-1))

        scores.insert(0, self.phase_scorer(initial_vad).squeeze(-1))
        return {
            "vad_trajectory": torch.stack(trajectory, dim=1),
            "phase_scores": torch.stack(scores, dim=1),
        }
