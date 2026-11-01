"""Music Foundation–Frontier Hybrid Model.

Combines the audio-feature foundation encoder (Tower 1) with the governed
music frontier bridge (Tower 2 interface) into a single inference pipeline.

    audio features + feature_ids
         │
         ▼
    MusicFoundationEncoder  ──→  CLS repr (d=128)
         │                           │
         ▼                           ▼
    MoodPredictor            MusicFoundationToFrontierProjector
         │                           │
         ▼                           ▼
    predicted VAD/features    MusicStructuredQuery
         │                           │
         ▼                           ▼
    ┌─────────────────────────────────────┐
    │    GovernedMusicBridge              │
    │    (domain firewall, lineage,      │
    │     pre-registration, calibration) │
    └─────────────────────────────────────┘
                    │
                    ▼
           GovernedMusicAnswer
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.foundation.music_foundation import (
    MusicFoundationConfig,
    MusicFoundationEncoder,
    MaskedFeatureModeling,
)
from models.foundation.music_frontier_bridge import (
    AUDIO_FEATURE_NAMES,
    GovernedMusicAnswer,
    GovernedMusicBridge,
    MockMusicFrontierCaller,
    MusicBridgeConfig,
    MusicEmbeddingSummariser,
    MusicFoundationToFrontierProjector,
    MusicFrontierResponse,
    MusicQueryType,
    MusicStructuredQuery,
)


@dataclass
class MusicHybridConfig:
    """Configuration for the full music hybrid pipeline."""
    music_config: MusicFoundationConfig = field(default_factory=MusicFoundationConfig)
    bridge_config: MusicBridgeConfig = field(default_factory=MusicBridgeConfig)
    prediction_hidden: int = 256
    prediction_dropout: float = 0.1
    use_frontier: bool = True
    projection_dim: int = 16
    n_mood_outputs: int = 5
    n_vad_outputs: int = 3


class MoodPredictor(nn.Module):
    """Predict target mood scores from CLS representation.

    Outputs:
        mood_scores: (B, 5)  — happy, energetic, calm, sad, tense
        vad: (B, 3)  — valence, arousal, dominance
    """

    def __init__(self, hidden_dim: int, n_mood: int = 5, n_vad: int = 3,
                 pred_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mood_head = nn.Sequential(
            nn.Linear(pred_hidden, n_mood),
            nn.Sigmoid(),
        )
        self.vad_head = nn.Sequential(
            nn.Linear(pred_hidden, n_vad),
            nn.Sigmoid(),
        )

    def forward(self, cls_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(cls_repr)
        return {
            "mood_scores": self.mood_head(h),
            "vad": self.vad_head(h),
        }


class FeaturePredictor(nn.Module):
    """Predict target audio features from CLS representation."""

    def __init__(self, hidden_dim: int, n_features: int,
                 pred_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, n_features),
        )

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        return self.predictor(cls_repr)


class MusicHybridModel(nn.Module):
    """The complete music foundation–frontier hybrid model.

    Forward pass:
        1. Encode audio features → CLS repr + per-feature repr
        2. Predict target mood/VAD scores
        3. Predict target audio features
        4. (Optional) Project CLS → frontier query features
        5. Build governed answer

    The model is trainable end-to-end on the foundation side.
    The frontier bridge is non-differentiable (LLM reasoning).
    """

    def __init__(self, config: MusicHybridConfig | None = None):
        super().__init__()
        self.config = config or MusicHybridConfig()
        c = self.config

        self.encoder = MusicFoundationEncoder(c.music_config)

        self.mood_predictor = MoodPredictor(
            c.music_config.hidden_dim,
            c.n_mood_outputs,
            c.n_vad_outputs,
            c.prediction_hidden,
            c.prediction_dropout,
        )

        self.feature_predictor = FeaturePredictor(
            c.music_config.hidden_dim,
            c.music_config.max_features,
            c.prediction_hidden,
            c.prediction_dropout,
        )

        if c.use_frontier:
            self.projector = MusicFoundationToFrontierProjector(
                c.music_config.hidden_dim,
                c.projection_dim,
            )

        self.bridge = GovernedMusicBridge(c.bridge_config)
        self.summariser = MusicEmbeddingSummariser()

    def forward(
        self,
        features: torch.Tensor,
        feature_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Foundation-only forward pass (differentiable, for training)."""
        cls_repr, seq_out = self.encoder(features, feature_indices, padding_mask)
        mood_out = self.mood_predictor(cls_repr)
        pred_features = self.feature_predictor(cls_repr)

        result = {
            "cls_repr": cls_repr,
            "seq_out": seq_out,
            "pred_mood": mood_out["mood_scores"],
            "pred_vad": mood_out["vad"],
            "pred_features": pred_features,
        }

        if self.config.use_frontier and hasattr(self, "projector"):
            proj = self.projector(cls_repr)
            result["frontier_projection"] = proj["projection"]
            result["foundation_confidence"] = proj["confidence"]

        return result

    def training_loss(
        self,
        features: torch.Tensor,
        feature_indices: torch.Tensor,
        target_vad: torch.Tensor,
        target_mood: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss: MSE on VAD + optional mood + feature reconstruction."""
        result = self.forward(features, feature_indices, padding_mask)

        vad_loss = F.mse_loss(result["pred_vad"], target_vad)
        total_loss = vad_loss

        losses = {"vad_loss": vad_loss.detach()}

        if target_mood is not None:
            mood_loss = F.mse_loss(result["pred_mood"], target_mood)
            total_loss = total_loss + mood_loss
            losses["mood_loss"] = mood_loss.detach()

        if target_features is not None:
            n_feats = min(target_features.shape[-1], result["pred_features"].shape[-1])
            feat_loss = F.mse_loss(
                result["pred_features"][..., :n_feats],
                target_features[..., :n_feats],
            )
            total_loss = total_loss + 0.5 * feat_loss
            losses["feature_loss"] = feat_loss.detach()

        return {"loss": total_loss, **losses, **result}

    @torch.no_grad()
    def predict_and_explain(
        self,
        features: torch.Tensor,
        feature_indices: torch.Tensor,
        feature_names: list[str],
        domain: str = "music",
        regime: str = "data-driven",
        hypothesis: str = "",
        study_id: str = "",
        frontier_call: Callable[[str], str | dict] | None = None,
        query_type: MusicQueryType = MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
        padding_mask: torch.Tensor | None = None,
    ) -> GovernedMusicAnswer:
        """Full hybrid inference: predict + explain."""
        self.eval()
        result = self.forward(features, feature_indices, padding_mask)

        baseline_np = features[0].cpu().numpy()
        pred_np = result["pred_features"][0].cpu().numpy()
        cls_np = result["cls_repr"][0].cpu().numpy()
        vad_np = result["pred_vad"][0].cpu().numpy()
        mood_np = result["pred_mood"][0].cpu().numpy()

        n_feats = min(len(feature_names), len(pred_np))
        feat_delta = self.summariser.summarise_feature_delta(
            baseline_np[:n_feats], pred_np[:n_feats],
            feature_names[:n_feats],
        )
        cls_summary = self.summariser.summarise_cls_embedding(cls_np)

        mood_labels = ["happy", "energetic", "calm", "sad", "tense"]
        foundation_prediction = {
            "predicted_features": pred_np[:n_feats].tolist(),
            "feature_delta": feat_delta,
            "cls_summary": cls_summary,
            "predicted_vad": {
                "valence": float(vad_np[0]),
                "arousal": float(vad_np[1]),
                "dominance": float(vad_np[2]) if len(vad_np) > 2 else 0.5,
            },
            "predicted_mood": {
                mood_labels[i]: float(mood_np[i])
                for i in range(min(len(mood_labels), len(mood_np)))
            },
        }
        if "foundation_confidence" in result:
            foundation_prediction["confidence"] = float(
                result["foundation_confidence"][0].cpu()
            )

        frontier_response = None
        if frontier_call is not None and self.config.use_frontier:
            query = self.bridge.build_query(
                query_type=query_type,
                feature_delta=feat_delta,
                cls_summary=cls_summary,
                domain=domain,
                regime=regime,
                hypothesis=hypothesis,
                study_id=study_id,
            )
            prompt = self.bridge.format_prompt(query)
            raw = frontier_call(prompt)
            frontier_response = self.bridge.parse_response(raw, query)

        return self.bridge.build_governed_answer(
            foundation_prediction=foundation_prediction,
            frontier_response=frontier_response,
            domain=domain,
            regime=regime,
            study_id=study_id,
        )

    def param_count(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        mood = sum(p.numel() for p in self.mood_predictor.parameters())
        feat = sum(p.numel() for p in self.feature_predictor.parameters())
        proj = sum(p.numel() for p in self.projector.parameters()) if hasattr(self, "projector") else 0
        return {
            "encoder": enc,
            "mood_predictor": mood,
            "feature_predictor": feat,
            "projector": proj,
            "total": enc + mood + feat + proj,
        }
