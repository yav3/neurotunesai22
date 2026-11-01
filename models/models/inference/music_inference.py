"""Music hybrid foundation–frontier inference service.

Provides the clean entry point for governed inference after training:
    1. Load a trained checkpoint
    2. Build a frontier-enabled model
    3. Run predict_and_explain with a real or mock frontier caller
    4. Return a GovernedMusicAnswer with full lineage

Usage:
    service = MusicInferenceService.from_checkpoint(
        ckpt_path, feature_names=feature_names,
    )
    answer = service.infer(
        features=feature_vector,
        hypothesis="Increasing tempo shifts arousal upward",
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from models.foundation.music_foundation import MusicFoundationConfig
from models.foundation.music_hybrid_model import MusicHybridModel, MusicHybridConfig
from models.foundation.music_frontier_bridge import (
    AUDIO_FEATURE_NAMES,
    GovernedMusicAnswer,
    MockMusicFrontierCaller,
    MusicQueryType,
)
from models.foundation.frontier_caller import (
    ClaudeFrontierCaller,
    FrontierCallerConfig,
)


@dataclass
class MusicInferenceConfig:
    """Configuration for music hybrid inference."""
    use_real_frontier: bool = False
    frontier_model: str = "claude-sonnet-4-5-20250514"
    frontier_api_key_env: str = "ANTHROPIC_API_KEY"
    default_regime: str = "data-driven"
    default_domain: str = "music"
    default_query_type: MusicQueryType = MusicQueryType.MOOD_SHIFT_HYPOTHESIS


class MusicInferenceService:
    """Governed inference from a trained music hybrid model."""

    def __init__(
        self,
        model: MusicHybridModel,
        feature_names: list[str],
        config: MusicInferenceConfig | None = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.config = config or MusicInferenceConfig()
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        feature_names: list[str] | None = None,
        config: MusicInferenceConfig | None = None,
    ) -> "MusicInferenceService":
        """Load a trained model from checkpoint."""
        ckpt_path = Path(ckpt_path)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        saved_config = payload.get("config", {})
        music_cfg_dict = saved_config.get("music_config", {})

        music_config = MusicFoundationConfig(
            max_features=music_cfg_dict.get("max_features", 64),
            hidden_dim=music_cfg_dict.get("hidden_dim", 128),
            num_layers=music_cfg_dict.get("num_layers", 4),
            num_heads=music_cfg_dict.get("num_heads", 4),
            ffn_dim=music_cfg_dict.get("ffn_dim", 512),
            dropout=music_cfg_dict.get("dropout", 0.1),
            n_value_bins=music_cfg_dict.get("n_value_bins", 32),
            mask_ratio=music_cfg_dict.get("mask_ratio", 0.15),
        )

        hybrid_config = MusicHybridConfig(
            music_config=music_config,
            prediction_hidden=saved_config.get("prediction_hidden", 256),
            use_frontier=saved_config.get("use_frontier", True),
            projection_dim=saved_config.get("projection_dim", 16),
        )

        model = MusicHybridModel(hybrid_config)
        model.load_state_dict(payload["model_state_dict"])

        names = feature_names or saved_config.get("feature_names", AUDIO_FEATURE_NAMES[:16])

        return cls(model, names, config)

    def _get_frontier_caller(self) -> Callable[[str], str | dict] | None:
        if not self.config.use_real_frontier:
            return MockMusicFrontierCaller()

        caller_config = FrontierCallerConfig(
            model=self.config.frontier_model,
            api_key_env=self.config.frontier_api_key_env,
        )
        return ClaudeFrontierCaller(caller_config)

    def infer(
        self,
        features: np.ndarray | torch.Tensor,
        hypothesis: str = "",
        domain: str | None = None,
        regime: str | None = None,
        study_id: str = "",
        query_type: MusicQueryType | None = None,
    ) -> GovernedMusicAnswer:
        """Run governed inference on a single audio feature vector.

        Args:
            features: (F,) or (1, F) audio feature values
            hypothesis: what the model should reason about
            domain: "music" or "therapeutic_audio"
            regime: "data-driven", "prior-contrast", or "therapeutic"
            study_id: pre-registered study ID
            query_type: type of frontier query
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if features.dim() == 1:
            features = features.unsqueeze(0)

        n_feats = features.shape[1]
        feature_indices = torch.arange(n_feats).unsqueeze(0)

        frontier_call = self._get_frontier_caller()

        return self.model.predict_and_explain(
            features=features,
            feature_indices=feature_indices,
            feature_names=self.feature_names[:n_feats],
            domain=domain or self.config.default_domain,
            regime=regime or self.config.default_regime,
            hypothesis=hypothesis or "Analyse the audio feature profile",
            study_id=study_id,
            frontier_call=frontier_call,
            query_type=query_type or self.config.default_query_type,
        )

    def batch_infer(
        self,
        features_batch: np.ndarray | torch.Tensor,
        hypothesis: str = "",
        domain: str | None = None,
        regime: str | None = None,
    ) -> list[GovernedMusicAnswer]:
        """Run inference on a batch of audio feature vectors."""
        if isinstance(features_batch, np.ndarray):
            features_batch = torch.from_numpy(features_batch).float()

        results = []
        for i in range(len(features_batch)):
            answer = self.infer(
                features_batch[i],
                hypothesis=hypothesis,
                domain=domain,
                regime=regime,
            )
            results.append(answer)
        return results
