"""End-to-end music foundation-frontier pipeline.

Raw audio → domain feature extraction → foundation encoder → predictions + frontier reasoning.

This is the complete pipeline that makes the model "working" end-to-end:
  1. AudioFeatureExtractor: waveform → 16-dim foundation features
  2. MusicFoundationEncoder: features → learned representation
  3. MoodPredictor: representation → mood + VAD predictions
  4. GovernedMusicBridge: representation → structured LLM reasoning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.foundation.audio_feature_extractor import (
    AudioFeatureExtractor,
    ExtractedAudioFeatures,
    FOUNDATION_FEATURE_NAMES,
)
from models.foundation.music_foundation import MusicFoundationConfig, MusicFoundationEncoder
from models.foundation.music_hybrid_model import MusicHybridModel, MusicHybridConfig
from models.foundation.music_frontier_bridge import (
    GovernedMusicAnswer,
    MockMusicFrontierCaller,
    MusicQueryType,
)


MOOD_LABELS = ["happy", "energetic", "calm", "sad", "tense"]
VAD_LABELS = ["valence", "arousal", "dominance"]


@dataclass
class PipelineConfig:
    sr: int = 22050
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    use_frontier: bool = True
    device: str = "cpu"


@dataclass
class PipelinePrediction:
    mood_scores: dict[str, float]
    vad_scores: dict[str, float]
    dominant_mood: str
    features: dict[str, float]
    metadata: dict[str, Any]
    frontier_answer: GovernedMusicAnswer | None = None


class MusicPipeline:
    """Complete audio → prediction pipeline."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.device = torch.device(self.config.device)

        self.extractor = AudioFeatureExtractor(sr=self.config.sr)

        music_cfg = MusicFoundationConfig(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
        )
        hybrid_cfg = MusicHybridConfig(
            music_config=music_cfg,
            use_frontier=self.config.use_frontier,
        )
        self.model = MusicHybridModel(hybrid_cfg).to(self.device)
        self.model.eval()

        self._feature_indices = torch.arange(16).unsqueeze(0).to(self.device)

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str | Path, config: PipelineConfig | None = None,
    ) -> "MusicPipeline":
        pipe = cls(config)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pipe.model.load_state_dict(payload["model_state_dict"])
        pipe.model.eval()
        return pipe

    def predict_audio(
        self,
        audio: np.ndarray,
        sr: int | None = None,
        hypothesis: str = "",
        query_type: MusicQueryType = MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
    ) -> PipelinePrediction:
        extracted = self.extractor.extract(audio, sr)
        return self._predict_from_features(extracted, hypothesis, query_type)

    def predict_features(
        self,
        features: np.ndarray,
        hypothesis: str = "",
        query_type: MusicQueryType = MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
    ) -> PipelinePrediction:
        extracted = ExtractedAudioFeatures(
            features=features.astype(np.float32),
            feature_names=FOUNDATION_FEATURE_NAMES,
            metadata={},
        )
        return self._predict_from_features(extracted, hypothesis, query_type)

    def _predict_from_features(
        self,
        extracted: ExtractedAudioFeatures,
        hypothesis: str,
        query_type: MusicQueryType,
    ) -> PipelinePrediction:
        feat_tensor = torch.from_numpy(extracted.features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(feat_tensor, self._feature_indices)
            mood_probs = torch.sigmoid(out["pred_mood"]).squeeze(0).cpu().numpy()
            vad_vals = out["pred_vad"].squeeze(0).cpu().numpy()

        mood_scores = {MOOD_LABELS[i]: float(mood_probs[i]) for i in range(len(MOOD_LABELS))}
        vad_scores = {VAD_LABELS[i]: float(vad_vals[i]) for i in range(len(VAD_LABELS))}
        dominant_mood = max(mood_scores, key=mood_scores.get)

        frontier_answer = None
        if self.config.use_frontier and hypothesis:
            frontier_call = MockMusicFrontierCaller()
            frontier_answer = self.model.predict_and_explain(
                features=feat_tensor,
                feature_indices=self._feature_indices,
                feature_names=extracted.feature_names,
                domain="music",
                regime="data-driven",
                hypothesis=hypothesis,
                frontier_call=frontier_call,
                query_type=query_type,
            )

        feature_dict = {
            extracted.feature_names[i]: float(extracted.features[i])
            for i in range(len(extracted.features))
        }

        return PipelinePrediction(
            mood_scores=mood_scores,
            vad_scores=vad_scores,
            dominant_mood=dominant_mood,
            features=feature_dict,
            metadata=extracted.metadata,
            frontier_answer=frontier_answer,
        )

    def predict_batch(
        self, audio_list: list[np.ndarray], sr: int | None = None,
    ) -> list[PipelinePrediction]:
        return [self.predict_audio(a, sr) for a in audio_list]

    def train_step(
        self,
        audio_list: list[np.ndarray],
        target_moods: np.ndarray,
        target_vad: np.ndarray,
        sr: int | None = None,
    ) -> dict[str, float]:
        self.model.train()
        features = self.extractor.extract_batch(audio_list, sr)
        feat_t = torch.from_numpy(features).to(self.device)
        idx_t = self._feature_indices.expand(len(audio_list), -1)

        out = self.model(feat_t, idx_t)

        mood_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["pred_mood"], torch.from_numpy(target_moods).float().to(self.device),
        )
        vad_loss = torch.nn.functional.mse_loss(
            out["pred_vad"], torch.from_numpy(target_vad).float().to(self.device),
        )
        total = mood_loss + vad_loss
        self.model.eval()

        return {
            "total_loss": total.item(),
            "mood_loss": mood_loss.item(),
            "vad_loss": vad_loss.item(),
        }
