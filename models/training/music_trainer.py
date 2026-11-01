"""Training pipeline for the Music Foundation–Frontier Hybrid Model.

Two training modes:
  1. Pretraining: MaskedFeatureModeling (self-supervised)
  2. Finetuning: Mood/VAD prediction from audio features (supervised)

Usage:
    from models.training.music_trainer import MusicTrainingPipeline
    pipeline = MusicTrainingPipeline()
    record = pipeline.run_governed_trial(study_id="MUSIC-001")
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models.foundation.music_foundation import (
    MusicFoundationConfig,
    MusicFoundationEncoder,
    MaskedFeatureModeling,
)
from models.foundation.music_hybrid_model import (
    MusicHybridModel,
    MusicHybridConfig,
)
from models.foundation.music_frontier_bridge import (
    AUDIO_FEATURE_NAMES,
    MusicQueryType,
    MockMusicFrontierCaller,
)
from models.data.music_dataset import (
    MusicFeatureDataset,
    generate_synthetic_music_data,
)


@dataclass
class MusicTrainConfig:
    pretrain_epochs: int = 10
    finetune_epochs: int = 20
    pretrain_lr: float = 1e-3
    finetune_lr: float = 5e-4
    batch_size: int = 32
    val_split: float = 0.2
    n_synthetic_samples: int = 2000
    n_features: int = 16
    seed: int = 2026
    checkpoint_dir: str = "checkpoints/music"
    use_frontier: bool = True


@dataclass
class MusicTrialRecord:
    study_id: str
    started_at: str = ""
    finished_at: str = ""
    pretrain_loss: float = 0.0
    finetune_loss: float = 0.0
    val_vad_mse: float = 0.0
    val_mood_mse: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    config: dict = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    content_hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps({
            "study_id": self.study_id,
            "pretrain_loss": self.pretrain_loss,
            "finetune_loss": self.finetune_loss,
            "val_vad_mse": self.val_vad_mse,
            "val_mood_mse": self.val_mood_mse,
        }, sort_keys=True)
        self.content_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.content_hash


class MusicTrainingPipeline:
    """End-to-end training pipeline for the music hybrid model."""

    def __init__(
        self,
        train_config: MusicTrainConfig | None = None,
        hybrid_config: MusicHybridConfig | None = None,
    ):
        self.train_config = train_config or MusicTrainConfig()
        self.hybrid_config = hybrid_config or MusicHybridConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_data(self) -> tuple[DataLoader, DataLoader, list[str]]:
        tc = self.train_config
        data = generate_synthetic_music_data(
            tc.n_synthetic_samples, tc.n_features, tc.seed,
        )
        dataset = MusicFeatureDataset(
            data["features"], data["feature_indices"],
            data["target_vad"], data["target_mood"],
        )
        n_val = int(len(dataset) * tc.val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(tc.seed),
        )
        train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=tc.batch_size)
        return train_loader, val_loader, data["feature_names"]

    def pretrain(self, model: MusicHybridModel, train_loader: DataLoader) -> float:
        """Self-supervised pretraining via masked feature modeling."""
        tc = self.train_config
        mfm = MaskedFeatureModeling(model.encoder).to(self.device)
        opt = torch.optim.AdamW(mfm.parameters(), lr=tc.pretrain_lr)

        mfm.train()
        final_loss = 0.0
        for epoch in range(tc.pretrain_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                feats = batch["features"].to(self.device)
                indices = batch["feature_indices"].to(self.device)
                out = mfm(feats, indices)
                loss = out["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)

        return final_loss

    def finetune(
        self, model: MusicHybridModel,
        train_loader: DataLoader, val_loader: DataLoader,
    ) -> dict[str, float]:
        """Supervised finetuning on VAD + mood prediction."""
        tc = self.train_config
        model = model.to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=tc.finetune_lr)

        final_train_loss = 0.0
        for epoch in range(tc.finetune_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                feats = batch["features"].to(self.device)
                indices = batch["feature_indices"].to(self.device)
                target_vad = batch["target_vad"].to(self.device)
                target_mood = batch["target_mood"].to(self.device)

                out = model.training_loss(
                    feats, indices, target_vad, target_mood,
                )
                loss = out["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_vad_mse = 0.0
        val_mood_mse = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(self.device)
                indices = batch["feature_indices"].to(self.device)
                target_vad = batch["target_vad"].to(self.device)
                target_mood = batch["target_mood"].to(self.device)

                out = model(feats, indices)
                val_vad_mse += F.mse_loss(out["pred_vad"], target_vad).item() * len(feats)
                val_mood_mse += F.mse_loss(out["pred_mood"], target_mood).item() * len(feats)
                n_val += len(feats)

        return {
            "train_loss": final_train_loss,
            "val_vad_mse": val_vad_mse / max(n_val, 1),
            "val_mood_mse": val_mood_mse / max(n_val, 1),
        }

    def run_governed_trial(self, study_id: str = "MUSIC-001") -> MusicTrialRecord:
        """Run a full governed training trial: pretrain → finetune → evaluate."""
        record = MusicTrialRecord(
            study_id=study_id,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        train_loader, val_loader, feature_names = self._build_data()
        model = MusicHybridModel(self.hybrid_config)

        pretrain_loss = self.pretrain(model, train_loader)
        record.pretrain_loss = pretrain_loss

        ft_results = self.finetune(model, train_loader, val_loader)
        record.finetune_loss = ft_results["train_loss"]
        record.val_vad_mse = ft_results["val_vad_mse"]
        record.val_mood_mse = ft_results["val_mood_mse"]
        record.n_samples = self.train_config.n_synthetic_samples
        record.n_features = self.train_config.n_features
        record.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.compute_hash()

        self._model = model
        self._feature_names = feature_names
        return record

    def run_and_serve(
        self,
        study_id: str = "MUSIC-001",
        checkpoint_dir: str | None = None,
    ) -> tuple[MusicTrialRecord, "MusicInferenceService"]:
        """Train → checkpoint → inference service in one call."""
        record = self.run_governed_trial(study_id)
        model = self._model
        feature_names = self._feature_names

        ckpt_dir = Path(checkpoint_dir or self.train_config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        ckpt_path = ckpt_dir / f"{study_id}_{ts}.pt"

        state = model.state_dict()
        weights_hash = hashlib.sha256()
        for key in sorted(state.keys()):
            weights_hash.update(key.encode())
            weights_hash.update(state[key].cpu().numpy().tobytes())

        config_dict = {
            "music_config": {
                "max_features": self.hybrid_config.music_config.max_features,
                "hidden_dim": self.hybrid_config.music_config.hidden_dim,
                "num_layers": self.hybrid_config.music_config.num_layers,
                "num_heads": self.hybrid_config.music_config.num_heads,
                "ffn_dim": self.hybrid_config.music_config.ffn_dim,
                "dropout": self.hybrid_config.music_config.dropout,
                "n_value_bins": self.hybrid_config.music_config.n_value_bins,
                "mask_ratio": self.hybrid_config.music_config.mask_ratio,
            },
            "prediction_hidden": self.hybrid_config.prediction_hidden,
            "use_frontier": self.hybrid_config.use_frontier,
            "projection_dim": self.hybrid_config.projection_dim,
            "feature_names": feature_names,
        }

        torch.save({
            "model_state_dict": state,
            "weights_hash": weights_hash.hexdigest(),
            "study_id": study_id,
            "content_hash": record.content_hash,
            "config": config_dict,
        }, ckpt_path)

        from models.inference.music_inference import MusicInferenceService
        service = MusicInferenceService.from_checkpoint(
            ckpt_path, feature_names=feature_names,
        )
        return record, service
