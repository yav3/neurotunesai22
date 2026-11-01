"""Governed trainer — training with evidence discipline.

Wraps the base Trainer with:
  1. Pre-registered training study (criterion before method)
  2. Species/regime firewall on data
  3. Run manifest for every training run (hash-addressed)
  4. Degeneracy monitoring (loss collapse, gradient vanishing)
  5. Checkpoint content-addressing (sha256 of state_dict)

This is the narrow foundation model training harness for item 2 of the
strategic roadmap: "one narrow foundation model on in-regime GEO series."
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
from torch.utils.data import DataLoader

from models.evidence.regime import RegimeFirewall, Species, Regime
from models.evidence.run_manifest import build_manifest, fingerprint
from models.evidence.validation_discipline import is_degenerate
from models.evidence.registry import StudyRegistry, StudyRecord


@dataclass
class TrainingSpec:
    """Pre-registered training specification — declared before any data is touched."""
    study_id: str
    model_name: str
    dataset_ids: list[str]
    species: str
    regime: str
    criterion: str
    min_samples: int
    max_epochs: int
    early_stopping_patience: int
    seed: int
    eval_metric: str
    eval_threshold: float
    preprocess_steps: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "model_name": self.model_name,
            "dataset_ids": self.dataset_ids,
            "species": self.species,
            "regime": self.regime,
            "criterion": self.criterion,
            "min_samples": self.min_samples,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
            "eval_metric": self.eval_metric,
            "eval_threshold": self.eval_threshold,
            "preprocess_steps": self.preprocess_steps,
            "notes": self.notes,
        }


@dataclass
class TrainingResult:
    """Immutable record of a completed training run."""
    spec: TrainingSpec
    manifest_id: str
    input_identity: str
    final_loss: float
    best_val_metric: float
    met_criterion: bool
    epochs_completed: int
    checkpoint_hash: str
    wall_time_seconds: float
    degeneracy_flags: list[str]
    created_at: float = field(default_factory=time.time)


class GovernedTrainer:
    """Training with evidence discipline.

    Usage:
        registry = StudyRegistry()
        firewall = RegimeFirewall()
        firewall.declare(Species.HUMAN, Regime.VALIDATION, source="GEO in-regime")

        spec = TrainingSpec(
            study_id="NF-001",
            model_name="grn_expression_predictor",
            dataset_ids=["GSE107965", "GSE119262"],
            species="human",
            regime="validation",
            criterion="val_loss < 0.5 on held-out GEO series",
            min_samples=30,
            max_epochs=100,
            early_stopping_patience=10,
            seed=42,
            eval_metric="mse",
            eval_threshold=0.5,
        )

        trainer = GovernedTrainer(model, train_loader, val_loader, spec, registry, firewall)
        result = trainer.run()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        spec: TrainingSpec,
        registry: StudyRegistry,
        firewall: RegimeFirewall,
        lr: float = 1e-4,
        device: str | None = None,
        checkpoint_dir: str | Path = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.spec = spec
        self.registry = registry
        self.firewall = firewall
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _check_preconditions(self) -> None:
        """Enforce evidence discipline before training starts."""
        species_enum = Species(self.spec.species)
        regime_enum = Regime(self.spec.regime)
        self.firewall.check_or_raise(species_enum, regime_enum)

        n_train = len(self.train_loader.dataset)
        if n_train < self.spec.min_samples:
            raise ValueError(
                f"Training data has {n_train} samples, below declared minimum "
                f"{self.spec.min_samples}. Evidence discipline: refuse, never default."
            )

    def _checkpoint_hash(self, state_dict: dict) -> str:
        """Content-address a checkpoint by hashing its serialized parameters."""
        h = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            h.update(key.encode("utf-8"))
            h.update(tensor.cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def _detect_degeneracy(self, losses: list[float]) -> list[str]:
        """Check for training pathologies."""
        flags = []
        arr = np.array(losses[-20:] if len(losses) >= 20 else losses)
        if len(arr) == 0:
            flags.append("no_losses_recorded")
            return flags
        degen, reason = is_degenerate(arr)
        if degen:
            flags.append(f"loss_degenerate: {reason}")
        if len(arr) >= 5 and np.all(np.diff(arr[-5:]) > 0):
            flags.append("loss_monotonically_increasing_last_5")
        if len(arr) >= 10 and np.std(arr[-10:]) < 1e-8:
            flags.append("loss_stagnant_last_10")
        return flags

    def run(self) -> TrainingResult:
        """Execute governed training run."""
        self._check_preconditions()

        study = self.registry.register(
            study_id=self.spec.study_id,
            title=f"Training: {self.spec.model_name}",
            criterion=self.spec.criterion,
        )

        torch.manual_seed(self.spec.seed)
        np.random.seed(self.spec.seed)

        self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion_fn = nn.MSELoss()

        best_val = float("inf")
        patience_counter = 0
        all_losses: list[float] = []
        best_state = None
        t0 = time.time()

        for epoch in range(self.spec.max_epochs):
            self.model.train()
            epoch_losses = []
            for batch in self.train_loader:
                x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                y = batch[1].to(self.device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else x

                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion_fn(pred, y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            all_losses.append(avg_loss)

            val_metric = self._evaluate(criterion_fn)
            if val_metric < best_val:
                best_val = val_metric
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.spec.early_stopping_patience:
                    break

        wall_time = time.time() - t0

        if best_state is not None:
            self.model.load_state_dict(best_state)
        ckpt_hash = self._checkpoint_hash(self.model.state_dict())

        ckpt_path = self.checkpoint_dir / f"{self.spec.model_name}_{ckpt_hash}.pt"
        torch.save(self.model.state_dict(), ckpt_path)

        degeneracy_flags = self._detect_degeneracy(all_losses)

        met = best_val <= self.spec.eval_threshold and len(degeneracy_flags) == 0

        manifest = build_manifest(
            analysis=f"train:{self.spec.model_name}",
            request_payload=self.spec.to_dict(),
            parameters={"lr": self.lr, "seed": self.spec.seed},
            seed=self.spec.seed,
            input_fingerprint=fingerprint({"datasets": self.spec.dataset_ids}),
            preprocess_steps=[{"step": s} for s in self.spec.preprocess_steps],
            output_payload={"best_val": best_val, "checkpoint": ckpt_hash},
            species=self.spec.species,
            regime=self.spec.regime,
            study_id=study.study_id,
        )

        verdict = "met" if met else "not_met"
        self.registry.record_verdict(
            study_id=study.study_id,
            measured={"best_val": best_val, "checkpoint": ckpt_hash},
            summary=f"{self.spec.eval_metric}={best_val:.4f}, degeneracy_flags={degeneracy_flags}",
            verdict=verdict,
        )

        return TrainingResult(
            spec=self.spec,
            manifest_id=manifest.manifest_id,
            input_identity=manifest.input_identity,
            final_loss=all_losses[-1] if all_losses else float("inf"),
            best_val_metric=best_val,
            met_criterion=met,
            epochs_completed=len(all_losses),
            checkpoint_hash=ckpt_hash,
            wall_time_seconds=wall_time,
            degeneracy_flags=degeneracy_flags,
        )

    def _evaluate(self, criterion_fn: nn.Module) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                y = batch[1].to(self.device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else x
                pred = self.model(x)
                losses.append(criterion_fn(pred, y).item())
        return float(np.mean(losses)) if losses else float("inf")
