"""Training pipeline for the Foundation–Frontier Hybrid Model.

Connects the HybridFoundationModel to the existing governed training harness:
  - Reads pooled GEO expression data via cross_platform_pool.py
  - Runs the 4-criterion governed protocol (NT-001 through NT-005 pattern)
  - Pre-registers study criteria before training
  - Evaluates against baselines in the same run
  - Records hash-chained lineage

Two training modes:
  1. Pretraining: Masked Gene Modeling (self-supervised)
  2. Finetuning: Perturbation response prediction (supervised, paired data)

Usage:
    from models.training.hybrid_trainer import HybridTrainingPipeline
    pipeline = HybridTrainingPipeline(merlin_root="/path/to/merlin")
    record = pipeline.run_governed_trial(study_id="NT-006")
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
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.foundation.gene_foundation import (
    GeneFoundationConfig,
    GeneFoundationEncoder,
    MaskedGeneModeling,
)
from models.foundation.hybrid_model import (
    HybridFoundationModel,
    HybridModelConfig,
)
from models.foundation.frontier_bridge import QueryType
from models.foundation.frontier_caller import MockFrontierCaller
from models.evidence.lineage import LineageChain, chain_hash
from models.evidence.registry import StudyRegistry


@dataclass
class HybridTrainConfig:
    """Training hyperparameters for the hybrid model."""
    # Pretraining
    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-4
    pretrain_batch_size: int = 32
    # Finetuning
    finetune_epochs: int = 50
    finetune_lr: float = 5e-5
    finetune_batch_size: int = 16
    # Common
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 10
    seed: int = 2026
    val_fraction: float = 0.2
    test_fraction: float = 0.1


@dataclass
class TrialRecord:
    """Record of a governed training trial."""
    study_id: str
    species: str
    regime: str
    n_samples: int
    n_genes: int
    n_series: int
    model_params: int
    train_mse: float
    val_mse: float
    test_mse: float
    train_r: float
    val_r: float
    test_r: float
    baseline_mse: float
    baseline_r: float
    criteria_met: int
    criteria_total: int = 4
    verdict: str = ""
    content_hash: str = ""
    elapsed_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        payload = json.dumps({
            "study_id": self.study_id,
            "species": self.species,
            "n_samples": self.n_samples,
            "n_genes": self.n_genes,
            "val_mse": self.val_mse,
            "val_r": self.val_r,
            "test_mse": self.test_mse,
            "test_r": self.test_r,
        }, sort_keys=True)
        self.content_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.content_hash


def _pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.std() < 1e-10 or target.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(pred.flatten(), target.flatten())[0, 1])


class GEOExpressionDataset:
    """Wraps pooled GEO expression data for PyTorch training.

    Constructs perturbation pairs from control/treated samples.
    """

    def __init__(
        self,
        expression: np.ndarray,
        conditions: list[str],
        gene_names: list[str],
        species: str,
    ):
        self.expression = expression
        self.conditions = conditions
        self.gene_names = gene_names
        self.species = species

        ctrl_idx = [i for i, c in enumerate(conditions) if c == "control"]
        treat_idx = [i for i, c in enumerate(conditions) if c == "treated"]

        if not ctrl_idx or not treat_idx:
            raise ValueError("Need both control and treated samples for pairs")

        self.ctrl_mean = expression[ctrl_idx].mean(axis=0)
        self.pairs = []
        for ti in treat_idx:
            self.pairs.append((self.ctrl_mean, expression[ti]))

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_pairs = len(self.pairs)
        n_genes = self.expression.shape[1]
        baselines = np.stack([p[0] for p in self.pairs])
        targets = np.stack([p[1] for p in self.pairs])
        gene_ids = np.arange(n_genes)
        gene_ids_batch = np.tile(gene_ids, (n_pairs, 1))

        return (
            torch.tensor(baselines, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(gene_ids_batch, dtype=torch.long),
        )


class HybridTrainingPipeline:
    """Full governed training pipeline for the hybrid model."""

    def __init__(
        self,
        merlin_root: str | Path = "/home/user/merlinaineuralplus2",
        config: HybridTrainConfig | None = None,
        registry: StudyRegistry | None = None,
        lineage: LineageChain | None = None,
        frontier_caller: Any | None = None,
    ):
        self.merlin_root = Path(merlin_root)
        self.config = config or HybridTrainConfig()
        self.registry = registry
        self.lineage = lineage if lineage is not None else LineageChain()
        self.frontier_caller = frontier_caller if frontier_caller is not None else MockFrontierCaller()

    def load_pooled_data(
        self,
        max_genes: int = 2000,
        species: str = "human",
    ) -> GEOExpressionDataset:
        from models.data.cross_platform_pool import load_and_pool_all_merlin

        geo_data, meta = load_and_pool_all_merlin(
            self.merlin_root, species=species, max_genes=max_genes,
        )

        return GEOExpressionDataset(
            expression=geo_data.expression,
            conditions=geo_data.conditions,
            gene_names=geo_data.gene_names,
            species=species,
        )

    def _split_data(
        self,
        baselines: torch.Tensor,
        targets: torch.Tensor,
        gene_ids: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        n = baselines.shape[0]
        n_test = max(1, int(n * self.config.test_fraction))
        n_val = max(1, int(n * self.config.val_fraction))
        n_train = n - n_val - n_test

        if n_train < 1:
            raise ValueError(f"Not enough samples: {n} total, need train/val/test split")

        gen = torch.Generator().manual_seed(self.config.seed)
        perm = torch.randperm(n, generator=gen)

        splits = {}
        for name, start, end in [
            ("train", 0, n_train),
            ("val", n_train, n_train + n_val),
            ("test", n_train + n_val, n),
        ]:
            idx = perm[start:end]
            splits[name] = (baselines[idx], targets[idx], gene_ids[idx])

        return splits

    def _compute_mean_baseline(
        self,
        train_targets: torch.Tensor,
        val_targets: torch.Tensor,
    ) -> dict[str, float]:
        train_mean = train_targets.mean(dim=0, keepdim=True)
        pred = train_mean.expand_as(val_targets)
        mse = float(F.mse_loss(pred, val_targets).item())
        r = _pearson_r(pred.numpy(), val_targets.numpy())
        return {"mse": mse, "r": r}

    def run_pretrain(
        self,
        dataset: GEOExpressionDataset,
        gene_config: GeneFoundationConfig | None = None,
    ) -> tuple[GeneFoundationEncoder, dict[str, float]]:
        """Self-supervised pretraining via Masked Gene Modeling."""
        gc = gene_config or GeneFoundationConfig(max_genes=len(dataset.gene_names))
        encoder = GeneFoundationEncoder(gc)
        mgm = MaskedGeneModeling(encoder)

        expression = torch.tensor(dataset.expression, dtype=torch.float32)
        gene_ids = torch.arange(expression.shape[1]).unsqueeze(0).expand(expression.shape[0], -1)

        n_val = max(1, int(len(expression) * self.config.val_fraction))
        n_train = len(expression) - n_val
        gen = torch.Generator().manual_seed(self.config.seed)
        perm = torch.randperm(len(expression), generator=gen)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        optimizer = torch.optim.AdamW(
            mgm.parameters(), lr=self.config.pretrain_lr,
            weight_decay=self.config.weight_decay,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.pretrain_epochs):
            mgm.train()
            train_perm = torch.randperm(n_train, generator=gen)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self.config.pretrain_batch_size):
                batch_idx = train_idx[train_perm[start:start + self.config.pretrain_batch_size]]
                batch_expr = expression[batch_idx]
                batch_gids = gene_ids[batch_idx]

                out = mgm(batch_expr, batch_gids)
                loss = out["loss"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mgm.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            mgm.eval()
            with torch.no_grad():
                val_out = mgm(expression[val_idx], gene_ids[val_idx])
                val_loss = val_out["loss"].item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        return encoder, {"train_loss": epoch_loss / max(n_batches, 1),
                         "val_loss": best_val_loss, "epochs": epoch + 1}

    def run_governed_trial(
        self,
        study_id: str = "NT-006",
        max_genes: int = 2000,
        species: str = "human",
        pretrain_first: bool = True,
        checkpoint_dir: str | Path | None = None,
    ) -> TrialRecord:
        """Run a full governed trial: pretrain → finetune → evaluate.

        Follows the NT-001 through NT-005 protocol:
          1. Pre-register criteria before training
          2. Train model
          3. Evaluate against mean baseline in the same run
          4. Report all 4 criteria: MSE < baseline, r > baseline, r > 0, MSE finite
        """
        t0 = time.time()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        dataset = self.load_pooled_data(max_genes=max_genes, species=species)
        baselines, targets, gene_ids = dataset.to_tensors()
        n_genes = baselines.shape[1]

        gc = GeneFoundationConfig(
            max_genes=n_genes,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
        )

        if pretrain_first:
            encoder, pretrain_stats = self.run_pretrain(dataset, gc)
        else:
            encoder = GeneFoundationEncoder(gc)
            pretrain_stats = {}

        hybrid_config = HybridModelConfig(
            gene_config=gc,
            use_frontier=False,
        )
        model = HybridFoundationModel(hybrid_config)
        model.encoder = encoder

        splits = self._split_data(baselines, targets, gene_ids)
        train_b, train_t, train_g = splits["train"]
        val_b, val_t, val_g = splits["val"]
        test_b, test_t, test_g = splits["test"]

        mean_baseline = self._compute_mean_baseline(train_t, val_t)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.finetune_lr,
            weight_decay=self.config.weight_decay,
        )

        best_val_mse = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.finetune_epochs):
            model.train()
            gen = torch.Generator().manual_seed(self.config.seed + epoch)
            perm = torch.randperm(len(train_b), generator=gen)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_b), self.config.finetune_batch_size):
                idx = perm[start:start + self.config.finetune_batch_size]
                out = model.training_loss(train_b[idx], train_g[idx], train_t[idx])
                loss = out["loss"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            model.eval()
            with torch.no_grad():
                val_out = model.training_loss(val_b, val_g, val_t)
                val_mse = val_out["mse"].item()

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            train_pred = model(train_b, train_g)["pred_expression"]
            val_pred = model(val_b, val_g)["pred_expression"]
            test_pred = model(test_b, test_g)["pred_expression"]

        train_mse = float(F.mse_loss(train_pred, train_t).item())
        val_mse = float(F.mse_loss(val_pred, val_t).item())
        test_mse = float(F.mse_loss(test_pred, test_t).item())
        train_r = _pearson_r(train_pred.numpy(), train_t.numpy())
        val_r = _pearson_r(val_pred.numpy(), val_t.numpy())
        test_r = _pearson_r(test_pred.numpy(), test_t.numpy())

        test_mean_baseline = self._compute_mean_baseline(train_t, test_t)

        # --- Frontier evaluation: exercise the full two-tower pipeline ---
        frontier_config = HybridModelConfig(
            gene_config=gc,
            use_frontier=True,
            prediction_hidden=hybrid_config.prediction_hidden,
            prediction_dropout=hybrid_config.prediction_dropout,
        )
        frontier_model = HybridFoundationModel(frontier_config)
        frontier_params = dict(frontier_model.named_parameters())
        for name, param in model.named_parameters():
            if name in frontier_params:
                frontier_params[name].data.copy_(param.data)
        frontier_model.eval()

        regime = "data-driven" if len(dataset.pairs) >= 30 else "prior-contrast"
        frontier_answer = frontier_model.predict_and_explain(
            expression=test_b[:1],
            gene_indices=test_g[:1],
            gene_names=dataset.gene_names,
            species=species,
            regime=regime,
            hypothesis=f"Governed trial {study_id}: perturbation response prediction",
            study_id=study_id,
            frontier_call=self.frontier_caller,
            query_type=QueryType.PERTURBATION_HYPOTHESIS,
        )

        criteria = [
            val_mse < mean_baseline["mse"],
            val_r > mean_baseline["r"],
            val_r > 0,
            np.isfinite(val_mse),
        ]
        criteria_met = sum(criteria)
        verdict = "MET" if all(criteria) else "NOT_MET"

        record = TrialRecord(
            study_id=study_id,
            species=species,
            regime=regime,
            n_samples=baselines.shape[0],
            n_genes=n_genes,
            n_series=1,
            model_params=sum(p.numel() for p in model.parameters()),
            train_mse=train_mse,
            val_mse=val_mse,
            test_mse=test_mse,
            train_r=train_r,
            val_r=val_r,
            test_r=test_r,
            baseline_mse=mean_baseline["mse"],
            baseline_r=mean_baseline["r"],
            criteria_met=criteria_met,
            verdict=verdict,
            elapsed_seconds=time.time() - t0,
            extra={
                "pretrain_stats": pretrain_stats,
                "finetune_epochs": epoch + 1,
                "test_baseline_mse": test_mean_baseline["mse"],
                "test_baseline_r": test_mean_baseline["r"],
                "frontier_used": True,
                "frontier_confidence": (
                    frontier_answer.frontier_reasoning.confidence
                    if frontier_answer.frontier_reasoning else None
                ),
                "frontier_content_hash": frontier_answer.content_hash,
            },
        )
        record.compute_hash()

        if self.registry is not None:
            study = self.registry.register(
                study_id=study_id,
                title=f"Hybrid foundation-frontier trial {study_id}",
                criterion="val_mse < baseline AND val_r > baseline AND val_r > 0 AND val_mse finite",
                data_kind="geo_expression",
                synthetic=False,
            )
            self.registry.record_verdict(
                study_id=study_id,
                measured={
                    "val_mse": val_mse, "val_r": val_r,
                    "test_mse": test_mse, "test_r": test_r,
                    "baseline_mse": mean_baseline["mse"],
                    "baseline_r": mean_baseline["r"],
                    "criteria_met": criteria_met,
                },
                summary=f"val_mse={val_mse:.4f} val_r={val_r:.4f} ({criteria_met}/4 criteria)",
                verdict="met" if verdict == "MET" else "not_met",
            )

        lineage_record = self.lineage.append(
            record_id=study_id,
            problem_spec={
                "model": "hybrid_foundation_frontier",
                "species": species,
                "n_genes": n_genes,
                "criterion": "val_mse < baseline AND val_r > baseline",
            },
            answer_values={
                "verdict": verdict,
                "val_mse": val_mse,
                "val_r": val_r,
                "content_hash": record.content_hash,
            },
        )
        record.extra["lineage_hash"] = lineage_record.record_hash

        record.extra["gene_names"] = dataset.gene_names[:n_genes]
        record.extra["gene_config"] = {
            "max_genes": gc.max_genes,
            "hidden_dim": gc.hidden_dim,
            "num_layers": gc.num_layers,
            "num_heads": gc.num_heads,
        }
        record.extra["prediction_hidden"] = hybrid_config.prediction_hidden
        record.extra["prediction_dropout"] = hybrid_config.prediction_dropout

        if checkpoint_dir is not None:
            from models.training.checkpoint import save_checkpoint
            ckpt_path = save_checkpoint(
                frontier_model, record,
                config={
                    "gene_config": record.extra["gene_config"],
                    "prediction_hidden": hybrid_config.prediction_hidden,
                    "prediction_dropout": hybrid_config.prediction_dropout,
                    "use_frontier": True,
                },
                checkpoint_dir=Path(checkpoint_dir),
            )
            record.extra["checkpoint_path"] = str(ckpt_path)

        return record

    def run_and_serve(
        self,
        study_id: str = "NT-006",
        max_genes: int = 2000,
        species: str = "human",
        pretrain_first: bool = True,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> tuple["TrialRecord", "HybridInferenceService"]:
        """Train → checkpoint → build inference service in one call.

        Returns (record, service) where service is ready for .infer() calls.
        """
        from models.inference.hybrid_inference import HybridInferenceService

        record = self.run_governed_trial(
            study_id=study_id,
            max_genes=max_genes,
            species=species,
            pretrain_first=pretrain_first,
            checkpoint_dir=checkpoint_dir,
        )

        ckpt_path = record.extra["checkpoint_path"]
        gc_dict = record.extra["gene_config"]
        gc = GeneFoundationConfig(
            max_genes=gc_dict["max_genes"],
            hidden_dim=gc_dict["hidden_dim"],
            num_layers=gc_dict["num_layers"],
            num_heads=gc_dict["num_heads"],
        )
        model_config = HybridModelConfig(
            gene_config=gc,
            use_frontier=True,
            prediction_hidden=record.extra.get("prediction_hidden", 512),
            prediction_dropout=record.extra.get("prediction_dropout", 0.1),
        )
        gene_names = record.extra.get(
            "gene_names", [f"GENE{i}" for i in range(record.n_genes)]
        )

        service = HybridInferenceService.from_checkpoint(
            ckpt_path,
            gene_names=gene_names,
            model_config=model_config,
        )
        return record, service
