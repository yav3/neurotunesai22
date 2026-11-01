"""Core training loop with mixed-precision, gradient accumulation, and
cosine-warmup scheduling."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Generic trainer for both pretraining and finetuning."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        grad_accum_steps: int = 1,
        fp16: bool = True,
        checkpoint_dir: str | Path = "checkpoints",
        patience: int = 10,
        device: str | torch.device | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_accum_steps = grad_accum_steps
        self.fp16 = fp16 and self.device != "cpu"
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        total = max_steps or (max_epochs or 1) * len(train_loader)
        self.scheduler = cosine_warmup_schedule(self.optimizer, warmup_steps, total)
        self.scaler = GradScaler(enabled=self.fp16)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()
        batch = self._to_device(batch)

        with autocast(device_type=self.device if isinstance(self.device, str) else "cuda",
                       enabled=self.fp16):
            outputs = self._forward(batch)
            loss = outputs["loss"] / self.grad_accum_steps

        self.scaler.scale(loss).backward()

        if (self.global_step + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        self.global_step += 1
        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items() if k != "logits" and k != "predictions"
                and k != "imputed" and k != "per_modality"}

    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclass must implement _forward")

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        accum: dict[str, float] = {}
        count = 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            outputs = self._forward(batch)
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    accum[k] = accum.get(k, 0.0) + v.item()
            count += 1
        return {k: v / max(count, 1) for k, v in accum.items()}

    def train(self) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {}
        epoch = 0

        while True:
            epoch += 1
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                metrics = self._train_step(batch)
                for k, v in metrics.items():
                    history.setdefault(f"train_{k}", []).append(v)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

                if self.max_steps and self.global_step >= self.max_steps:
                    break

            val_metrics = self._validate()
            for k, v in val_metrics.items():
                history.setdefault(f"val_{k}", []).append(v)

            val_loss = val_metrics.get("loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                self.patience_counter += 1

            if self.max_steps and self.global_step >= self.max_steps:
                break
            if self.max_epochs and epoch >= self.max_epochs:
                break
            if self.patience_counter >= self.patience:
                break

        self._save_checkpoint("final.pt")
        return history

    def _save_checkpoint(self, name: str):
        path = self.checkpoint_dir / name
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, dict):
                out[k] = {kk: vv.to(self.device) if isinstance(vv, torch.Tensor)
                          else vv for kk, vv in v.items()}
            else:
                out[k] = v
        return out
