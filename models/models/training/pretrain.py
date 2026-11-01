"""Pretraining entry point for NeuroFoundation.

Usage:
    python -m models.training.pretrain --data-dir synthetic_data/quarantine \\
        --max-steps 100000 --batch-size 512
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from models.config import FoundationConfig, PretrainConfig
from models.foundation.neurofoundation import NeuroFoundation
from models.foundation.objectives import PretrainingObjective
from models.data.dataset import NeuroDataset
from models.data.collator import ModalityMaskingCollator
from models.training.trainer import Trainer


class PretrainTrainer(Trainer):
    """Specialises Trainer for self-supervised pretraining."""

    def __init__(self, objective: PretrainingObjective, **kwargs):
        super().__init__(model=objective, **kwargs)
        self.objective = objective

    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        return self.objective(batch["x_dict"])


def main():
    parser = argparse.ArgumentParser(description="Pretrain NeuroFoundation")
    parser.add_argument("--data-dir", type=str, default="synthetic_data/quarantine")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fc = FoundationConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    pc = PretrainConfig(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        fp16=args.fp16,
        seed=args.seed,
    )

    dataset = NeuroDataset(args.data_dir, max_samples=args.max_samples)
    val_size = min(len(dataset) // 10, 50_000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    collator = ModalityMaskingCollator()
    train_loader = DataLoader(
        train_ds, batch_size=pc.batch_size, shuffle=True,
        num_workers=4, collate_fn=collator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=pc.batch_size, shuffle=False,
        num_workers=2, collate_fn=collator, pin_memory=True,
    )

    model = NeuroFoundation(fc)
    objective = PretrainingObjective(model, pc)

    print(f"Model parameters: {model.num_parameters:,}")
    print(f"Training samples: {train_size:,}  Validation: {val_size:,}")

    trainer = PretrainTrainer(
        objective=objective,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=pc.lr,
        weight_decay=pc.weight_decay,
        warmup_steps=pc.warmup_steps,
        max_steps=pc.max_steps,
        grad_accum_steps=pc.grad_accum_steps,
        fp16=pc.fp16,
        checkpoint_dir=args.checkpoint_dir,
    )
    history = trainer.train()
    print("Pretraining complete.")


if __name__ == "__main__":
    main()
