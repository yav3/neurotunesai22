"""Finetuning entry point for downstream tasks.

Usage:
    python -m models.training.finetune --task classification \\
        --data-dir synthetic_data/quarantine \\
        --checkpoint checkpoints/pretrain/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models.config import FoundationConfig, FinetuneConfig
from models.foundation.neurofoundation import NeuroFoundation
from models.downstream.classification import DiseaseClassifier
from models.downstream.progression import ProgressionPredictor
from models.downstream.treatment import TreatmentResponsePredictor
from models.downstream.imputation import ModalityImputer
from models.data.dataset import NeuroDataset
from models.data.collator import ModalityMaskingCollator
from models.training.trainer import Trainer


class ClassificationTrainer(Trainer):
    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        return self.model.loss(batch["x_dict"], batch["labels"], batch.get("mask_dict"))


class ImputationTrainer(Trainer):
    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        return self.model.loss(batch["x_dict"], batch["x_dict"], batch.get("mask_dict"))


TASK_MAP = {
    "classification": (DiseaseClassifier, ClassificationTrainer),
    "imputation": (ModalityImputer, ImputationTrainer),
}


def main():
    parser = argparse.ArgumentParser(description="Finetune NeuroFoundation")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_MAP.keys()))
    parser.add_argument("--data-dir", type=str, default="synthetic_data/quarantine")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fc = FoundationConfig()
    foundation = NeuroFoundation(fc)
    state = ckpt.get("model_state_dict", ckpt)
    # Strip 'model.' prefix if saved from PretrainingObjective
    cleaned = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
               for k, v in state.items()}
    foundation.load_state_dict(cleaned, strict=False)

    ftc = FinetuneConfig(
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        freeze_encoder_epochs=args.freeze_epochs, seed=args.seed,
    )

    HeadCls, TrainerCls = TASK_MAP[args.task]
    head = HeadCls(foundation, ftc)

    dataset = NeuroDataset(args.data_dir, max_samples=args.max_samples)
    val_size = min(len(dataset) // 10, 20_000)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    collator = ModalityMaskingCollator()
    train_loader = DataLoader(
        train_ds, batch_size=ftc.batch_size, shuffle=True,
        num_workers=4, collate_fn=collator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=ftc.batch_size, shuffle=False,
        num_workers=2, collate_fn=collator, pin_memory=True,
    )

    ckpt_dir = args.checkpoint_dir or f"checkpoints/finetune_{args.task}"
    print(f"Task: {args.task} | Params: {sum(p.numel() for p in head.parameters()):,}")

    trainer = TrainerCls(
        model=head,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=ftc.lr,
        weight_decay=ftc.weight_decay,
        warmup_steps=int(ftc.warmup_ratio * ftc.epochs * len(train_loader)),
        max_epochs=ftc.epochs,
        fp16=True,
        checkpoint_dir=ckpt_dir,
        patience=ftc.patience,
    )
    history = trainer.train()
    print(f"Finetuning ({args.task}) complete.")


if __name__ == "__main__":
    main()
