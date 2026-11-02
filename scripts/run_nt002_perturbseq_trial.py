"""NT-002 Governed Training Trial — Xaira Perturb-seq panel (pseudo-bulk).

Pre-registered criterion: val_loss < 2.0 (MSE in log1p space).
Dataset: Xaira panel (human, HCT116, 30 knockout targets, 30 genes, data-driven regime).
Species: human (firewall-enforced).
Seed: 2026.
Split: 20% val, 10% test, 70% train.

This is the first Perturb-seq governed training run — single-cell data aggregated
to pseudo-bulk (one sample per knockout target) then trained through the same
perturbation-response cVAE as NT-001.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from models.data.perturbseq_loader import (
    load_perturbseq_parquet,
    pseudobulk_by_target,
)
from models.data.geo_loader import content_hash
from models.foundation.perturbation_response import (
    PerturbationModelConfig,
    PerturbationResponseModel,
    PerturbationType,
)
from models.training.governed_harness import (
    TrainingCriterion,
    TrainingSpec,
    select_training_regime,
)
from models.training.geo_training_pipeline import (
    prepare_datasets,
    train_epoch,
    evaluate,
    GEOTrainingConfig,
)


PERTURBSEQ_PATH = Path(
    "/home/user/merlinaineuralplus2/backend/data/perturbseq/human/xaira_panel_cells_subset.parquet"
)


def main():
    t0 = time.time()

    print(f"Loading {PERTURBSEQ_PATH.name}...")
    expr, targets, genes, batches, meta = load_perturbseq_parquet(PERTURBSEQ_PATH)
    print(f"  Cells: {meta.n_cells}")
    print(f"  Genes: {meta.n_genes}")
    print(f"  Targets: {meta.n_targets}")
    print(f"  Control cells: {meta.n_control_cells}")
    print(f"  Batches: {meta.n_batches}")
    print(f"  SHA-256: {meta.content_hash[:16]}...")

    # Pseudo-bulk aggregation
    data = pseudobulk_by_target(expr, targets, genes, log1p=True, min_cells=10)
    n_control = sum(1 for c in data.conditions if c == "control")
    n_treated = sum(1 for c in data.conditions if c == "treated")

    print(f"\nPseudo-bulk samples: {data.expression.shape[0]} ({n_control}C / {n_treated}T)")
    print(f"  Genes: {len(data.gene_names)}")

    # Regime selection
    regime = select_training_regime(
        n_samples=data.expression.shape[0],
        n_conditions=2,
    )
    print(f"  Regime: {regime}")

    # Pre-register training spec
    spec = TrainingSpec(
        model_id="PerturbationResponseModel",
        dataset_id="xaira_perturbseq_panel",
        seed=2026,
        held_out_fraction=0.2,
        species="human",
        regime=regime,
        criteria=(
            TrainingCriterion(metric="val_loss", op="<", threshold=2.0),
        ),
    )
    spec_hash = spec.content_hash()
    print(f"\nSpec hash: {spec_hash}")

    # Prepare datasets
    config = GEOTrainingConfig(
        seed=2026,
        held_out_fraction=0.2,
        test_fraction=0.1,
        batch_size=8,
        epochs=50,
        learning_rate=1e-3,
        latent_dim=32,
        val_loss_threshold=2.0,
        perturbation_type="knockout",
        log1p=False,  # already log1p'd in pseudobulk
    )
    train_ds, val_ds, test_ds = prepare_datasets([data], config)
    print(f"\nSplit: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    if len(train_ds) == 0:
        print("WARNING: No training pairs — insufficient samples for split.")
        record = _make_record(spec_hash, meta, data, config, regime,
                              [], {}, {}, "not_met", "Insufficient samples",
                              time.time() - t0)
        _write_record(record)
        return

    # Build model
    n_genes = data.expression.shape[1]
    model_config = PerturbationModelConfig(
        n_genes=n_genes,
        latent_dim=config.latent_dim,
        encoder_hidden=128,
        decoder_hidden=128,
        n_drugs=1,
    )
    model = PerturbationResponseModel(model_config)
    device = torch.device("cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters ({n_params * 4 / 1e6:.1f} MB float32)")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size)

    print(f"\nTraining {config.epochs} epochs...")
    history = []
    best_val = float("inf")
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  best={best_val:.4f}")

    # Test evaluation
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size)
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest metrics: {test_metrics}")

    # Evaluate criteria
    criteria_results = []
    for criterion in spec.criteria:
        ok, msg = criterion.evaluate({"model": val_metrics})
        criteria_results.append({"criterion": str(criterion), "met": ok, "detail": msg})
        print(f"Criterion: {msg} → {'MET' if ok else 'NOT MET'}")

    verdict = "met" if all(r["met"] for r in criteria_results) else "not_met"
    elapsed = time.time() - t0

    record = _make_record(
        spec_hash, meta, data, config, regime,
        history, val_metrics, test_metrics,
        verdict, "", elapsed,
    )
    _write_record(record)


def _make_record(spec_hash, meta, data, config, regime,
                 history, val_metrics, test_metrics, verdict, note, elapsed):
    n_genes = data.expression.shape[1]
    n_params = 0
    if history:
        model_config = PerturbationModelConfig(
            n_genes=n_genes, latent_dim=config.latent_dim,
            encoder_hidden=128, decoder_hidden=128, n_drugs=1,
        )
        model = PerturbationResponseModel(model_config)
        n_params = sum(p.numel() for p in model.parameters())

    return {
        "trial_id": "NT-002",
        "spec_hash": spec_hash,
        "dataset": {
            "source": "xaira_perturbseq_panel",
            "species": "human",
            "modality": "perturbseq",
            "n_cells": meta.n_cells,
            "n_genes": int(n_genes),
            "n_targets": meta.n_targets,
            "n_pseudobulk_samples": int(data.expression.shape[0]),
            "sha256": meta.content_hash,
            "regime": regime,
        },
        "model": {
            "id": "PerturbationResponseModel",
            "n_params": n_params,
            "latent_dim": config.latent_dim,
            "encoder_hidden": 128,
            "decoder_hidden": 128,
        },
        "training": {
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.learning_rate,
            "n_train": len(history) if history else 0,
            "n_val": 0,
            "n_test": 0,
        },
        "results": {
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": val_metrics.get("val_loss") if val_metrics else None,
            "best_val_loss": min((h["val_loss"] for h in history), default=None) if history else None,
            "test_loss": test_metrics.get("val_loss") if test_metrics else None,
        },
        "criteria_results": [],
        "verdict": verdict,
        "verdict_note": note,
        "elapsed_seconds": round(elapsed, 1),
    }


def _write_record(record):
    record_path = Path(__file__).parent.parent / "docs" / "trials" / "NT-002_perturbseq.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nRecord written: {record_path}")
    print(f"Verdict: {record['verdict']}")


if __name__ == "__main__":
    main()
