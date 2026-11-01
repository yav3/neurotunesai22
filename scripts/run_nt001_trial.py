"""NT-001 Governed Training Trial — GSE119262 in-regime.

Pre-registered criterion: val_loss < 2.0 (MSE in log1p space).
Dataset: GSE119262 (human, everolimus, 44 samples, 17183 genes, data-driven regime).
Species: human (firewall-enforced).
Seed: 2026.
Split: 20% val, 10% test, 70% train.

This is the first in-regime governed training run of the perturbation-response cVAE.
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from models.data.geo_loader import load_geo_csv, content_hash
from models.foundation.perturbation_response import (
    GEOExpressionData,
    PerturbationDataset,
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


MERLIN_DATA = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")
TARGET_CSV = MERLIN_DATA / "gse119262_autophagy_full.csv"


def main():
    t0 = time.time()

    # 1. Load the in-regime dataset
    print(f"Loading {TARGET_CSV.name}...")
    data = load_geo_csv(
        TARGET_CSV,
        accession="GSE119262",
        species="human",
        log1p=False,  # PerturbationDataset handles log1p
    )
    n_control = sum(1 for c in data.conditions if c == "control")
    n_treated = sum(1 for c in data.conditions if c == "treated")
    sha = content_hash(TARGET_CSV)

    print(f"  Samples: {data.expression.shape[0]} ({n_control}C / {n_treated}T)")
    print(f"  Genes: {len(data.gene_names)}")
    print(f"  SHA-256: {sha[:16]}...")

    # 2. Regime selection (pre-registered before method)
    regime = select_training_regime(
        n_samples=data.expression.shape[0],
        n_conditions=2,  # control + treated
    )
    print(f"  Regime: {regime}")
    assert regime == "data-driven", f"Expected data-driven, got {regime}"

    # 3. Pre-register the training spec
    spec = TrainingSpec(
        model_id="PerturbationResponseModel",
        dataset_id="GSE119262",
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

    # 4. Prepare datasets (with evidence discipline split)
    config = GEOTrainingConfig(
        seed=2026,
        held_out_fraction=0.2,
        test_fraction=0.1,
        batch_size=16,
        epochs=50,
        learning_rate=1e-3,
        latent_dim=64,
        val_loss_threshold=2.0,
        perturbation_type="drug",
        log1p=True,
    )
    train_ds, val_ds, test_ds = prepare_datasets([data], config)
    print(f"\nSplit: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    # 5. Build model
    n_genes = data.expression.shape[1]
    model_config = PerturbationModelConfig(
        n_genes=n_genes,
        latent_dim=config.latent_dim,
        encoder_hidden=256,
        decoder_hidden=256,
        n_drugs=10,
    )
    model = PerturbationResponseModel(model_config)
    device = torch.device("cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters ({n_params * 4 / 1e6:.1f} MB float32)")

    # 6. Train
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

    # 7. Test evaluation
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size)
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest metrics: {test_metrics}")

    # 8. Evaluate criteria
    criteria_results = []
    for criterion in spec.criteria:
        ok, msg = criterion.evaluate({"model": val_metrics})
        criteria_results.append({"criterion": str(criterion), "met": ok, "detail": msg})
        print(f"Criterion: {msg} → {'MET' if ok else 'NOT MET'}")

    verdict = "met" if all(r["met"] for r in criteria_results) else "not_met"
    elapsed = time.time() - t0

    # 9. Write record
    record = {
        "trial_id": "NT-001",
        "spec_hash": spec_hash,
        "dataset": {
            "accession": "GSE119262",
            "species": "human",
            "n_samples": int(data.expression.shape[0]),
            "n_genes": int(n_genes),
            "n_control": n_control,
            "n_treated": n_treated,
            "sha256": sha,
            "regime": regime,
        },
        "model": {
            "id": "PerturbationResponseModel",
            "n_params": n_params,
            "latent_dim": config.latent_dim,
            "encoder_hidden": 256,
            "decoder_hidden": 256,
        },
        "training": {
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.learning_rate,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
        },
        "results": {
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": val_metrics["val_loss"],
            "best_val_loss": best_val,
            "test_loss": test_metrics["val_loss"],
        },
        "criteria_results": criteria_results,
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 1),
    }

    record_path = Path(__file__).parent.parent / "docs" / "trials" / "NT-001_GSE119262.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nRecord written: {record_path}")
    print(f"Verdict: {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
