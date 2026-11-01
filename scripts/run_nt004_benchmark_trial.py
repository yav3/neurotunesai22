"""NT-004 Governed Benchmark Trial — DREAM5 gold-standard validation.

Pre-registered criterion: AUROC > 0.55 (better than random at 0.50).
Dataset: DREAM5 Network1 (in_silico, 100 genes, 195 TFs, 805 samples).
Regime: benchmark (dedicated validation regime).
Seed: 2026.

This trial tests whether the perturbation-response cVAE encodes regulatory
structure. The model is trained on DREAM5 expression data, then each TF is
virtually perturbed and downstream effects scored against the gold standard.

A correlation baseline (|Pearson r| between TF and target) runs as the
pre-registered control in the same invocation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from models.data.prior_network_loader import load_dream5_benchmark
from models.foundation.perturbation_response import (
    GEOExpressionData,
    PerturbationModelConfig,
    PerturbationResponseModel,
    PerturbationType,
)
from models.training.governed_harness import TrainingCriterion, TrainingSpec
from models.training.geo_training_pipeline import (
    prepare_datasets,
    train_epoch,
    evaluate,
    GEOTrainingConfig,
)
from models.validation.benchmark_evaluator import (
    infer_regulatory_scores,
    score_against_gold_standard,
    evaluate_random_baseline,
    evaluate_correlation_baseline,
)


DREAM5_DIR = Path("/home/user/merlinaineuralplus2/backend/data/benchmarks/dream5")


def main():
    t0 = time.time()

    print("Loading DREAM5 benchmark...")
    bench = load_dream5_benchmark(DREAM5_DIR)
    print(f"  Genes: {len(bench.gene_names)}")
    print(f"  TFs: {len(bench.tf_names)}")
    print(f"  Samples: {bench.expression.shape[0]}")
    print(f"  True edges: {bench.n_true_edges}")
    print(f"  False edges: {bench.n_false_edges}")
    print(f"  SHA-256: {bench.content_hash[:16]}...")

    # Wrap as GEOExpressionData for the training pipeline
    n_samples = bench.expression.shape[0]
    n_genes = bench.expression.shape[1]
    conditions = ["control"] * (n_samples // 2) + ["treated"] * (n_samples - n_samples // 2)

    data = GEOExpressionData(
        expression=bench.expression,
        gene_names=bench.gene_names,
        conditions=conditions,
        species="in_silico",
        accession="DREAM5_network1",
    )

    # Pre-register spec
    spec = TrainingSpec(
        model_id="PerturbationResponseModel",
        dataset_id="DREAM5_network1",
        seed=2026,
        held_out_fraction=0.2,
        species="in_silico",
        regime="benchmark",
        criteria=(
            TrainingCriterion(metric="auroc", op=">", threshold=0.55,
                              description="AUROC > 0.55 (random baseline ≈ 0.50)"),
            TrainingCriterion(metric="auroc", op=">", versus="correlation",
                              description="Model AUROC beats correlation baseline"),
        ),
    )
    spec_hash = spec.content_hash()
    print(f"\nSpec hash: {spec_hash}")

    # Training config
    config = GEOTrainingConfig(
        seed=2026,
        held_out_fraction=0.2,
        test_fraction=0.1,
        batch_size=32,
        epochs=80,
        learning_rate=5e-4,
        latent_dim=32,
        val_loss_threshold=2.0,
        perturbation_type="drug",
        log1p=True,
    )

    # Prepare datasets and train
    train_ds, val_ds, test_ds = prepare_datasets([data], config)
    print(f"\nSplit: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    model_config = PerturbationModelConfig(
        n_genes=n_genes,
        latent_dim=config.latent_dim,
        encoder_hidden=256,
        decoder_hidden=256,
        n_drugs=1,
    )
    model = PerturbationResponseModel(model_config)
    device = torch.device("cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

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

    # Benchmark evaluation — infer regulatory scores from trained model
    print("\nInferring regulatory scores from trained model...")
    gene_idx = {g: i for i, g in enumerate(bench.gene_names)}
    tf_indices = [gene_idx[tf] for tf in bench.tf_names if tf in gene_idx]

    regulatory_scores = infer_regulatory_scores(
        model=model,
        expression=bench.expression,
        tf_indices=tf_indices,
        gene_names=bench.gene_names,
        perturbation_magnitude=2.0,
        device="cpu",
    )

    model_result = score_against_gold_standard(regulatory_scores, bench)
    print(f"  Model AUROC: {model_result.auroc:.4f}")
    print(f"  Model AUPRC: {model_result.auprc:.4f}")
    print(f"  Precision@10: {model_result.precision_at_k.get(10, 0):.4f}")
    print(f"  Precision@50: {model_result.precision_at_k.get(50, 0):.4f}")

    # Control: correlation baseline (same invocation)
    print("\nCorrelation baseline (control)...")
    corr_result = evaluate_correlation_baseline(bench)
    print(f"  Correlation AUROC: {corr_result.auroc:.4f}")
    print(f"  Correlation AUPRC: {corr_result.auprc:.4f}")

    # Control: random baseline
    print("\nRandom baseline (control)...")
    rand_result = evaluate_random_baseline(bench)
    print(f"  Random AUROC: {rand_result.auroc:.4f}")

    # Evaluate criteria
    arms = {
        "model": {"auroc": model_result.auroc, "auprc": model_result.auprc},
        "correlation": {"auroc": corr_result.auroc, "auprc": corr_result.auprc},
        "random": {"auroc": rand_result.auroc, "auprc": rand_result.auprc},
    }

    criteria_results = []
    for criterion in spec.criteria:
        ok, msg = criterion.evaluate(arms)
        criteria_results.append({"criterion": str(criterion), "met": ok, "detail": msg})
        print(f"\nCriterion: {msg} → {'MET' if ok else 'NOT MET'}")

    verdict = "met" if all(r["met"] for r in criteria_results) else "not_met"
    elapsed = time.time() - t0

    record = {
        "trial_id": "NT-004",
        "spec_hash": spec_hash,
        "dataset": {
            "source": "DREAM5_network1",
            "species": "in_silico",
            "modality": "benchmark",
            "n_genes": len(bench.gene_names),
            "n_tfs": len(bench.tf_names),
            "n_samples": int(bench.expression.shape[0]),
            "n_true_edges": bench.n_true_edges,
            "n_false_edges": bench.n_false_edges,
            "sha256": bench.content_hash,
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
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": val_metrics.get("val_loss") if val_metrics else None,
            "best_val_loss": min((h["val_loss"] for h in history), default=None) if history else None,
        },
        "results": {
            "model_auroc": model_result.auroc,
            "model_auprc": model_result.auprc,
            "model_precision_at_k": {str(k): v for k, v in model_result.precision_at_k.items()},
            "correlation_auroc": corr_result.auroc,
            "correlation_auprc": corr_result.auprc,
            "random_auroc": rand_result.auroc,
        },
        "criteria_results": criteria_results,
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 1),
    }

    record_path = Path(__file__).parent.parent / "docs" / "trials" / "NT-004_benchmark.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nRecord written: {record_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
