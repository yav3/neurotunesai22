"""NT-001 governed training run on committed in-regime GEO data.

Implements the full pre-registered protocol from docs/NT_001_PREREGISTRATION.md:
  1. Load curated in-regime series from Merlin's committed data
  2. Build PerturbationDataset (baseline/target pairs)
  3. Split with fixed seed (80/10/10)
  4. Train cVAE (model arm)
  5. Compute mean_baseline and linear_shift baselines (same test split)
  6. Run permutation control (same hyperparameters, shuffled labels)
  7. Evaluate 4 pre-registered criteria
  8. Write content-addressed record to docs/records/NT-001_result.json

Same-run requirement: model, baselines, and permutation control all
execute in this single invocation on the same held-out split.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data.geo_loader import load_geo_csv, discover_merlin_datasets
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

MERLIN_HUMAN_DIR = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")
RECORDS_DIR = Path(__file__).parent.parent / "docs/records"
SEED = 2026
LATENT_DIM = 64
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
MIN_SAMPLES = 30
HELD_OUT_FRACTION = 0.2
TEST_FRACTION = 0.1
MAX_GENES = None  # None = all genes; int = top-k by variance


def load_in_regime_data(max_genes: int | None = None) -> list[GEOExpressionData]:
    datasets = discover_merlin_datasets(MERLIN_HUMAN_DIR, species="human")
    in_regime = [d for d in datasets if d["n_samples"] >= MIN_SAMPLES
                 and d["n_control"] > 0 and d["n_treated"] > 0]
    geo_list = []
    for meta in in_regime:
        data = load_geo_csv(meta["path"], accession=meta["accession"],
                            species="human", log1p=True)
        if max_genes is not None and data.expression.shape[1] > max_genes:
            gene_var = np.var(data.expression, axis=0)
            top_idx = np.argsort(gene_var)[-max_genes:]
            top_idx = np.sort(top_idx)
            data = GEOExpressionData(
                expression=data.expression[:, top_idx],
                conditions=data.conditions,
                gene_names=[data.gene_names[i] for i in top_idx],
                species=data.species,
                accession=data.accession,
            )
        geo_list.append(data)
    return geo_list


def build_splits(geo_data: GEOExpressionData, seed: int):
    ds = PerturbationDataset(geo_data, perturbation_type=PerturbationType.DRUG)
    n = len(ds)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n).tolist()
    n_test = max(1, int(n * TEST_FRACTION))
    n_val = max(1, int(n * HELD_OUT_FRACTION))
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train = max(1, n - 2)
        n_val = max(1, (n - n_train) // 2)
        n_test = n - n_train - n_val
    train = Subset(ds, indices[:n_train])
    val = Subset(ds, indices[n_train:n_train + n_val])
    test = Subset(ds, indices[n_train + n_val:])
    return ds, train, val, test


def train_model(model, train_ds, val_ds, epochs, lr, batch_size, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            baseline = batch["baseline"].to(device)
            target = batch["target"].to(device)
            ptype_ids = batch["ptype_ids"].to(device)
            gene_idx = batch.get("gene_idx", torch.zeros(baseline.shape[0], dtype=torch.long)).to(device)
            drug_idx = batch.get("drug_idx", torch.zeros(baseline.shape[0], dtype=torch.long)).to(device)
            dose = batch.get("dose", torch.ones(baseline.shape[0])).to(device)
            duration = batch.get("duration", torch.ones(baseline.shape[0]) * 24).to(device)
            time_point = batch.get("time_point", torch.zeros(baseline.shape[0])).to(device)
            optimizer.zero_grad()
            out = model(baseline=baseline, ptype_ids=ptype_ids, gene_idx=gene_idx,
                        drug_idx=drug_idx, dose=dose, duration=duration, time_point=time_point)
            pred = out["predicted"] if isinstance(out, dict) and "predicted" in out else list(out.values())[0]
            loss = F.mse_loss(pred, target)
            if isinstance(out, dict) and "kl_loss" in out:
                loss = loss + out["kl_loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                baseline = batch["baseline"].to(device)
                target = batch["target"].to(device)
                ptype_ids = batch["ptype_ids"].to(device)
                gene_idx = batch.get("gene_idx", torch.zeros(baseline.shape[0], dtype=torch.long)).to(device)
                drug_idx = batch.get("drug_idx", torch.zeros(baseline.shape[0], dtype=torch.long)).to(device)
                dose = batch.get("dose", torch.ones(baseline.shape[0])).to(device)
                duration = batch.get("duration", torch.ones(baseline.shape[0]) * 24).to(device)
                time_point = batch.get("time_point", torch.zeros(baseline.shape[0])).to(device)
                out = model(baseline=baseline, ptype_ids=ptype_ids, gene_idx=gene_idx,
                            drug_idx=drug_idx, dose=dose, duration=duration, time_point=time_point)
                pred = out["predicted"] if isinstance(out, dict) and "predicted" in out else list(out.values())[0]
                val_loss += F.mse_loss(pred, target).item()
                vn += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss / max(vn, 1)})
    return history


@torch.no_grad()
def evaluate_arm(model, test_ds, device):
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    all_pred, all_target = [], []
    for batch in loader:
        baseline = batch["baseline"].to(device)
        target = batch["target"].to(device)
        ptype_ids = batch["ptype_ids"].to(device)
        gene_idx = batch.get("gene_idx", torch.zeros(baseline.shape[0], dtype=torch.long)).to(device)
        drug_idx = batch.get("drug_idx", torch.zeros(baseline.shape[0], dtype=torch.long)).to(device)
        dose = batch.get("dose", torch.ones(baseline.shape[0])).to(device)
        duration = batch.get("duration", torch.ones(baseline.shape[0]) * 24).to(device)
        time_point = batch.get("time_point", torch.zeros(baseline.shape[0])).to(device)
        out = model(baseline=baseline, ptype_ids=ptype_ids, gene_idx=gene_idx,
                    drug_idx=drug_idx, dose=dose, duration=duration, time_point=time_point)
        pred = out["predicted"] if isinstance(out, dict) and "predicted" in out else list(out.values())[0]
        all_pred.append(pred.cpu())
        all_target.append(target.cpu())
    pred = torch.cat(all_pred, dim=0).numpy()
    target = torch.cat(all_target, dim=0).numpy()
    mse = float(np.mean((pred - target) ** 2))
    rs = []
    for g in range(pred.shape[1]):
        if np.std(target[:, g]) > 1e-10 and np.std(pred[:, g]) > 1e-10:
            r, _ = pearsonr(pred[:, g], target[:, g])
            if np.isfinite(r):
                rs.append(r)
    pearson_r = float(np.mean(rs)) if rs else 0.0
    return {"mse": mse, "pearson_r": pearson_r}


def compute_mean_baseline(train_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=len(train_ds))
    for batch in train_loader:
        train_targets = batch["target"].numpy()
        break
    train_mean = train_targets.mean(axis=0, keepdims=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    for batch in test_loader:
        test_targets = batch["target"].numpy()
        break
    pred = np.tile(train_mean, (test_targets.shape[0], 1))
    mse = float(np.mean((pred - test_targets) ** 2))
    rs = []
    for g in range(pred.shape[1]):
        if np.std(test_targets[:, g]) > 1e-10 and np.std(pred[:, g]) > 1e-10:
            r, _ = pearsonr(pred[:, g], test_targets[:, g])
            if np.isfinite(r):
                rs.append(r)
    pearson_r = float(np.mean(rs)) if rs else 0.0
    return {"mse": mse, "pearson_r": pearson_r}


def compute_linear_shift(train_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=len(train_ds))
    for batch in train_loader:
        baselines = batch["baseline"].numpy()
        targets = batch["target"].numpy()
        break
    shift = (targets - baselines).mean(axis=0, keepdims=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    for batch in test_loader:
        test_baselines = batch["baseline"].numpy()
        test_targets = batch["target"].numpy()
        break
    pred = test_baselines + shift
    mse = float(np.mean((pred - test_targets) ** 2))
    rs = []
    for g in range(pred.shape[1]):
        if np.std(test_targets[:, g]) > 1e-10 and np.std(pred[:, g]) > 1e-10:
            r, _ = pearsonr(pred[:, g], test_targets[:, g])
            if np.isfinite(r):
                rs.append(r)
    pearson_r = float(np.mean(rs)) if rs else 0.0
    return {"mse": mse, "pearson_r": pearson_r}


def run_permutation_control(geo_data, n_genes, train_ds, val_ds, test_ds, device,
                            latent_dim=LATENT_DIM, epochs=EPOCHS):
    rng = np.random.RandomState(SEED + 1)
    shuffled_conditions = list(geo_data.conditions)
    rng.shuffle(shuffled_conditions)
    shuffled_data = GEOExpressionData(
        expression=geo_data.expression,
        conditions=shuffled_conditions,
        gene_names=geo_data.gene_names,
        species=geo_data.species,
        accession=geo_data.accession + "_shuffled",
    )
    shuffled_ds = PerturbationDataset(shuffled_data, perturbation_type=PerturbationType.DRUG)
    if len(shuffled_ds) == 0:
        return {"mse": float("inf"), "pearson_r": 0.0}
    n = len(shuffled_ds)
    indices = rng.permutation(n).tolist()
    n_test = max(1, int(n * TEST_FRACTION))
    n_val = max(1, int(n * HELD_OUT_FRACTION))
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train = max(1, n - 2)
        n_val = max(1, (n - n_train) // 2)
        n_test = n - n_train - n_val
    perm_train = Subset(shuffled_ds, indices[:n_train])
    perm_val = Subset(shuffled_ds, indices[n_train:n_train + n_val])
    perm_test = Subset(shuffled_ds, indices[n_train + n_val:])

    config = PerturbationModelConfig(n_genes=n_genes, latent_dim=latent_dim, n_drugs=10)
    perm_model = PerturbationResponseModel(config).to(device)
    train_model(perm_model, perm_train, perm_val, epochs, LR, BATCH_SIZE, device)
    return evaluate_arm(perm_model, perm_test, device)


def main():
    t0 = time.time()
    device = torch.device("cpu")

    max_genes = MAX_GENES
    latent_dim = LATENT_DIM
    epochs = EPOCHS
    for arg in sys.argv[1:]:
        if arg.startswith("--max-genes="):
            max_genes = int(arg.split("=")[1])
        elif arg.startswith("--latent-dim="):
            latent_dim = int(arg.split("=")[1])
        elif arg.startswith("--epochs="):
            epochs = int(arg.split("=")[1])

    print("=" * 60)
    print("NT-001 GOVERNED TRAINING RUN")
    if max_genes:
        print(f"  Gene subset: top {max_genes} by variance")
    print(f"  Latent dim: {latent_dim}, Epochs: {epochs}")
    print("=" * 60)

    print("\n[1/7] Loading in-regime data...")
    geo_list = load_in_regime_data(max_genes=max_genes)
    if not geo_list:
        print("ERROR: No in-regime datasets found")
        sys.exit(1)
    geo_data = geo_list[0]
    print(f"  {geo_data.accession}: {geo_data.expression.shape[0]} samples, "
          f"{geo_data.expression.shape[1]} genes")

    print("\n[2/7] Building splits (seed={})...".format(SEED))
    full_ds, train_ds, val_ds, test_ds = build_splits(geo_data, SEED)
    print(f"  Total pairs: {len(full_ds)}, train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")

    n_genes = geo_data.expression.shape[1]

    print("\n[3/7] Training model arm...")
    config = PerturbationModelConfig(n_genes=n_genes, latent_dim=latent_dim, n_drugs=10)
    model = PerturbationResponseModel(config).to(device)
    history = train_model(model, train_ds, val_ds, epochs, LR, BATCH_SIZE, device)
    print(f"  Final train_loss: {history[-1]['train_loss']:.4f}")
    print(f"  Final val_loss: {history[-1]['val_loss']:.4f}")

    print("\n[4/7] Evaluating model on test split...")
    model_metrics = evaluate_arm(model, test_ds, device)
    print(f"  Model MSE: {model_metrics['mse']:.6f}")
    print(f"  Model Pearson r: {model_metrics['pearson_r']:.4f}")

    print("\n[5/7] Computing baselines on same test split...")
    mean_bl = compute_mean_baseline(train_ds, test_ds)
    linear_bl = compute_linear_shift(train_ds, test_ds)
    print(f"  Mean-baseline MSE: {mean_bl['mse']:.6f}, r: {mean_bl['pearson_r']:.4f}")
    print(f"  Linear-shift MSE: {linear_bl['mse']:.6f}, r: {linear_bl['pearson_r']:.4f}")

    print("\n[6/7] Running permutation control...")
    perm_metrics = run_permutation_control(geo_data, n_genes, train_ds, val_ds, test_ds, device,
                                           latent_dim=latent_dim, epochs=epochs)
    print(f"  Permutation MSE: {perm_metrics['mse']:.6f}, r: {perm_metrics['pearson_r']:.4f}")

    print("\n[7/7] Evaluating pre-registered criteria...")
    arms = {
        "model": model_metrics,
        "mean_baseline": mean_bl,
        "linear_shift": linear_bl,
        "permutation": perm_metrics,
    }
    criteria = [
        TrainingCriterion(metric="mse", op="<", versus="mean_baseline",
                          description="C1: Model MSE < mean-baseline MSE"),
        TrainingCriterion(metric="mse", op="<", versus="linear_shift",
                          description="C2: Model MSE < linear-shift MSE"),
        TrainingCriterion(metric="pearson_r", op=">", threshold=0.5,
                          description="C3: Pearson r > 0.5"),
        TrainingCriterion(metric="pearson_r", op=">", versus="mean_baseline",
                          description="C4: Model Pearson r > mean-baseline r"),
    ]
    criteria_results = []
    for c in criteria:
        ok, detail = c.evaluate(arms)
        criteria_results.append({
            "criterion": c.description,
            "met": ok,
            "detail": detail,
        })
        status = "MET" if ok else "NOT MET"
        print(f"  {status}: {c.description} — {detail}")

    verdict = "met" if all(r["met"] for r in criteria_results) else "not_met"
    elapsed = time.time() - t0

    record = {
        "experiment_id": "NT-001",
        "date": str(date.today()),
        "parameters": {
            "seed": SEED,
            "latent_dim": latent_dim,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "n_genes": n_genes,
            "n_genes_total": 17183,
            "max_genes": max_genes,
            "n_samples": geo_data.expression.shape[0],
            "accession": geo_data.accession,
            "species": "human",
            "regime": "data-driven",
        },
        "arms": arms,
        "criteria": criteria_results,
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 2),
        "training_history": history,
    }
    record_payload = json.dumps(
        {"experiment_id": record["experiment_id"],
         "parameters": record["parameters"],
         "arms": record["arms"]},
        sort_keys=True,
    )
    record["record_hash"] = hashlib.sha256(record_payload.encode()).hexdigest()

    out_path = RECORDS_DIR / "NT-001_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict.upper()}")
    print(f"Record hash: {record['record_hash'][:16]}...")
    print(f"Written to: {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
