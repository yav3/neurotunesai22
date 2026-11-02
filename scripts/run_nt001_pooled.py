"""NT-001 governed training with multi-series pooling.

Pools all committed human GEO series that have both control and treated
conditions, using their gene intersection. Perturbation pairs form
within each study (no cross-study pairing).

Pre-registered criteria are the same 4 from NT_001_PREREGISTRATION.md:
  C1: Model MSE < mean-baseline MSE
  C2: Model MSE < linear-shift MSE
  C3: Pearson r > 0.5
  C4: Model Pearson r > mean-baseline Pearson r
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
from torch.utils.data import DataLoader, ConcatDataset, Subset
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data.geo_loader import load_geo_csv, discover_merlin_datasets
from models.data.multi_series_pool import (
    find_gene_intersection,
    subset_to_genes,
    build_study_aware_dataset,
)
from models.foundation.perturbation_response import (
    GEOExpressionData,
    PerturbationDataset,
    PerturbationModelConfig,
    PerturbationResponseModel,
    PerturbationType,
)
from models.training.governed_harness import TrainingCriterion

MERLIN_DIR = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")
RECORDS_DIR = Path(__file__).parent.parent / "docs/records"
SEED = 2026
LATENT_DIM = 32
EPOCHS = 100
BATCH_SIZE = 8
LR = 5e-4
HELD_OUT = 0.2
TEST_FRAC = 0.1


def load_qualifying_series() -> list[GEOExpressionData]:
    metas = discover_merlin_datasets(MERLIN_DIR, species="human")
    datasets = []
    for m in sorted(metas, key=lambda x: -x["n_samples"]):
        if m["n_control"] > 0 and m["n_treated"] > 0:
            d = load_geo_csv(m["path"], accession=m["accession"],
                             species="human", log1p=True)
            datasets.append(d)
    return datasets


def split_dataset(ds, seed):
    n = len(ds)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n).tolist()
    n_test = max(1, int(n * TEST_FRAC))
    n_val = max(1, int(n * HELD_OUT))
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train = max(1, n - 2)
        n_val = max(1, (n - n_train) // 2)
        n_test = n - n_train - n_val
    return (Subset(ds, idx[:n_train]),
            Subset(ds, idx[n_train:n_train + n_val]),
            Subset(ds, idx[n_train + n_val:]))


def train_model(model, train_ds, val_ds, epochs, lr, batch_size, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    history = []
    for epoch in range(epochs):
        model.train()
        tloss, nb = 0.0, 0
        for batch in train_loader:
            bl = batch["baseline"].to(device)
            tgt = batch["target"].to(device)
            pt = batch["ptype_ids"].to(device)
            gi = batch.get("gene_idx", torch.zeros(bl.shape[0], dtype=torch.long)).to(device)
            di = batch.get("drug_idx", torch.zeros(bl.shape[0], dtype=torch.long)).to(device)
            dose = batch.get("dose", torch.ones(bl.shape[0])).to(device)
            dur = batch.get("duration", torch.ones(bl.shape[0]) * 24).to(device)
            tp = batch.get("time_point", torch.zeros(bl.shape[0])).to(device)
            opt.zero_grad()
            out = model(baseline=bl, ptype_ids=pt, gene_idx=gi, drug_idx=di,
                        dose=dose, duration=dur, time_point=tp)
            pred = out["predicted"] if isinstance(out, dict) and "predicted" in out else list(out.values())[0]
            loss = F.mse_loss(pred, tgt)
            if isinstance(out, dict) and "kl_loss" in out:
                loss = loss + out["kl_loss"]
            loss.backward()
            opt.step()
            tloss += loss.item()
            nb += 1
        tl = tloss / max(nb, 1)
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                bl = batch["baseline"].to(device)
                tgt = batch["target"].to(device)
                pt = batch["ptype_ids"].to(device)
                gi = batch.get("gene_idx", torch.zeros(bl.shape[0], dtype=torch.long)).to(device)
                di = batch.get("drug_idx", torch.zeros(bl.shape[0], dtype=torch.long)).to(device)
                dose = batch.get("dose", torch.ones(bl.shape[0])).to(device)
                dur = batch.get("duration", torch.ones(bl.shape[0]) * 24).to(device)
                tp = batch.get("time_point", torch.zeros(bl.shape[0])).to(device)
                out = model(baseline=bl, ptype_ids=pt, gene_idx=gi, drug_idx=di,
                            dose=dose, duration=dur, time_point=tp)
                pred = out["predicted"] if isinstance(out, dict) and "predicted" in out else list(out.values())[0]
                vloss += F.mse_loss(pred, tgt).item()
                vn += 1
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vloss / max(vn, 1)})
    return history


@torch.no_grad()
def evaluate_arm(model, test_ds, device):
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    all_pred, all_tgt = [], []
    for batch in loader:
        bl = batch["baseline"].to(device)
        tgt = batch["target"].to(device)
        pt = batch["ptype_ids"].to(device)
        gi = batch.get("gene_idx", torch.zeros(bl.shape[0], dtype=torch.long)).to(device)
        di = batch.get("drug_idx", torch.zeros(bl.shape[0], dtype=torch.long)).to(device)
        dose = batch.get("dose", torch.ones(bl.shape[0])).to(device)
        dur = batch.get("duration", torch.ones(bl.shape[0]) * 24).to(device)
        tp = batch.get("time_point", torch.zeros(bl.shape[0])).to(device)
        out = model(baseline=bl, ptype_ids=pt, gene_idx=gi, drug_idx=di,
                    dose=dose, duration=dur, time_point=tp)
        pred = out["predicted"] if isinstance(out, dict) and "predicted" in out else list(out.values())[0]
        all_pred.append(pred.cpu())
        all_tgt.append(tgt.cpu())
    pred = torch.cat(all_pred).numpy()
    tgt = torch.cat(all_tgt).numpy()
    mse = float(np.mean((pred - tgt) ** 2))
    rs = []
    for g in range(pred.shape[1]):
        if np.std(tgt[:, g]) > 1e-10 and np.std(pred[:, g]) > 1e-10:
            r, _ = pearsonr(pred[:, g], tgt[:, g])
            if np.isfinite(r):
                rs.append(r)
    return {"mse": mse, "pearson_r": float(np.mean(rs)) if rs else 0.0}


def compute_mean_baseline(train_ds, test_ds):
    tl = DataLoader(train_ds, batch_size=len(train_ds))
    for b in tl:
        train_tgt = b["target"].numpy(); break
    mean = train_tgt.mean(axis=0, keepdims=True)
    tl2 = DataLoader(test_ds, batch_size=len(test_ds))
    for b in tl2:
        test_tgt = b["target"].numpy(); break
    pred = np.tile(mean, (test_tgt.shape[0], 1))
    mse = float(np.mean((pred - test_tgt) ** 2))
    rs = []
    for g in range(pred.shape[1]):
        if np.std(test_tgt[:, g]) > 1e-10 and np.std(pred[:, g]) > 1e-10:
            r, _ = pearsonr(pred[:, g], test_tgt[:, g])
            if np.isfinite(r): rs.append(r)
    return {"mse": mse, "pearson_r": float(np.mean(rs)) if rs else 0.0}


def compute_linear_shift(train_ds, test_ds):
    tl = DataLoader(train_ds, batch_size=len(train_ds))
    for b in tl:
        bl = b["baseline"].numpy(); tgt = b["target"].numpy(); break
    shift = (tgt - bl).mean(axis=0, keepdims=True)
    tl2 = DataLoader(test_ds, batch_size=len(test_ds))
    for b in tl2:
        test_bl = b["baseline"].numpy(); test_tgt = b["target"].numpy(); break
    pred = test_bl + shift
    mse = float(np.mean((pred - test_tgt) ** 2))
    rs = []
    for g in range(pred.shape[1]):
        if np.std(test_tgt[:, g]) > 1e-10 and np.std(pred[:, g]) > 1e-10:
            r, _ = pearsonr(pred[:, g], test_tgt[:, g])
            if np.isfinite(r): rs.append(r)
    return {"mse": mse, "pearson_r": float(np.mean(rs)) if rs else 0.0}


def main():
    t0 = time.time()
    device = torch.device("cpu")

    print("=" * 60)
    print("NT-001 POOLED GOVERNED TRAINING")
    print("=" * 60)

    print("\n[1/7] Loading qualifying series...")
    all_series = load_qualifying_series()
    print(f"  {len(all_series)} series with both control and treated conditions")
    for d in all_series:
        nc = sum(1 for c in d.conditions if c == "control")
        nt = sum(1 for c in d.conditions if c == "treated")
        print(f"    {d.accession}: {d.expression.shape[0]} samples ({nc}C/{nt}T), "
              f"{d.expression.shape[1]} genes")

    common = find_gene_intersection(all_series)
    print(f"\n  Gene intersection: {len(common)} genes")

    if len(common) == 0:
        print("\n  No common genes — trying pairwise pooling with largest series...")
        largest = all_series[0]  # sorted by n_samples desc
        best_pair = None
        best_overlap = 0
        for other in all_series[1:]:
            overlap = len(set(largest.gene_names) & set(other.gene_names))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = other
        if best_pair and best_overlap > 0:
            all_series = [largest, best_pair]
            common = find_gene_intersection(all_series)
            print(f"  Best pair: {largest.accession} + {best_pair.accession}, "
                  f"{len(common)} common genes")
        else:
            print("  ERROR: No pairwise gene overlap found")
            sys.exit(1)

    print(f"\n[2/7] Building study-aware perturbation dataset...")
    combined, genes, n_pairs = build_study_aware_dataset(
        all_series, perturbation_type=PerturbationType.DRUG,
    )
    n_genes = len(genes)
    print(f"  {n_pairs} perturbation pairs on {n_genes} common genes")

    print(f"\n[3/7] Splitting (seed={SEED})...")
    train_ds, val_ds, test_ds = split_dataset(combined, SEED)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    print(f"\n[4/7] Training model (latent={LATENT_DIM}, epochs={EPOCHS})...")
    cfg = PerturbationModelConfig(n_genes=n_genes, latent_dim=LATENT_DIM, n_drugs=10)
    model = PerturbationResponseModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")
    history = train_model(model, train_ds, val_ds, EPOCHS, LR, BATCH_SIZE, device)
    print(f"  Final train_loss: {history[-1]['train_loss']:.4f}")
    print(f"  Final val_loss: {history[-1]['val_loss']:.4f}")

    print(f"\n[5/7] Evaluating on test split...")
    model_m = evaluate_arm(model, test_ds, device)
    print(f"  Model MSE: {model_m['mse']:.6f}, r: {model_m['pearson_r']:.4f}")

    mean_bl = compute_mean_baseline(train_ds, test_ds)
    linear_bl = compute_linear_shift(train_ds, test_ds)
    print(f"  Mean-baseline MSE: {mean_bl['mse']:.6f}, r: {mean_bl['pearson_r']:.4f}")
    print(f"  Linear-shift MSE: {linear_bl['mse']:.6f}, r: {linear_bl['pearson_r']:.4f}")

    print(f"\n[6/7] Permutation control...")
    rng = np.random.RandomState(SEED + 1)
    all_items = [combined[i] for i in range(len(combined))]
    shuffled_targets = [item["target"].clone() for item in all_items]
    perm_idx = rng.permutation(len(shuffled_targets))
    shuffled_targets = [shuffled_targets[i] for i in perm_idx]

    class PermutedDataset(torch.utils.data.Dataset):
        def __init__(self, items, shuffled_tgts):
            self.items = items
            self.shuffled = shuffled_tgts
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            item = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in self.items[idx].items()}
            item["target"] = self.shuffled[idx]
            return item

    perm_ds = PermutedDataset(all_items, shuffled_targets)
    perm_train, perm_val, perm_test = split_dataset(perm_ds, SEED + 1)
    perm_model = PerturbationResponseModel(cfg).to(device)
    train_model(perm_model, perm_train, perm_val, EPOCHS, LR, BATCH_SIZE, device)
    perm_m = evaluate_arm(perm_model, perm_test, device)
    print(f"  Permutation MSE: {perm_m['mse']:.6f}, r: {perm_m['pearson_r']:.4f}")

    print(f"\n[7/7] Evaluating criteria...")
    arms = {"model": model_m, "mean_baseline": mean_bl,
            "linear_shift": linear_bl, "permutation": perm_m}
    criteria = [
        TrainingCriterion(metric="mse", op="<", versus="mean_baseline",
                          description="C1: Model MSE < mean-baseline MSE"),
        TrainingCriterion(metric="mse", op="<", versus="linear_shift",
                          description="C2: Model MSE < linear-shift MSE"),
        TrainingCriterion(metric="pearson_r", op=">", threshold=0.5,
                          description="C3: Pearson r > 0.5"),
        TrainingCriterion(metric="pearson_r", op=">", versus="mean_baseline",
                          description="C4: Model r > mean-baseline r"),
    ]
    results = []
    for c in criteria:
        ok, detail = c.evaluate(arms)
        results.append({"criterion": c.description, "met": ok, "detail": detail})
        print(f"  {'MET' if ok else 'NOT MET'}: {c.description} — {detail}")

    verdict = "met" if all(r["met"] for r in results) else "not_met"
    elapsed = time.time() - t0

    accessions = sorted({d.accession for d in all_series})
    record = {
        "experiment_id": "NT-001-pooled",
        "date": str(date.today()),
        "parameters": {
            "seed": SEED, "latent_dim": LATENT_DIM, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR,
            "n_genes": n_genes, "n_pairs": n_pairs,
            "n_series": len(all_series),
            "accessions": accessions,
            "species": "human", "regime": "pooled-multi-series",
        },
        "arms": arms,
        "criteria": results,
        "verdict": verdict,
        "elapsed_seconds": round(elapsed, 2),
        "training_history": [history[0], history[-1]],
    }
    payload = json.dumps({"experiment_id": record["experiment_id"],
                          "parameters": record["parameters"],
                          "arms": record["arms"]}, sort_keys=True)
    record["record_hash"] = hashlib.sha256(payload.encode()).hexdigest()

    out = RECORDS_DIR / "NT-001_pooled_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict.upper()}")
    print(f"Record: {out}")
    print(f"Hash: {record['record_hash'][:16]}...")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
