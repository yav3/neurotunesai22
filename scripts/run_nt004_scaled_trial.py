"""NT-004 Governed Training Trial -- scaled multi-series pooled perturbation response.

NT-003 proved pooling helps via diversity (heterogeneous perturbation pairs)
but was limited to 9 common genes between a full matrix and a small subset.

NT-004 uses FULL expression matrices for 5 series (now available in Merlin's
committed data):
  GSE119262: 17183 genes x 44 samples (autophagy, gene symbols)
  GSE26927:  20590 genes x 118 samples (neuroinflammation, Illumina probes)
  GSE15222:  24355 genes x 363 samples (Alzheimer genetics, GenBank IDs)
  GSE5281:   54676 genes x 161 samples (Alzheimer brain, Affymetrix probes)
  GSE30174:  54676 genes x 80 samples (cancer radiation, Affymetrix probes)

All series are human, all have >=30 samples (data-driven regime).

Finding from first run (2026-10-31): cross-platform gene-ID namespaces are
incompatible. Only same-platform pairs share IDs (GSE5281+GSE30174 share
54,675 Affymetrix probes). Probe-to-gene-symbol mapping requires platform
annotation files not available in this environment. Also, 54K features with
121 pairs (p/n=452) caused NaN training collapse in the VAE.

Strategy (revised):
  1. Find gene intersection; expect cross-platform zeros
  2. Top-variance feature selection (max 2000) to control p/n ratio
  3. Pool perturbation pairs (baseline=per-study mean, target=each sample)
  4. Arm A: single series with most pairs; Arm B: multi-series pooled
  5. Gradient clipping to prevent NaN training collapse

Pre-registered criteria (same 4-criterion protocol as NT-001/002/003):
  C1: Model MSE < mean-baseline MSE
  C2: Model MSE < linear-shift MSE
  C3: Pearson r(predicted, target) > 0.3
  C4: Model Pearson r > mean-baseline r

Seed: 2026. Split: 20% val, 10% test. Species: human (firewall-enforced).
Regime: data-driven (all series >=30 samples).
Max features: 2000 (variance-selected to keep p/n manageable).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data.geo_loader import load_geo_csv
from models.foundation.perturbation_response import GEOExpressionData

MERLIN_HUMAN_DIR = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")
RECORDS_DIR = Path(__file__).parent.parent / "docs" / "records"
SEED = 2026
MAX_FEATURES = 2000

FULL_MATRICES = [
    ("gse119262_autophagy_full.csv", "GSE119262", "autophagy"),
    ("gse26927_neurodegeneration_full.csv", "GSE26927", "neurodegeneration"),
    ("gse15222_neurodegeneration_full.csv", "GSE15222", "neurodegeneration"),
    ("gse5281_neurodegeneration_full.csv", "GSE5281", "neurodegeneration"),
    ("gse30174_drug_response_full.csv", "GSE30174", "drug_response"),
]


def find_gene_intersection(datasets: list[GEOExpressionData]) -> list[str]:
    """Find genes common to all datasets, preserving first dataset's order."""
    if not datasets:
        return []
    common = set(datasets[0].gene_names)
    for ds in datasets[1:]:
        common &= set(ds.gene_names)
    return [g for g in datasets[0].gene_names if g in common]


def subset_to_common_genes(
    ds: GEOExpressionData, common_genes: list[str]
) -> GEOExpressionData:
    """Subset a dataset to common genes."""
    gene_idx = {g: i for i, g in enumerate(ds.gene_names)}
    indices = [gene_idx[g] for g in common_genes if g in gene_idx]
    kept_genes = [g for g in common_genes if g in gene_idx]
    return GEOExpressionData(
        expression=ds.expression[:, indices],
        conditions=ds.conditions,
        gene_names=kept_genes,
        species=ds.species,
        accession=ds.accession,
    )


def select_top_variance_features(
    datasets: list[GEOExpressionData], max_features: int = MAX_FEATURES
) -> list[GEOExpressionData]:
    """Select top-variance features across pooled data to control p/n ratio."""
    n_genes = datasets[0].expression.shape[1]
    if n_genes <= max_features:
        return datasets

    pooled = np.vstack([ds.expression for ds in datasets])
    gene_var = np.var(pooled, axis=0)
    top_idx = np.argsort(gene_var)[-max_features:]
    top_idx_sorted = np.sort(top_idx)

    gene_names = datasets[0].gene_names
    kept_genes = [gene_names[i] for i in top_idx_sorted]

    result = []
    for ds in datasets:
        result.append(GEOExpressionData(
            expression=ds.expression[:, top_idx_sorted],
            conditions=ds.conditions,
            gene_names=kept_genes,
            species=ds.species,
            accession=ds.accession,
        ))

    print(f"  Variance selection: {n_genes} -> {max_features} features "
          f"(var range: {gene_var[top_idx_sorted[0]]:.4f} to {gene_var[top_idx_sorted[-1]]:.4f})")
    return result


def make_pairs(ds: GEOExpressionData) -> list[tuple]:
    """Build (baseline, target) pairs. Baseline = per-study mean."""
    ctrl_idx = [i for i, c in enumerate(ds.conditions) if c == "control"]
    treat_idx = [i for i, c in enumerate(ds.conditions) if c == "treated"]

    if ctrl_idx and treat_idx:
        ctrl_mean = ds.expression[ctrl_idx].mean(axis=0)
        return [(ctrl_mean, ds.expression[ti]) for ti in treat_idx]

    n = ds.expression.shape[0]
    if n < 4:
        return []
    half = n // 2
    baseline = ds.expression[:half].mean(axis=0)
    return [(baseline, ds.expression[i]) for i in range(half, n)]


def split_pairs(pairs, seed=SEED, val_frac=0.2, test_frac=0.1):
    """Deterministic train/val/test split."""
    rng = np.random.RandomState(seed)
    n = len(pairs)
    idx = rng.permutation(n).tolist()
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train = max(1, n - 2)
        n_val = max(1, (n - n_train) // 2)
        n_test = n - n_train - n_val
    train = [pairs[i] for i in idx[:n_train]]
    val = [pairs[i] for i in idx[n_train : n_train + n_val]]
    test = [pairs[i] for i in idx[n_train + n_val :]]
    return train, val, test


def compute_metrics(predictions, targets):
    """MSE and Pearson r."""
    preds = np.array(predictions)
    targs = np.array(targets)
    mse = float(np.mean((preds - targs) ** 2))
    if np.isnan(mse) or np.isinf(mse):
        return {"mse": float("nan"), "pearson_r": float("nan")}
    fp, ft = preds.flatten(), targs.flatten()
    if np.std(fp) < 1e-10 or np.std(ft) < 1e-10:
        r = 0.0
    else:
        r = float(np.corrcoef(fp, ft)[0, 1])
    return {"mse": round(mse, 6), "pearson_r": round(r, 6)}


def run_baselines(train, val, test):
    """Mean, linear-shift, and permutation baselines."""
    all_train_targets = np.array([t for _, t in train])
    mean_target = all_train_targets.mean(axis=0)

    val_t = np.array([t for _, t in val])
    test_t = np.array([t for _, t in test])
    val_b = np.array([b for b, _ in val])
    test_b = np.array([b for b, _ in test])

    mean_val = compute_metrics(np.tile(mean_target, (len(val), 1)), val_t)
    mean_test = compute_metrics(np.tile(mean_target, (len(test), 1)), test_t)

    all_train_b = np.array([b for b, _ in train])
    shift = mean_target - all_train_b.mean(axis=0)
    shift_val = compute_metrics(val_b + shift, val_t)
    shift_test = compute_metrics(test_b + shift, test_t)

    rng = np.random.RandomState(SEED + 1)
    perm_val = compute_metrics(
        all_train_targets[rng.randint(0, len(all_train_targets), len(val))], val_t
    )
    perm_test = compute_metrics(
        all_train_targets[rng.randint(0, len(all_train_targets), len(test))], test_t
    )

    return {
        "mean_baseline": {"val": mean_val, "test": mean_test},
        "linear_shift": {"val": shift_val, "test": shift_test},
        "permutation": {"val": perm_val, "test": perm_test},
    }


def run_linear_vae(train, val, test, n_genes, latent_dim=16, epochs=300, lr=1e-3):
    """Linear VAE with gradient clipping."""
    import torch
    import torch.nn as nn

    class LinearVAE(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.enc_mu = nn.Linear(input_dim, latent_dim)
            self.enc_logvar = nn.Linear(input_dim, latent_dim)
            self.dec = nn.Linear(latent_dim, input_dim)

        def forward(self, baseline, target):
            delta = target - baseline
            mu = self.enc_mu(delta)
            logvar = torch.clamp(self.enc_logvar(delta), -10, 10)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            pred = baseline + self.dec(z)
            recon = nn.functional.mse_loss(pred, target)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return pred, recon + 0.1 * kl

    torch.manual_seed(SEED)

    model = LinearVAE(n_genes, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_b = torch.tensor(np.array([b for b, _ in train]), dtype=torch.float32)
    train_t = torch.tensor(np.array([t for _, t in train]), dtype=torch.float32)
    val_b = torch.tensor(np.array([b for b, _ in val]), dtype=torch.float32)
    val_t = torch.tensor(np.array([t for _, t in val]), dtype=torch.float32)
    test_b = torch.tensor(np.array([b for b, _ in test]), dtype=torch.float32)
    test_t = torch.tensor(np.array([t for _, t in test]), dtype=torch.float32)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, loss = model(train_b, train_t)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    NaN/Inf loss at epoch {epoch+1}, stopping")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                _, vl = model(val_b, val_t)
            vl_val = vl.item()
            if vl_val < best_val_loss:
                best_val_loss = vl_val
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 3:
                print(f"    Early stop at epoch {epoch+1}")
                break

    model.eval()
    with torch.no_grad():
        val_pred, _ = model(val_b, val_t)
        test_pred, _ = model(test_b, test_t)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": "LinearVAE",
        "latent_dim": latent_dim,
        "n_params": n_params,
        "epochs": epoch + 1,
        "val": compute_metrics(val_pred.numpy(), val_t.numpy()),
        "test": compute_metrics(test_pred.numpy(), test_t.numpy()),
    }


def evaluate_criteria(arm, baselines):
    """4-criterion protocol."""
    m_mse = arm["val"]["mse"]
    m_r = arm["val"]["pearson_r"]
    mean_mse = baselines["mean_baseline"]["val"]["mse"]
    shift_mse = baselines["linear_shift"]["val"]["mse"]
    mean_r = baselines["mean_baseline"]["val"]["pearson_r"]

    nan_present = any(
        x != x for x in [m_mse, m_r, mean_mse, shift_mse, mean_r]
    )
    if nan_present:
        return [
            {"criterion": "C1", "met": False, "detail": "NaN in metrics"},
            {"criterion": "C2", "met": False, "detail": "NaN in metrics"},
            {"criterion": "C3", "met": False, "detail": "NaN in metrics"},
            {"criterion": "C4", "met": False, "detail": "NaN in metrics"},
        ]

    return [
        {"criterion": "C1", "met": m_mse < mean_mse,
         "detail": f"model={m_mse:.6f} vs mean={mean_mse:.6f}"},
        {"criterion": "C2", "met": m_mse < shift_mse,
         "detail": f"model={m_mse:.6f} vs shift={shift_mse:.6f}"},
        {"criterion": "C3", "met": m_r > 0.3,
         "detail": f"model_r={m_r:.4f} vs 0.3"},
        {"criterion": "C4", "met": m_r > mean_r,
         "detail": f"model_r={m_r:.4f} vs mean_r={mean_r:.4f}"},
    ]


def main():
    t0 = time.time()
    print("NT-004: Scaled multi-series pooled trial (revised)")
    print(f"  SEED={SEED}, MAX_FEATURES={MAX_FEATURES}")
    print()

    # -- Load all full matrices --
    datasets: list[GEOExpressionData] = []
    programs: dict[str, str] = {}
    for fname, acc, prog in FULL_MATRICES:
        fpath = MERLIN_HUMAN_DIR / fname
        if not fpath.exists():
            print(f"  SKIP {acc}: file not found ({fname})")
            continue
        ds = load_geo_csv(fpath, accession=acc, species="human", log1p=True)
        n_ctrl = sum(1 for c in ds.conditions if c == "control")
        n_treat = sum(1 for c in ds.conditions if c == "treated")
        print(f"  {acc}: {ds.expression.shape[1]} genes x {ds.expression.shape[0]} samples "
              f"({n_ctrl}c/{n_treat}t) [{prog}]")
        datasets.append(ds)
        programs[acc] = prog

    if len(datasets) < 2:
        print("Need at least 2 datasets. Exiting.")
        return

    # -- Find gene intersection --
    common_genes = find_gene_intersection(datasets)
    print(f"\nGene intersection across {len(datasets)} series: {len(common_genes)} genes")

    cross_platform_note = ""
    if len(common_genes) < 100:
        print(f"  Only {len(common_genes)} common genes -- cross-platform ID mismatch")
        cross_platform_note = (
            "Cross-platform gene-ID namespaces incompatible: gene symbols (GSE119262), "
            "Illumina probes (GSE26927), GenBank IDs (GSE15222), Affymetrix probes "
            "(GSE5281/GSE30174). Only same-platform pairs share IDs."
        )
        print(f"  NOTE: {cross_platform_note}")

        best_pair = None
        best_count = 0
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                cg = find_gene_intersection([datasets[i], datasets[j]])
                if len(cg) > best_count:
                    best_count = len(cg)
                    best_pair = (i, j)
                print(f"    {datasets[i].accession} ^ {datasets[j].accession}: {len(cg)} genes")

        if best_pair and best_count >= 500:
            i, j = best_pair
            print(f"\n  Best pair: {datasets[i].accession} + {datasets[j].accession} "
                  f"({best_count} genes)")
            datasets = [datasets[i], datasets[j]]
            common_genes = find_gene_intersection(datasets)
        else:
            print("  No pair with >=500 common genes. Cannot proceed.")
            return

    # -- Subset all datasets to common genes --
    subsetted = [subset_to_common_genes(ds, common_genes) for ds in datasets]

    # -- Variance-based feature selection --
    subsetted = select_top_variance_features(subsetted, MAX_FEATURES)
    n_genes = subsetted[0].expression.shape[1]
    print(f"\nWorking with {n_genes} features across {len(subsetted)} series")

    # -- Build pairs --
    all_pairs: list[tuple] = []
    pair_sources: list[str] = []
    pair_counts: dict[str, int] = {}
    for ds in subsetted:
        ps = make_pairs(ds)
        print(f"  {ds.accession}: {len(ps)} pairs")
        all_pairs.extend(ps)
        pair_sources.extend([ds.accession] * len(ps))
        pair_counts[ds.accession] = len(ps)

    n_total_pairs = len(all_pairs)
    print(f"\nTotal pairs: {n_total_pairs}")
    print(f"p/n ratio: {n_genes}/{n_total_pairs} = {n_genes/max(n_total_pairs,1):.1f}")

    # -- Arm A: Single-series (series with most pairs) --
    arm_a_acc = max(pair_counts, key=pair_counts.get)
    arm_a_ds = next(ds for ds in subsetted if ds.accession == arm_a_acc)
    pairs_a = make_pairs(arm_a_ds)
    if len(pairs_a) < 5:
        print(f"WARNING: Too few single-series pairs for {arm_a_acc}.")
        return

    train_a, val_a, test_a = split_pairs(pairs_a)
    print(f"\nArm A (single-series, {arm_a_acc}):")
    print(f"  {len(pairs_a)} pairs -> {len(train_a)} train / {len(val_a)} val / {len(test_a)} test")

    baselines_a = run_baselines(train_a, val_a, test_a)
    lvae_a = run_linear_vae(train_a, val_a, test_a, n_genes, latent_dim=16, epochs=300)
    criteria_a = evaluate_criteria(lvae_a, baselines_a)
    verdict_a = "met" if all(c["met"] for c in criteria_a) else "not_met"

    print(f"  LinearVAE({lvae_a['n_params']} params): "
          f"MSE={lvae_a['val']['mse']}, r={lvae_a['val']['pearson_r']}")
    for c in criteria_a:
        status = "MET" if c["met"] else "NOT_MET"
        print(f"    {c['criterion']}: {status} -- {c['detail']}")
    print(f"  Verdict: {verdict_a.upper()}")

    # -- Arm B: Multi-series pooled --
    train_b, val_b, test_b = split_pairs(all_pairs)
    print(f"\nArm B (pooled, {len(subsetted)} series):")
    print(f"  {n_total_pairs} pairs -> {len(train_b)} train / {len(val_b)} val / {len(test_b)} test")

    baselines_b = run_baselines(train_b, val_b, test_b)
    lvae_b = run_linear_vae(train_b, val_b, test_b, n_genes, latent_dim=16, epochs=300)
    criteria_b = evaluate_criteria(lvae_b, baselines_b)
    verdict_b = "met" if all(c["met"] for c in criteria_b) else "not_met"

    print(f"  LinearVAE({lvae_b['n_params']} params): "
          f"MSE={lvae_b['val']['mse']}, r={lvae_b['val']['pearson_r']}")
    for c in criteria_b:
        status = "MET" if c["met"] else "NOT_MET"
        print(f"    {c['criterion']}: {status} -- {c['detail']}")
    print(f"  Verdict: {verdict_b.upper()}")

    elapsed = time.time() - t0

    # -- Build record --
    record = {
        "experiment_id": "NT-004",
        "date": str(date.today()),
        "revision": "variance-selected (max 2000 features)",
        "cross_platform_note": cross_platform_note,
        "parameters": {
            "seed": SEED,
            "species": "human",
            "regime": "data-driven (all series >=30 samples)",
            "max_features": MAX_FEATURES,
            "n_features_used": n_genes,
            "series_used": [ds.accession for ds in subsetted],
            "series_programs": {ds.accession: programs.get(ds.accession, "unknown")
                                for ds in subsetted},
            "n_series": len(subsetted),
            "total_pairs": n_total_pairs,
            "pair_counts": pair_counts,
            "p_n_ratio": round(n_genes / max(n_total_pairs, 1), 2),
            "latent_dim": 16,
            "feature_selection": "top variance across pooled samples",
        },
        "arm_a_single_series": {
            **lvae_a,
            "series": arm_a_acc,
            "n_pairs": len(pairs_a),
            "criteria": criteria_a,
            "verdict": verdict_a,
            "baselines": baselines_a,
        },
        "arm_b_multi_series_pooled": {
            **lvae_b,
            "n_series": len(subsetted),
            "n_pairs": n_total_pairs,
            "criteria": criteria_b,
            "verdict": verdict_b,
            "baselines": baselines_b,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    record["record_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORDS_DIR / "NT-004_result.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Record: {out}")
    print(f"Hash: {record['record_hash'][:16]}...")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
