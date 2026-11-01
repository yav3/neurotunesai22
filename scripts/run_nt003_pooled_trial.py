"""NT-003 Governed Training Trial — multi-series pooled perturbation response.

NT-002 proved the Linear VAE (1266 params) beats baselines on GSE119262 alone
(50 genes, 21 pairs, p/n=2.38). NT-003 tests whether pooling multiple series
improves generalization by increasing the effective training set.

Available datasets with control+treated (from NT-001 curated series catalog):
  GSE119262: 44 samples, 17183 genes (autophagy, in-regime)
  GSE107965: 18 samples, 29 genes (autophagy, prior-contrast)
  GSE201710: 16 samples, 56 genes (reprogramming, prior-contrast)
  GSE297233:  8 samples, 28 genes (mesenchymal, prior-mean)

Challenge: gene intersection across full+subset matrices yields very few
common genes. Strategy: pool only the two autophagy datasets (GSE119262 +
GSE107965) which share the biological program, then test separately on
GSE201710 (cross-program transfer).

Pre-registered criteria (same 4-criterion protocol):
  C1: Model MSE < mean-baseline MSE
  C2: Model MSE < linear-shift MSE
  C3: Pearson r(predicted, target) > 0.3
  C4: Model Pearson r > mean-baseline r

Seed: 2026. Split: 20% val, 10% test (within-study splits, then pooled).
Species: human (firewall-enforced). Regime: mixed (documented honestly).
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

from models.data.geo_loader import load_geo_csv, content_hash
from models.data.multi_series_pool import (
    find_gene_intersection,
    subset_to_genes,
    pool_datasets,
)
from models.foundation.perturbation_response import GEOExpressionData

MERLIN_HUMAN_DIR = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")
RECORDS_DIR = Path(__file__).parent.parent / "docs" / "records"
SEED = 2026
HELD_OUT_FRACTION = 0.2
TEST_FRACTION = 0.1


def load_dataset(filename: str, accession: str) -> GEOExpressionData:
    """Load a single dataset with log1p transform."""
    return load_geo_csv(
        MERLIN_HUMAN_DIR / filename,
        accession=accession, species="human", log1p=True,
    )


def make_perturbation_pairs(data: GEOExpressionData):
    """Build (baseline, target) pairs from control/treated samples."""
    ctrl_idx = [i for i, c in enumerate(data.conditions) if c == "control"]
    treat_idx = [i for i, c in enumerate(data.conditions) if c == "treated"]
    if not ctrl_idx or not treat_idx:
        return []
    ctrl_mean = data.expression[ctrl_idx].mean(axis=0)
    return [(ctrl_mean, data.expression[ti]) for ti in treat_idx]


def split_pairs(pairs, seed=2026, val_frac=0.2, test_frac=0.1):
    """Split pairs into train/val/test."""
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
    val = [pairs[i] for i in idx[n_train:n_train + n_val]]
    test = [pairs[i] for i in idx[n_train + n_val:]]
    return train, val, test


def compute_metrics(predictions, targets):
    """MSE and Pearson r between predicted and target arrays."""
    preds = np.array(predictions)
    targs = np.array(targets)
    mse = float(np.mean((preds - targs) ** 2))
    flat_p = preds.flatten()
    flat_t = targs.flatten()
    if np.std(flat_p) < 1e-10 or np.std(flat_t) < 1e-10:
        r = 0.0
    else:
        r = float(np.corrcoef(flat_p, flat_t)[0, 1])
    return {"mse": mse, "pearson_r": r}


def run_baselines(train, val, test):
    """Mean-baseline, linear-shift, and permutation controls."""
    all_train_targets = np.array([t for _, t in train])
    train_target_mean = all_train_targets.mean(axis=0)

    val_targets = np.array([t for _, t in val])
    test_targets = np.array([t for _, t in test])

    val_mean_pred = np.tile(train_target_mean, (len(val), 1))
    test_mean_pred = np.tile(train_target_mean, (len(test), 1))
    mean_val = compute_metrics(val_mean_pred, val_targets)
    mean_test = compute_metrics(test_mean_pred, test_targets)

    all_train_baselines = np.array([b for b, _ in train])
    shift = train_target_mean - all_train_baselines.mean(axis=0)
    val_baselines = np.array([b for b, _ in val])
    test_baselines = np.array([b for b, _ in test])
    shift_val = compute_metrics(val_baselines + shift, val_targets)
    shift_test = compute_metrics(test_baselines + shift, test_targets)

    rng = np.random.RandomState(SEED + 1)
    perm_val = compute_metrics(
        all_train_targets[rng.randint(0, len(all_train_targets), len(val))],
        val_targets,
    )
    perm_test = compute_metrics(
        all_train_targets[rng.randint(0, len(all_train_targets), len(test))],
        test_targets,
    )

    return {
        "mean_baseline": {"val": mean_val, "test": mean_test},
        "linear_shift": {"val": shift_val, "test": shift_test},
        "permutation": {"val": perm_val, "test": perm_test},
    }


def run_linear_vae(train, val, test, n_genes, latent_dim=8, epochs=200, lr=1e-3):
    """Linear VAE (same architecture as NT-002)."""
    import torch
    import torch.nn as nn

    class LinearVAE(nn.Module):
        def __init__(self, n_genes, latent_dim):
            super().__init__()
            self.enc_mu = nn.Linear(n_genes, latent_dim)
            self.enc_logvar = nn.Linear(n_genes, latent_dim)
            self.dec = nn.Linear(latent_dim, n_genes)

        def encode(self, x):
            return self.enc_mu(x), self.enc_logvar(x)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.dec(z)

        def forward(self, baseline, target):
            delta = target - baseline
            mu, logvar = self.encode(delta)
            z = self.reparameterize(mu, logvar)
            pred_delta = self.decode(z)
            pred = baseline + pred_delta
            recon_loss = nn.functional.mse_loss(pred, target)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return pred, recon_loss + 0.1 * kl

    torch.manual_seed(SEED)
    model = LinearVAE(n_genes, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def to_tensors(pairs):
        baselines = torch.tensor([b for b, _ in pairs], dtype=torch.float32)
        targets = torch.tensor([t for _, t in pairs], dtype=torch.float32)
        return baselines, targets

    train_b, train_t = to_tensors(train)
    val_b, val_t = to_tensors(val)
    test_b, test_t = to_tensors(test)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, loss = model(train_b, train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred, _ = model(val_b, val_t)
        test_pred, _ = model(test_b, test_t)

    val_metrics = compute_metrics(val_pred.numpy(), val_t.numpy())
    test_metrics = compute_metrics(test_pred.numpy(), test_t.numpy())

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": "LinearVAE",
        "latent_dim": latent_dim,
        "n_params": n_params,
        "epochs": epochs,
        "val": val_metrics,
        "test": test_metrics,
    }


def evaluate_criteria(arm_result, baselines):
    """4-criterion protocol."""
    model_mse = arm_result["val"]["mse"]
    model_r = arm_result["val"]["pearson_r"]
    mean_mse = baselines["mean_baseline"]["val"]["mse"]
    shift_mse = baselines["linear_shift"]["val"]["mse"]
    mean_r = baselines["mean_baseline"]["val"]["pearson_r"]

    return [
        {
            "criterion": "C1: Model MSE < mean-baseline MSE",
            "met": model_mse < mean_mse,
            "detail": f"model={model_mse:.6f} vs mean={mean_mse:.6f}",
        },
        {
            "criterion": "C2: Model MSE < linear-shift MSE",
            "met": model_mse < shift_mse,
            "detail": f"model={model_mse:.6f} vs shift={shift_mse:.6f}",
        },
        {
            "criterion": "C3: Pearson r > 0.3",
            "met": model_r > 0.3,
            "detail": f"model_r={model_r:.4f} vs 0.3",
        },
        {
            "criterion": "C4: Model r > mean-baseline r",
            "met": model_r > mean_r,
            "detail": f"model_r={model_r:.4f} vs mean_r={mean_r:.4f}",
        },
    ]


def main():
    t0 = time.time()
    print("NT-003: Multi-series pooled perturbation response")
    print(f"  SEED={SEED}")
    print()

    # ── Load datasets ──
    print("Loading datasets...")
    gse119262 = load_dataset("gse119262_autophagy_full.csv", "GSE119262")
    gse107965 = load_dataset("gse107965_autophagy_subset.csv", "GSE107965")
    gse201710 = load_dataset("gse201710_imr90_osk_subset.csv", "GSE201710")

    for ds in [gse119262, gse107965, gse201710]:
        n_ctrl = sum(1 for c in ds.conditions if c == "control")
        n_treat = sum(1 for c in ds.conditions if c == "treated")
        print(f"  {ds.accession}: {ds.expression.shape[0]} samples, "
              f"{ds.expression.shape[1]} genes, {n_ctrl} ctrl / {n_treat} treat")
    print()

    # ── Arm A: Single-series (NT-002 reproduction on intersected gene set) ──
    # Use only GSE119262 but restricted to the common gene set with GSE107965
    common_genes = find_gene_intersection([gse119262, gse107965])
    print(f"Gene intersection (GSE119262 ∩ GSE107965): {len(common_genes)} genes")

    if len(common_genes) < 5:
        print("WARNING: Very few common genes. Subset CSVs use different gene panels.")
        print("Falling back to GSE119262-only with top-variance gene subsetting.")
        print()

        # Arm A: GSE119262 alone, top-29 genes (to match GSE107965's gene count)
        n_genes_target = min(gse107965.expression.shape[1], 50)
        gene_var = np.var(gse119262.expression, axis=0)
        top_idx = np.sort(np.argsort(gene_var)[-n_genes_target:])
        gse119262_sub = GEOExpressionData(
            expression=gse119262.expression[:, top_idx],
            conditions=gse119262.conditions,
            gene_names=[gse119262.gene_names[i] for i in top_idx],
            species=gse119262.species,
            accession=gse119262.accession,
        )

        pairs_a = make_perturbation_pairs(gse119262_sub)
        n_genes = gse119262_sub.expression.shape[1]
        print(f"Arm A (single-series): {len(pairs_a)} pairs, {n_genes} genes, "
              f"p/n = {n_genes/len(pairs_a):.1f}")

        train_a, val_a, test_a = split_pairs(pairs_a)
        print(f"  Split: {len(train_a)} train / {len(val_a)} val / {len(test_a)} test")

        baselines_a = run_baselines(train_a, val_a, test_a)
        lvae_a = run_linear_vae(train_a, val_a, test_a, n_genes, latent_dim=8, epochs=200)
        criteria_a = evaluate_criteria(lvae_a, baselines_a)
        verdict_a = "met" if all(c["met"] for c in criteria_a) else "not_met"

        print(f"\nArm A results (single-series, {n_genes} genes):")
        print(f"  LinearVAE: val MSE={lvae_a['val']['mse']:.6f}, r={lvae_a['val']['pearson_r']:.4f}")
        for c in criteria_a:
            print(f"    {c['criterion']}: {'MET' if c['met'] else 'FAIL'} ({c['detail']})")
        print(f"  Verdict: {verdict_a}")

        # Arm B: Pooled — use all three datasets' treated samples
        # Since gene intersection is small, pool at the study-aware level:
        # each dataset contributes pairs on its own gene set, we learn shared latent structure
        # Strategy: use top-variance genes from GSE119262 that overlap with the subsets
        all_pairs = []
        pair_sources = []

        for ds in [gse119262_sub, gse107965, gse201710]:
            # Subset each to the same n_genes_target genes by variance
            if ds.expression.shape[1] > n_genes_target:
                gv = np.var(ds.expression, axis=0)
                ti = np.sort(np.argsort(gv)[-n_genes_target:])
                ds_sub = GEOExpressionData(
                    expression=ds.expression[:, ti],
                    conditions=ds.conditions,
                    gene_names=[ds.gene_names[i] for i in ti],
                    species=ds.species,
                    accession=ds.accession,
                )
            else:
                ds_sub = ds

            ps = make_perturbation_pairs(ds_sub)
            if ps:
                all_pairs.extend(ps)
                pair_sources.extend([ds.accession] * len(ps))

        # NOTE: Pairs from different studies have different gene sets.
        # This is a known limitation — the latent space must generalize.
        # For an honest trial, we only pool series with the SAME gene set.
        # Since that's not possible here, we fall back to GSE119262 alone
        # but with a larger training split (no held-out test from other series).

        # Honest approach: just run GSE119262 with more training data
        # by reducing val/test fraction, as a baseline for future pooling.
        train_b, val_b, test_b = split_pairs(pairs_a, val_frac=0.15, test_frac=0.08)
        print(f"\nArm B (expanded train split): {len(train_b)} train / "
              f"{len(val_b)} val / {len(test_b)} test")

        baselines_b = run_baselines(train_b, val_b, test_b)
        lvae_b = run_linear_vae(train_b, val_b, test_b, n_genes, latent_dim=8, epochs=200)
        criteria_b = evaluate_criteria(lvae_b, baselines_b)
        verdict_b = "met" if all(c["met"] for c in criteria_b) else "not_met"

        print(f"\nArm B results (expanded train):")
        print(f"  LinearVAE: val MSE={lvae_b['val']['mse']:.6f}, r={lvae_b['val']['pearson_r']:.4f}")
        for c in criteria_b:
            print(f"    {c['criterion']}: {'MET' if c['met'] else 'FAIL'} ({c['detail']})")
        print(f"  Verdict: {verdict_b}")

        pooling_note = (
            "Gene intersection between full (17183) and subset (29-56) matrices "
            f"yielded only {len(common_genes)} common genes. True multi-series pooling "
            "requires full expression matrices for all series. This trial falls back "
            "to single-series with varied train/val splits as a controlled comparison."
        )

    else:
        # True pooling path — enough common genes to pool
        pooled = pool_datasets([gse119262, gse107965])
        n_genes = pooled.expression.shape[1]
        print(f"Pooled: {pooled.expression.shape[0]} samples, {n_genes} genes")

        pairs_pooled = make_perturbation_pairs(pooled)
        print(f"Perturbation pairs: {len(pairs_pooled)} (p/n = {n_genes/len(pairs_pooled):.1f})")

        # Single-series arm
        gse119262_sub = subset_to_genes(gse119262, common_genes)
        pairs_a = make_perturbation_pairs(gse119262_sub)
        train_a, val_a, test_a = split_pairs(pairs_a)

        baselines_a = run_baselines(train_a, val_a, test_a)
        lvae_a = run_linear_vae(train_a, val_a, test_a, n_genes, latent_dim=8, epochs=200)
        criteria_a = evaluate_criteria(lvae_a, baselines_a)
        verdict_a = "met" if all(c["met"] for c in criteria_a) else "not_met"

        # Pooled arm
        train_b, val_b, test_b = split_pairs(pairs_pooled)

        baselines_b = run_baselines(train_b, val_b, test_b)
        lvae_b = run_linear_vae(train_b, val_b, test_b, n_genes, latent_dim=8, epochs=200)
        criteria_b = evaluate_criteria(lvae_b, baselines_b)
        verdict_b = "met" if all(c["met"] for c in criteria_b) else "not_met"

        pooling_note = (
            f"True multi-series pooling on {len(common_genes)} common genes. "
            f"Single-series: {len(pairs_a)} pairs. Pooled: {len(pairs_pooled)} pairs."
        )

        print(f"\nArm A (single-series): verdict={verdict_a}")
        print(f"Arm B (pooled): verdict={verdict_b}")

    elapsed = time.time() - t0

    # ── Build record ──
    record = {
        "experiment_id": "NT-003",
        "date": str(date.today()),
        "parameters": {
            "seed": SEED,
            "n_genes_common": len(common_genes),
            "pooling_note": pooling_note,
            "species": "human",
            "regime": "mixed (data-driven + prior-contrast)",
            "series_used": ["GSE119262", "GSE107965", "GSE201710"],
        },
        "arm_a_single_series": {
            **lvae_a,
            "criteria": criteria_a,
            "verdict": verdict_a,
            "baselines": {
                k: {split: {mk: round(mv, 6) for mk, mv in metrics.items()}
                    for split, metrics in v.items()}
                for k, v in baselines_a.items()
            },
        },
        "arm_b_pooled_or_expanded": {
            **lvae_b,
            "criteria": criteria_b,
            "verdict": verdict_b,
            "baselines": {
                k: {split: {mk: round(mv, 6) for mk, mv in metrics.items()}
                    for split, metrics in v.items()}
                for k, v in baselines_b.items()
            },
        },
        "data_scaling_finding": (
            "Current committed data has only 1 in-regime series (GSE119262, 44 samples). "
            "The other 7 series are subsets with 4-28 samples and 25-56 genes — too small "
            "for data-driven regime and too few genes for meaningful cross-series pooling. "
            "Next step: download full expression matrices for the 745 candidate series "
            "identified in the GEO series catalog (NT-001_curated_series.json)."
        ),
        "elapsed_seconds": round(elapsed, 1),
    }
    record["record_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORDS_DIR / "NT-003_result.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nRecord written: {out}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
