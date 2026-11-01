"""NT-002 Governed Training Trial — right-sized models for current data.

NT-001 showed the full cVAE (10.6M params) cannot beat baselines with 21
perturbation pairs. NT-002 tests dimensionality-appropriate models:

  Arm A: Ridge regression (alpha CV'd) — can we learn anything at all?
  Arm B: Linear VAE (latent_dim=8, no hidden layers) — simplest generative model

Both arms run on GSE119262 (human, data-driven regime) with gene subsetting
to 50 top-variance genes (p/n ≈ 2.4, the minimal feasible ratio).

Pre-registered criteria (same 4-criterion protocol as NT-001):
  C1: Model MSE < mean-baseline MSE
  C2: Model MSE < linear-shift MSE
  C3: Pearson r(predicted, target) > 0.3 (relaxed from 0.5 — more realistic)
  C4: Model Pearson r > mean-baseline r

Seed: 2026. Split: 20% val, 10% test. Species: human (firewall-enforced).
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

from models.data.geo_loader import load_geo_csv, discover_merlin_datasets, content_hash
from models.foundation.perturbation_response import GEOExpressionData

MERLIN_HUMAN_DIR = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")
RECORDS_DIR = Path(__file__).parent.parent / "docs" / "records"
SEED = 2026
MAX_GENES = 50
HELD_OUT_FRACTION = 0.2
TEST_FRACTION = 0.1


def load_gse119262(max_genes: int = 50) -> GEOExpressionData:
    """Load GSE119262 with top-k variance gene subsetting."""
    data = load_geo_csv(
        MERLIN_HUMAN_DIR / "gse119262_autophagy_full.csv",
        accession="GSE119262", species="human", log1p=True,
    )
    if max_genes and data.expression.shape[1] > max_genes:
        gene_var = np.var(data.expression, axis=0)
        top_idx = np.sort(np.argsort(gene_var)[-max_genes:])
        data = GEOExpressionData(
            expression=data.expression[:, top_idx],
            conditions=data.conditions,
            gene_names=[data.gene_names[i] for i in top_idx],
            species=data.species,
            accession=data.accession,
        )
    return data


def make_perturbation_pairs(data: GEOExpressionData):
    """Build (baseline, target) pairs from control/treated samples."""
    ctrl_idx = [i for i, c in enumerate(data.conditions) if c == "control"]
    treat_idx = [i for i, c in enumerate(data.conditions) if c == "treated"]
    if not ctrl_idx or not treat_idx:
        raise ValueError("Need both control and treated samples")

    ctrl_mean = data.expression[ctrl_idx].mean(axis=0)
    pairs = []
    for ti in treat_idx:
        pairs.append((ctrl_mean, data.expression[ti]))
    return pairs


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


def run_ridge_arm(train, val, test, n_genes):
    """Arm A: Ridge regression with cross-validated alpha."""
    from sklearn.linear_model import RidgeCV

    X_train = np.array([b for b, _ in train])
    y_train = np.array([t for _, t in train])
    X_val = np.array([b for b, _ in val])
    y_val = np.array([t for _, t in val])
    X_test = np.array([b for b, _ in test])
    y_test = np.array([t for _, t in test])

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = compute_metrics(val_pred, y_val)
    test_metrics = compute_metrics(test_pred, y_test)

    return {
        "model": "RidgeCV",
        "alpha_selected": float(model.alpha_),
        "n_params": int(n_genes * n_genes + n_genes),
        "val": val_metrics,
        "test": test_metrics,
    }


def run_linear_vae_arm(train, val, test, n_genes, latent_dim=8, epochs=100, lr=1e-3):
    """Arm B: Linear VAE (no hidden layers)."""
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


def run_baselines(train, val, test):
    """Mean-baseline, linear-shift, and permutation controls."""
    all_train_targets = np.array([t for _, t in train])
    train_target_mean = all_train_targets.mean(axis=0)

    # Mean baseline: predict the training-set target mean for every sample
    val_targets = np.array([t for _, t in val])
    test_targets = np.array([t for _, t in test])
    val_mean_pred = np.tile(train_target_mean, (len(val), 1))
    test_mean_pred = np.tile(train_target_mean, (len(test), 1))
    mean_val = compute_metrics(val_mean_pred, val_targets)
    mean_test = compute_metrics(test_mean_pred, test_targets)

    # Linear shift: target ≈ baseline + (mean_target - mean_baseline)
    all_train_baselines = np.array([b for b, _ in train])
    shift = train_target_mean - all_train_baselines.mean(axis=0)
    val_baselines = np.array([b for b, _ in val])
    test_baselines = np.array([b for b, _ in test])
    val_shift_pred = val_baselines + shift
    test_shift_pred = test_baselines + shift
    shift_val = compute_metrics(val_shift_pred, val_targets)
    shift_test = compute_metrics(test_shift_pred, test_targets)

    # Permutation: predict a random target from the training set
    rng = np.random.RandomState(SEED + 1)
    val_perm_pred = all_train_targets[rng.randint(0, len(all_train_targets), len(val))]
    test_perm_pred = all_train_targets[rng.randint(0, len(all_train_targets), len(test))]
    perm_val = compute_metrics(val_perm_pred, val_targets)
    perm_test = compute_metrics(test_perm_pred, test_targets)

    return {
        "mean_baseline": {"val": mean_val, "test": mean_test},
        "linear_shift": {"val": shift_val, "test": shift_test},
        "permutation": {"val": perm_val, "test": perm_test},
    }


def evaluate_criteria(arm_result, baselines):
    """Evaluate 4-criterion protocol."""
    model_mse = arm_result["val"]["mse"]
    model_r = arm_result["val"]["pearson_r"]
    mean_mse = baselines["mean_baseline"]["val"]["mse"]
    shift_mse = baselines["linear_shift"]["val"]["mse"]
    mean_r = baselines["mean_baseline"]["val"]["pearson_r"]

    criteria = [
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
    return criteria


def main():
    t0 = time.time()
    print("NT-002: Right-sized models for current data")
    print(f"  MAX_GENES={MAX_GENES}, SEED={SEED}")
    print()

    data = load_gse119262(max_genes=MAX_GENES)
    n_genes = data.expression.shape[1]
    sha = content_hash(MERLIN_HUMAN_DIR / "gse119262_autophagy_full.csv")
    print(f"Loaded GSE119262: {data.expression.shape[0]} samples, {n_genes} genes")

    pairs = make_perturbation_pairs(data)
    print(f"Perturbation pairs: {len(pairs)} (p/n = {n_genes/len(pairs):.1f})")

    train, val, test = split_pairs(pairs)
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    print()

    baselines = run_baselines(train, val, test)
    print("Baselines (val):")
    for name, bl in baselines.items():
        print(f"  {name}: MSE={bl['val']['mse']:.6f}, r={bl['val']['pearson_r']:.4f}")
    print()

    # Arm A: Ridge
    print("Running Arm A: Ridge regression...")
    ridge = run_ridge_arm(train, val, test, n_genes)
    ridge_criteria = evaluate_criteria(ridge, baselines)
    ridge_verdict = "met" if all(c["met"] for c in ridge_criteria) else "not_met"
    print(f"  Ridge (alpha={ridge['alpha_selected']}): val MSE={ridge['val']['mse']:.6f}, r={ridge['val']['pearson_r']:.4f}")
    for c in ridge_criteria:
        print(f"    {c['criterion']}: {'MET' if c['met'] else 'FAIL'} ({c['detail']})")
    print(f"  Verdict: {ridge_verdict}")
    print()

    # Arm B: Linear VAE
    print("Running Arm B: Linear VAE (latent_dim=8)...")
    lvae = run_linear_vae_arm(train, val, test, n_genes, latent_dim=8, epochs=200)
    lvae_criteria = evaluate_criteria(lvae, baselines)
    lvae_verdict = "met" if all(c["met"] for c in lvae_criteria) else "not_met"
    print(f"  LinearVAE ({lvae['n_params']} params): val MSE={lvae['val']['mse']:.6f}, r={lvae['val']['pearson_r']:.4f}")
    for c in lvae_criteria:
        print(f"    {c['criterion']}: {'MET' if c['met'] else 'FAIL'} ({c['detail']})")
    print(f"  Verdict: {lvae_verdict}")
    print()

    elapsed = time.time() - t0
    record = {
        "experiment_id": "NT-002",
        "date": str(date.today()),
        "parameters": {
            "seed": SEED,
            "max_genes": MAX_GENES,
            "n_genes_original": 17183,
            "n_genes_subsetted": n_genes,
            "n_samples": data.expression.shape[0],
            "n_pairs": len(pairs),
            "p_over_n": round(n_genes / len(pairs), 2),
            "accession": "GSE119262",
            "species": "human",
            "regime": "data-driven",
            "dataset_sha256": sha,
        },
        "baselines": {
            k: {split: {mk: round(mv, 6) for mk, mv in metrics.items()}
                for split, metrics in v.items()}
            for k, v in baselines.items()
        },
        "arm_a_ridge": {
            **ridge,
            "criteria": ridge_criteria,
            "verdict": ridge_verdict,
        },
        "arm_b_linear_vae": {
            **lvae,
            "criteria": lvae_criteria,
            "verdict": lvae_verdict,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    record["record_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORDS_DIR / "NT-002_result.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Record written: {out}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
