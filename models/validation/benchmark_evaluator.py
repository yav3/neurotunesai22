"""Benchmark evaluator — score gene regulatory network inference against gold standards.

Uses the DREAM5 gold-standard network to evaluate how well a trained
perturbation-response model captures regulatory relationships. The approach:

1. Train cVAE on DREAM5 expression data (100 genes, 805 samples)
2. For each TF, perturb its expression and measure downstream effects
3. Build an inferred adjacency matrix from perturbation responses
4. Score against the gold standard via AUROC and precision@k

This is not network inference from scratch — it measures whether the cVAE's
latent space encodes regulatory structure that the gold standard confirms.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from models.data.prior_network_loader import (
    BenchmarkNetwork,
    load_dream5_benchmark,
)


@dataclass
class BenchmarkResult:
    """Result of evaluating a model against a benchmark network."""
    benchmark: str
    species: str
    n_genes: int
    n_tfs: int
    n_true_edges: int
    n_false_edges: int
    auroc: float
    auprc: float
    precision_at_k: dict[int, float]
    n_predicted_edges: int
    content_hash: str = ""


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under ROC curve, no sklearn dependency."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0.0
    fp = 0.0
    auc = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under precision-recall curve."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    if n_pos == 0:
        return 0.0
    tp = 0.0
    fp = 0.0
    area = 0.0
    prev_recall = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        if label == 1:
            area += precision * (recall - prev_recall)
            prev_recall = recall
    return area


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_values: list[int]) -> dict[int, float]:
    """Precision at k for several k values."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    result = {}
    for k in k_values:
        if k > len(y_sorted):
            k = len(y_sorted)
        result[k] = float(y_sorted[:k].sum() / k) if k > 0 else 0.0
    return result


def infer_regulatory_scores(
    model: Any,
    expression: np.ndarray,
    tf_indices: list[int],
    gene_names: list[str],
    perturbation_magnitude: float = 2.0,
    device: str = "cpu",
) -> np.ndarray:
    """Infer TF→target regulatory scores from a trained cVAE.

    For each TF, compute the mean baseline expression, then perturb the TF
    by ±perturbation_magnitude and measure the L1 response of each gene.
    Returns a (n_tfs, n_genes) score matrix.
    """
    import torch

    model.eval()
    baseline = expression.mean(axis=0)
    n_genes = len(gene_names)
    scores = np.zeros((len(tf_indices), n_genes), dtype=np.float32)

    with torch.no_grad():
        baseline_t = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to(device)
        ptype_ids = torch.zeros(1, dtype=torch.long).to(device)
        gene_idx_t = torch.zeros(1, dtype=torch.long).to(device)
        drug_idx = torch.zeros(1, dtype=torch.long).to(device)
        dose = torch.ones(1, dtype=torch.float32).to(device)
        duration = torch.ones(1, dtype=torch.float32).to(device)
        time_point = torch.zeros(1, dtype=torch.float32).to(device)

        pred_baseline = model.predict(
            baseline_t, ptype_ids, gene_idx_t, drug_idx, dose, duration, time_point,
        )
        baseline_pred = pred_baseline.squeeze(0).cpu().numpy()

        for i, tf_idx in enumerate(tf_indices):
            perturbed = baseline.copy()
            perturbed[tf_idx] += perturbation_magnitude

            perturbed_t = torch.tensor(perturbed, dtype=torch.float32).unsqueeze(0).to(device)
            pred_perturbed = model.predict(
                perturbed_t, ptype_ids, gene_idx_t, drug_idx, dose, duration, time_point,
            )
            perturbed_pred = pred_perturbed.squeeze(0).cpu().numpy()

            scores[i] = np.abs(perturbed_pred - baseline_pred)

    return scores


def score_against_gold_standard(
    regulatory_scores: np.ndarray,
    bench: BenchmarkNetwork,
) -> BenchmarkResult:
    """Score inferred regulatory scores against DREAM5 gold standard.

    Args:
        regulatory_scores: (n_tfs, n_genes) predicted regulatory strengths
        bench: BenchmarkNetwork with gold_standard array

    Returns:
        BenchmarkResult with AUROC, AUPRC, and precision@k
    """
    gene_idx = {g: i for i, g in enumerate(bench.gene_names)}
    tf_idx_map = {g: i for i, g in enumerate(bench.tf_names)}

    y_true = []
    y_score = []

    for row in bench.gold_standard:
        tf_i, target_i, label = int(row[0]), int(row[1]), int(row[2])
        tf_name = bench.gene_names[tf_i]
        if tf_name not in tf_idx_map:
            continue
        tf_row = tf_idx_map[tf_name]
        if tf_row >= regulatory_scores.shape[0] or target_i >= regulatory_scores.shape[1]:
            continue
        y_true.append(label)
        y_score.append(regulatory_scores[tf_row, target_i])

    y_true = np.array(y_true, dtype=np.int32)
    y_score = np.array(y_score, dtype=np.float32)

    if len(y_true) == 0:
        return BenchmarkResult(
            benchmark="DREAM5_network1",
            species=bench.species,
            n_genes=len(bench.gene_names),
            n_tfs=len(bench.tf_names),
            n_true_edges=bench.n_true_edges,
            n_false_edges=bench.n_false_edges,
            auroc=0.5,
            auprc=0.0,
            precision_at_k={},
            n_predicted_edges=0,
        )

    auroc = _auroc(y_true, y_score)
    auprc = _auprc(y_true, y_score)
    k_values = [10, 20, 50, 100]
    prec_at_k = _precision_at_k(y_true, y_score, k_values)

    h = hashlib.sha256()
    h.update(y_true.tobytes())
    h.update(y_score.tobytes())

    return BenchmarkResult(
        benchmark="DREAM5_network1",
        species=bench.species,
        n_genes=len(bench.gene_names),
        n_tfs=len(bench.tf_names),
        n_true_edges=bench.n_true_edges,
        n_false_edges=bench.n_false_edges,
        auroc=auroc,
        auprc=auprc,
        precision_at_k=prec_at_k,
        n_predicted_edges=int((y_score > 0).sum()),
        content_hash=h.hexdigest(),
    )


def evaluate_random_baseline(bench: BenchmarkNetwork, seed: int = 2026) -> BenchmarkResult:
    """Random baseline for comparison — expected AUROC ≈ 0.5."""
    rng = np.random.RandomState(seed)
    n_tfs = len(bench.tf_names)
    n_genes = len(bench.gene_names)
    random_scores = rng.rand(n_tfs, n_genes).astype(np.float32)
    return score_against_gold_standard(random_scores, bench)


def evaluate_correlation_baseline(bench: BenchmarkNetwork) -> BenchmarkResult:
    """Pearson correlation baseline — a simple non-model comparator."""
    expr = bench.expression  # (samples, genes)
    tf_idx_map = {g: i for i, g in enumerate(bench.tf_names)}
    gene_idx = {g: i for i, g in enumerate(bench.gene_names)}

    n_tfs = len(bench.tf_names)
    n_genes = len(bench.gene_names)
    scores = np.zeros((n_tfs, n_genes), dtype=np.float32)

    for ti, tf_name in enumerate(bench.tf_names):
        if tf_name not in gene_idx:
            continue
        tf_col = gene_idx[tf_name]
        tf_expr = expr[:, tf_col]
        tf_std = tf_expr.std()
        if tf_std < 1e-10:
            continue
        for gi in range(n_genes):
            g_expr = expr[:, gi]
            g_std = g_expr.std()
            if g_std < 1e-10:
                continue
            corr = np.corrcoef(tf_expr, g_expr)[0, 1]
            scores[ti, gi] = abs(corr)

    return score_against_gold_standard(scores, bench)
