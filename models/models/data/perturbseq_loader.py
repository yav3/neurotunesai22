"""Perturb-seq data loader — single-cell CRISPR screen data for perturbation cVAE.

Bridges Merlin's committed Perturb-seq parquet files (Xaira panel format) to
NeuroTunesAI's PerturbationDataset training format.

Data flow:
    Merlin parquet → read (cells × genes + gene_target + batch)
    → pseudo-bulk per target (mean expression per knockout)
    → OR single-cell pairs (control cell, perturbed cell)
    → PerturbationDataset (baseline/target pairs for training)

Species: human only (firewall-enforced).
Regime: data-driven (Xaira subset has 62,898 cells).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from models.foundation.perturbation_response import (
    GEOExpressionData,
    PerturbationDataset,
    PerturbationType,
)


CONTROL_LABEL = "Non-Targeting"


@dataclass
class PerturbSeqMeta:
    """Metadata for a loaded Perturb-seq dataset."""
    source_path: str
    n_cells: int
    n_genes: int
    n_targets: int
    n_control_cells: int
    n_batches: int
    gene_targets: list[str]
    gene_names: list[str]
    species: str = "human"
    content_hash: str = ""


def _content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_perturbseq_parquet(
    parquet_path: str | Path,
    species: str = "human",
    min_cells_per_target: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], PerturbSeqMeta]:
    """Load a Perturb-seq parquet into raw arrays.

    Returns
    -------
    expression : (n_cells, n_genes) float32 array
    targets : (n_cells,) string array — gene_target per cell
    gene_names : list of gene column names
    batches : (n_cells,) string array — batch per cell
    meta : PerturbSeqMeta
    """
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)
    table = pq.read_table(parquet_path)
    df_cols = table.column_names

    meta_cols = {"gene_target", "total_counts", "batch"}
    gene_names = [c for c in df_cols if c not in meta_cols]

    targets_arr = table.column("gene_target").to_pylist()
    batches_arr = table.column("batch").to_pylist() if "batch" in df_cols else [""] * len(targets_arr)

    expression = np.column_stack([
        table.column(g).to_numpy().astype(np.float32)
        for g in gene_names
    ])

    unique_targets = sorted(set(targets_arr) - {CONTROL_LABEL})
    target_counts = {}
    for t in targets_arr:
        target_counts[t] = target_counts.get(t, 0) + 1
    valid_targets = [t for t in unique_targets if target_counts.get(t, 0) >= min_cells_per_target]

    n_control = sum(1 for t in targets_arr if t == CONTROL_LABEL)

    meta = PerturbSeqMeta(
        source_path=str(parquet_path),
        n_cells=len(targets_arr),
        n_genes=len(gene_names),
        n_targets=len(valid_targets),
        n_control_cells=n_control,
        n_batches=len(set(batches_arr)),
        gene_targets=valid_targets,
        gene_names=gene_names,
        species=species,
        content_hash=_content_hash(parquet_path),
    )

    return expression, np.array(targets_arr), gene_names, np.array(batches_arr), meta


def pseudobulk_by_target(
    expression: np.ndarray,
    targets: np.ndarray,
    gene_names: list[str],
    species: str = "human",
    log1p: bool = True,
    min_cells: int = 10,
) -> GEOExpressionData:
    """Aggregate single-cell data to pseudo-bulk per knockout target.

    Each row in the output is the mean expression across all cells with
    that gene_target label. This gives one sample per perturbation —
    compatible with the bulk-expression PerturbationDataset.
    """
    unique_targets = sorted(set(targets))
    rows = []
    conditions = []
    sample_names = []

    for target in unique_targets:
        mask = targets == target
        n_cells = mask.sum()
        if n_cells < min_cells:
            continue
        mean_expr = expression[mask].mean(axis=0)
        rows.append(mean_expr)
        conditions.append("control" if target == CONTROL_LABEL else "treated")
        sample_names.append(target)

    matrix = np.array(rows, dtype=np.float32)
    if log1p:
        matrix = np.log1p(np.maximum(matrix, 0.0)).astype(np.float32)

    return GEOExpressionData(
        expression=matrix,
        conditions=conditions,
        gene_names=gene_names,
        species=species,
        accession="perturbseq",
        metadata={
            "source": "perturbseq_pseudobulk",
            "n_targets": len(sample_names),
            "sample_names": sample_names,
            "log1p": log1p,
        },
    )


def single_cell_pairs(
    expression: np.ndarray,
    targets: np.ndarray,
    gene_names: list[str],
    max_pairs_per_target: int = 100,
    seed: int = 2026,
    log1p: bool = True,
) -> list[dict[str, Any]]:
    """Generate (control_cell, perturbed_cell) pairs for single-cell training.

    For each knockout target, pairs a random control cell with a random
    perturbed cell. Returns dicts suitable for direct cVAE training.
    """
    rng = np.random.RandomState(seed)
    control_mask = targets == CONTROL_LABEL
    control_indices = np.where(control_mask)[0]

    if len(control_indices) == 0:
        return []

    unique_targets = sorted(set(targets) - {CONTROL_LABEL})
    pairs = []

    for target in unique_targets:
        perturbed_mask = targets == target
        perturbed_indices = np.where(perturbed_mask)[0]

        if len(perturbed_indices) == 0:
            continue

        n_pairs = min(max_pairs_per_target, len(perturbed_indices))
        ctrl_sample = rng.choice(control_indices, size=n_pairs, replace=True)
        pert_sample = rng.choice(perturbed_indices, size=n_pairs, replace=False if n_pairs <= len(perturbed_indices) else True)

        for ci, pi in zip(ctrl_sample, pert_sample):
            baseline = expression[ci].astype(np.float32)
            perturbed = expression[pi].astype(np.float32)
            if log1p:
                baseline = np.log1p(np.maximum(baseline, 0.0))
                perturbed = np.log1p(np.maximum(perturbed, 0.0))

            gene_idx = gene_names.index(target) if target in gene_names else -1
            pairs.append({
                "baseline": baseline,
                "perturbed": perturbed,
                "gene_target": target,
                "gene_idx": gene_idx,
                "ptype": PerturbationType.KNOCKOUT,
            })

    return pairs


def discover_perturbseq_datasets(
    merlin_data_dir: str | Path | None = None,
    species: str = "human",
) -> list[dict[str, Any]]:
    """Discover Perturb-seq parquet files in Merlin's data directory."""
    if merlin_data_dir is None:
        merlin_data_dir = (
            Path(__file__).parent.parent.parent.parent
            / "merlinaineuralplus2" / "backend" / "data" / "perturbseq" / species
        )
    else:
        merlin_data_dir = Path(merlin_data_dir)

    if not merlin_data_dir.exists():
        return []

    datasets = []
    for pq_file in sorted(merlin_data_dir.glob("*.parquet")):
        try:
            expr, targets, genes, batches, meta = load_perturbseq_parquet(pq_file, species=species)
            datasets.append({
                "source_path": str(pq_file),
                "n_cells": meta.n_cells,
                "n_genes": meta.n_genes,
                "n_targets": meta.n_targets,
                "n_control_cells": meta.n_control_cells,
                "n_batches": meta.n_batches,
                "species": species,
                "content_hash": meta.content_hash,
                "modality": "perturbseq",
                "in_regime": meta.n_cells >= 30,
            })
        except Exception:
            continue

    return datasets
