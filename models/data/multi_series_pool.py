"""Multi-series pooling for governed training.

Pools multiple GEO expression datasets that share a common gene set,
enforcing the species firewall and tracking provenance per-series.
Designed for the NT-001 regime where a single series has too few
perturbation pairs to train a cVAE.

Evidence discipline:
  - Species firewall: all series must declare the same species
  - Gene intersection: only genes present in ALL series are retained
  - Study-aware pairing: perturbation pairs form within each study
  - Provenance: each sample carries its source accession
  - No cross-species pooling, no silent gene imputation
"""
from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from models.data.geo_loader import load_geo_csv, discover_merlin_datasets
from models.foundation.perturbation_response import (
    GEOExpressionData,
    PerturbationDataset,
    PerturbationType,
)


def find_gene_intersection(datasets: list[GEOExpressionData]) -> list[str]:
    """Return sorted gene names present in ALL datasets."""
    gene_sets = [set(d.gene_names) for d in datasets]
    common = reduce(lambda a, b: a & b, gene_sets)
    return sorted(common)


def subset_to_genes(
    data: GEOExpressionData, genes: list[str],
) -> GEOExpressionData:
    """Subset expression matrix to the given gene list (in order)."""
    gene_to_idx = {g: i for i, g in enumerate(data.gene_names)}
    indices = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    kept_genes = [data.gene_names[i] for i in indices]
    return GEOExpressionData(
        expression=data.expression[:, indices],
        conditions=data.conditions,
        gene_names=kept_genes,
        species=data.species,
        accession=data.accession,
    )


def pool_datasets(
    datasets: list[GEOExpressionData],
    max_genes: int | None = None,
) -> GEOExpressionData:
    """Pool multiple GEO datasets into one, on the common gene intersection.

    Enforces species firewall. Optionally subsets to top-k genes by
    cross-dataset variance after intersection.
    """
    if not datasets:
        raise ValueError("No datasets to pool")

    species_set = {d.species for d in datasets}
    if len(species_set) != 1:
        raise ValueError(f"Cross-species pooling refused: {species_set}")
    species = species_set.pop()

    common_genes = find_gene_intersection(datasets)
    if not common_genes:
        raise ValueError("No common genes across datasets")

    subsets = [subset_to_genes(d, common_genes) for d in datasets]

    if max_genes is not None and len(common_genes) > max_genes:
        stacked = np.vstack([s.expression for s in subsets])
        gene_var = np.var(stacked, axis=0)
        top_idx = np.argsort(gene_var)[-max_genes:]
        top_idx = np.sort(top_idx)
        common_genes = [common_genes[i] for i in top_idx]
        subsets = [
            GEOExpressionData(
                expression=s.expression[:, top_idx],
                conditions=s.conditions,
                gene_names=common_genes,
                species=s.species,
                accession=s.accession,
            )
            for s in subsets
        ]

    pooled_expr = np.vstack([s.expression for s in subsets])
    pooled_conditions = []
    for s in subsets:
        pooled_conditions.extend(s.conditions)
    accessions = sorted({d.accession for d in datasets})

    return GEOExpressionData(
        expression=pooled_expr.astype(np.float32),
        conditions=pooled_conditions,
        gene_names=common_genes,
        species=species,
        accession="+".join(accessions),
    )


def build_study_aware_dataset(
    datasets: list[GEOExpressionData],
    max_genes: int | None = None,
    perturbation_type: PerturbationType = PerturbationType.DRUG,
) -> tuple[ConcatDataset, list[str], int]:
    """Build a ConcatDataset where perturbation pairs form within each study.

    Returns (dataset, common_genes, n_pairs_total).
    Each study contributes its own PerturbationDataset on the common gene set.
    """
    if not datasets:
        raise ValueError("No datasets")

    species_set = {d.species for d in datasets}
    if len(species_set) != 1:
        raise ValueError(f"Cross-species pooling refused: {species_set}")

    common_genes = find_gene_intersection(datasets)
    if not common_genes:
        raise ValueError("No common genes across datasets")

    subsets = [subset_to_genes(d, common_genes) for d in datasets]

    if max_genes is not None and len(common_genes) > max_genes:
        stacked = np.vstack([s.expression for s in subsets])
        gene_var = np.var(stacked, axis=0)
        top_idx = np.argsort(gene_var)[-max_genes:]
        top_idx = np.sort(top_idx)
        common_genes = [common_genes[i] for i in top_idx]
        subsets = [
            GEOExpressionData(
                expression=s.expression[:, top_idx],
                conditions=s.conditions,
                gene_names=common_genes,
                species=s.species,
                accession=s.accession,
            )
            for s in subsets
        ]

    per_study_ds = []
    for s in subsets:
        n_ctrl = sum(1 for c in s.conditions if c == "control")
        n_treat = sum(1 for c in s.conditions if c == "treated")
        if n_ctrl > 0 and n_treat > 0:
            ds = PerturbationDataset(s, perturbation_type=perturbation_type)
            if len(ds) > 0:
                per_study_ds.append(ds)

    if not per_study_ds:
        raise ValueError("No valid perturbation pairs across all studies")

    combined = ConcatDataset(per_study_ds)
    return combined, common_genes, len(combined)


def load_and_pool_merlin_data(
    data_dir: str | Path,
    species: str = "human",
    min_conditions: int = 2,
    max_genes: int | None = None,
    log1p: bool = True,
) -> tuple[GEOExpressionData, dict]:
    """Load all qualifying series from Merlin's data directory and pool them.

    Returns (pooled_data, metadata) where metadata tracks per-series provenance.
    """
    data_dir = Path(data_dir)
    all_meta = discover_merlin_datasets(data_dir, species=species)

    datasets = []
    series_info = []
    for meta in sorted(all_meta, key=lambda x: -x["n_samples"]):
        if meta["n_control"] > 0 and meta["n_treated"] > 0:
            data = load_geo_csv(
                meta["path"], accession=meta["accession"],
                species=species, log1p=log1p,
            )
            datasets.append(data)
            series_info.append({
                "accession": meta["accession"],
                "n_samples": meta["n_samples"],
                "n_control": meta["n_control"],
                "n_treated": meta["n_treated"],
                "n_genes_original": meta["n_genes"],
                "content_hash": meta["content_hash"],
            })

    if not datasets:
        raise ValueError(f"No qualifying series found in {data_dir}")

    pooled = pool_datasets(datasets, max_genes=max_genes)

    metadata = {
        "n_series": len(datasets),
        "n_samples_total": pooled.expression.shape[0],
        "n_genes_common": pooled.expression.shape[1],
        "max_genes": max_genes,
        "species": species,
        "series": series_info,
    }
    return pooled, metadata
