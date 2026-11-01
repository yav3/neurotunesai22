"""Cross-platform multi-series pooling with automatic probe-to-gene mapping.

Extends multi_series_pool.py to handle datasets with different ID namespaces
(Affymetrix, Illumina, GI accessions, gene symbols) by applying probe→gene
mapping before the gene intersection step.

Pipeline:
  1. Load each GEO CSV (raw probe/gene IDs)
  2. Detect ID format per dataset
  3. Apply platform-specific probe→gene mapping (pass-through for symbols)
  4. Collapse multi-probe genes (mean by default)
  5. Intersect on common gene symbols across all mapped datasets
  6. Pool expression matrices (samples × common_genes)

Species firewall enforced. No cross-species pooling.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from models.data.geo_loader import load_geo_csv, infer_condition, discover_merlin_datasets
from models.data.probe_mapper import (
    IDFormat,
    PlatformMapping,
    MappedExpression,
    CrossPlatformPool,
    detect_id_format,
    load_cached_mapping,
    map_probes_to_genes,
    align_to_common_genes,
)
from models.foundation.perturbation_response import GEOExpressionData

_PLATFORM_FOR_ACCESSION: dict[str, str] = {
    "GSE5281": "GPL570",
    "GSE30174": "GPL570",
    "GSE26927": "GPL6255",
    "GSE15222": "GPL2700",
}


def _detect_platform(probe_ids: list[str], accession: str = "") -> str | None:
    """Detect the microarray platform from probe IDs or known accession mapping."""
    if accession in _PLATFORM_FOR_ACCESSION:
        return _PLATFORM_FOR_ACCESSION[accession]
    fmt = detect_id_format(probe_ids)
    if fmt == IDFormat.AFFYMETRIX:
        return "GPL570"
    if fmt == IDFormat.ILLUMINA_ILMN:
        return "GPL6255"
    if fmt == IDFormat.ILLUMINA_GI:
        return "GPL2700"
    return None


@dataclass
class MappedGEODataset:
    """A GEO dataset after probe-to-gene mapping."""
    expression: np.ndarray  # (samples, genes)
    gene_names: list[str]
    conditions: list[str]
    species: str
    accession: str
    id_format: IDFormat
    n_original_ids: int
    n_mapped: int
    n_unmapped: int
    n_collapsed: int
    platform: str


def load_and_map_geo(
    csv_path: str | Path,
    accession: str = "",
    species: str = "human",
    log1p: bool = True,
    collapse: str = "mean",
) -> MappedGEODataset:
    """Load a GEO CSV and apply probe-to-gene mapping if needed.

    For gene-symbol-indexed files, this is a pass-through.
    For probe-indexed files, loads the cached platform mapping and collapses
    multi-probe genes.
    """
    csv_path = Path(csv_path)
    if not accession:
        m = re.search(r"(GSE\d+)", csv_path.stem, re.IGNORECASE)
        accession = m.group(1).upper() if m else csv_path.stem

    probe_ids: list[str] = []
    sample_names: list[str] = []
    rows: list[list[float]] = []

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        sample_names = [s.strip() for s in header[1:]]
        for row in reader:
            if not row or not row[0].strip():
                continue
            probe_ids.append(row[0].strip())
            vals = []
            for v in row[1:]:
                try:
                    vals.append(float(v.strip()) if v.strip() else 0.0)
                except ValueError:
                    vals.append(0.0)
            rows.append(vals)

    matrix = np.array(rows, dtype=np.float32)  # (probes, samples)
    matrix = np.nan_to_num(matrix, nan=0.0)
    if log1p:
        matrix = np.log1p(np.maximum(matrix, 0.0)).astype(np.float32)
    expression = matrix.T  # (samples, probes)

    conditions = [infer_condition(s) for s in sample_names]
    fmt = detect_id_format(probe_ids)

    if fmt == IDFormat.GENE_SYMBOL:
        return MappedGEODataset(
            expression=expression,
            gene_names=probe_ids,
            conditions=conditions,
            species=species,
            accession=accession,
            id_format=fmt,
            n_original_ids=len(probe_ids),
            n_mapped=len(probe_ids),
            n_unmapped=0,
            n_collapsed=0,
            platform="identity",
        )

    platform_id = _detect_platform(probe_ids, accession)
    if platform_id is None:
        raise ValueError(
            f"Cannot detect platform for {accession} "
            f"(format={fmt.value}, sample IDs: {probe_ids[:3]})"
        )

    mapping = load_cached_mapping(platform_id)
    if mapping is None:
        raise ValueError(
            f"No cached mapping for platform {platform_id}. "
            f"Run the platform annotation download first."
        )

    mapped = map_probes_to_genes(probe_ids, expression, mapping, collapse=collapse)

    return MappedGEODataset(
        expression=mapped.expression,
        gene_names=mapped.gene_names,
        conditions=conditions,
        species=species,
        accession=accession,
        id_format=fmt,
        n_original_ids=len(probe_ids),
        n_mapped=mapped.n_probes_mapped,
        n_unmapped=mapped.n_probes_unmapped,
        n_collapsed=mapped.n_genes_collapsed,
        platform=platform_id,
    )


def pool_cross_platform(
    datasets: list[MappedGEODataset],
    max_genes: int | None = None,
) -> tuple[GEOExpressionData, dict[str, Any]]:
    """Pool multiple mapped datasets into one GEOExpressionData on common genes.

    All datasets must declare the same species (firewall enforced).
    Optionally subsets to top-k genes by cross-dataset variance.
    """
    if not datasets:
        raise ValueError("No datasets to pool")

    species_set = {d.species for d in datasets}
    if len(species_set) != 1:
        raise ValueError(f"Cross-species pooling refused: {species_set}")
    species = species_set.pop()

    gene_lists = [d.gene_names for d in datasets]
    gene_sets = [set(gl) for gl in gene_lists]
    common = sorted(set.intersection(*gene_sets))
    if not common:
        raise ValueError("No common genes across datasets after mapping")

    if max_genes is not None and len(common) > max_genes:
        stacked = []
        for d in datasets:
            g2i = {g: j for j, g in enumerate(d.gene_names)}
            idx = [g2i[g] for g in common]
            stacked.append(d.expression[:, idx])
        all_expr = np.vstack(stacked)
        gene_var = np.var(all_expr, axis=0)
        top_idx = np.argsort(gene_var)[-max_genes:]
        top_idx = np.sort(top_idx)
        common = [common[i] for i in top_idx]

    aligned = []
    pooled_conditions: list[str] = []
    per_dataset_stats = []

    for d in datasets:
        g2i = {g: j for j, g in enumerate(d.gene_names)}
        idx = [g2i[g] for g in common]
        aligned.append(d.expression[:, idx])
        pooled_conditions.extend(d.conditions)
        per_dataset_stats.append({
            "accession": d.accession,
            "platform": d.platform,
            "id_format": d.id_format.value,
            "n_original_ids": d.n_original_ids,
            "n_mapped": d.n_mapped,
            "n_unmapped": d.n_unmapped,
            "n_collapsed": d.n_collapsed,
            "n_genes_after_mapping": len(d.gene_names),
            "n_common_genes": len(common),
            "gene_retention": len(common) / max(len(d.gene_names), 1),
            "n_samples": d.expression.shape[0],
        })

    pooled_expr = np.vstack(aligned).astype(np.float32)
    accessions = [d.accession for d in datasets]

    pooled_data = GEOExpressionData(
        expression=pooled_expr,
        conditions=pooled_conditions,
        gene_names=common,
        species=species,
        accession="+".join(accessions),
    )

    metadata = {
        "n_datasets": len(datasets),
        "n_samples_total": pooled_expr.shape[0],
        "n_common_genes": len(common),
        "max_genes": max_genes,
        "species": species,
        "platforms": sorted({d.platform for d in datasets}),
        "per_dataset": per_dataset_stats,
    }
    return pooled_data, metadata


def load_and_pool_all_merlin(
    merlin_root: str | Path,
    species: str = "human",
    max_genes: int | None = None,
    log1p: bool = True,
    collapse: str = "mean",
) -> tuple[GEOExpressionData, dict[str, Any]]:
    """Load all qualifying GEO series from Merlin and pool across platforms.

    This is the top-level function for NT-005 multi-program pooling.
    Discovers all expression CSVs, applies probe-to-gene mapping per platform,
    and pools on the common gene intersection.

    Returns (pooled_data, metadata) with per-series provenance and mapping stats.
    """
    merlin_root = Path(merlin_root)
    expr_dir = merlin_root / "backend" / "data" / "expression" / species

    if not expr_dir.exists():
        raise ValueError(f"Expression directory not found: {expr_dir}")

    datasets: list[MappedGEODataset] = []
    for csv_file in sorted(expr_dir.glob("*.csv")):
        try:
            d = load_and_map_geo(csv_file, species=species, log1p=log1p, collapse=collapse)
            n_ctrl = sum(1 for c in d.conditions if c == "control")
            n_treat = sum(1 for c in d.conditions if c == "treated")
            if n_ctrl > 0 and n_treat > 0 and len(d.gene_names) > 0:
                datasets.append(d)
        except Exception:
            continue

    if not datasets:
        raise ValueError(f"No qualifying series found in {expr_dir}")

    return pool_cross_platform(datasets, max_genes=max_genes)
