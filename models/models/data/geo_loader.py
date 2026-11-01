"""GEO expression data loader — reads Merlin's committed CSVs into PerturbationDataset format.

Bridges the gap between Merlin's validated GEO expression data (committed as
genes×samples CSVs with condition-encoding column names) and NeuroTunesAI's
PerturbationDataset training format for the NT-001 conditional VAE.

Data flow:
    Merlin CSV → parse (genes×samples) → transpose to (samples×genes)
    → infer conditions from column names → GEOExpressionData
    → PerturbationDataset (baseline/target pairs for training)

Species: human only (firewall-enforced, no cross-species pooling).
Regime: data-driven (≥30 samples) or prior-contrast (<30 samples, ≥2 conditions).
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from models.foundation.perturbation_response import GEOExpressionData


# ── Condition inference from Merlin's CSV column naming conventions ──────────

# GSE107965: hESCs_control_rep1, Fibroblasts_2days_rapamycin_rep2
_RAPAMYCIN_RE = re.compile(r"(?i)(rapamycin|treated)")
_CONTROL_RE = re.compile(r"(?i)(control|untreated|baseline|vehicle|dmso)")

# GSE201710: D20_GFP_R1, D50_P16OSK_R1
_OSK_RE = re.compile(r"(?i)(OSK|P16OSK|treated)")
_GFP_RE = re.compile(r"(?i)(GFP)")

# GSE119262: Patient1.Pre, Patient3.Post
_PRE_POST_RE = re.compile(r"(?i)\.(Pre|Post|Baseline|Treatment)$")


def infer_condition(sample_name: str) -> str:
    """Infer treatment/control label from Merlin's column naming conventions."""
    if _RAPAMYCIN_RE.search(sample_name):
        return "treated"
    if _CONTROL_RE.search(sample_name):
        return "control"
    if _OSK_RE.search(sample_name):
        return "treated"
    if _GFP_RE.search(sample_name):
        return "control"
    m = _PRE_POST_RE.search(sample_name)
    if m:
        label = m.group(1).lower()
        return "control" if label in ("pre", "baseline") else "treated"
    return "unknown"


@dataclass
class GEODatasetMeta:
    """Metadata for a loaded GEO dataset, matching Merlin's registry format."""
    accession: str
    path: str
    n_genes: int
    n_samples: int
    n_control: int
    n_treated: int
    species: str
    content_hash: str
    conditions: list[str] = field(default_factory=list)


def load_geo_csv(
    csv_path: str | Path,
    accession: str = "",
    species: str = "human",
    log1p: bool = True,
    min_variance: float = 0.0,
    condition_override: dict[str, str] | None = None,
) -> GEOExpressionData:
    """Load a Merlin-format GEO expression CSV into GEOExpressionData.

    CSV format: first column = gene names/IDs, remaining columns = samples.
    Values are expression counts (raw or normalized). Gene names occupy rows.

    Parameters
    ----------
    csv_path : path to CSV file
    accession : GEO accession (e.g. "GSE107965"), auto-inferred from filename if empty
    species : "human" (enforced; no cross-species)
    log1p : apply log1p transformation (appropriate for raw counts)
    min_variance : drop genes with variance below this threshold (0 = keep all)
    condition_override : explicit {sample_name: condition} map (bypasses inference)
    """
    csv_path = Path(csv_path)
    if not accession:
        m = re.search(r"(GSE\d+)", csv_path.stem, re.IGNORECASE)
        accession = m.group(1).upper() if m else csv_path.stem

    gene_names: list[str] = []
    sample_names: list[str] = []
    rows: list[list[float]] = []

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        sample_names = [s.strip() for s in header[1:]]
        for row in reader:
            if not row or not row[0].strip():
                continue
            gene_names.append(row[0].strip())
            vals = []
            for v in row[1:]:
                try:
                    vals.append(float(v.strip()) if v.strip() else np.nan)
                except ValueError:
                    vals.append(np.nan)
            rows.append(vals)

    # genes × samples matrix
    matrix = np.array(rows, dtype=np.float32)
    # handle NaN: replace with 0 (conservative for raw counts)
    matrix = np.nan_to_num(matrix, nan=0.0)

    if log1p:
        matrix = np.log1p(np.maximum(matrix, 0.0)).astype(np.float32)

    # filter low-variance genes
    if min_variance > 0:
        variances = matrix.var(axis=1)
        keep = variances >= min_variance
        matrix = matrix[keep]
        gene_names = [g for g, k in zip(gene_names, keep) if k]

    # transpose to (samples × genes) for PerturbationDataset
    expression = matrix.T  # (n_samples, n_genes)

    # infer conditions
    if condition_override:
        conditions = [condition_override.get(s, infer_condition(s)) for s in sample_names]
    else:
        conditions = [infer_condition(s) for s in sample_names]

    return GEOExpressionData(
        expression=expression,
        conditions=conditions,
        gene_names=gene_names,
        species=species,
        accession=accession,
        metadata={
            "source_path": str(csv_path),
            "log1p": log1p,
            "n_unknown_conditions": sum(1 for c in conditions if c == "unknown"),
        },
    )


def content_hash(csv_path: str | Path) -> str:
    """SHA-256 content hash matching Merlin's registry contract."""
    h = hashlib.sha256()
    with open(csv_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_merlin_datasets(
    merlin_data_dir: str | Path | None = None,
    species: str = "human",
) -> list[dict[str, Any]]:
    """Discover GEO expression CSVs in Merlin's data directory.

    Returns metadata dicts compatible with NeuroTunesAI's data registry.
    """
    if merlin_data_dir is None:
        # default: look relative to this file for the sibling repo
        merlin_data_dir = Path(__file__).parent.parent.parent.parent / "merlinaineuralplus2" / "backend" / "data" / "expression" / species
    else:
        merlin_data_dir = Path(merlin_data_dir)

    if not merlin_data_dir.exists():
        return []

    datasets = []
    for csv_file in sorted(merlin_data_dir.glob("*.csv")):
        data = load_geo_csv(csv_file, species=species, log1p=False)
        n_control = sum(1 for c in data.conditions if c == "control")
        n_treated = sum(1 for c in data.conditions if c == "treated")

        datasets.append({
            "accession": data.accession,
            "path": str(csv_file),
            "n_genes": len(data.gene_names),
            "n_samples": data.expression.shape[0],
            "n_control": n_control,
            "n_treated": n_treated,
            "species": species,
            "content_hash": content_hash(csv_file),
            "conditions": data.conditions,
            "in_regime": data.expression.shape[0] >= 30,
        })

    return datasets
