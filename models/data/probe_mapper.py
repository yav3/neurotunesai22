"""Cross-platform probe-to-gene mapper for NT-005 multi-program pooling.

Maps microarray probe IDs to HGNC gene symbols, enabling cross-platform
pooling of GEO expression datasets that use different ID namespaces.

Known platforms in Merlin's committed data:
  - Gene symbols (GSE119262, all subsets): pass-through
  - Affymetrix GPL570 / HG-U133 Plus 2.0 (GSE5281, GSE30174): 54675 probes
  - Illumina ILMN_ format (GSE26927): ~20589 probes
  - GI_ accession format (GSE15222): ~24354 probes

The mapper:
  1. Detects ID format from the first N gene/probe identifiers
  2. Loads or builds a probe→gene mapping for the detected platform
  3. Collapses many-to-one probes (multiple probes per gene) by mean
  4. Returns a gene-symbol-indexed expression matrix ready for pooling
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class IDFormat(str, Enum):
    GENE_SYMBOL = "gene_symbol"
    AFFYMETRIX = "affymetrix"
    ILLUMINA_ILMN = "illumina_ilmn"
    ILLUMINA_GI = "illumina_gi"
    ENSEMBL = "ensembl"
    UNKNOWN = "unknown"


# ── Format detection ─────────────────────────────────────────────────────

_AFFY_RE = re.compile(r"^\d+(_[a-z])?_at$", re.IGNORECASE)
_ILMN_RE = re.compile(r"^ILMN_\d+$", re.IGNORECASE)
_GI_RE = re.compile(r"^GI_\d+(-[AS])?$", re.IGNORECASE)
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{11}$")
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9\-]{0,19}$")


def detect_id_format(ids: list[str], sample_n: int = 50) -> IDFormat:
    """Classify the ID namespace from a sample of identifiers."""
    sample = ids[:sample_n]
    if not sample:
        return IDFormat.UNKNOWN

    counts = {fmt: 0 for fmt in IDFormat}
    for sid in sample:
        sid = sid.strip()
        if _AFFY_RE.match(sid):
            counts[IDFormat.AFFYMETRIX] += 1
        elif _ILMN_RE.match(sid):
            counts[IDFormat.ILLUMINA_ILMN] += 1
        elif _GI_RE.match(sid):
            counts[IDFormat.ILLUMINA_GI] += 1
        elif _ENSEMBL_RE.match(sid):
            counts[IDFormat.ENSEMBL] += 1
        elif _SYMBOL_RE.match(sid):
            counts[IDFormat.GENE_SYMBOL] += 1
        else:
            counts[IDFormat.UNKNOWN] += 1

    best = max(counts, key=counts.get)  # type: ignore[arg-type]
    if counts[best] < len(sample) * 0.5:
        return IDFormat.UNKNOWN
    return best


# ── Probe mapping tables ─────────────────────────────────────────────────

@dataclass
class PlatformMapping:
    """A probe-to-gene mapping for a specific microarray platform."""
    platform: str
    probe_to_gene: dict[str, str]
    n_probes: int = 0
    n_genes: int = 0
    content_hash: str = ""

    def __post_init__(self):
        self.n_probes = len(self.probe_to_gene)
        self.n_genes = len(set(self.probe_to_gene.values()))

    def map_id(self, probe_id: str) -> str | None:
        return self.probe_to_gene.get(probe_id)


_MAPPING_CACHE_DIR = Path(__file__).parent / "_platform_cache"


def _cache_path(platform: str) -> Path:
    return _MAPPING_CACHE_DIR / f"{platform}_mapping.json"


def save_mapping(mapping: PlatformMapping) -> Path:
    """Persist a platform mapping to the local cache."""
    _MAPPING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(mapping.platform)
    data = {
        "platform": mapping.platform,
        "probe_to_gene": mapping.probe_to_gene,
    }
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path


def load_cached_mapping(platform: str) -> PlatformMapping | None:
    """Load a cached platform mapping if available."""
    path = _cache_path(platform)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return PlatformMapping(
        platform=data["platform"],
        probe_to_gene=data["probe_to_gene"],
    )


def parse_geo_soft_annotation(
    soft_text: str,
    platform: str = "",
) -> PlatformMapping:
    """Parse a GEO SOFT platform annotation into a probe→gene mapping.

    SOFT format: tab-delimited, header line starts with !platform_table_begin,
    column headers follow, then probe rows, ending with !platform_table_end.
    The 'Gene Symbol' column (or 'Symbol') holds the HGNC symbol.
    """
    lines = soft_text.split("\n")
    in_table = False
    headers: list[str] = []
    probe_col = -1
    gene_col = -1
    mapping: dict[str, str] = {}

    for line in lines:
        line = line.rstrip()
        if line.startswith("!platform_table_begin"):
            in_table = True
            continue
        if line.startswith("!platform_table_end"):
            break
        if not in_table:
            continue

        if not headers:
            headers = line.split("\t")
            for i, h in enumerate(headers):
                hl = h.strip().lower()
                if hl == "id":
                    probe_col = i
                if hl in ("gene symbol", "symbol", "gene_symbol"):
                    gene_col = i
            if probe_col < 0:
                probe_col = 0
            continue

        parts = line.split("\t")
        if probe_col >= len(parts) or gene_col < 0 or gene_col >= len(parts):
            continue

        probe_id = parts[probe_col].strip()
        gene_raw = parts[gene_col].strip()
        if not probe_id or not gene_raw or gene_raw == "---":
            continue

        gene = gene_raw.split("///")[0].strip()
        if gene and _SYMBOL_RE.match(gene):
            mapping[probe_id] = gene

    return PlatformMapping(platform=platform, probe_to_gene=mapping)


def build_mapping_from_csv(
    csv_path: str | Path,
    probe_col: int = 0,
    gene_col: int = 1,
    platform: str = "",
    delimiter: str = "\t",
    skip_header: bool = True,
) -> PlatformMapping:
    """Build a probe→gene mapping from a two-column annotation file."""
    mapping: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        for row in reader:
            if len(row) <= max(probe_col, gene_col):
                continue
            probe_id = row[probe_col].strip()
            gene = row[gene_col].strip()
            if probe_id and gene and gene != "---":
                gene = gene.split("///")[0].strip()
                if _SYMBOL_RE.match(gene):
                    mapping[probe_id] = gene

    return PlatformMapping(platform=platform, probe_to_gene=mapping)


# ── Expression matrix mapping ────────────────────────────────────────────

@dataclass
class MappedExpression:
    """Gene-symbol-indexed expression matrix after probe mapping."""
    gene_names: list[str]
    expression: np.ndarray  # (n_samples, n_genes)
    original_ids: list[str]
    id_format: IDFormat
    n_probes_mapped: int
    n_probes_unmapped: int
    n_genes_collapsed: int  # genes that had multiple probes averaged
    mapping_platform: str


def map_probes_to_genes(
    probe_ids: list[str],
    expression: np.ndarray,
    mapping: PlatformMapping,
    collapse: str = "mean",
) -> MappedExpression:
    """Map probe-level expression to gene-level by collapsing multi-probe genes.

    Parameters
    ----------
    probe_ids : probe/gene identifiers, length G (matching expression columns)
    expression : (n_samples, G) array
    mapping : PlatformMapping with probe→gene
    collapse : 'mean' (default), 'median', or 'max' for multi-probe genes
    """
    gene_to_cols: dict[str, list[int]] = {}
    n_mapped = 0
    n_unmapped = 0

    for i, pid in enumerate(probe_ids):
        gene = mapping.map_id(pid)
        if gene is None:
            n_unmapped += 1
            continue
        n_mapped += 1
        gene_to_cols.setdefault(gene, []).append(i)

    genes_sorted = sorted(gene_to_cols.keys())
    n_samples = expression.shape[0]
    result = np.empty((n_samples, len(genes_sorted)), dtype=np.float32)
    n_collapsed = 0

    for j, gene in enumerate(genes_sorted):
        cols = gene_to_cols[gene]
        if len(cols) == 1:
            result[:, j] = expression[:, cols[0]]
        else:
            n_collapsed += 1
            sub = expression[:, cols]
            if collapse == "mean":
                result[:, j] = sub.mean(axis=1)
            elif collapse == "median":
                result[:, j] = np.median(sub, axis=1)
            elif collapse == "max":
                result[:, j] = sub.max(axis=1)
            else:
                result[:, j] = sub.mean(axis=1)

    return MappedExpression(
        gene_names=genes_sorted,
        expression=result,
        original_ids=probe_ids,
        id_format=detect_id_format(probe_ids),
        n_probes_mapped=n_mapped,
        n_probes_unmapped=n_unmapped,
        n_genes_collapsed=n_collapsed,
        mapping_platform=mapping.platform,
    )


# ── Cross-platform intersection ──────────────────────────────────────────

def find_common_genes(*gene_lists: list[str]) -> list[str]:
    """Return sorted intersection of multiple gene name lists."""
    if not gene_lists:
        return []
    common = set(gene_lists[0])
    for gl in gene_lists[1:]:
        common &= set(gl)
    return sorted(common)


@dataclass
class CrossPlatformPool:
    """Result of cross-platform gene intersection and alignment."""
    common_genes: list[str]
    n_datasets: int
    expressions: list[np.ndarray]  # each (n_samples_i, n_common_genes)
    accessions: list[str]
    per_dataset_stats: list[dict[str, Any]]

    @property
    def total_samples(self) -> int:
        return sum(e.shape[0] for e in self.expressions)

    @property
    def n_common_genes(self) -> int:
        return len(self.common_genes)

    def pooled_expression(self) -> np.ndarray:
        """Concatenate all datasets vertically on the common gene set."""
        return np.concatenate(self.expressions, axis=0)


def align_to_common_genes(
    mapped_datasets: list[MappedExpression],
    accessions: list[str] | None = None,
) -> CrossPlatformPool:
    """Align multiple mapped datasets to their common gene intersection."""
    if accessions is None:
        accessions = [f"dataset_{i}" for i in range(len(mapped_datasets))]

    gene_lists = [ds.gene_names for ds in mapped_datasets]
    common = find_common_genes(*gene_lists)

    expressions = []
    stats = []
    for i, ds in enumerate(mapped_datasets):
        gene_to_idx = {g: j for j, g in enumerate(ds.gene_names)}
        indices = [gene_to_idx[g] for g in common]
        aligned = ds.expression[:, indices]
        expressions.append(aligned)
        stats.append({
            "accession": accessions[i],
            "original_genes": len(ds.gene_names),
            "common_genes": len(common),
            "gene_retention": len(common) / max(len(ds.gene_names), 1),
            "n_samples": ds.expression.shape[0],
            "id_format": ds.id_format.value,
            "n_probes_mapped": ds.n_probes_mapped,
            "n_probes_unmapped": ds.n_probes_unmapped,
        })

    return CrossPlatformPool(
        common_genes=common,
        n_datasets=len(mapped_datasets),
        expressions=expressions,
        accessions=accessions,
        per_dataset_stats=stats,
    )
