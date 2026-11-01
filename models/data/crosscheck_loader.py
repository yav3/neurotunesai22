"""Crosscheck data loader — AlphaFold DB structure + STRING PPI validation data.

Loads Merlin's committed crosscheck data, which provides independent
validation signals for GRN-nominated targets:

  - AlphaFold DB: per-protein pLDDT confidence (structural actionability)
  - STRING PPI: detailed protein-protein interactions with sub-scores
    (neighborhood, fusion, cooccurrence, co-expression, experimental,
    database, text-mining)

These are used as triangulation evidence in the governed evidence layer:
a gene nominated by GRN inference is more credible if it also has high
structural confidence (AlphaFold) and strong interaction evidence (STRING).

Species: human.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AlphaFoldCrosscheck:
    """AlphaFold DB structural readiness for a set of genes."""
    genes: dict[str, dict[str, float]]  # gene -> {plddt metrics + grn scores}
    n_genes: int
    source_path: str
    fetched_date: str = ""
    description: str = ""
    content_hash: str = ""

    def gene_names(self) -> list[str]:
        return sorted(self.genes.keys())

    def plddt_vector(self) -> tuple[np.ndarray, list[str]]:
        """Mean pLDDT confidence per gene as a vector."""
        names = self.gene_names()
        vals = np.array([self.genes[g].get("mean_plddt", 0.0) for g in names], dtype=np.float32)
        return vals, names

    def structurally_actionable(self, threshold: float = 70.0) -> list[str]:
        """Genes with mean pLDDT >= threshold."""
        return [g for g in self.gene_names()
                if self.genes[g].get("mean_plddt", 0.0) >= threshold]


@dataclass
class StringPPICrosscheck:
    """STRING PPI interaction network for a gene panel."""
    genes: list[str]
    rows: list[dict[str, str]]  # raw STRING rows with all sub-scores
    source: str
    source_path: str
    fetched_for: str = ""
    content_hash: str = ""

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def n_interactions(self) -> int:
        return len(self.rows)

    def adjacency_matrix(self, score_col: str = "score") -> tuple[np.ndarray, list[str]]:
        """Build a weighted adjacency matrix from STRING interactions.

        Args:
            score_col: Which STRING score column to use. Options:
                score (combined), nscore (neighborhood), fscore (fusion),
                pscore (phylogenetic), ascore (coexpression),
                escore (experimental), dscore (database), tscore (textmining).
        """
        gene_idx = {g: i for i, g in enumerate(self.genes)}
        n = len(self.genes)
        adj = np.zeros((n, n), dtype=np.float32)
        for row in self.rows:
            a = row.get("preferredName_A", "")
            b = row.get("preferredName_B", "")
            score = float(row.get(score_col, "0"))
            ia = gene_idx.get(a)
            ib = gene_idx.get(b)
            if ia is not None and ib is not None:
                adj[ia, ib] = max(adj[ia, ib], score)
                adj[ib, ia] = max(adj[ib, ia], score)
        return adj, self.genes

    def high_confidence_edges(self, threshold: float = 0.7) -> list[tuple[str, str, float]]:
        """Edges with combined score above threshold."""
        edges = []
        for row in self.rows:
            score = float(row.get("score", "0"))
            if score >= threshold:
                edges.append((
                    row.get("preferredName_A", ""),
                    row.get("preferredName_B", ""),
                    score,
                ))
        return edges


def load_alphafold_crosscheck(json_path: str | Path) -> AlphaFoldCrosscheck:
    """Load an AlphaFold DB crosscheck JSON."""
    json_path = Path(json_path)
    with open(json_path) as f:
        raw = json.load(f)

    h = hashlib.sha256()
    h.update(json.dumps(raw.get("genes", {}), sort_keys=True).encode())

    return AlphaFoldCrosscheck(
        genes=raw.get("genes", {}),
        n_genes=raw.get("n_genes", len(raw.get("genes", {}))),
        source_path=str(json_path),
        fetched_date=raw.get("fetched_date", ""),
        description=raw.get("description", ""),
        content_hash=h.hexdigest(),
    )


def load_string_ppi_crosscheck(json_path: str | Path) -> StringPPICrosscheck:
    """Load a STRING PPI crosscheck JSON."""
    json_path = Path(json_path)
    with open(json_path) as f:
        raw = json.load(f)

    h = hashlib.sha256()
    h.update(json.dumps(raw.get("genes", []), sort_keys=True).encode())
    h.update(str(len(raw.get("rows", []))).encode())

    return StringPPICrosscheck(
        genes=raw.get("genes", []),
        rows=raw.get("rows", []),
        source=raw.get("source", ""),
        source_path=str(json_path),
        fetched_for=raw.get("fetched_for", ""),
        content_hash=h.hexdigest(),
    )


def discover_crosschecks(
    merlin_crosscheck_dir: str | Path | None = None,
    species: str = "human",
) -> list[dict[str, Any]]:
    """Discover crosscheck data files in Merlin's data directory."""
    if merlin_crosscheck_dir is None:
        merlin_crosscheck_dir = (
            Path(__file__).parent.parent.parent.parent
            / "merlinaineuralplus2" / "backend" / "data" / "crosschecks" / species
        )
    else:
        merlin_crosscheck_dir = Path(merlin_crosscheck_dir)

    if not merlin_crosscheck_dir.exists():
        return []

    results: list[dict[str, Any]] = []
    for json_file in sorted(merlin_crosscheck_dir.glob("*.json")):
        try:
            name = json_file.stem
            if "alphafold" in name.lower():
                af = load_alphafold_crosscheck(json_file)
                results.append({
                    "source_path": str(json_file),
                    "name": name,
                    "crosscheck_type": "alphafold_structure",
                    "n_genes": af.n_genes,
                    "content_hash": af.content_hash,
                    "modality": "crosscheck",
                })
            elif "string" in name.lower() or "ppi" in name.lower():
                sp = load_string_ppi_crosscheck(json_file)
                results.append({
                    "source_path": str(json_file),
                    "name": name,
                    "crosscheck_type": "string_ppi",
                    "n_genes": sp.n_genes,
                    "n_interactions": sp.n_interactions,
                    "content_hash": sp.content_hash,
                    "modality": "crosscheck",
                })
        except Exception:
            continue

    return results
