"""Prior network and benchmark loader — STRING PPI priors and DREAM5 gold standards.

Bridges Merlin's committed network data into NeuroTunesAI's training and
validation formats. Two data types:

  - STRING PPI priors: per-dataset JSON files with gene lists and weighted
    edges from the STRING protein-protein interaction database. Used as
    prior adjacency matrices for prior-contrast regime training (< 30 samples).
  - DREAM5 benchmark: gold-standard regulatory networks with expression data
    from the DREAM5 network inference challenge. Used for model validation.

Species: human (STRING priors), in_silico (DREAM5).
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PriorNetwork:
    """A protein-protein interaction prior network."""
    genes: list[str]
    adjacency: np.ndarray  # (n_genes, n_genes) symmetric, values in [0, 1]
    species: str
    source: str  # e.g. "STRING"
    required_score: int
    n_edges: int
    content_hash: str = ""


@dataclass
class BenchmarkNetwork:
    """A gold-standard regulatory network with paired expression data."""
    expression: np.ndarray  # (n_samples, n_genes)
    gene_names: list[str]
    tf_names: list[str]
    gold_standard: np.ndarray  # (n_edges, 3) — [tf_idx, target_idx, label]
    n_true_edges: int
    n_false_edges: int
    species: str
    source: str
    content_hash: str = ""


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_string_prior(
    json_path: str | Path,
    species: str = "human",
    normalize: bool = True,
) -> PriorNetwork:
    """Load a STRING PPI prior network from Merlin's JSON format.

    The JSON has: {genes: [...], species: str, required_score: int,
    edges: [[gene_a, gene_b, score], ...]}.

    Returns a PriorNetwork with a symmetric adjacency matrix.
    Edge scores are in [0, 1000]; normalize=True maps to [0, 1].
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    genes = data["genes"]
    n = len(genes)
    gene_idx = {g: i for i, g in enumerate(genes)}

    adj = np.zeros((n, n), dtype=np.float32)
    n_edges = 0
    for edge in data["edges"]:
        g_a, g_b, score = edge[0], edge[1], float(edge[2])
        if g_a in gene_idx and g_b in gene_idx:
            i, j = gene_idx[g_a], gene_idx[g_b]
            val = score / 1000.0 if normalize else score
            adj[i, j] = val
            adj[j, i] = val
            n_edges += 1

    return PriorNetwork(
        genes=genes,
        adjacency=adj,
        species=data.get("species", species),
        source="STRING",
        required_score=data.get("required_score", 400),
        n_edges=n_edges,
        content_hash=_file_hash(json_path),
    )


def load_dream5_benchmark(
    data_dir: str | Path,
) -> BenchmarkNetwork:
    """Load DREAM5 network inference benchmark data.

    Expects three files in data_dir:
      - network1_subset_expression.csv (gene × samples)
      - network1_subset_tfs.txt (transcription factor names)
      - network1_subset_gold_standard.tsv (gene_a, gene_b, label)

    The expression CSV is genes-as-rows, samples-as-columns (Merlin format).
    """
    data_dir = Path(data_dir)

    expr_path = data_dir / "network1_subset_expression.csv"
    tfs_path = data_dir / "network1_subset_tfs.txt"
    gold_path = data_dir / "network1_subset_gold_standard.tsv"

    # Expression: genes × samples → transpose to samples × genes
    with open(expr_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        gene_names = []
        rows = []
        for row in reader:
            gene_names.append(row[0])
            rows.append([float(x) for x in row[1:]])

    expression = np.array(rows, dtype=np.float32).T  # (samples, genes)

    # Transcription factors
    with open(tfs_path) as f:
        tf_names = [line.strip() for line in f if line.strip()]

    # Gold standard: gene_a \t gene_b \t label (0 or 1)
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    gold_rows = []
    n_true = 0
    n_false = 0
    with open(gold_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            g_a, g_b, label = parts[0], parts[1], int(parts[2])
            if g_a in gene_idx and g_b in gene_idx:
                gold_rows.append([gene_idx[g_a], gene_idx[g_b], label])
                if label == 1:
                    n_true += 1
                else:
                    n_false += 1

    gold_standard = np.array(gold_rows, dtype=np.int32) if gold_rows else np.zeros((0, 3), dtype=np.int32)

    # Content hash over all three files
    h = hashlib.sha256()
    for p in [expr_path, tfs_path, gold_path]:
        h.update(_file_hash(p).encode())
    combined_hash = h.hexdigest()

    return BenchmarkNetwork(
        expression=expression,
        gene_names=gene_names,
        tf_names=tf_names,
        gold_standard=gold_standard,
        n_true_edges=n_true,
        n_false_edges=n_false,
        species="in_silico",
        source="DREAM5_network1",
        content_hash=combined_hash,
    )


def prior_to_adjacency(
    prior: PriorNetwork,
    target_genes: list[str],
    threshold: float = 0.0,
) -> np.ndarray:
    """Extract a submatrix from a prior network aligned to target genes.

    Returns a (len(target_genes), len(target_genes)) adjacency matrix.
    Genes not in the prior get zero rows/columns.
    """
    n = len(target_genes)
    adj = np.zeros((n, n), dtype=np.float32)
    gene_idx = {g: i for i, g in enumerate(prior.genes)}

    for i, g_i in enumerate(target_genes):
        for j, g_j in enumerate(target_genes):
            if g_i in gene_idx and g_j in gene_idx:
                val = prior.adjacency[gene_idx[g_i], gene_idx[g_j]]
                if val > threshold:
                    adj[i, j] = val

    return adj


def discover_string_priors(
    merlin_priors_dir: str | Path | None = None,
    species: str = "human",
) -> list[dict[str, Any]]:
    """Discover STRING PPI prior JSON files in Merlin's data directory."""
    if merlin_priors_dir is None:
        merlin_priors_dir = (
            Path(__file__).parent.parent.parent.parent
            / "merlinaineuralplus2" / "backend" / "data" / "priors" / "string" / species
        )
    else:
        merlin_priors_dir = Path(merlin_priors_dir)

    if not merlin_priors_dir.exists():
        return []

    priors = []
    for json_file in sorted(merlin_priors_dir.glob("*.json")):
        try:
            prior = load_string_prior(json_file, species=species)
            priors.append({
                "source_path": str(json_file),
                "dataset": json_file.stem,
                "species": prior.species,
                "n_genes": len(prior.genes),
                "n_edges": prior.n_edges,
                "required_score": prior.required_score,
                "content_hash": prior.content_hash,
                "modality": "ppi_prior",
            })
        except Exception:
            continue

    return priors


def discover_benchmarks(
    merlin_benchmarks_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Discover benchmark datasets in Merlin's data directory."""
    if merlin_benchmarks_dir is None:
        merlin_benchmarks_dir = (
            Path(__file__).parent.parent.parent.parent
            / "merlinaineuralplus2" / "backend" / "data" / "benchmarks"
        )
    else:
        merlin_benchmarks_dir = Path(merlin_benchmarks_dir)

    if not merlin_benchmarks_dir.exists():
        return []

    benchmarks = []

    # DREAM5
    dream5_dir = merlin_benchmarks_dir / "dream5"
    if (dream5_dir / "network1_subset_expression.csv").exists():
        try:
            bench = load_dream5_benchmark(dream5_dir)
            benchmarks.append({
                "source_path": str(dream5_dir),
                "benchmark": "DREAM5_network1",
                "species": bench.species,
                "n_genes": len(bench.gene_names),
                "n_samples": bench.expression.shape[0],
                "n_tfs": len(bench.tf_names),
                "n_true_edges": bench.n_true_edges,
                "n_false_edges": bench.n_false_edges,
                "content_hash": bench.content_hash,
                "modality": "benchmark",
            })
        except Exception:
            pass

    return benchmarks
