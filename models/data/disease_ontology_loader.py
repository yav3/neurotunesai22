"""Disease ontology loader — MONDO/HPO disease-phenotype graphs from Merlin.

Loads Merlin's committed disease ontology JSON files into structured graph
objects for NeuroTunesAI. The ontology encodes:

  - Disease subtypes (MONDO IDs) with HPO phenotype annotations
  - Hierarchical (subtype_of) and lateral (comorbid, overlap) edges
  - Cross-references to OMIM, Orphanet, GARD, ICD-10, SNOMED

This is the phenotype-level complement to the gene-level data (expression,
PPI priors). Together they support disease-gene-phenotype triangulation
in the governed evidence layer.

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
class DiseaseNode:
    """A disease or phenotype-group node in the ontology."""
    mondo_id: str
    label: str
    kind: str  # eds_subtype, adjacent_candidate, comorbidity_candidate, formal_ancestor, etc.
    hpo_ids: list[str] = field(default_factory=list)
    xrefs: list[str] = field(default_factory=list)
    hpo_sources: dict[str, str] = field(default_factory=dict)  # source_id -> disease_name


@dataclass
class DiseaseEdge:
    """A directed edge in the disease ontology."""
    source: str  # MONDO ID
    target: str  # MONDO ID
    edge_type: str  # subtype_of, comorbid_with_literature, overlap_syndrome_with, etc.
    edge_source: str = ""


@dataclass
class DiseaseOntology:
    """A disease ontology graph with phenotype annotations."""
    nodes: dict[str, DiseaseNode]
    edges: list[DiseaseEdge]
    species: str
    source_path: str
    built_date: str = ""
    sources: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_disease_subtypes(self) -> int:
        return sum(1 for n in self.nodes.values() if n.kind == "eds_subtype")

    @property
    def n_with_hpo(self) -> int:
        return sum(1 for n in self.nodes.values() if n.hpo_ids)

    def all_hpo_ids(self) -> set[str]:
        """All unique HPO IDs across all nodes."""
        ids: set[str] = set()
        for node in self.nodes.values():
            ids.update(node.hpo_ids)
        return ids

    def adjacency_matrix(self, edge_types: set[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Build a binary adjacency matrix from the ontology edges.

        Args:
            edge_types: If given, include only these edge types. None = all.

        Returns:
            (adjacency, node_ids): binary matrix and ordered MONDO IDs.
        """
        node_ids = sorted(self.nodes.keys())
        idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        adj = np.zeros((n, n), dtype=np.float32)
        for edge in self.edges:
            if edge_types and edge.edge_type not in edge_types:
                continue
            si = idx.get(edge.source)
            ti = idx.get(edge.target)
            if si is not None and ti is not None:
                adj[si, ti] = 1.0
                adj[ti, si] = 1.0
        return adj, node_ids

    def phenotype_overlap_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Jaccard similarity of HPO phenotype sets between disease nodes.

        Returns:
            (similarity, node_ids): Jaccard matrix and ordered MONDO IDs
            for nodes that have HPO annotations.
        """
        annotated = [(nid, set(n.hpo_ids)) for nid, n in sorted(self.nodes.items()) if n.hpo_ids]
        node_ids = [nid for nid, _ in annotated]
        n = len(annotated)
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                si = annotated[i][1]
                sj = annotated[j][1]
                inter = len(si & sj)
                union = len(si | sj)
                val = inter / union if union > 0 else 0.0
                sim[i, j] = val
                sim[j, i] = val
        return sim, node_ids


def load_disease_ontology(
    json_path: str | Path,
    species: str = "human",
) -> DiseaseOntology:
    """Load a disease ontology JSON from Merlin's data directory.

    Args:
        json_path: Path to the ontology JSON file.
        species: Species label (default: human).

    Returns:
        DiseaseOntology with nodes, edges, and phenotype annotations.
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        raw = json.load(f)

    about = raw.get("_about", {})

    # Parse nodes
    nodes: dict[str, DiseaseNode] = {}
    for mondo_id, ndata in raw.get("nodes", {}).items():
        all_hpo: list[str] = []
        hpo_sources: dict[str, str] = {}
        for src_id, src_data in ndata.get("hpo_symptom_sets", {}).items():
            hpo_sources[src_id] = src_data.get("disease_name", "")
            all_hpo.extend(src_data.get("hpo_ids", []))
        # Deduplicate HPO IDs while preserving order
        seen: set[str] = set()
        unique_hpo: list[str] = []
        for hid in all_hpo:
            if hid not in seen:
                seen.add(hid)
                unique_hpo.append(hid)

        nodes[mondo_id] = DiseaseNode(
            mondo_id=mondo_id,
            label=ndata.get("label", ""),
            kind=ndata.get("kind", "unknown"),
            hpo_ids=unique_hpo,
            xrefs=ndata.get("xref", []),
            hpo_sources=hpo_sources,
        )

    # Parse edges
    edges: list[DiseaseEdge] = []
    for edata in raw.get("edges", []):
        edges.append(DiseaseEdge(
            source=edata["source"],
            target=edata["target"],
            edge_type=edata["type"],
            edge_source=edata.get("edge_source", ""),
        ))

    # Content hash
    h = hashlib.sha256()
    h.update(json.dumps(sorted(nodes.keys()), sort_keys=True).encode())
    h.update(json.dumps([(e.source, e.target, e.edge_type) for e in edges], sort_keys=True).encode())
    for nid in sorted(nodes.keys()):
        h.update(json.dumps(nodes[nid].hpo_ids, sort_keys=True).encode())

    return DiseaseOntology(
        nodes=nodes,
        edges=edges,
        species=species,
        source_path=str(json_path),
        built_date=about.get("built", ""),
        sources=about.get("sources", {}),
        content_hash=h.hexdigest(),
    )


def discover_disease_ontologies(
    merlin_ontology_dir: str | Path | None = None,
    species: str = "human",
) -> list[dict[str, Any]]:
    """Discover disease ontology JSON files in Merlin's data directory."""
    if merlin_ontology_dir is None:
        merlin_ontology_dir = (
            Path(__file__).parent.parent.parent.parent
            / "merlinaineuralplus2" / "backend" / "data" / "ontology" / species
        )
    else:
        merlin_ontology_dir = Path(merlin_ontology_dir)

    if not merlin_ontology_dir.exists():
        return []

    ontologies: list[dict[str, Any]] = []
    for json_file in sorted(merlin_ontology_dir.glob("*_ontology.json")):
        try:
            onto = load_disease_ontology(json_file, species=species)
            ontologies.append({
                "source_path": str(json_file),
                "name": json_file.stem,
                "species": onto.species,
                "n_nodes": onto.n_nodes,
                "n_edges": onto.n_edges,
                "n_disease_subtypes": onto.n_disease_subtypes,
                "n_with_hpo": onto.n_with_hpo,
                "n_unique_hpo": len(onto.all_hpo_ids()),
                "content_hash": onto.content_hash,
                "modality": "disease_ontology",
            })
        except Exception:
            continue

    return ontologies
