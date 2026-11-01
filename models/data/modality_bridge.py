"""Modality bridge — unified data discovery and loading across data types.

Bridges all available data modalities from Merlin's committed data into
NeuroTunesAI's training formats. Each modality has its own loader;
the bridge provides a unified discovery + regime classification interface.

Supported modalities:
  - bulk_expression: GEO CSVs (genes × samples), via geo_loader
  - perturbseq: Xaira-format parquet (cells × genes + knockout targets), via perturbseq_loader
  - ppi_prior: STRING PPI prior networks, via prior_network_loader
  - benchmark: DREAM5 gold-standard evaluation, via prior_network_loader
  - disease_ontology: MONDO/HPO disease-phenotype graphs, via disease_ontology_loader
  - crosscheck: AlphaFold DB structure + STRING PPI validation, via crosscheck_loader
  - literature_evidence: PubMed/ClinVar/ClinicalTrials.gov digests, via literature_loader
  - molecule: quantum chemistry archives (QCSchema/XYZ), via molecule_loader

Planned (not yet implemented):
  - methylation: beta-value arrays (CpG × samples)
  - imaging: extracted feature vectors (samples × features)
  - single_cell_atlas: h5ad/AnnData format
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from models.training.governed_harness import select_training_regime


class DataModality(str, Enum):
    BULK_EXPRESSION = "bulk_expression"
    PERTURBSEQ = "perturbseq"
    PPI_PRIOR = "ppi_prior"
    BENCHMARK = "benchmark"
    DISEASE_ONTOLOGY = "disease_ontology"
    CROSSCHECK = "crosscheck"
    LITERATURE_EVIDENCE = "literature_evidence"
    MOLECULE = "molecule"
    METHYLATION = "methylation"
    IMAGING = "imaging"
    SINGLE_CELL_ATLAS = "single_cell_atlas"


@dataclass
class DatasetEntry:
    """Unified dataset entry across modalities."""
    modality: DataModality
    path: str
    species: str
    n_features: int
    n_samples: int
    content_hash: str
    regime: str
    in_regime: bool
    accession: str = ""
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "modality": self.modality.value,
            "path": self.path,
            "species": self.species,
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "content_hash": self.content_hash,
            "regime": self.regime,
            "in_regime": self.in_regime,
            "accession": self.accession,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


def discover_all_datasets(
    merlin_root: str | Path | None = None,
    species: str = "human",
) -> list[DatasetEntry]:
    """Discover all available datasets across all modalities.

    Scans Merlin's data directory tree for each supported modality
    and returns a unified list with regime classification.
    """
    if merlin_root is None:
        merlin_root = Path(__file__).parent.parent.parent.parent / "merlinaineuralplus2"
    else:
        merlin_root = Path(merlin_root)

    entries: list[DatasetEntry] = []

    # Bulk expression
    from models.data.geo_loader import discover_merlin_datasets
    expr_dir = merlin_root / "backend" / "data" / "expression" / species
    for d in discover_merlin_datasets(expr_dir, species=species):
        regime = select_training_regime(
            n_samples=d["n_samples"],
            n_conditions=2 if d["n_control"] > 0 and d["n_treated"] > 0 else 1,
        )
        entries.append(DatasetEntry(
            modality=DataModality.BULK_EXPRESSION,
            path=d["path"],
            species=species,
            n_features=d["n_genes"],
            n_samples=d["n_samples"],
            content_hash=d["content_hash"],
            regime=regime,
            in_regime=regime == "data-driven",
            accession=d["accession"],
            extra={
                "n_control": d["n_control"],
                "n_treated": d["n_treated"],
            },
        ))

    # Perturb-seq
    from models.data.perturbseq_loader import discover_perturbseq_datasets
    ps_dir = merlin_root / "backend" / "data" / "perturbseq" / species
    for d in discover_perturbseq_datasets(ps_dir, species=species):
        entries.append(DatasetEntry(
            modality=DataModality.PERTURBSEQ,
            path=d["source_path"],
            species=species,
            n_features=d["n_genes"],
            n_samples=d["n_cells"],
            content_hash=d["content_hash"],
            regime="data-driven",
            in_regime=True,
            extra={
                "n_targets": d["n_targets"],
                "n_control_cells": d["n_control_cells"],
                "n_batches": d["n_batches"],
            },
        ))

    # STRING PPI priors
    from models.data.prior_network_loader import discover_string_priors
    priors_dir = merlin_root / "backend" / "data" / "priors" / "string" / species
    for d in discover_string_priors(priors_dir, species=species):
        entries.append(DatasetEntry(
            modality=DataModality.PPI_PRIOR,
            path=d["source_path"],
            species=species,
            n_features=d["n_genes"],
            n_samples=d["n_edges"],
            content_hash=d["content_hash"],
            regime="not_applicable",
            in_regime=True,
            extra={
                "dataset": d["dataset"],
                "required_score": d["required_score"],
            },
        ))

    # Benchmarks (species-agnostic)
    from models.data.prior_network_loader import discover_benchmarks
    benchmarks_dir = merlin_root / "backend" / "data" / "benchmarks"
    for d in discover_benchmarks(benchmarks_dir):
        entries.append(DatasetEntry(
            modality=DataModality.BENCHMARK,
            path=d["source_path"],
            species=d["species"],
            n_features=d["n_genes"],
            n_samples=d["n_samples"],
            content_hash=d["content_hash"],
            regime="data-driven",
            in_regime=True,
            extra={
                "benchmark": d["benchmark"],
                "n_tfs": d["n_tfs"],
                "n_true_edges": d["n_true_edges"],
                "n_false_edges": d["n_false_edges"],
            },
        ))

    # Disease ontologies
    from models.data.disease_ontology_loader import discover_disease_ontologies
    ontology_dir = merlin_root / "backend" / "data" / "ontology" / species
    for d in discover_disease_ontologies(ontology_dir, species=species):
        entries.append(DatasetEntry(
            modality=DataModality.DISEASE_ONTOLOGY,
            path=d["source_path"],
            species=species,
            n_features=d["n_unique_hpo"],
            n_samples=d["n_nodes"],
            content_hash=d["content_hash"],
            regime="not_applicable",
            in_regime=True,
            extra={
                "name": d["name"],
                "n_edges": d["n_edges"],
                "n_disease_subtypes": d["n_disease_subtypes"],
                "n_with_hpo": d["n_with_hpo"],
            },
        ))

    # Crosschecks (AlphaFold structure, STRING PPI validation)
    from models.data.crosscheck_loader import discover_crosschecks
    crosscheck_dir = merlin_root / "backend" / "data" / "crosschecks" / species
    for d in discover_crosschecks(crosscheck_dir, species=species):
        entries.append(DatasetEntry(
            modality=DataModality.CROSSCHECK,
            path=d["source_path"],
            species=species,
            n_features=d["n_genes"],
            n_samples=d.get("n_interactions", d["n_genes"]),
            content_hash=d["content_hash"],
            regime="not_applicable",
            in_regime=True,
            extra={
                "name": d["name"],
                "crosscheck_type": d["crosscheck_type"],
            },
        ))

    # Literature evidence digests (ClinVar, PubMed, ClinicalTrials.gov)
    from models.data.literature_loader import discover_literature_digests
    literature_dir = merlin_root / "backend" / "data" / "literature" / species
    for d in discover_literature_digests(literature_dir, species=species):
        entries.append(DatasetEntry(
            modality=DataModality.LITERATURE_EVIDENCE,
            path=d["source_path"],
            species=species,
            n_features=d["n_clinvar_genes"],
            n_samples=d["n_pubmed_total"],
            content_hash=d["content_hash"],
            regime="not_applicable",
            in_regime=True,
            extra={
                "name": d["name"],
                "n_pubmed_records": d["n_pubmed_records"],
                "n_trials_total": d["n_trials_total"],
                "n_clinical_trials": d["n_clinical_trials"],
            },
        ))

    # Molecule datasets (quantum chemistry archives)
    from models.data.molecule_loader import discover_molecule_datasets
    molecules_dir = merlin_root / "backend" / "data" / "molecules"
    for d in discover_molecule_datasets(molecules_dir):
        entries.append(DatasetEntry(
            modality=DataModality.MOLECULE,
            path=d["source_path"],
            species="not_applicable",
            n_features=len(d.get("unique_elements", [])),
            n_samples=d["n_files_in_archive"],
            content_hash=d["content_hash"],
            regime="not_applicable",
            in_regime=True,
            extra={
                "name": d["name"],
                "n_molecules": d["n_molecules"],
                "unique_elements": d["unique_elements"],
                "atom_count_range": d["atom_count_range"],
                "archive_format": d["archive_format"],
            },
        ))

    return entries


def regime_summary(entries: list[DatasetEntry]) -> dict[str, Any]:
    """Summarise discovered datasets by modality and regime."""
    by_modality: dict[str, list[DatasetEntry]] = {}
    for e in entries:
        by_modality.setdefault(e.modality.value, []).append(e)

    summary = {}
    for mod, datasets in by_modality.items():
        in_regime = [d for d in datasets if d.in_regime]
        summary[mod] = {
            "total": len(datasets),
            "in_regime": len(in_regime),
            "total_samples": sum(d.n_samples for d in datasets),
            "in_regime_samples": sum(d.n_samples for d in in_regime),
            "regimes": list(set(d.regime for d in datasets)),
        }

    summary["_total"] = {
        "datasets": len(entries),
        "in_regime": sum(1 for e in entries if e.in_regime),
        "total_samples": sum(e.n_samples for e in entries),
        "modalities": list(by_modality.keys()),
    }
    return summary
