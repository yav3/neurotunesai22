"""Literature evidence loader — PubMed, ClinVar, ClinicalTrials.gov digests.

Loads Merlin's committed evidence digest JSON files into structured objects
for NeuroTunesAI. Each digest aggregates multi-source evidence for a disease
or condition:

  - ClinVar: per-gene variant pathogenicity counts
  - PubMed: article metadata (PMID, title, journal, year)
  - ClinicalTrials.gov: trial metadata (NCT ID, phase, status, title)
  - BioGPS: dataset identifiers
  - LOVD: variant database reachability

These provide bibliometric evidence density for the governed evidence layer:
a gene or disease subtype with deep ClinVar + PubMed coverage carries more
triangulation weight than one with sparse literature.

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
class ClinVarGeneEntry:
    """ClinVar pathogenicity variant counts for one gene."""
    gene: str
    pathogenic: int = 0
    likely_pathogenic: int = 0
    benign: int = 0
    uncertain_significance: int = 0
    total_submitted: int = 0


@dataclass
class PubMedRecord:
    """Minimal PubMed article record."""
    pmid: str
    title: str
    journal: str = ""
    year: int = 0


@dataclass
class ClinicalTrial:
    """ClinicalTrials.gov study record."""
    nct_id: str
    title: str
    status: str = ""
    phases: list[str] = field(default_factory=list)


@dataclass
class LiteratureEvidenceDigest:
    """Multi-source literature evidence for a disease/condition."""
    clinvar_genes: list[ClinVarGeneEntry]
    pubmed_records: list[PubMedRecord]
    clinical_trials: list[ClinicalTrial]
    total_pubmed: int
    total_trials: int
    source_path: str
    built_date: str = ""
    content_hash: str = ""

    @property
    def n_clinvar_genes(self) -> int:
        return len(self.clinvar_genes)

    @property
    def n_pubmed_records(self) -> int:
        return len(self.pubmed_records)

    @property
    def n_clinical_trials(self) -> int:
        return len(self.clinical_trials)

    def clinvar_gene_names(self) -> list[str]:
        return sorted(g.gene for g in self.clinvar_genes)

    def pathogenic_genes(self, min_pathogenic: int = 1) -> list[str]:
        """Genes with at least min_pathogenic pathogenic + likely pathogenic variants."""
        return sorted(
            g.gene for g in self.clinvar_genes
            if (g.pathogenic + g.likely_pathogenic) >= min_pathogenic
        )

    def pathogenicity_vector(self) -> tuple[np.ndarray, list[str]]:
        """Pathogenic variant counts as a vector, sorted by gene."""
        genes = self.clinvar_gene_names()
        by_name = {g.gene: g for g in self.clinvar_genes}
        vals = np.array(
            [by_name[g].pathogenic + by_name[g].likely_pathogenic for g in genes],
            dtype=np.float32,
        )
        return vals, genes

    def publication_years(self) -> np.ndarray:
        """Array of publication years (0 excluded) for temporal analysis."""
        years = [r.year for r in self.pubmed_records if r.year > 0]
        return np.array(years, dtype=np.int32) if years else np.array([], dtype=np.int32)

    def trials_by_phase(self) -> dict[str, int]:
        """Count of clinical trials by phase."""
        counts: dict[str, int] = {}
        for t in self.clinical_trials:
            for phase in t.phases:
                counts[phase] = counts.get(phase, 0) + 1
            if not t.phases:
                counts["UNPHASED"] = counts.get("UNPHASED", 0) + 1
        return counts

    def active_trials(self) -> list[ClinicalTrial]:
        """Trials with recruiting or active status."""
        active_statuses = {
            "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION",
            "NOT_YET_RECRUITING",
        }
        return [t for t in self.clinical_trials if t.status in active_statuses]


def load_literature_digest(json_path: str | Path) -> LiteratureEvidenceDigest:
    """Load a literature evidence digest JSON from Merlin's data directory."""
    json_path = Path(json_path)
    with open(json_path) as f:
        raw = json.load(f)

    about = raw.get("_about", {})

    # ClinVar genes
    clinvar_genes: list[ClinVarGeneEntry] = []
    for gene, counts in raw.get("clinvar", {}).get("genes", {}).items():
        clinvar_genes.append(ClinVarGeneEntry(
            gene=gene,
            pathogenic=counts.get("pathogenic", 0),
            likely_pathogenic=counts.get("likely pathogenic", 0),
            benign=counts.get("benign", 0),
            uncertain_significance=counts.get("uncertain significance", 0),
            total_submitted=counts.get("total_submitted", 0),
        ))

    # PubMed records
    pubmed_records: list[PubMedRecord] = []
    for rec in raw.get("pubmed", {}).get("records", []):
        pubmed_records.append(PubMedRecord(
            pmid=str(rec.get("pmid", "")),
            title=rec.get("title", ""),
            journal=rec.get("journal", ""),
            year=rec.get("year", 0),
        ))

    # Clinical trials
    clinical_trials: list[ClinicalTrial] = []
    for study in raw.get("clinicaltrials", {}).get("studies", []):
        clinical_trials.append(ClinicalTrial(
            nct_id=study.get("nct_id", ""),
            title=study.get("title", ""),
            status=study.get("status", ""),
            phases=study.get("phases", []),
        ))

    # Content hash
    h = hashlib.sha256()
    h.update(json.dumps(sorted(g.gene for g in clinvar_genes)).encode())
    h.update(str(len(pubmed_records)).encode())
    h.update(str(len(clinical_trials)).encode())
    for g in sorted(clinvar_genes, key=lambda x: x.gene):
        h.update(f"{g.gene}:{g.pathogenic}:{g.likely_pathogenic}".encode())

    return LiteratureEvidenceDigest(
        clinvar_genes=clinvar_genes,
        pubmed_records=pubmed_records,
        clinical_trials=clinical_trials,
        total_pubmed=raw.get("pubmed", {}).get("total_records", len(pubmed_records)),
        total_trials=raw.get("clinicaltrials", {}).get("total_studies", len(clinical_trials)),
        source_path=str(json_path),
        built_date=about.get("built", ""),
        content_hash=h.hexdigest(),
    )


def discover_literature_digests(
    merlin_literature_dir: str | Path | None = None,
    species: str = "human",
) -> list[dict[str, Any]]:
    """Discover literature evidence digest JSON files in Merlin's data directory."""
    if merlin_literature_dir is None:
        merlin_literature_dir = (
            Path(__file__).parent.parent.parent.parent
            / "merlinaineuralplus2" / "backend" / "data" / "literature" / species
        )
    else:
        merlin_literature_dir = Path(merlin_literature_dir)

    if not merlin_literature_dir.exists():
        return []

    results: list[dict[str, Any]] = []
    for json_file in sorted(merlin_literature_dir.glob("*_digest.json")):
        try:
            digest = load_literature_digest(json_file)
            results.append({
                "source_path": str(json_file),
                "name": json_file.stem,
                "species": species,
                "n_clinvar_genes": digest.n_clinvar_genes,
                "n_pubmed_total": digest.total_pubmed,
                "n_trials_total": digest.total_trials,
                "n_pubmed_records": digest.n_pubmed_records,
                "n_clinical_trials": digest.n_clinical_trials,
                "content_hash": digest.content_hash,
                "modality": "literature_evidence",
            })
        except Exception:
            continue

    return results
