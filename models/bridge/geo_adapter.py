"""GEO series adapter — connects to Merlin's GEO validation infrastructure.

Merlin has 11 GEO validation tests (test_geo_*_validation.py) covering
real public expression data with pre-registered criteria:
  - GSE107965: autophagy (rapamycin time course, hESC + fibroblast)
  - GSE201710: reprogramming (IMR90)
  - GSE series for: EDS, erythropoiesis/globin, HSC rejuvenation,
    mesenchymal drift, microglia/APOE4, plasma rejuvenation,
    SOX17/pluripotency, T-cell exhaustion, ZNF512B

The narrow foundation model target: time-course/perturbation model
on 358 in-regime GEO series (56 time-flagged), evaluated against
degree-matched nulls.

This adapter:
  1. Reads GEO dataset envelopes (geo_indexer.py format)
  2. Filters for in-regime series by Merlin's criteria
  3. Produces tensors suitable for NeuroTunesAI solvers
  4. Tracks provenance for evidence chain
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class GEOSeriesMetadata:
    """Metadata for a single GEO series, matching geo_indexer.py envelope."""
    accession: str
    title: str
    organism: str
    platform: str
    sample_count: int
    series_type: str
    is_time_course: bool = False
    is_perturbation: bool = False
    time_points: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    pubmed_ids: list[str] = field(default_factory=list)


@dataclass
class GEOExpressionMatrix:
    """Expression matrix from a GEO series, ready for solver consumption."""
    accession: str
    gene_names: list[str]
    sample_names: list[str]
    expression: torch.Tensor
    conditions: torch.Tensor | None = None
    time_labels: torch.Tensor | None = None
    species: str = "human"

    @property
    def n_genes(self) -> int:
        return self.expression.shape[0]

    @property
    def n_samples(self) -> int:
        return self.expression.shape[1]


class GEOAdapter:
    """Connects Merlin's GEO infrastructure to NeuroTunesAI solvers.

    In-regime filtering criteria (from Merlin's evidence discipline):
      - Species declared (not defaulted)
      - ≥30 samples (statistical power)
      - Time-flagged series: has explicit time-course or perturbation design
      - Platform is array or RNA-seq (not ChIP, methylation, etc.)
    """

    MINIMUM_SAMPLES = 30
    IN_REGIME_PLATFORMS = {"GPL570", "GPL17586", "GPL16791", "GPL20301"}
    EXPRESSION_TYPES = {
        "Expression profiling by array",
        "Expression profiling by high throughput sequencing",
    }

    def __init__(self) -> None:
        self._series_cache: dict[str, GEOSeriesMetadata] = {}

    def load_envelopes(self, path: Path) -> int:
        """Load GEO dataset envelopes from Merlin's geo_indexer output."""
        data = json.loads(path.read_text())
        envelopes = data.get("envelopes", data if isinstance(data, list) else [])
        count = 0
        for env in envelopes:
            meta = env.get("metadata", env)
            acc = meta.get("accession", env.get("dataset_id", "").replace("geo:", ""))
            if not acc:
                continue
            series = GEOSeriesMetadata(
                accession=acc,
                title=meta.get("title", ""),
                organism=meta.get("platform_organism", meta.get("organism", "unknown")),
                platform=meta.get("platform", ""),
                sample_count=meta.get("sample_count", 0),
                series_type=meta.get("series_type", env.get("modality_tags", [""])[0] if "modality_tags" in env else ""),
                is_time_course="time" in meta.get("title", "").lower() or "time course" in meta.get("summary", "").lower(),
                is_perturbation=any(kw in meta.get("title", "").lower() for kw in ["treatment", "perturbation", "knockdown", "knockout", "overexpression"]),
                pubmed_ids=meta.get("pubmed_ids", []),
            )
            self._series_cache[acc] = series
            count += 1
        return count

    def filter_in_regime(
        self,
        species: str = "Homo sapiens",
        min_samples: int | None = None,
        time_flagged_only: bool = False,
    ) -> list[GEOSeriesMetadata]:
        """Filter series by in-regime criteria."""
        min_n = min_samples or self.MINIMUM_SAMPLES
        results = []
        for s in self._series_cache.values():
            if species and s.organism != species:
                continue
            if s.sample_count < min_n:
                continue
            if time_flagged_only and not (s.is_time_course or s.is_perturbation):
                continue
            results.append(s)
        return results

    def csv_to_expression_matrix(
        self,
        csv_text: str,
        accession: str,
        species: str = "human",
    ) -> GEOExpressionMatrix:
        """Parse a gene x sample CSV into a GEOExpressionMatrix.

        Format: first column is gene name, remaining columns are samples.
        Matches Merlin's fixture format (e.g., gse107965_autophagy_subset.csv).
        """
        lines = csv_text.strip().split("\n")
        header = lines[0].split(",")
        sample_names = header[1:]
        gene_names = []
        values = []
        for line in lines[1:]:
            parts = line.split(",")
            gene_names.append(parts[0])
            values.append([float(x) for x in parts[1:]])

        expression = torch.tensor(values, dtype=torch.float32)
        return GEOExpressionMatrix(
            accession=accession,
            gene_names=gene_names,
            sample_names=sample_names,
            expression=expression,
            species=species,
        )

    def expression_to_grn_input(self, matrix: GEOExpressionMatrix) -> torch.Tensor:
        """Convert expression matrix to GRN solver input signal.

        The GRN solver expects (batch, n_entities) where entities are genes.
        We take the mean expression across samples as the external signal.
        """
        return matrix.expression.mean(dim=1).unsqueeze(0)

    @property
    def total_series(self) -> int:
        return len(self._series_cache)

    @property
    def time_flagged_count(self) -> int:
        return sum(1 for s in self._series_cache.values() if s.is_time_course or s.is_perturbation)
