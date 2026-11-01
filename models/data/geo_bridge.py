"""GEO dataset bridge — connects Merlin's validated GEO series to NeuroTunesAI training.

Merlin (merlinaineuralplus2) has 11 GEO adapters with real validation results.
This module registers those datasets as training candidates, with species/regime
metadata, sample counts, and validation verdicts from Merlin's study registry.

The strategic purpose: item 2 of the roadmap — "one narrow foundation model on
in-regime GEO series (358 with >= 30 samples, 56 time-flagged)."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GEOSeriesRecord:
    """A GEO series with metadata for training regime selection."""
    accession: str
    species: str
    n_samples: int
    n_conditions: int
    program: str
    has_time_axis: bool
    validation_verdict: str | None
    validation_study_id: str | None
    merlin_adapter: str | None
    notes: str = ""

    @property
    def in_data_driven_regime(self) -> bool:
        return self.n_samples >= 30 and not self.has_time_axis

    @property
    def in_longitudinal_regime(self) -> bool:
        return self.has_time_axis

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        d = asdict(self)
        d["in_data_driven_regime"] = self.in_data_driven_regime
        d["in_longitudinal_regime"] = self.in_longitudinal_regime
        return d


MERLIN_GEO_CATALOG: list[GEOSeriesRecord] = [
    GEOSeriesRecord(
        accession="GSE201710",
        species="human",
        n_samples=16,
        n_conditions=2,
        program="reprogramming",
        has_time_axis=False,
        validation_verdict="mixed",
        validation_study_id="RB-001",
        merlin_adapter="geo_reprogramming_adapter",
        notes="IMR90 fibroblasts, AAV-OSK vs GFP. POU5F1/SOX2 rank 2/3 at 54 genes (GENIE3), "
              "but 79/184 at 184 genes (ARACNE). Method-specific, not panel-size-specific. "
              "Authorship: Altos Labs (Retro competitor).",
    ),
    GEOSeriesRecord(
        accession="GSE107965",
        species="human",
        n_samples=18,
        n_conditions=2,
        program="autophagy",
        has_time_axis=False,
        validation_verdict="met",
        validation_study_id="RB-002",
        merlin_adapter="geo_autophagy_adapter",
        notes="hESCs + fibroblasts, rapamycin vs control. 7/9 pre-specified genes confirm "
              "(FOXO3 1st, MTOR 2nd, ULK1 4th). Strongest single replication.",
    ),
    GEOSeriesRecord(
        accession="GSE119262",
        species="human",
        n_samples=44,
        n_conditions=2,
        program="autophagy",
        has_time_axis=False,
        validation_verdict="not_met",
        validation_study_id="RB-021",
        merlin_adapter=None,
        notes="Everolimus vs baseline, 23 patients. First data-driven regime run. "
              "3/9 autophagy genes in top half vs 9/9 by STRING degree. "
              "Dominance alone 0.778 but composite dragged down by out-degree weight.",
    ),
    GEOSeriesRecord(
        accession="GSE297233",
        species="human",
        n_samples=8,
        n_conditions=2,
        program="reprogramming",
        has_time_axis=False,
        validation_verdict="mixed",
        validation_study_id=None,
        merlin_adapter="geo_mesenchymal_drift_adapter",
        notes="GM00731 aged fibroblast OSK vs control. Independent replication of GSE201710 "
              "on same 8-gene core panel. Small sample (4/arm).",
    ),
    GEOSeriesRecord(
        accession="GSE115678",
        species="human",
        n_samples=24,
        n_conditions=2,
        program="erythropoiesis",
        has_time_axis=False,
        validation_verdict=None,
        validation_study_id=None,
        merlin_adapter="geo_erythropoiesis_globin_adapter",
        notes="Terminal erythropoiesis RNA-seq. Subseries of GSE115684.",
    ),
    GEOSeriesRecord(
        accession="GSE252062",
        species="mouse",
        n_samples=12,
        n_conditions=2,
        program="hsc_rejuvenation",
        has_time_axis=False,
        validation_verdict=None,
        validation_study_id=None,
        merlin_adapter="geo_hsc_rejuvenation_adapter",
        notes="Myeloid-biased HSC depletion, aged mouse. Species firewall: MOUSE.",
    ),
]


class GEOBridge:
    """Bridge between Merlin's GEO adapters and NeuroTunesAI training pipeline."""

    def __init__(self, catalog: list[GEOSeriesRecord] | None = None):
        self._catalog = {r.accession: r for r in (catalog or MERLIN_GEO_CATALOG)}

    def get(self, accession: str) -> GEOSeriesRecord | None:
        return self._catalog.get(accession)

    def list_all(self) -> list[GEOSeriesRecord]:
        return list(self._catalog.values())

    def filter_in_regime(self, min_samples: int = 30) -> list[GEOSeriesRecord]:
        """Return series that qualify for data-driven inference."""
        return [r for r in self._catalog.values()
                if r.n_samples >= min_samples and not r.has_time_axis]

    def filter_by_species(self, species: str) -> list[GEOSeriesRecord]:
        return [r for r in self._catalog.values() if r.species == species]

    def filter_validated(self, verdict: str | None = None) -> list[GEOSeriesRecord]:
        """Return series with validation results. If verdict given, filter to that verdict."""
        if verdict:
            return [r for r in self._catalog.values()
                    if r.validation_verdict == verdict]
        return [r for r in self._catalog.values()
                if r.validation_verdict is not None]

    def training_candidates(self, species: str = "human", min_samples: int = 30) -> list[GEOSeriesRecord]:
        """Return series suitable for narrow foundation model training.

        Criteria: correct species, enough samples, not longitudinal.
        The strategic target: 358 human GEO series with >= 30 samples.
        """
        return [r for r in self._catalog.values()
                if r.species == species
                and r.n_samples >= min_samples
                and not r.has_time_axis]

    def summary(self) -> dict[str, Any]:
        all_r = list(self._catalog.values())
        return {
            "total": len(all_r),
            "human": len([r for r in all_r if r.species == "human"]),
            "mouse": len([r for r in all_r if r.species == "mouse"]),
            "in_regime": len([r for r in all_r if r.in_data_driven_regime]),
            "validated": len([r for r in all_r if r.validation_verdict]),
            "met": len([r for r in all_r if r.validation_verdict == "met"]),
            "not_met": len([r for r in all_r if r.validation_verdict == "not_met"]),
            "mixed": len([r for r in all_r if r.validation_verdict == "mixed"]),
        }
