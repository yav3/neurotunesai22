"""Cross-repo bridge to Merlin's (merlinaineuralplus2) governed evidence system.

Provides typed access to Merlin's:
  - Data registry (backend/data/registry.json): species-partitioned datasets
  - Study records (docs/partnerships/records/*.json): content-addressed results
  - Method router (backend/methods/method_router.py): regime selection
  - GEO dataset scout (docs/data/geo_dataset_scout.json): in-regime series
  - Species firewall (backend/methods/species.py): hard human/mouse boundary
  - Invention harness (backend/app/inventions/harness.py): experiment framework

This bridge reads Merlin's data; it does not write to it. NeuroTunesAI's
training outputs feed back into Merlin via its REST endpoints or by
committing study records in Merlin's format.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

MERLIN_ROOT = Path(__file__).resolve().parents[3] / "merlinaineuralplus2"
MERLIN_DATA = MERLIN_ROOT / "backend" / "data"
MERLIN_RECORDS = MERLIN_ROOT / "docs" / "partnerships" / "records"
MERLIN_GEO_SCOUT = MERLIN_ROOT / "docs" / "data" / "geo_dataset_scout.json"


Species = Literal["human", "mouse", "in_silico", "not_applicable", "unknown"]
Regime = Literal["data-driven", "prior-contrast", "prior-mean", "longitudinal"]

MIN_DATA_DRIVEN_SAMPLES = 30


@dataclass(frozen=True)
class DataRegistryEntry:
    entry_id: str
    path: str
    species: Species
    kind: str
    licence: str
    source: str
    product_use_class: str
    sha256: str
    bytes_: int
    studies: tuple[str, ...] = ()
    accession: str | None = None
    program: str | None = None


@dataclass(frozen=True)
class GEOSeriesInfo:
    accession: str
    title: str
    n_samples: int
    organism: str
    program: str


@dataclass(frozen=True)
class RegimeSelection:
    regime: Regime
    inference_method: str
    state_mode: str
    reason: str
    out_of_regime: bool = False


class MerlinDataRegistry:
    """Read-only access to Merlin's backend/data/registry.json."""

    def __init__(self, registry_path: Path | None = None):
        self._path = registry_path or (MERLIN_DATA / "registry.json")
        self._entries: dict[str, DataRegistryEntry] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text())
        for eid, info in raw.get("entries", {}).items():
            self._entries[eid] = DataRegistryEntry(
                entry_id=eid,
                path=info.get("path", ""),
                species=info.get("species", "unknown"),
                kind=info.get("kind", ""),
                licence=info.get("licence", ""),
                source=info.get("source", ""),
                product_use_class=info.get("product_use_class", ""),
                sha256=info.get("sha256", ""),
                bytes_=info.get("bytes", 0),
                studies=tuple(info.get("studies", [])),
                accession=info.get("accession"),
                program=info.get("program"),
            )

    def get(self, entry_id: str) -> DataRegistryEntry | None:
        return self._entries.get(entry_id)

    def list_by_species(self, species: Species) -> list[DataRegistryEntry]:
        return [e for e in self._entries.values() if e.species == species]

    def list_by_kind(self, kind: str) -> list[DataRegistryEntry]:
        return [e for e in self._entries.values() if e.kind == kind]

    def list_by_program(self, program: str) -> list[DataRegistryEntry]:
        return [e for e in self._entries.values() if e.program == program]

    def list_expression_data(self, species: Species | None = None) -> list[DataRegistryEntry]:
        entries = [e for e in self._entries.values() if e.kind == "expression"]
        if species:
            entries = [e for e in entries if e.species == species]
        return entries

    def resolve_path(self, entry: DataRegistryEntry) -> Path:
        return MERLIN_DATA / entry.path

    @property
    def all_entries(self) -> dict[str, DataRegistryEntry]:
        return dict(self._entries)


class MerlinGEOScout:
    """Read-only access to Merlin's GEO dataset scout results."""

    def __init__(self, scout_path: Path | None = None):
        self._path = scout_path or MERLIN_GEO_SCOUT
        self._data: dict[str, Any] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    @property
    def total_series(self) -> int:
        return self._data.get("total_series", 0)

    @property
    def programs(self) -> list[str]:
        return list(self._data.get("programs", {}).keys())

    def series_for_program(self, program: str) -> list[GEOSeriesInfo]:
        prog_data = self._data.get("programs", {}).get(program, {})
        result = []
        for s in prog_data.get("series", []):
            result.append(GEOSeriesInfo(
                accession=s.get("accession", ""),
                title=s.get("title", ""),
                n_samples=s.get("n_samples", 0),
                organism=s.get("organism", ""),
                program=program,
            ))
        return result

    def in_regime_series(self, min_samples: int = MIN_DATA_DRIVEN_SAMPLES,
                         species: str | None = None) -> list[GEOSeriesInfo]:
        result = []
        for program in self.programs:
            for s in self.series_for_program(program):
                if s.n_samples >= min_samples:
                    if species is None or species.lower() in s.organism.lower():
                        result.append(s)
        return result


def select_regime(*, n_samples: int, n_conditions: int = 0,
                  has_time_axis: bool = False) -> RegimeSelection:
    """Mirror of Merlin's method_router.select_regime for local use."""
    if has_time_axis:
        return RegimeSelection(
            "longitudinal", "rtp", "mean",
            "time axis calls for the temporal driver kernel"
        )
    if n_samples >= MIN_DATA_DRIVEN_SAMPLES:
        state = "contrast" if n_conditions == 2 else "mean"
        return RegimeSelection(
            "data-driven", "genie3", state,
            f"{n_samples} samples supports data-driven inference"
        )
    if n_conditions == 2:
        return RegimeSelection(
            "prior-contrast", "string", "contrast",
            f"{n_samples} samples too few for data-driven; using prior + contrast",
            out_of_regime=False
        )
    return RegimeSelection(
        "prior-mean", "string", "mean",
        f"{n_samples} samples, no conditions: prior edges, magnitude state",
        out_of_regime=False
    )


def infer_species(gene_names: list[str]) -> Species:
    """Simplified species inference from gene names (mirrors Merlin's species.py)."""
    human_count = mouse_count = 0
    for g in gene_names:
        s = str(g).strip()
        if s.upper().startswith("ENSG"):
            human_count += 1
        elif s.upper().startswith("ENSMUSG"):
            mouse_count += 1
        elif len(s) > 1 and s[0].isupper() and s[1:] != s[1:].upper():
            mouse_count += 1
    if human_count > 0 and mouse_count == 0:
        return "human"
    if mouse_count > 0 and human_count == 0:
        return "mouse"
    if human_count > 0 and mouse_count > 0:
        raise ValueError(
            f"Mixed species signals: {human_count} human-like, {mouse_count} mouse-like. "
            "Declare species explicitly."
        )
    return "unknown"


def validate_species_boundary(declared: Species, gene_names: list[str]) -> None:
    """Enforce the species firewall: declared species must match gene-name inference."""
    if declared in ("in_silico", "not_applicable", "unknown"):
        return
    inferred = infer_species(gene_names)
    if inferred == "unknown":
        return
    if inferred != declared:
        raise ValueError(
            f"Species firewall violation: declared={declared}, "
            f"inferred={inferred} from gene names"
        )


class MerlinStudyRecords:
    """Read-only access to Merlin's content-addressed study records."""

    def __init__(self, records_dir: Path | None = None):
        self._dir = records_dir or MERLIN_RECORDS

    def list_records(self) -> list[str]:
        if not self._dir.exists():
            return []
        return [f.stem for f in self._dir.glob("*.json")]

    def read_record(self, record_id: str) -> dict[str, Any] | None:
        path = self._dir / f"{record_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def verify_record(self, record_id: str) -> bool:
        """Verify a record's content hash matches its stored hash."""
        record = self.read_record(record_id)
        if record is None or "content_hash" not in record:
            return False
        stored_hash = record.pop("content_hash")
        computed = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        record["content_hash"] = stored_hash
        return computed == stored_hash


def format_study_record(
    study_id: str,
    title: str,
    question: str,
    data_kind: str,
    species: Species,
    regime: str,
    arms: dict[str, dict[str, float]],
    criteria_results: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Create a content-addressed study record in Merlin's format."""
    record = {
        "study_id": study_id,
        "title": title,
        "question": question,
        "data_kind": data_kind,
        "species": species,
        "regime": regime,
        "arms": arms,
        "criteria_results": criteria_results,
        "parameters": parameters,
        "repo": "yav3/NeuroTunesAI",
    }
    content = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["content_hash"] = hashlib.sha256(content.encode()).hexdigest()
    return record
