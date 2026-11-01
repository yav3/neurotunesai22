"""Training bridge — connects Merlin's curated GEO data to the governed harness.

Reads committed expression CSVs from merlinaineuralplus2/backend/data/expression/,
constructs PerturbationDatasets, and runs the GovernedTrainer with pre-registered
criteria from NT-001.

Usage:
    from models.bridge.training_bridge import MerlinTrainingBridge
    bridge = MerlinTrainingBridge(merlin_root="/path/to/merlinaineuralplus2")
    datasets = bridge.load_human_perturbation_datasets()
    record = bridge.run_governed_training(datasets, study_id="NT-001")
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.foundation.perturbation_response import (
    GEOExpressionData,
    PerturbationDataset,
    PerturbationModelConfig,
    PerturbationResponseModel,
    PerturbationType,
)
from models.training.governed_harness import (
    CheckpointStore,
    GovernedTrainer,
    TrainingCriterion,
    TrainingRecord,
    TrainingSpec,
    infer_species_from_genes,
    select_training_regime,
)


@dataclass(frozen=True)
class MerlinDatasetRef:
    """Reference to a committed dataset in Merlin's data registry."""
    registry_id: str
    path: str
    species: str
    accession: str
    kind: str = "expression"
    sha256: str = ""


def _parse_merlin_csv(csv_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Parse Merlin's gene×sample CSV format.

    Returns (gene_names, sample_names, expression[n_samples, n_genes]).
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        sample_names = header[1:]
        gene_names = []
        rows = []
        for row in reader:
            gene_names.append(row[0])
            rows.append([float(x) for x in row[1:]])
    expr = np.array(rows, dtype=np.float32).T
    return gene_names, sample_names, expr


def _infer_conditions(sample_names: list[str]) -> list[str]:
    """Infer control/treated conditions from Merlin's sample naming convention.

    Merlin's GEO fixtures use patterns like:
      hESCs_control_rep1, hESCs_2days_rapamycin_rep1
      IMR90_GFP_rep1, IMR90_OSK_rep1
    """
    conditions = []
    control_keywords = {"control", "vehicle", "dmso", "untreated", "baseline", "gfp", "mock"}
    for name in sample_names:
        parts = name.lower().replace("-", "_").split("_")
        if any(kw in parts for kw in control_keywords):
            conditions.append("control")
        else:
            conditions.append("treated")
    return conditions


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class MerlinTrainingBridge:
    """Loads Merlin's committed GEO data and feeds the governed training harness."""

    HUMAN_DATA_DIR = "backend/data/expression/human"

    def __init__(self, merlin_root: str | Path) -> None:
        self.merlin_root = Path(merlin_root)
        self._data_dir = self.merlin_root / self.HUMAN_DATA_DIR
        if not self._data_dir.exists():
            raise FileNotFoundError(
                f"Merlin data directory not found: {self._data_dir}. "
                f"Clone merlinaineuralplus2 first."
            )

    def list_datasets(self) -> list[MerlinDatasetRef]:
        """List all committed human expression datasets."""
        refs = []
        for csv_path in sorted(self._data_dir.glob("*.csv")):
            acc = csv_path.stem.split("_")[0].upper()
            refs.append(MerlinDatasetRef(
                registry_id=f"expression:human:{csv_path.stem}",
                path=str(csv_path.relative_to(self.merlin_root)),
                species="human",
                accession=acc,
                sha256=_sha256_file(csv_path),
            ))
        return refs

    def load_dataset(self, ref: MerlinDatasetRef) -> GEOExpressionData:
        """Load a single dataset into NeuroTunesAI's GEOExpressionData format.

        Species comes from the registry declaration, not gene-name inference.
        Per Merlin's species.py: upper-case symbols do not decide species.
        Per evidence discipline: declare or refuse, never default.
        If gene names *can* be inferred and contradict the registry, we raise.
        """
        csv_path = self.merlin_root / ref.path
        gene_names, sample_names, expression = _parse_merlin_csv(csv_path)

        try:
            inferred = infer_species_from_genes(gene_names[:20])
            if inferred != ref.species:
                raise ValueError(
                    f"Species mismatch for {ref.accession}: registry declares "
                    f"'{ref.species}', gene names infer '{inferred}'"
                )
        except ValueError as e:
            if "Cannot infer" in str(e):
                pass  # ambiguous symbols — trust the registry declaration
            else:
                raise

        conditions = _infer_conditions(sample_names)

        return GEOExpressionData(
            expression=expression,
            conditions=conditions,
            gene_names=gene_names,
            species=ref.species,
            accession=ref.accession,
        )

    def load_human_perturbation_datasets(
        self,
        perturbation_type: PerturbationType = PerturbationType.DRUG,
    ) -> list[tuple[MerlinDatasetRef, PerturbationDataset]]:
        """Load all human datasets as PerturbationDatasets.

        Filters out datasets with no control samples (species firewall
        requires declared species; regime requires ≥2 conditions).
        """
        results = []
        for ref in self.list_datasets():
            data = self.load_dataset(ref)
            ds = PerturbationDataset(data, perturbation_type=perturbation_type)
            if len(ds) == 0:
                continue
            results.append((ref, ds))
        return results

    def build_training_spec(
        self,
        dataset_refs: list[MerlinDatasetRef],
        n_genes: int,
        study_id: str = "NT-001",
        seed: int = 2026,
    ) -> TrainingSpec:
        """Build a governed TrainingSpec from loaded datasets."""
        n_total = sum(1 for r in dataset_refs for _ in [1])
        regime = select_training_regime(n_samples=n_total * 30)

        return TrainingSpec(
            model_id="PerturbationResponseModel",
            dataset_id="|".join(r.accession for r in dataset_refs),
            seed=seed,
            held_out_fraction=0.2,
            species="human",
            regime=regime,
            criteria=(
                TrainingCriterion(
                    metric="mse", op="<", versus="mean_baseline",
                    description="Model MSE < mean-baseline MSE on held-out test",
                ),
                TrainingCriterion(
                    metric="mse", op="<", versus="linear_shift",
                    description="Model MSE < linear-shift MSE on held-out test",
                ),
                TrainingCriterion(
                    metric="pearson_r", op=">", threshold=0.5,
                    description="Mean per-gene Pearson r > 0.5 on held-out test",
                ),
                TrainingCriterion(
                    metric="pearson_r", op=">", versus="mean_baseline",
                    description="Model Pearson r > mean-baseline Pearson r",
                ),
            ),
            lr=1e-4,
            max_epochs=100,
            batch_size=64,
            patience=10,
            data_kind="perturbation_expression",
            description=f"NT-001: cVAE on {len(dataset_refs)} GEO series, {n_genes} genes",
        )

    def build_model_factory(self, n_genes: int) -> callable:
        """Return a factory function for PerturbationResponseModel."""
        def factory():
            config = PerturbationModelConfig(n_genes=n_genes)
            return PerturbationResponseModel(config)
        return factory

    def provenance_manifest(
        self,
        dataset_refs: list[MerlinDatasetRef],
        spec: TrainingSpec,
    ) -> dict[str, Any]:
        """Build a provenance manifest for the training run."""
        return {
            "study_id": "NT-001",
            "spec_hash": spec.content_hash(),
            "merlin_repo": str(self.merlin_root),
            "datasets": [
                {
                    "registry_id": r.registry_id,
                    "accession": r.accession,
                    "species": r.species,
                    "sha256": r.sha256,
                    "path": r.path,
                }
                for r in dataset_refs
            ],
            "cross_repo_contract": {
                "species_firewall": "models.training.governed_harness.infer_species_from_genes",
                "regime_selection": "models.training.governed_harness.select_training_regime",
                "data_format": "merlinaineuralplus2/backend/data/expression/human/*.csv",
                "evidence_pattern": "merlinaineuralplus2/backend/app/inventions/harness.py",
            },
        }
