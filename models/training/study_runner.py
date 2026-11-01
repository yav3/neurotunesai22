"""Pre-registered study runner — governed training with evidence records.

Connects NeuroTunesAI's StudyRegistry + LineageChain to the GEO training
pipeline, producing hash-chained, tamper-evident training records that
satisfy the evidence discipline:

1. Pre-register the study (criterion, regime, data source) before running
2. Discover and validate available datasets against declared regime
3. Execute governed training with deterministic seed
4. Record verdict immutably (met/not met) with full provenance
5. Chain the result into the lineage for replay verification

Usage:
    runner = StudyRunner(merlin_data_dir="/path/to/merlin/data/expression/human")
    result = runner.run_study(
        study_id="NT-GEO-001",
        title="NT-001 cVAE on GSE119262 (data-driven regime)",
        criterion="val_loss < 2.0",
        accessions=["GSE119262"],
    )
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from models.data.geo_loader import discover_merlin_datasets, load_geo_csv, content_hash
from models.evidence.lineage import LineageChain
from models.evidence.registry import StudyRegistry, StudyRecord
from models.training.geo_training_pipeline import (
    GEOTrainingConfig,
    build_training_spec,
    prepare_datasets,
    run_governed_training,
)
from models.training.governed_harness import select_training_regime


@dataclass
class DataSufficiency:
    """Result of checking whether available data meets regime requirements."""
    sufficient: bool
    regime: str
    total_samples: int
    total_control: int
    total_treated: int
    datasets_found: int
    datasets_qualified: int
    accessions: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "regime": self.regime,
            "total_samples": self.total_samples,
            "total_control": self.total_control,
            "total_treated": self.total_treated,
            "datasets_found": self.datasets_found,
            "datasets_qualified": self.datasets_qualified,
            "accessions": self.accessions,
            "reason": self.reason,
        }


def check_data_sufficiency(
    merlin_data_dir: str | Path,
    species: str = "human",
    min_samples: int = 10,
    accessions: list[str] | None = None,
) -> DataSufficiency:
    """Check whether available Merlin datasets are sufficient for governed training.

    Parameters
    ----------
    merlin_data_dir : path to Merlin's expression data directory
    species : species filter (enforced; no cross-species)
    min_samples : minimum samples per dataset
    accessions : if given, restrict to these GEO accessions only
    """
    all_datasets = discover_merlin_datasets(merlin_data_dir, species)
    if not all_datasets:
        return DataSufficiency(
            sufficient=False, regime="unknown", total_samples=0,
            total_control=0, total_treated=0, datasets_found=0,
            datasets_qualified=0, reason="No datasets found in Merlin data directory",
        )

    if accessions:
        all_datasets = [d for d in all_datasets if d["accession"] in accessions]

    qualified = [
        d for d in all_datasets
        if d["n_samples"] >= min_samples and d["n_control"] > 0 and d["n_treated"] > 0
    ]

    total_samples = sum(d["n_samples"] for d in qualified)
    total_control = sum(d["n_control"] for d in qualified)
    total_treated = sum(d["n_treated"] for d in qualified)
    n_conditions = 2 if (total_control > 0 and total_treated > 0) else 1

    regime = select_training_regime(
        n_samples=total_samples, n_conditions=n_conditions,
    ) if qualified else "unknown"

    sufficient = len(qualified) > 0 and total_control > 0 and total_treated > 0
    reason = ""
    if not qualified:
        reason = f"No datasets meet min_samples={min_samples} with control+treated pairs"
    elif total_control == 0:
        reason = "No control samples across qualified datasets"
    elif total_treated == 0:
        reason = "No treated samples across qualified datasets"

    return DataSufficiency(
        sufficient=sufficient,
        regime=regime,
        total_samples=total_samples,
        total_control=total_control,
        total_treated=total_treated,
        datasets_found=len(all_datasets),
        datasets_qualified=len(qualified),
        accessions=[d["accession"] for d in qualified],
        reason=reason,
    )


@dataclass
class StudyResult:
    """Complete result of a governed study run."""
    study_id: str
    verdict: str
    training_result: dict[str, Any]
    data_sufficiency: DataSufficiency
    provenance: dict[str, Any]
    lineage_hash: str = ""
    study_record: StudyRecord | None = None

    def to_dict(self) -> dict:
        return {
            "study_id": self.study_id,
            "verdict": self.verdict,
            "training_result": self.training_result,
            "data_sufficiency": self.data_sufficiency.to_dict(),
            "provenance": self.provenance,
            "lineage_hash": self.lineage_hash,
        }


class StudyRunner:
    """Governed study runner — pre-register, validate, train, record.

    Also exported as GovernedStudyRunner for cross-repo manifest compatibility.
    """

    def __init__(
        self,
        merlin_data_dir: str | Path | None = None,
        records_dir: Path | None = None,
    ):
        self.merlin_data_dir = Path(merlin_data_dir) if merlin_data_dir else None
        self.registry = StudyRegistry(records_dir)
        self.lineage = LineageChain()

    def check_sufficiency(
        self,
        species: str = "human",
        min_samples: int = 10,
        accessions: list[str] | None = None,
    ) -> DataSufficiency:
        if self.merlin_data_dir is None:
            return DataSufficiency(
                sufficient=False, regime="unknown", total_samples=0,
                total_control=0, total_treated=0, datasets_found=0,
                datasets_qualified=0, reason="No Merlin data directory configured",
            )
        return check_data_sufficiency(
            self.merlin_data_dir, species, min_samples, accessions,
        )

    def run_study(
        self,
        study_id: str,
        title: str,
        criterion: str = "val_loss < 2.0",
        accessions: list[str] | None = None,
        species: str = "human",
        epochs: int = 50,
        batch_size: int = 16,
        latent_dim: int = 64,
        seed: int = 2026,
        min_samples: int = 10,
    ) -> StudyResult:
        """Execute a complete governed study run.

        Steps:
        1. Pre-register the study in the registry
        2. Validate data sufficiency
        3. Run governed training
        4. Record verdict immutably
        5. Chain into lineage
        """
        # 1. Pre-register
        study = self.registry.register(
            study_id=study_id,
            title=title,
            criterion=criterion,
            data_kind="geo_expression",
            synthetic=False,
        )

        # 2. Validate sufficiency
        sufficiency = self.check_sufficiency(species, min_samples, accessions)
        if not sufficiency.sufficient:
            self.registry.record_verdict(
                study_id=study_id,
                measured={"data_sufficiency": sufficiency.to_dict()},
                summary=f"Data insufficient: {sufficiency.reason}",
                verdict="not_met",
                verdict_note="Pre-run data validation failed",
            )
            return StudyResult(
                study_id=study_id,
                verdict="not_met",
                training_result={},
                data_sufficiency=sufficiency,
                provenance={"stage": "data_validation", "reason": sufficiency.reason},
                study_record=study,
            )

        # 3. Run training
        config = GEOTrainingConfig(
            merlin_data_dir=str(self.merlin_data_dir),
            species=species,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            min_samples=min_samples,
        )
        training_result = run_governed_training(config)

        if "error" in training_result:
            self.registry.record_verdict(
                study_id=study_id,
                measured={"error": training_result["error"]},
                summary=f"Training failed: {training_result['error']}",
                verdict="not_met",
                verdict_note="Training execution error",
            )
            return StudyResult(
                study_id=study_id,
                verdict="not_met",
                training_result=training_result,
                data_sufficiency=sufficiency,
                provenance={"stage": "training", "error": training_result["error"]},
                study_record=study,
            )

        # 4. Evaluate criterion and record verdict
        val_loss = training_result.get("final_val_loss")
        criterion_met = training_result.get("verdict") == "met"

        measured = {
            "final_val_loss": val_loss,
            "final_train_loss": training_result.get("final_train_loss"),
            "n_train": training_result.get("n_train"),
            "n_val": training_result.get("n_val"),
            "n_test": training_result.get("n_test"),
            "n_genes": training_result.get("n_genes"),
            "regime": training_result.get("spec", {}).get("regime"),
            "criteria_results": training_result.get("criteria_results"),
            "test_metrics": training_result.get("test_metrics"),
        }

        verdict = "met" if criterion_met else "not_met"
        summary = (
            f"val_loss={val_loss:.4f}, regime={sufficiency.regime}, "
            f"n_train={training_result.get('n_train')}"
        ) if val_loss is not None else "Training produced no val_loss"

        self.registry.record_verdict(
            study_id=study_id,
            measured=measured,
            summary=summary,
            verdict=verdict,
        )

        # 5. Chain into lineage
        provenance = {
            "spec_hash": training_result.get("spec_hash"),
            "data_hashes": {
                d["accession"]: d["content_hash"]
                for d in discover_merlin_datasets(self.merlin_data_dir, species)
                if accessions is None or d["accession"] in accessions
            },
            "config": {
                "seed": seed, "epochs": epochs, "batch_size": batch_size,
                "latent_dim": latent_dim, "species": species,
            },
            "sufficiency": sufficiency.to_dict(),
        }

        record = self.lineage.append(
            record_id=study_id,
            problem_spec={
                "criterion": criterion,
                "regime": sufficiency.regime,
                "accessions": sufficiency.accessions,
            },
            answer_values={
                "verdict": verdict,
                "val_loss": float(val_loss) if val_loss is not None else None,
                "spec_hash": training_result.get("spec_hash"),
            },
        )

        return StudyResult(
            study_id=study_id,
            verdict=verdict,
            training_result=training_result,
            data_sufficiency=sufficiency,
            provenance=provenance,
            lineage_hash=record.record_hash,
            study_record=study,
        )
GovernedStudyRunner = StudyRunner
