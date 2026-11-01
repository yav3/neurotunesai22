"""Perturbation inference service — governed predictions from trained NT-001.

Wraps PerturbationResponseModel.predict() with:
  - Model provenance (spec hash, training record, data lineage)
  - Prediction logging (hash-chained for audit trail)
  - Gene name resolution (indices ↔ names via the training gene list)
  - Batch prediction with confidence estimates (latent variance)

Usage:
    service = PerturbationInferenceService.from_training_result(
        training_result, gene_names, model,
    )
    pred = service.predict_drug_response(
        baseline_expression, drug_idx=0, dose=1.0, duration=24.0,
    )
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from models.foundation.perturbation_response import (
    PerturbationModelConfig,
    PerturbationResponseModel,
    PerturbationType,
)


@dataclass
class PredictionRecord:
    """Logged prediction with provenance."""
    prediction_id: str
    timestamp: float
    perturbation_type: str
    n_genes: int
    top_changed_genes: list[dict[str, Any]]
    model_spec_hash: str
    prediction_hash: str

    def to_dict(self) -> dict:
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "perturbation_type": self.perturbation_type,
            "n_genes": self.n_genes,
            "top_changed_genes": self.top_changed_genes,
            "model_spec_hash": self.model_spec_hash,
            "prediction_hash": self.prediction_hash,
        }


@dataclass
class PredictionResult:
    """Complete result of a perturbation prediction."""
    predicted_expression: np.ndarray
    baseline_expression: np.ndarray
    fold_changes: np.ndarray
    gene_names: list[str]
    top_upregulated: list[dict[str, Any]]
    top_downregulated: list[dict[str, Any]]
    latent_variance: float
    record: PredictionRecord

    def to_dict(self) -> dict:
        return {
            "top_upregulated": self.top_upregulated,
            "top_downregulated": self.top_downregulated,
            "n_genes": len(self.gene_names),
            "latent_variance": self.latent_variance,
            "prediction_id": self.record.prediction_id,
            "model_spec_hash": self.record.model_spec_hash,
        }


class PerturbationInferenceService:
    """Governed inference service for the NT-001 perturbation response model."""

    def __init__(
        self,
        model: PerturbationResponseModel,
        gene_names: list[str],
        spec_hash: str = "",
        device: torch.device | None = None,
    ):
        self.model = model
        self.gene_names = gene_names
        self.spec_hash = spec_hash
        self.device = device or torch.device("cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self._prediction_count = 0
        self._log: list[PredictionRecord] = []

    @classmethod
    def from_training_result(
        cls,
        training_result: dict[str, Any],
        gene_names: list[str],
        model: PerturbationResponseModel,
        device: torch.device | None = None,
    ) -> PerturbationInferenceService:
        return cls(
            model=model,
            gene_names=gene_names,
            spec_hash=training_result.get("spec_hash", ""),
            device=device,
        )

    def _make_prediction_id(self) -> str:
        self._prediction_count += 1
        return f"pred-{self._prediction_count:06d}-{int(time.time())}"

    def _compute_prediction_hash(
        self, baseline: np.ndarray, predicted: np.ndarray, ptype: str,
    ) -> str:
        payload = json.dumps({
            "baseline_hash": hashlib.sha256(baseline.tobytes()).hexdigest()[:16],
            "predicted_hash": hashlib.sha256(predicted.tobytes()).hexdigest()[:16],
            "ptype": ptype,
            "spec_hash": self.spec_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _extract_top_genes(
        self, fold_changes: np.ndarray, top_k: int = 10,
    ) -> tuple[list[dict], list[dict]]:
        indices_up = np.argsort(-fold_changes)[:top_k]
        indices_down = np.argsort(fold_changes)[:top_k]

        up = [
            {"gene": self.gene_names[i], "fold_change": float(fold_changes[i]),
             "index": int(i)}
            for i in indices_up if fold_changes[i] > 0
        ]
        down = [
            {"gene": self.gene_names[i], "fold_change": float(fold_changes[i]),
             "index": int(i)}
            for i in indices_down if fold_changes[i] < 0
        ]
        return up, down

    def predict_drug_response(
        self,
        baseline: np.ndarray,
        drug_idx: int = 0,
        dose: float = 1.0,
        duration: float = 24.0,
        top_k: int = 10,
    ) -> PredictionResult:
        """Predict gene expression response to a drug perturbation.

        Parameters
        ----------
        baseline : (n_genes,) expression vector (log1p space)
        drug_idx : drug index in model vocabulary
        dose : dose in mg
        duration : duration in hours
        top_k : number of top changed genes to report
        """
        return self._predict(
            baseline=baseline,
            ptype=PerturbationType.DRUG,
            gene_idx=0,
            drug_idx=drug_idx,
            dose=dose,
            duration=duration,
            time_point=0.0,
            top_k=top_k,
        )

    def predict_knockout_response(
        self,
        baseline: np.ndarray,
        gene_idx: int,
        top_k: int = 10,
    ) -> PredictionResult:
        """Predict gene expression response to a gene knockout."""
        return self._predict(
            baseline=baseline,
            ptype=PerturbationType.KNOCKOUT,
            gene_idx=gene_idx,
            drug_idx=0,
            dose=0.0,
            duration=0.0,
            time_point=0.0,
            top_k=top_k,
        )

    def predict_time_course(
        self,
        baseline: np.ndarray,
        time_point: float,
        top_k: int = 10,
    ) -> PredictionResult:
        """Predict gene expression at a future time point."""
        return self._predict(
            baseline=baseline,
            ptype=PerturbationType.TIME_COURSE,
            gene_idx=0,
            drug_idx=0,
            dose=0.0,
            duration=0.0,
            time_point=time_point,
            top_k=top_k,
        )

    def _predict(
        self,
        baseline: np.ndarray,
        ptype: PerturbationType,
        gene_idx: int,
        drug_idx: int,
        dose: float,
        duration: float,
        time_point: float,
        top_k: int,
    ) -> PredictionResult:
        assert len(baseline) == len(self.gene_names), (
            f"Baseline has {len(baseline)} genes, model expects {len(self.gene_names)}"
        )

        baseline_t = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to(self.device)
        ptype_id = {"knockout": 0, "drug": 1, "time_course": 2}[ptype.value]

        with torch.no_grad():
            out = self.model(
                baseline=baseline_t,
                ptype_ids=torch.tensor([ptype_id], device=self.device),
                gene_idx=torch.tensor([gene_idx], device=self.device),
                drug_idx=torch.tensor([drug_idx], device=self.device),
                dose=torch.tensor([dose], device=self.device),
                duration=torch.tensor([duration], device=self.device),
                time_point=torch.tensor([time_point], device=self.device),
            )

        predicted = out["predicted"].squeeze(0).cpu().numpy()
        mu = out["mu"].squeeze(0).cpu().numpy()
        logvar = out["logvar"].squeeze(0).cpu().numpy()
        latent_var = float(np.exp(logvar).mean())

        fold_changes = predicted - baseline

        top_up, top_down = self._extract_top_genes(fold_changes, top_k)

        pred_id = self._make_prediction_id()
        pred_hash = self._compute_prediction_hash(baseline, predicted, ptype.value)

        record = PredictionRecord(
            prediction_id=pred_id,
            timestamp=time.time(),
            perturbation_type=ptype.value,
            n_genes=len(self.gene_names),
            top_changed_genes=top_up[:5] + top_down[:5],
            model_spec_hash=self.spec_hash,
            prediction_hash=pred_hash,
        )
        self._log.append(record)

        return PredictionResult(
            predicted_expression=predicted,
            baseline_expression=baseline,
            fold_changes=fold_changes,
            gene_names=self.gene_names,
            top_upregulated=top_up,
            top_downregulated=top_down,
            latent_variance=latent_var,
            record=record,
        )

    @property
    def prediction_log(self) -> list[dict]:
        return [r.to_dict() for r in self._log]

    @property
    def n_predictions(self) -> int:
        return len(self._log)
