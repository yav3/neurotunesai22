"""Tests for perturbation inference service.

Validates:
  - Drug, knockout, and time-course prediction paths
  - Gene name resolution (top up/downregulated)
  - Prediction logging and provenance hashing
  - Latent variance as a confidence proxy
  - End-to-end: train on real Merlin data → predict
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from models.foundation.perturbation_response import (
    PerturbationModelConfig,
    PerturbationResponseModel,
)
from models.inference.perturbation_inference import (
    PerturbationInferenceService,
    PredictionResult,
)

MERLIN_DATA_DIR = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")


# ── Synthetic model tests ─────────────────────────────────────────────

def _make_service(n_genes: int = 20) -> PerturbationInferenceService:
    config = PerturbationModelConfig(n_genes=n_genes, latent_dim=8, n_drugs=5)
    model = PerturbationResponseModel(config)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return PerturbationInferenceService(
        model=model, gene_names=gene_names, spec_hash="test-hash",
    )


class TestDrugPrediction:

    def test_basic_prediction(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        result = service.predict_drug_response(baseline, drug_idx=0, dose=1.0)
        assert isinstance(result, PredictionResult)
        assert result.predicted_expression.shape == (20,)
        assert result.fold_changes.shape == (20,)
        assert len(result.gene_names) == 20

    def test_top_genes_reported(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        result = service.predict_drug_response(baseline, top_k=5)
        assert len(result.top_upregulated) <= 5
        assert len(result.top_downregulated) <= 5
        for gene_info in result.top_upregulated:
            assert "gene" in gene_info
            assert "fold_change" in gene_info
            assert gene_info["fold_change"] > 0

    def test_prediction_logged(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        service.predict_drug_response(baseline)
        assert service.n_predictions == 1
        log = service.prediction_log
        assert len(log) == 1
        assert log[0]["perturbation_type"] == "drug"
        assert log[0]["model_spec_hash"] == "test-hash"

    def test_prediction_hash_deterministic(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        r1 = service.predict_drug_response(baseline, drug_idx=0, dose=1.0)
        # same inputs should produce same prediction hash
        r2 = service.predict_drug_response(baseline, drug_idx=0, dose=1.0)
        assert r1.record.prediction_hash == r2.record.prediction_hash

    def test_different_doses_different_predictions(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        r1 = service.predict_drug_response(baseline, dose=0.1)
        r2 = service.predict_drug_response(baseline, dose=10.0)
        # different doses should produce different predictions
        assert not np.allclose(r1.predicted_expression, r2.predicted_expression)


class TestKnockoutPrediction:

    def test_knockout_prediction(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        result = service.predict_knockout_response(baseline, gene_idx=5)
        assert result.record.perturbation_type == "knockout"
        assert result.predicted_expression.shape == (20,)

    def test_different_genes_different_predictions(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        r1 = service.predict_knockout_response(baseline, gene_idx=0)
        r2 = service.predict_knockout_response(baseline, gene_idx=10)
        assert not np.allclose(r1.predicted_expression, r2.predicted_expression)


class TestTimeCoursePrediction:

    def test_time_course_prediction(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        result = service.predict_time_course(baseline, time_point=24.0)
        assert result.record.perturbation_type == "time_course"

    def test_different_times_different_predictions(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        r1 = service.predict_time_course(baseline, time_point=0.0)
        r2 = service.predict_time_course(baseline, time_point=100.0)
        assert not np.allclose(r1.predicted_expression, r2.predicted_expression)


class TestLatentVariance:

    def test_variance_is_finite(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        result = service.predict_drug_response(baseline)
        assert np.isfinite(result.latent_variance)
        assert result.latent_variance >= 0


class TestResultSerialization:

    def test_to_dict(self):
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        result = service.predict_drug_response(baseline)
        d = result.to_dict()
        assert "top_upregulated" in d
        assert "top_downregulated" in d
        assert "latent_variance" in d
        assert "prediction_id" in d
        assert "model_spec_hash" in d

    def test_json_serializable(self):
        import json
        service = _make_service()
        baseline = np.random.randn(20).astype(np.float32)
        result = service.predict_drug_response(baseline)
        serialized = json.dumps(result.to_dict(), default=str)
        assert len(serialized) > 50

    def test_gene_dimension_mismatch_raises(self):
        service = _make_service(n_genes=20)
        wrong_baseline = np.random.randn(10).astype(np.float32)
        with pytest.raises(AssertionError, match="Baseline has 10 genes"):
            service.predict_drug_response(wrong_baseline)


# ── End-to-end with real Merlin data ──────────────────────────────────

@pytest.mark.skipif(not MERLIN_DATA_DIR.exists(), reason="Merlin data not available")
class TestEndToEndInference:

    def test_train_then_predict_gse107965(self):
        """Train on real GSE107965, then run inference."""
        from models.data.geo_loader import load_geo_csv
        from models.training.geo_training_pipeline import (
            GEOTrainingConfig,
            run_governed_training,
        )

        config = GEOTrainingConfig(
            merlin_data_dir=str(MERLIN_DATA_DIR),
            epochs=3,
            batch_size=4,
            latent_dim=16,
            min_samples=5,
        )
        result = run_governed_training(config)
        assert "error" not in result

        # load the data to get gene names and a baseline
        data = load_geo_csv(
            MERLIN_DATA_DIR / "gse107965_autophagy_subset.csv",
            accession="GSE107965",
        )

        n_genes = data.expression.shape[1]
        model_config = PerturbationModelConfig(
            n_genes=n_genes, latent_dim=16, n_drugs=10,
        )
        model = PerturbationResponseModel(model_config)

        service = PerturbationInferenceService.from_training_result(
            training_result=result,
            gene_names=data.gene_names,
            model=model,
        )

        # use the first control sample as baseline
        control_idx = [i for i, c in enumerate(data.conditions) if c == "control"]
        baseline = np.log1p(data.expression[control_idx[0]])

        pred = service.predict_drug_response(
            baseline, drug_idx=0, dose=1.0, duration=24.0,
        )
        assert pred.predicted_expression.shape == (n_genes,)
        assert len(pred.gene_names) == n_genes
        assert service.n_predictions == 1
        assert pred.record.model_spec_hash == result["spec_hash"]

    def test_knockout_on_real_data(self):
        """Predict knockout effect on a real baseline."""
        from models.data.geo_loader import load_geo_csv

        data = load_geo_csv(
            MERLIN_DATA_DIR / "gse107965_autophagy_subset.csv",
            accession="GSE107965",
        )

        n_genes = data.expression.shape[1]
        config = PerturbationModelConfig(
            n_genes=n_genes, latent_dim=16, n_drugs=10,
        )
        model = PerturbationResponseModel(config)

        service = PerturbationInferenceService(
            model=model, gene_names=data.gene_names, spec_hash="test",
        )

        baseline = np.log1p(data.expression[0])
        result = service.predict_knockout_response(baseline, gene_idx=5)
        assert result.predicted_expression.shape == (n_genes,)
        assert result.record.perturbation_type == "knockout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
