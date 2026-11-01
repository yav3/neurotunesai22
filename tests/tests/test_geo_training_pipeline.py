"""Tests for GEO data loading and governed training pipeline.

Validates:
  - CSV loading from Merlin's committed expression data
  - Condition inference from column naming conventions
  - Species firewall enforcement
  - PerturbationDataset construction from real GEO data
  - Training pipeline end-to-end on real committed data
  - Cross-repo dataset discovery
"""
from __future__ import annotations

import os
import sys
import csv
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from models.data.geo_loader import (
    load_geo_csv,
    infer_condition,
    content_hash,
    discover_merlin_datasets,
)
from models.foundation.perturbation_response import (
    PerturbationDataset,
    PerturbationType,
)


# ── Condition inference ──────────────────────────────────────────────

def test_infer_condition_rapamycin():
    assert infer_condition("hESCs_2days_rapamycin_rep1") == "treated"

def test_infer_condition_control():
    assert infer_condition("hESCs_control_rep1") == "control"

def test_infer_condition_osk():
    assert infer_condition("D20_P16OSK_R1") == "treated"

def test_infer_condition_gfp():
    assert infer_condition("D20_GFP_R1") == "control"

def test_infer_condition_pre_post():
    assert infer_condition("Patient1.Pre") == "control"
    assert infer_condition("Patient3.Post") == "treated"

def test_infer_condition_unknown():
    assert infer_condition("SRR12345") == "unknown"


# ── CSV loading ──────────────────────────────────────────────────────

def _write_test_csv(tmpdir: str, n_genes: int = 10, n_control: int = 3, n_treated: int = 5) -> str:
    path = os.path.join(tmpdir, "test_expression.csv")
    rng = np.random.RandomState(42)
    sample_names = (
        [f"control_rep{i+1}" for i in range(n_control)]
        + [f"treated_rep{i+1}" for i in range(n_treated)]
    )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene"] + sample_names)
        for g in range(n_genes):
            vals = rng.randint(0, 5000, size=n_control + n_treated).tolist()
            writer.writerow([f"GENE{g}"] + vals)
    return path


def test_load_geo_csv_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_test_csv(tmpdir, n_genes=20, n_control=3, n_treated=5)
        data = load_geo_csv(path, accession="GSE000001")
        assert data.expression.shape == (8, 20)  # 8 samples, 20 genes
        assert len(data.gene_names) == 20
        assert data.accession == "GSE000001"
        assert data.species == "human"


def test_load_geo_csv_conditions():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_test_csv(tmpdir, n_genes=5, n_control=2, n_treated=3)
        data = load_geo_csv(path)
        assert sum(1 for c in data.conditions if c == "control") == 2
        assert sum(1 for c in data.conditions if c == "treated") == 3


def test_load_geo_csv_log1p():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_test_csv(tmpdir, n_genes=5, n_control=2, n_treated=2)
        data_raw = load_geo_csv(path, log1p=False)
        data_log = load_geo_csv(path, log1p=True)
        # log1p values should be smaller
        assert data_log.expression.max() < data_raw.expression.max()
        # log1p(0) = 0, log1p(x) > 0 for x > 0
        assert np.all(data_log.expression >= 0)


def test_load_geo_csv_accession_inference():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "gse107965_subset.csv")
        rng = np.random.RandomState(0)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gene", "control_rep1", "treated_rep1"])
            writer.writerow(["TP53"] + rng.randint(0, 100, 2).tolist())
        data = load_geo_csv(path)
        assert data.accession == "GSE107965"


def test_content_hash_deterministic():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_test_csv(tmpdir, n_genes=3, n_control=1, n_treated=1)
        h1 = content_hash(path)
        h2 = content_hash(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


# ── Real Merlin data loading ────────────────────────────────────────

MERLIN_DATA_DIR = Path("/home/user/merlinaineuralplus2/backend/data/expression/human")


@pytest.mark.skipif(
    not MERLIN_DATA_DIR.exists(),
    reason="Merlin data directory not available",
)
class TestRealMerlinData:

    def test_load_gse107965(self):
        path = MERLIN_DATA_DIR / "gse107965_autophagy_subset.csv"
        data = load_geo_csv(path, accession="GSE107965")
        assert data.expression.shape[0] == 18  # 18 samples
        assert data.accession == "GSE107965"
        n_control = sum(1 for c in data.conditions if c == "control")
        n_treated = sum(1 for c in data.conditions if c == "treated")
        assert n_control == 6  # 3 hESC + 3 fibroblast control
        assert n_treated == 12  # 6 hESC + 6 fibroblast rapamycin

    def test_load_gse201710(self):
        path = MERLIN_DATA_DIR / "gse201710_imr90_osk_subset.csv"
        data = load_geo_csv(path, accession="GSE201710")
        assert data.expression.shape[0] > 0
        conditions = set(data.conditions)
        assert "control" in conditions  # GFP
        assert "treated" in conditions  # OSK

    def test_load_gse119262_full(self):
        path = MERLIN_DATA_DIR / "gse119262_autophagy_full.csv"
        data = load_geo_csv(path, accession="GSE119262")
        # 44 samples (23 patients × pre/post, minus 2 missing)
        assert data.expression.shape[0] >= 30  # in-regime for data-driven
        assert data.expression.shape[1] > 1000  # full transcriptome

    def test_perturbation_dataset_from_real_data(self):
        path = MERLIN_DATA_DIR / "gse107965_autophagy_subset.csv"
        data = load_geo_csv(path, accession="GSE107965")
        ds = PerturbationDataset(data, perturbation_type=PerturbationType.DRUG)
        assert len(ds) > 0
        item = ds[0]
        assert "baseline" in item
        assert "target" in item
        assert item["baseline"].shape == item["target"].shape

    def test_discover_merlin_datasets(self):
        datasets = discover_merlin_datasets(MERLIN_DATA_DIR)
        assert len(datasets) >= 8  # 8 committed CSVs
        for d in datasets:
            assert "accession" in d
            assert "n_genes" in d
            assert "content_hash" in d
            assert d["species"] == "human"

    def test_in_regime_filtering(self):
        datasets = discover_merlin_datasets(MERLIN_DATA_DIR)
        in_regime = [d for d in datasets if d["in_regime"]]
        # GSE119262 (44 samples) should be in-regime
        accessions = {d["accession"] for d in in_regime}
        assert "GSE119262" in accessions


# ── Training pipeline ────────────────────────────────────────────────

def test_training_spec_from_datasets():
    from models.training.geo_training_pipeline import build_training_spec, GEOTrainingConfig
    from models.foundation.perturbation_response import GEOExpressionData

    rng = np.random.RandomState(42)
    data = GEOExpressionData(
        expression=rng.randn(40, 20).astype(np.float32),
        conditions=["control"] * 10 + ["treated"] * 30,
        gene_names=[f"G{i}" for i in range(20)],
        species="human",
        accession="GSE000001",
    )
    config = GEOTrainingConfig()
    spec = build_training_spec([data], config)
    assert spec.model_id == "PerturbationResponseModel"
    assert spec.species == "human"
    assert spec.regime == "data-driven"  # 40 samples ≥ 30


def test_training_spec_cross_species_refused():
    from models.training.geo_training_pipeline import build_training_spec, GEOTrainingConfig
    from models.foundation.perturbation_response import GEOExpressionData

    rng = np.random.RandomState(0)
    d1 = GEOExpressionData(
        expression=rng.randn(10, 5).astype(np.float32),
        conditions=["control"] * 5 + ["treated"] * 5,
        gene_names=[f"G{i}" for i in range(5)],
        species="human",
        accession="GSE1",
    )
    d2 = GEOExpressionData(
        expression=rng.randn(10, 5).astype(np.float32),
        conditions=["control"] * 5 + ["treated"] * 5,
        gene_names=[f"G{i}" for i in range(5)],
        species="mouse",
        accession="GSE2",
    )
    with pytest.raises(ValueError, match="Cross-species"):
        build_training_spec([d1, d2], GEOTrainingConfig())


def test_prepare_datasets_split():
    from models.training.geo_training_pipeline import prepare_datasets, GEOTrainingConfig
    from models.foundation.perturbation_response import GEOExpressionData

    n_genes = 10
    rng = np.random.RandomState(42)
    data = GEOExpressionData(
        expression=np.vstack([
            rng.randn(5, n_genes),
            rng.randn(15, n_genes),
        ]).astype(np.float32),
        conditions=["control"] * 5 + ["treated"] * 15,
        gene_names=[f"G{i}" for i in range(n_genes)],
        species="human",
        accession="GSE000001",
    )
    config = GEOTrainingConfig(held_out_fraction=0.2, test_fraction=0.1)
    train, val, test = prepare_datasets([data], config)
    total = len(train) + len(val) + len(test)
    assert total > 0
    assert len(val) >= 1
    assert len(test) >= 1


@pytest.mark.skipif(
    not MERLIN_DATA_DIR.exists(),
    reason="Merlin data directory not available",
)
def test_end_to_end_training_on_real_data():
    """End-to-end: load real GSE107965, build PerturbationDataset, train 2 epochs."""
    from models.training.geo_training_pipeline import (
        GEOTrainingConfig,
        run_governed_training,
    )
    config = GEOTrainingConfig(
        merlin_data_dir=str(MERLIN_DATA_DIR),
        epochs=2,
        batch_size=4,
        min_samples=5,
        latent_dim=16,
    )
    result = run_governed_training(config)
    assert "error" not in result, result.get("error", "")
    assert result["n_train"] > 0
    assert result["final_val_loss"] is not None
    assert isinstance(result["final_val_loss"], float)
    assert np.isfinite(result["final_val_loss"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
