"""Tests for the Merlin ↔ NeuroTunesAI training bridge.

Validates that real committed GEO data from merlinaineuralplus2 loads
correctly through the governed harness pipeline.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pytest
import torch

MERLIN_ROOT = Path(__file__).parent.parent.parent / "merlinaineuralplus2"
MERLIN_AVAILABLE = (MERLIN_ROOT / "backend" / "data" / "expression" / "human").exists()


@pytest.mark.skipif(not MERLIN_AVAILABLE, reason="merlinaineuralplus2 not cloned")
class TestMerlinTrainingBridge:

    def _bridge(self):
        from models.bridge.training_bridge import MerlinTrainingBridge
        return MerlinTrainingBridge(MERLIN_ROOT)

    def test_list_datasets(self):
        bridge = self._bridge()
        refs = bridge.list_datasets()
        assert len(refs) >= 1
        for ref in refs:
            assert ref.species == "human"
            assert ref.accession.startswith("GSE")
            assert ref.sha256  # content-addressed

    def test_load_gse107965(self):
        bridge = self._bridge()
        refs = bridge.list_datasets()
        autophagy = [r for r in refs if "gse107965" in r.registry_id]
        assert len(autophagy) == 1, f"Expected GSE107965, got {[r.registry_id for r in refs]}"
        data = bridge.load_dataset(autophagy[0])
        assert data.species == "human"
        assert data.accession == "GSE107965"
        assert data.expression.shape[0] > 0  # has samples
        assert len(data.gene_names) > 0
        assert "control" in data.conditions

    def test_conditions_inferred_correctly(self):
        bridge = self._bridge()
        refs = bridge.list_datasets()
        autophagy = [r for r in refs if "gse107965" in r.registry_id]
        data = bridge.load_dataset(autophagy[0])
        n_control = sum(1 for c in data.conditions if c == "control")
        n_treated = sum(1 for c in data.conditions if c == "treated")
        assert n_control >= 1
        assert n_treated >= 1

    def test_perturbation_dataset_from_merlin_data(self):
        from models.foundation.perturbation_response import (
            PerturbationDataset, PerturbationType,
        )
        bridge = self._bridge()
        refs = bridge.list_datasets()
        autophagy = [r for r in refs if "gse107965" in r.registry_id]
        data = bridge.load_dataset(autophagy[0])
        ds = PerturbationDataset(data, perturbation_type=PerturbationType.DRUG)
        assert len(ds) > 0
        item = ds[0]
        assert item["baseline"].shape == (len(data.gene_names),)
        assert item["target"].shape == (len(data.gene_names),)

    def test_load_all_human_datasets(self):
        bridge = self._bridge()
        results = bridge.load_human_perturbation_datasets()
        assert len(results) >= 1
        for ref, ds in results:
            assert ref.species == "human"
            assert len(ds) > 0

    def test_species_firewall_rejects_mouse_as_human(self):
        """Mouse datasets in the mouse/ dir must not load via the human bridge."""
        bridge = self._bridge()
        refs = bridge.list_datasets()
        for ref in refs:
            assert ref.species == "human"

    def test_build_training_spec(self):
        bridge = self._bridge()
        refs = bridge.list_datasets()
        spec = bridge.build_training_spec(refs, n_genes=50)
        assert spec.species == "human"
        assert spec.regime == "data-driven"
        assert len(spec.criteria) == 4
        assert spec.content_hash()  # hashable

    def test_provenance_manifest(self):
        bridge = self._bridge()
        refs = bridge.list_datasets()
        spec = bridge.build_training_spec(refs, n_genes=50)
        manifest = bridge.provenance_manifest(refs, spec)
        assert manifest["study_id"] == "NT-001"
        assert manifest["spec_hash"] == spec.content_hash()
        assert len(manifest["datasets"]) == len(refs)
        for ds in manifest["datasets"]:
            assert ds["sha256"]  # content-addressed

    def test_model_factory(self):
        bridge = self._bridge()
        factory = bridge.build_model_factory(n_genes=50)
        model = factory()
        assert model is not None
        x = torch.randn(2, 50)
        out = model(
            baseline=x,
            ptype_ids=torch.zeros(2, dtype=torch.long),
            gene_idx=torch.zeros(2, dtype=torch.long),
            drug_idx=torch.zeros(2, dtype=torch.long),
            dose=torch.ones(2),
            duration=torch.ones(2),
            time_point=torch.ones(2),
        )
        assert "predicted" in out


@pytest.mark.skipif(not MERLIN_AVAILABLE, reason="merlinaineuralplus2 not cloned")
class TestCrossRepoSpeciesFirewall:
    """Verify species firewall is consistent between NeuroTunesAI and Merlin."""

    def test_human_genes_from_merlin_data(self):
        """All human datasets load with species='human'.

        Some datasets have all-uppercase gene symbols which are ambiguous
        per species.py ("upper-case symbols do not decide"). The bridge
        trusts the registry declaration in those cases.
        """
        from models.training.governed_harness import infer_species_from_genes
        from models.bridge.training_bridge import MerlinTrainingBridge
        bridge = MerlinTrainingBridge(MERLIN_ROOT)
        refs = bridge.list_datasets()
        for ref in refs:
            data = bridge.load_dataset(ref)
            assert data.species == "human", f"{ref.accession}: got {data.species}"
            try:
                inferred = infer_species_from_genes(data.gene_names[:20])
                assert inferred == "human", f"{ref.accession}: inferred {inferred}"
            except ValueError:
                pass  # ambiguous — acceptable, registry declaration governs

    def test_regime_selection_matches_merlin(self):
        from models.training.governed_harness import select_training_regime
        assert select_training_regime(n_samples=50) == "data-driven"
        assert select_training_regime(n_samples=10, n_conditions=3) == "prior-contrast"
        assert select_training_regime(n_samples=10) == "prior-mean"
        assert select_training_regime(n_samples=100, has_time_axis=True) == "longitudinal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
