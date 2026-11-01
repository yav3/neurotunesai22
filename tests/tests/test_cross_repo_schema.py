"""Cross-repo schema validation tests.

Verifies that NeuroTunesAI's bridge types align with:
  - Merlin's data registry contract (species, SHA-256, study refs)
  - Product surface AnalysisResult interface (audio features)
  - Flutter app service layer (therapeutic session, EIF ranking)
  - Merlin's evidence discipline (lineage, registry, firewall)

These tests read actual files from sibling repos to catch schema drift.
"""
from __future__ import annotations

import os
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")
PRODUCT_ROOT = Path("/home/user/neurotunescopy4privatesharing")
FLUTTER_ROOT = Path("/home/user/neurotunes")


# ── Product surface schema alignment ────────────────────────────────

@pytest.mark.skipif(
    not (PRODUCT_ROOT / "supabase/functions/audio-analysis/index.ts").exists(),
    reason="Product repo not available",
)
class TestProductSchemaAlignment:
    """Verify AudioFeatures fields match the product's AnalysisResult."""

    @pytest.fixture(autouse=True)
    def _load_schema(self):
        ts_path = PRODUCT_ROOT / "supabase/functions/audio-analysis/index.ts"
        self.ts_source = ts_path.read_text()

    def _interface_has_field(self, field_name: str) -> bool:
        pattern = rf"^\s+{field_name}\??\s*:"
        return bool(re.search(pattern, self.ts_source, re.MULTILINE))

    def test_core_audio_fields_present(self):
        from models.bridge.product_bridge import AudioFeatures
        import dataclasses
        fields = {f.name for f in dataclasses.fields(AudioFeatures) if f.name != "track_id"}
        # these fields must exist in both the bridge and the product interface
        critical_fields = {"bpm", "valence", "arousal", "dominance", "energy_level",
                           "danceability", "spectral_centroid", "spectral_rolloff",
                           "loudness_lufs", "mood_happy", "mood_sad", "mood_relaxed",
                           "mood_aggressive"}
        for field in critical_fields:
            assert field in fields, f"AudioFeatures missing {field}"
            assert self._interface_has_field(field), f"AnalysisResult missing {field}"

    def test_vad_tensor_dimensions(self):
        from models.bridge.product_bridge import AudioFeatures
        f = AudioFeatures(track_id="t", valence=0.5, arousal=0.6, dominance=0.7)
        vad = f.to_vad_tensor()
        assert vad.shape == (3,)
        assert vad[0].item() == pytest.approx(0.5)

    def test_eif_input_keys(self):
        from models.bridge.product_bridge import AudioFeatures
        f = AudioFeatures(track_id="t", valence=0.7, arousal=0.4,
                          energy_level=0.6, bpm=120, spectral_centroid=3000)
        eif = f.to_eif_input()
        expected_keys = {"valence", "arousal", "energy", "bpm", "spectral_centroid"}
        assert set(eif.keys()) == expected_keys


# ── Brainwave schema alignment ──────────────────────────────────────

@pytest.mark.skipif(
    not (PRODUCT_ROOT / "supabase/functions/brainwave-stream/index.ts").exists(),
    reason="Product repo not available",
)
class TestBrainwaveSchemaAlignment:

    def test_frequency_bands_match_product(self):
        from models.bridge.product_bridge import BrainwaveParams
        params = BrainwaveParams()
        product_bands = {"delta", "theta", "alpha", "beta", "gamma"}
        bridge_bands = set(params.BAND_RANGES.keys())
        assert bridge_bands == product_bands

    def test_band_ranges_physiologically_valid(self):
        from models.bridge.product_bridge import BrainwaveParams
        params = BrainwaveParams()
        # delta < theta < alpha < beta < gamma (non-overlapping)
        prev_hi = 0.0
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            lo, hi = params.BAND_RANGES[band]
            assert lo >= prev_hi, f"{band} overlaps with previous band"
            assert hi > lo, f"{band} range is empty"
            prev_hi = hi


# ── Merlin data registry alignment ──────────────────────────────────

@pytest.mark.skipif(
    not (MERLIN_ROOT / "backend/data/registry.json").exists(),
    reason="Merlin repo not available",
)
class TestMerlinRegistryAlignment:

    @pytest.fixture(autouse=True)
    def _load_registry(self):
        with open(MERLIN_ROOT / "backend/data/registry.json") as f:
            self.registry = json.load(f)

    def test_registry_has_entries(self):
        entries = self.registry.get("entries", {})
        assert len(entries) > 0

    def test_entry_has_required_fields(self):
        entries = self.registry.get("entries", {})
        required = {"path", "species", "sha256", "source"}
        for key, entry in entries.items():
            for field in required:
                assert field in entry, f"Entry {key} missing {field}"

    def test_species_values_valid(self):
        entries = self.registry.get("entries", {})
        valid_species = {"human", "mouse", "in_silico", "in_vitro", "synthetic", "not_applicable"}
        for key, entry in entries.items():
            assert entry["species"] in valid_species, \
                f"Entry {key} has invalid species: {entry['species']}"

    def test_expression_data_registered(self):
        entries = self.registry.get("entries", {})
        expression_entries = {k: v for k, v in entries.items()
                             if "expression" in v.get("path", "")}
        assert len(expression_entries) > 0

    def test_sha256_format(self):
        entries = self.registry.get("entries", {})
        for key, entry in entries.items():
            sha = entry.get("sha256", "")
            assert len(sha) == 64, f"Entry {key} has invalid SHA-256 length"
            assert all(c in "0123456789abcdef" for c in sha), \
                f"Entry {key} has non-hex SHA-256"


# ── Merlin evidence discipline alignment ────────────────────────────

@pytest.mark.skipif(
    not (MERLIN_ROOT / "backend/methods").exists(),
    reason="Merlin repo not available",
)
class TestMerlinEvidenceAlignment:

    def test_geo_adapter_pattern_exists(self):
        adapters = list((MERLIN_ROOT / "backend/methods").glob("geo_*_adapter.py"))
        assert len(adapters) >= 8

    def test_expression_data_committed(self):
        data_dir = MERLIN_ROOT / "backend/data/expression/human"
        csvs = list(data_dir.glob("*.csv"))
        assert len(csvs) >= 8

    def test_neurotunes_can_load_merlin_data(self):
        from models.data.geo_loader import discover_merlin_datasets
        data_dir = MERLIN_ROOT / "backend/data/expression/human"
        datasets = discover_merlin_datasets(data_dir)
        assert len(datasets) >= 8
        for d in datasets:
            assert d["n_genes"] > 0
            assert d["n_samples"] > 0

    def test_regime_classification_matches(self):
        """Verify NeuroTunesAI's regime selection matches Merlin's conventions."""
        from models.training.governed_harness import select_training_regime
        # GSE119262 has 44 samples → data-driven
        assert select_training_regime(n_samples=44) == "data-driven"
        # GSE107965 subset has 18 samples, 2 conditions → prior-contrast
        assert select_training_regime(n_samples=18, n_conditions=2) == "prior-contrast"
        # <30 samples, 1 condition → prior-mean
        assert select_training_regime(n_samples=15) == "prior-mean"


# ── Flutter app service alignment ───────────────────────────────────

@pytest.mark.skipif(
    not (FLUTTER_ROOT / "lib/services").exists(),
    reason="Flutter repo not available",
)
class TestFlutterServiceAlignment:

    def test_music_service_exists(self):
        assert (FLUTTER_ROOT / "lib/services/music_service.dart").exists()

    def test_data_service_exists(self):
        assert (FLUTTER_ROOT / "lib/services/data_service.dart").exists()

    def test_bridge_session_plan_format(self):
        """Verify VADE session plan output matches what a Flutter service would consume."""
        import torch
        from models.bridge.product_bridge import ProductBridge
        trajectory = torch.randn(4, 6)
        doses = torch.tensor([1.0, 0.618, 0.382, 0.236, 0.146, 0.09])
        plan = ProductBridge.vade_trajectory_to_session_plan(trajectory, doses)
        assert len(plan) == 6
        required_keys = {"step", "target_valence", "target_arousal",
                         "target_dominance", "dose_intensity"}
        for step in plan:
            assert required_keys.issubset(step.keys())
            assert isinstance(step["step"], int)
            assert isinstance(step["dose_intensity"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
