"""Tests for model checkpoint save/load."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
from models.music_foundation.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_npz,
    load_npz,
)


@pytest.fixture
def model():
    cfg = EncoderConfig(audio_dim=16, semantic_dim=8, hidden_dim=32, latent_dim=16, n_heads=2)
    return MusicFoundationModel(cfg)


class TestJSONCheckpoint:
    def test_save_creates_file(self, model):
        with tempfile.TemporaryDirectory() as td:
            path = save_checkpoint(model, Path(td) / "model.json")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_roundtrip_preserves_params(self, model):
        audio = np.random.randn(4, 16)
        semantic = np.random.randn(4, 8)
        emb_before = model.encode(audio, semantic).copy()

        with tempfile.TemporaryDirectory() as td:
            save_checkpoint(model, Path(td) / "model.json")
            loaded, meta = load_checkpoint(Path(td) / "model.json")

        emb_after = loaded.encode(audio, semantic)
        np.testing.assert_allclose(emb_before, emb_after, atol=1e-10)

    def test_metadata_preserved(self, model):
        meta = {"epoch": 5, "loss": 0.42}
        with tempfile.TemporaryDirectory() as td:
            save_checkpoint(model, Path(td) / "model.json", metadata=meta)
            _, loaded_meta = load_checkpoint(Path(td) / "model.json")
        assert loaded_meta["epoch"] == 5
        assert loaded_meta["loss"] == 0.42

    def test_config_preserved(self, model):
        with tempfile.TemporaryDirectory() as td:
            save_checkpoint(model, Path(td) / "model.json")
            loaded, _ = load_checkpoint(Path(td) / "model.json")
        assert loaded.config.audio_dim == 16
        assert loaded.config.latent_dim == 16
        assert loaded.config.n_heads == 2


class TestNPZCheckpoint:
    def test_save_creates_file(self, model):
        with tempfile.TemporaryDirectory() as td:
            path = save_npz(model, Path(td) / "model.npz")
            assert path.exists()

    def test_roundtrip_preserves_params(self, model):
        audio = np.random.randn(4, 16)
        semantic = np.random.randn(4, 8)
        emb_before = model.encode(audio, semantic).copy()

        with tempfile.TemporaryDirectory() as td:
            save_npz(model, Path(td) / "model.npz")
            fresh = MusicFoundationModel(model.config)
            load_npz(fresh, Path(td) / "model.npz")

        emb_after = fresh.encode(audio, semantic)
        np.testing.assert_allclose(emb_before, emb_after, atol=1e-10)

    def test_npz_smaller_than_json(self, model):
        with tempfile.TemporaryDirectory() as td:
            json_path = save_checkpoint(model, Path(td) / "model.json")
            npz_path = save_npz(model, Path(td) / "model.npz")
            assert npz_path.stat().st_size < json_path.stat().st_size
