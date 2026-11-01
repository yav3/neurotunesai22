"""Tests for the Music Foundation–Frontier Hybrid Model.

Covers:
  - MusicFoundationEncoder forward pass
  - MaskedFeatureModeling pretraining
  - MoodPredictor / FeaturePredictor outputs
  - MusicHybridModel end-to-end
  - GovernedMusicBridge query/parse/answer
  - MusicTrainingPipeline governed trial
  - MusicInferenceService from_checkpoint + infer
  - Synthetic data generation
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from models.foundation.music_foundation import (
    MusicFoundationConfig,
    MusicFoundationEncoder,
    MaskedFeatureModeling,
    ValueBinner,
    TrackFeatureEmbedding,
)
from models.foundation.music_frontier_bridge import (
    AUDIO_FEATURE_NAMES,
    GovernedMusicBridge,
    MockMusicFrontierCaller,
    MusicBridgeConfig,
    MusicEmbeddingSummariser,
    MusicQueryType,
)
from models.foundation.music_hybrid_model import (
    MoodPredictor,
    FeaturePredictor,
    MusicHybridConfig,
    MusicHybridModel,
)
from models.data.music_dataset import (
    MusicFeatureDataset,
    generate_synthetic_music_data,
)
from models.training.music_trainer import (
    MusicTrainConfig,
    MusicTrainingPipeline,
)
from models.inference.music_inference import (
    MusicInferenceConfig,
    MusicInferenceService,
)


# ── Foundation encoder ──────────────────────────────────────────────────

class TestMusicFoundationEncoder:
    def test_forward_shape(self):
        cfg = MusicFoundationConfig(max_features=32, hidden_dim=64,
                                     num_layers=2, num_heads=2, ffn_dim=128)
        enc = MusicFoundationEncoder(cfg)
        B, F = 4, 16
        feats = torch.randn(B, F)
        indices = torch.arange(F).unsqueeze(0).expand(B, -1)
        cls_repr, seq_out = enc(feats, indices)
        assert cls_repr.shape == (B, 64)
        assert seq_out.shape == (B, F, 64)

    def test_with_padding_mask(self):
        cfg = MusicFoundationConfig(max_features=32, hidden_dim=64,
                                     num_layers=2, num_heads=2, ffn_dim=128)
        enc = MusicFoundationEncoder(cfg)
        B, F = 3, 12
        feats = torch.randn(B, F)
        indices = torch.arange(F).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, F, dtype=torch.bool)
        mask[0, 8:] = False
        cls_repr, seq_out = enc(feats, indices, mask)
        assert cls_repr.shape == (B, 64)

    def test_no_cls_token(self):
        cfg = MusicFoundationConfig(max_features=32, hidden_dim=64,
                                     num_layers=2, num_heads=2, ffn_dim=128,
                                     use_cls_token=False)
        enc = MusicFoundationEncoder(cfg)
        B, F = 2, 8
        feats = torch.randn(B, F)
        indices = torch.arange(F).unsqueeze(0).expand(B, -1)
        cls_repr, seq_out = enc(feats, indices)
        assert cls_repr.shape == (B, 64)
        assert seq_out.shape == (B, F, 64)

    def test_param_count(self):
        cfg = MusicFoundationConfig(max_features=32, hidden_dim=64,
                                     num_layers=2, num_heads=2, ffn_dim=128)
        enc = MusicFoundationEncoder(cfg)
        assert enc.param_count() > 0


class TestValueBinner:
    def test_output_shape(self):
        binner = ValueBinner(n_bins=16, hidden_dim=32)
        feats = torch.randn(4, 10)
        out = binner(feats)
        assert out.shape == (4, 10, 32)


class TestMaskedFeatureModeling:
    def test_forward_produces_loss(self):
        cfg = MusicFoundationConfig(max_features=32, hidden_dim=64,
                                     num_layers=2, num_heads=2, ffn_dim=128,
                                     mask_ratio=0.3)
        enc = MusicFoundationEncoder(cfg)
        mfm = MaskedFeatureModeling(enc)
        B, F = 4, 16
        feats = torch.randn(B, F)
        indices = torch.arange(F).unsqueeze(0).expand(B, -1)
        out = mfm(feats, indices)
        assert "loss" in out
        assert out["loss"].item() >= 0
        assert out["n_masked"] > 0
        assert out["pred"].shape == (B, F)


# ── Predictors ──────────────────────────────────────────────────────────

class TestMoodPredictor:
    def test_output_shapes(self):
        pred = MoodPredictor(hidden_dim=64, n_mood=5, n_vad=3, pred_hidden=32)
        cls = torch.randn(4, 64)
        out = pred(cls)
        assert out["mood_scores"].shape == (4, 5)
        assert out["vad"].shape == (4, 3)
        assert (out["mood_scores"] >= 0).all() and (out["mood_scores"] <= 1).all()
        assert (out["vad"] >= 0).all() and (out["vad"] <= 1).all()


class TestFeaturePredictor:
    def test_output_shape(self):
        pred = FeaturePredictor(hidden_dim=64, n_features=16, pred_hidden=32)
        cls = torch.randn(4, 64)
        out = pred(cls)
        assert out.shape == (4, 16)


# ── Hybrid model ────────────────────────────────────────────────────────

class TestMusicHybridModel:
    def _make_model(self):
        cfg = MusicHybridConfig(
            music_config=MusicFoundationConfig(
                max_features=32, hidden_dim=64,
                num_layers=2, num_heads=2, ffn_dim=128,
            ),
            prediction_hidden=64,
            projection_dim=8,
        )
        return MusicHybridModel(cfg)

    def test_forward(self):
        model = self._make_model()
        B, F = 4, 16
        feats = torch.randn(B, F)
        indices = torch.arange(F).unsqueeze(0).expand(B, -1)
        out = model(feats, indices)
        assert "cls_repr" in out
        assert "pred_mood" in out
        assert "pred_vad" in out
        assert "pred_features" in out
        assert out["pred_vad"].shape == (B, 3)
        assert out["pred_mood"].shape == (B, 5)

    def test_training_loss(self):
        model = self._make_model()
        B, F = 4, 16
        feats = torch.randn(B, F)
        indices = torch.arange(F).unsqueeze(0).expand(B, -1)
        target_vad = torch.rand(B, 3)
        target_mood = torch.rand(B, 5)
        out = model.training_loss(feats, indices, target_vad, target_mood)
        assert "loss" in out
        assert out["loss"].requires_grad

    def test_predict_and_explain_mock(self):
        model = self._make_model()
        feats = torch.randn(1, 16)
        indices = torch.arange(16).unsqueeze(0)
        names = AUDIO_FEATURE_NAMES[:16]
        caller = MockMusicFrontierCaller()
        answer = model.predict_and_explain(
            feats, indices, names,
            hypothesis="Test hypothesis",
            frontier_call=caller,
        )
        assert answer.is_governed
        assert answer.content_hash
        assert answer.frontier_reasoning is not None
        assert answer.frontier_reasoning.confidence > 0

    def test_predict_and_explain_no_frontier(self):
        model = self._make_model()
        feats = torch.randn(1, 16)
        indices = torch.arange(16).unsqueeze(0)
        names = AUDIO_FEATURE_NAMES[:16]
        answer = model.predict_and_explain(
            feats, indices, names,
            hypothesis="Test hypothesis",
            frontier_call=None,
        )
        assert answer.is_governed
        assert answer.frontier_reasoning is None

    def test_param_count(self):
        model = self._make_model()
        counts = model.param_count()
        assert counts["total"] > 0
        assert counts["encoder"] > 0
        assert counts["mood_predictor"] > 0


# ── Bridge ──────────────────────────────────────────────────────────────

class TestGovernedMusicBridge:
    def test_build_query(self):
        bridge = GovernedMusicBridge()
        delta = {
            "top_features": [{"feature": "valence", "delta": 0.2,
                              "baseline": 0.4, "predicted": 0.6}],
            "mean_abs_delta": 0.15,
            "vad_delta": {"valence": 0.2, "arousal": 0.1, "dominance": 0.0},
            "total_features": 16,
        }
        cls_summary = {"norm": 1.0, "mean": 0.1, "std": 0.3,
                       "max": 0.9, "min": -0.5, "sparsity": 0.1}
        query = bridge.build_query(
            MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
            delta, cls_summary, "music", "data-driven", "Test hypothesis",
        )
        assert query.domain == "music"
        assert query.lineage_hash

    def test_domain_firewall(self):
        bridge = GovernedMusicBridge()
        with pytest.raises(ValueError, match="Domain.*not in allowed"):
            bridge.build_query(
                MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
                {}, {}, "genomics", "data-driven", "hypothesis",
            )

    def test_regime_firewall(self):
        bridge = GovernedMusicBridge()
        with pytest.raises(ValueError, match="Regime.*not in allowed"):
            bridge.build_query(
                MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
                {}, {}, "music", "invalid_regime", "hypothesis",
            )

    def test_format_prompt(self):
        bridge = GovernedMusicBridge()
        delta = {
            "top_features": [{"feature": "valence", "delta": 0.2,
                              "baseline": 0.4, "predicted": 0.6}],
            "mean_abs_delta": 0.15,
            "vad_delta": {"valence": 0.2, "arousal": 0.1, "dominance": 0.0},
            "total_features": 16,
        }
        cls_summary = {"norm": 1.0, "mean": 0.1, "std": 0.3,
                       "max": 0.9, "min": -0.5, "sparsity": 0.1}
        query = bridge.build_query(
            MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
            delta, cls_summary, "music", "data-driven", "Test",
        )
        prompt = bridge.format_prompt(query)
        assert "music" in prompt
        assert "valence" in prompt

    def test_parse_response(self):
        bridge = GovernedMusicBridge()
        query = bridge.build_query(
            MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
            {"top_features": [], "mean_abs_delta": 0, "vad_delta": {}, "total_features": 0},
            {}, "music", "data-driven", "Test",
        )
        raw = {
            "reasoning": "Good interpretation",
            "confidence": 0.8,
            "evidence_citations": ["DOI:test"],
            "suggested_tracks": ["Track A"],
            "mood_trajectory": [{"valence": 0.5, "arousal": 0.5, "time_pct": 0.5}],
            "caveats": ["Test caveat"],
        }
        resp = bridge.parse_response(raw, query)
        assert resp.confidence == 0.8
        assert resp.response_hash
        assert len(resp.mood_trajectory) == 1

    def test_lineage_chain_grows(self):
        bridge = GovernedMusicBridge()
        assert len(bridge.lineage) == 0
        bridge.build_query(
            MusicQueryType.MOOD_SHIFT_HYPOTHESIS,
            {"top_features": [], "mean_abs_delta": 0, "vad_delta": {}, "total_features": 0},
            {}, "music", "data-driven", "H1",
        )
        assert len(bridge.lineage) == 1
        bridge.build_query(
            MusicQueryType.TRACK_RECOMMENDATION,
            {"top_features": [], "mean_abs_delta": 0, "vad_delta": {}, "total_features": 0},
            {}, "music", "data-driven", "H2",
        )
        assert len(bridge.lineage) == 2


# ── Dataset ─────────────────────────────────────────────────────────────

class TestMusicDataset:
    def test_synthetic_generation(self):
        data = generate_synthetic_music_data(100, 16, seed=42)
        assert data["features"].shape == (100, 16)
        assert data["target_vad"].shape == (100, 3)
        assert data["target_mood"].shape == (100, 5)
        assert len(data["feature_names"]) == 16

    def test_dataset_from_synthetic(self):
        ds = MusicFeatureDataset.from_synthetic(50, 16, seed=42)
        assert len(ds) == 50
        sample = ds[0]
        assert sample["features"].shape == (16,)
        assert sample["target_vad"].shape == (3,)
        assert sample["target_mood"].shape == (5,)

    def test_vad_in_range(self):
        data = generate_synthetic_music_data(500, 16, seed=42)
        assert (data["target_vad"] >= 0).all()
        assert (data["target_vad"] <= 1).all()
        assert (data["target_mood"] >= 0).all()
        assert (data["target_mood"] <= 1).all()


# ── Training pipeline ───────────────────────────────────────────────────

class TestMusicTrainingPipeline:
    def test_governed_trial(self):
        cfg = MusicTrainConfig(
            pretrain_epochs=2,
            finetune_epochs=3,
            n_synthetic_samples=100,
            n_features=16,
            batch_size=16,
        )
        hybrid_cfg = MusicHybridConfig(
            music_config=MusicFoundationConfig(
                max_features=32, hidden_dim=64,
                num_layers=2, num_heads=2, ffn_dim=128,
            ),
            prediction_hidden=32,
        )
        pipeline = MusicTrainingPipeline(cfg, hybrid_cfg)
        record = pipeline.run_governed_trial("TEST-001")
        assert record.study_id == "TEST-001"
        assert record.pretrain_loss > 0
        assert record.finetune_loss > 0
        assert record.val_vad_mse > 0
        assert record.content_hash

    def test_run_and_serve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MusicTrainConfig(
                pretrain_epochs=1,
                finetune_epochs=2,
                n_synthetic_samples=64,
                n_features=16,
                batch_size=16,
                checkpoint_dir=tmpdir,
            )
            hybrid_cfg = MusicHybridConfig(
                music_config=MusicFoundationConfig(
                    max_features=32, hidden_dim=64,
                    num_layers=2, num_heads=2, ffn_dim=128,
                ),
                prediction_hidden=32,
            )
            pipeline = MusicTrainingPipeline(cfg, hybrid_cfg)
            record, service = pipeline.run_and_serve("TEST-002", tmpdir)
            assert record.content_hash
            assert isinstance(service, MusicInferenceService)

            answer = service.infer(
                np.random.rand(16).astype(np.float32),
                hypothesis="Test hypothesis",
            )
            assert answer.is_governed


# ── Inference service ───────────────────────────────────────────────────

class TestMusicInferenceService:
    def _make_service(self, tmpdir: str) -> MusicInferenceService:
        cfg = MusicTrainConfig(
            pretrain_epochs=1, finetune_epochs=1,
            n_synthetic_samples=32, n_features=16, batch_size=8,
            checkpoint_dir=tmpdir,
        )
        hybrid_cfg = MusicHybridConfig(
            music_config=MusicFoundationConfig(
                max_features=32, hidden_dim=64,
                num_layers=2, num_heads=2, ffn_dim=128,
            ),
            prediction_hidden=32,
        )
        pipeline = MusicTrainingPipeline(cfg, hybrid_cfg)
        _, service = pipeline.run_and_serve("INFER-TEST", tmpdir)
        return service

    def test_infer_numpy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            answer = service.infer(
                np.random.rand(16).astype(np.float32),
                hypothesis="Test",
            )
            assert answer.is_governed
            assert "predicted_vad" in answer.foundation_prediction
            assert "predicted_mood" in answer.foundation_prediction

    def test_infer_torch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            answer = service.infer(
                torch.randn(16),
                hypothesis="Test torch input",
            )
            assert answer.is_governed

    def test_batch_infer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            answers = service.batch_infer(
                np.random.rand(3, 16).astype(np.float32),
                hypothesis="Batch test",
            )
            assert len(answers) == 3
            assert all(a.is_governed for a in answers)

    def test_from_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MusicTrainConfig(
                pretrain_epochs=1, finetune_epochs=1,
                n_synthetic_samples=32, n_features=16, batch_size=8,
                checkpoint_dir=tmpdir,
            )
            hybrid_cfg = MusicHybridConfig(
                music_config=MusicFoundationConfig(
                    max_features=32, hidden_dim=64,
                    num_layers=2, num_heads=2, ffn_dim=128,
                ),
                prediction_hidden=32,
            )
            pipeline = MusicTrainingPipeline(cfg, hybrid_cfg)
            record, service1 = pipeline.run_and_serve("ROUNDTRIP", tmpdir)

            ckpts = list(Path(tmpdir).glob("*.pt"))
            assert len(ckpts) == 1

            service2 = MusicInferenceService.from_checkpoint(ckpts[0])
            feats = np.random.rand(16).astype(np.float32)
            a1 = service1.infer(feats, hypothesis="H")
            a2 = service2.infer(feats, hypothesis="H")
            np.testing.assert_allclose(
                a1.foundation_prediction["predicted_vad"]["valence"],
                a2.foundation_prediction["predicted_vad"]["valence"],
                atol=1e-5,
            )


# ── Summariser ──────────────────────────────────────────────────────────

class TestMusicEmbeddingSummariser:
    def test_feature_delta(self):
        baseline = np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float32)
        predicted = np.array([0.7, 0.2, 0.9, 0.4], dtype=np.float32)
        names = ["valence", "arousal", "dominance", "energy"]
        result = MusicEmbeddingSummariser.summarise_feature_delta(
            baseline, predicted, names,
        )
        assert len(result["top_features"]) == 4
        assert result["total_features"] == 4
        assert result["mean_abs_delta"] > 0

    def test_cls_summary(self):
        cls = np.random.randn(64).astype(np.float32)
        result = MusicEmbeddingSummariser.summarise_cls_embedding(cls)
        assert "norm" in result
        assert "sparsity" in result
