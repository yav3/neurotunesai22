"""Tests for the music foundation training loop."""
import pytest
import numpy as np

from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
from models.music_foundation.training import (
    Trainer, TrainingConfig, TrainingMetrics,
    cosine_lr, clip_gradients, numerical_gradient,
)
from models.music_foundation.pretraining import PretrainingObjective


@pytest.fixture
def small_model():
    cfg = EncoderConfig(audio_dim=16, semantic_dim=8, hidden_dim=32, latent_dim=16, n_heads=2)
    return MusicFoundationModel(cfg)


@pytest.fixture
def synth_data():
    rng = np.random.RandomState(0)
    audio = rng.randn(64, 16)
    semantic = rng.randn(64, 8)
    return audio, semantic


class TestCosineSchedule:
    def test_warmup_ramp(self):
        assert cosine_lr(0, 10, 100, 1e-3) < cosine_lr(9, 10, 100, 1e-3)

    def test_start_of_warmup(self):
        lr = cosine_lr(0, 10, 100, 1e-3)
        assert lr == pytest.approx(1e-4, rel=0.01)

    def test_end_is_min(self):
        lr = cosine_lr(100, 10, 100, 1e-3, min_lr=1e-6)
        assert lr == pytest.approx(1e-6, rel=0.01)

    def test_mid_between(self):
        lr = cosine_lr(55, 10, 100, 1e-3)
        assert 1e-6 < lr < 1e-3


class TestClipGradients:
    def test_no_clip_below_threshold(self):
        grads = {"a": np.array([0.1, 0.2])}
        clipped = clip_gradients(grads, 10.0)
        np.testing.assert_array_equal(clipped["a"], grads["a"])

    def test_clips_above_threshold(self):
        grads = {"a": np.array([10.0, 10.0])}
        clipped = clip_gradients(grads, 1.0)
        norm = np.linalg.norm(clipped["a"])
        assert norm <= 1.0 + 1e-6


class TestNumericalGradient:
    def test_gradient_shape_matches_params(self, small_model):
        audio = np.random.randn(4, 16)
        semantic = np.random.randn(4, 8)
        obj = PretrainingObjective()
        grads = numerical_gradient(small_model, audio, semantic, obj)
        params = small_model.params()
        for k in grads:
            assert grads[k].shape == params[k].shape


class TestTrainer:
    def test_single_step(self, small_model, synth_data):
        audio, semantic = synth_data
        trainer = Trainer(small_model, TrainingConfig(batch_size=8, epochs=1))
        losses = trainer.train_step(audio[:8], semantic[:8])
        assert "total" in losses
        assert losses["total"] >= 0
        assert trainer.metrics.step == 1

    def test_train_epoch(self, small_model, synth_data):
        audio, semantic = synth_data
        trainer = Trainer(small_model, TrainingConfig(batch_size=16, epochs=1))
        epoch_loss = trainer.train_epoch(audio, semantic)
        assert isinstance(epoch_loss, float)
        assert trainer.metrics.epoch == 1
        assert len(trainer.metrics.loss_history) == 4  # 64/16 = 4 steps

    def test_fit_multiple_epochs(self, small_model, synth_data):
        audio, semantic = synth_data
        cfg = TrainingConfig(batch_size=32, epochs=2)
        trainer = Trainer(small_model, cfg)
        metrics = trainer.fit(audio, semantic)
        assert metrics.epoch == 2
        assert len(metrics.loss_history) == 4  # 2 epochs * 2 batches

    def test_fit_with_validation(self, small_model, synth_data):
        audio, semantic = synth_data
        cfg = TrainingConfig(batch_size=32, epochs=2)
        trainer = Trainer(small_model, cfg)
        metrics = trainer.fit(audio[:48], semantic[:48], audio[48:], semantic[48:])
        assert len(metrics.eval_losses) == 2

    def test_lr_recorded(self, small_model, synth_data):
        audio, semantic = synth_data
        trainer = Trainer(small_model, TrainingConfig(batch_size=32, epochs=1))
        trainer.fit(audio, semantic)
        assert len(trainer.metrics.lr_history) > 0
        assert all(lr > 0 for lr in trainer.metrics.lr_history)


class TestMetrics:
    def test_summary_empty(self):
        m = TrainingMetrics()
        s = m.summary()
        assert s["final_loss"] is None

    def test_summary_populated(self):
        m = TrainingMetrics(step=10, epoch=2, loss_history=[0.5, 0.4, 0.3])
        s = m.summary()
        assert s["final_loss"] == 0.3
        assert s["best_loss"] == 0.3
        assert s["steps"] == 10
