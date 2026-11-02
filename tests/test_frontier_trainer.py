"""Tests for frontier trainer: cosine schedule, AudioDataset, FrontierTrainer."""
import numpy as np
import pytest
import tempfile
import os


class TestCosineSchedule:
    def test_warmup_ramp(self):
        from models.music_foundation.frontier_trainer import cosine_schedule
        lr0 = cosine_schedule(0, warmup=10, total=100, base_lr=0.01)
        lr5 = cosine_schedule(5, warmup=10, total=100, base_lr=0.01)
        lr9 = cosine_schedule(9, warmup=10, total=100, base_lr=0.01)
        assert lr0 < lr5 < lr9
        assert lr9 == pytest.approx(0.01, rel=0.05)

    def test_cosine_decay(self):
        from models.music_foundation.frontier_trainer import cosine_schedule
        lr_start = cosine_schedule(10, warmup=10, total=100, base_lr=0.01)
        lr_mid = cosine_schedule(55, warmup=10, total=100, base_lr=0.01)
        lr_end = cosine_schedule(99, warmup=10, total=100, base_lr=0.01)
        assert lr_start > lr_mid > lr_end

    def test_min_lr_floor(self):
        from models.music_foundation.frontier_trainer import cosine_schedule
        lr = cosine_schedule(999, warmup=10, total=1000, base_lr=0.01, min_lr=1e-5)
        assert lr >= 1e-5

    def test_no_warmup(self):
        from models.music_foundation.frontier_trainer import cosine_schedule
        lr = cosine_schedule(0, warmup=0, total=100, base_lr=0.01)
        assert lr == pytest.approx(0.01, rel=0.01)


class TestAudioDataset:
    def test_single_audio(self):
        from models.music_foundation.frontier_trainer import AudioDataset
        audio = np.random.randn(50000)
        ds = AudioDataset([audio], segment_length=22050)
        assert len(ds) >= 2

    def test_short_audio_padded(self):
        from models.music_foundation.frontier_trainer import AudioDataset
        audio = np.ones(1000)
        ds = AudioDataset([audio], segment_length=22050)
        assert len(ds) == 1
        assert ds[0].shape == (22050,)
        assert np.allclose(ds[0][:1000], 1.0)
        assert np.allclose(ds[0][1000:], 0.0)

    def test_2d_array_input(self):
        from models.music_foundation.frontier_trainer import AudioDataset
        data = np.random.randn(4, 22050)
        ds = AudioDataset(data, segment_length=22050)
        assert len(ds) == 4

    def test_1d_array_input(self):
        from models.music_foundation.frontier_trainer import AudioDataset
        data = np.random.randn(44100)
        ds = AudioDataset(data, segment_length=22050)
        assert len(ds) == 2

    def test_batch_iterator(self):
        from models.music_foundation.frontier_trainer import AudioDataset
        data = [np.random.randn(22050) for _ in range(8)]
        ds = AudioDataset(data, segment_length=22050)
        batches = list(ds.batch_iterator(batch_size=3, shuffle=False))
        total = sum(len(b) for b in batches)
        assert total == 8

    def test_batch_iterator_shuffle(self):
        from models.music_foundation.frontier_trainer import AudioDataset
        data = [np.random.randn(22050) for _ in range(8)]
        ds = AudioDataset(data, segment_length=22050, seed=0)
        b1 = list(ds.batch_iterator(3, shuffle=True))
        ds2 = AudioDataset(data, segment_length=22050, seed=1)
        b2 = list(ds2.batch_iterator(3, shuffle=True))
        first_batch_1 = b1[0]
        first_batch_2 = b2[0]
        assert not np.allclose(first_batch_1, first_batch_2)


class TestFrontierTrainer:
    def _make_trainer(self):
        from models.music_foundation.frontier import MusicFrontierModel, FrontierConfig
        from models.music_foundation.frontier_trainer import (
            FrontierTrainer, FrontierTrainerConfig,
        )
        model_cfg = FrontierConfig(
            n_mels=32, n_fft=512, hop_length=128, latent_dim=16,
            hidden_dim=32, n_heads=4, spec_layers=1, dec_layers=1,
            diff_layers=1, n_timesteps=10, codebook_size=32, n_codebooks=2,
            max_gen_frames=16, griffin_lim_iters=2,
        )
        model = MusicFrontierModel(model_cfg, seed=42)
        train_cfg = FrontierTrainerConfig(
            lr=1e-3, epochs=2, batch_size=2, warmup_steps=2, total_steps=20,
            clip_norm=1.0, perturb_per_param=2, sr=22050,
            eval_every=1, patience=100, save_every=100,
        )
        trainer = FrontierTrainer(model, train_cfg)
        return trainer

    def test_compute_losses(self):
        trainer = self._make_trainer()
        batch = np.random.randn(2, 4410).astype(np.float64)
        losses = trainer._compute_losses(batch)
        assert "total" in losses
        assert "reconstruction" in losses
        assert "commitment" in losses
        assert "diffusion" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_lr_warmup(self):
        trainer = self._make_trainer()
        trainer._step = 0
        lr0 = trainer._lr()
        trainer._step = 1
        lr1 = trainer._lr()
        assert lr1 > lr0

    def test_train_step(self):
        trainer = self._make_trainer()
        batch = np.random.randn(2, 4410).astype(np.float64)
        losses = trainer.train_step(batch)
        assert "total" in losses
        assert trainer._step == 1
        assert len(trainer.train_losses) == 1

    def test_clip_gradients(self):
        trainer = self._make_trainer()
        grads = {"a": np.ones((10, 10)) * 100.0}
        clipped = trainer._clip_gradients(grads)
        norm = np.sqrt(np.sum(clipped["a"] ** 2))
        assert norm <= trainer.config.clip_norm + 1e-6

    def test_evaluate(self):
        from models.music_foundation.frontier_trainer import AudioDataset
        trainer = self._make_trainer()
        data = [np.random.randn(4410) for _ in range(4)]
        ds = AudioDataset(data, segment_length=4410)
        val = trainer.evaluate(ds)
        assert "total" in val
        assert len(trainer.val_losses) == 1

    def test_save_load_roundtrip(self):
        trainer = self._make_trainer()
        trainer._step = 42
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt")
            trainer.save_state(path)
            trainer._step = 0
            trainer.load_state(path)
            assert trainer._step == 42

    def test_early_stopping(self):
        trainer = self._make_trainer()
        trainer.config.patience = 2
        assert not trainer._check_early_stop(1.0)
        assert not trainer._check_early_stop(0.9)
        assert not trainer._check_early_stop(1.0)
        assert trainer._check_early_stop(1.0)


class TestEndToEnd:
    def test_synthetic_train_loop(self):
        from models.music_foundation.frontier import MusicFrontierModel, FrontierConfig
        from models.music_foundation.frontier_trainer import (
            FrontierTrainer, FrontierTrainerConfig, AudioDataset,
        )
        from models.music_foundation.audio_pipeline import SyntheticAudioGenerator

        model_cfg = FrontierConfig(
            n_mels=32, n_fft=512, hop_length=128, latent_dim=16,
            hidden_dim=32, n_heads=4, spec_layers=1, dec_layers=1,
            diff_layers=1, n_timesteps=10, codebook_size=32, n_codebooks=2,
            max_gen_frames=16, griffin_lim_iters=2,
        )
        model = MusicFrontierModel(model_cfg, seed=42)

        gen = SyntheticAudioGenerator(seed=0)
        samples = gen.generate_dataset(4, duration_s=0.2, sr=22050)
        train_ds = AudioDataset(samples, segment_length=4410)
        assert len(train_ds) >= 4

        train_cfg = FrontierTrainerConfig(
            lr=1e-3, epochs=1, batch_size=2, warmup_steps=1, total_steps=10,
            clip_norm=1.0, perturb_per_param=2, sr=22050,
            eval_every=1, patience=100,
        )
        trainer = FrontierTrainer(model, train_cfg)
        result = trainer.fit(train_ds)
        assert "epochs_run" in result
        assert result["epochs_run"] == 1
        assert "total" in result["final_train_loss"]
        assert np.isfinite(result["final_train_loss"]["total"])
