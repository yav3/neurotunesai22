"""Tests for frontier music model: spectrogram, conditioning, diffusion, frontier."""
import numpy as np
import pytest


class TestSpectrogramEncoder:
    def test_encode_audio_shape(self):
        from models.music_foundation.spectrogram import SpectrogramEncoder, SpectrogramEncoderConfig
        cfg = SpectrogramEncoderConfig(n_mels=64, hidden_dim=128, latent_dim=64, n_layers=2, n_heads=4)
        enc = SpectrogramEncoder(cfg)
        audio = np.random.randn(22050).astype(np.float64)
        frames = enc.encode_audio(audio, sr=22050)
        assert frames.ndim == 2
        assert frames.shape[1] == 64

    def test_encode_to_single(self):
        from models.music_foundation.spectrogram import SpectrogramEncoder, SpectrogramEncoderConfig
        cfg = SpectrogramEncoderConfig(n_mels=64, hidden_dim=128, latent_dim=64, n_layers=1, n_heads=4)
        enc = SpectrogramEncoder(cfg)
        audio = np.random.randn(11025).astype(np.float64)
        emb = enc.encode_to_single(audio, sr=22050)
        assert emb.ndim == 1
        assert emb.shape[0] == 64

    def test_mel_computation(self):
        from models.music_foundation.spectrogram import _compute_mel_spectrogram
        audio = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
        mel = _compute_mel_spectrogram(audio, 22050, 2048, 512, 128)
        assert mel.ndim == 2
        assert mel.shape[1] == 128
        assert mel.shape[0] > 0
        assert np.all(np.isfinite(mel))

    def test_decoder_shape(self):
        from models.music_foundation.spectrogram import SpectrogramDecoder, SpectrogramEncoderConfig
        cfg = SpectrogramEncoderConfig(n_mels=64, hidden_dim=128, latent_dim=64, n_layers=1, n_heads=4)
        dec = SpectrogramDecoder(cfg)
        z = np.random.randn(10, 64)
        mel = dec.decode(z)
        assert mel.shape == (10, 64)

    def test_params_nonempty(self):
        from models.music_foundation.spectrogram import SpectrogramEncoder, SpectrogramEncoderConfig
        cfg = SpectrogramEncoderConfig(n_mels=64, hidden_dim=128, latent_dim=64, n_layers=1, n_heads=4)
        enc = SpectrogramEncoder(cfg)
        p = enc.params()
        assert len(p) > 0
        assert all(isinstance(v, np.ndarray) for v in p.values())


class TestConditioning:
    def test_genre_encode_name(self):
        from models.music_foundation.conditioning import GenreEncoder
        enc = GenreEncoder(20, 32)
        emb = enc.encode_name("jazz")
        assert emb.shape == (32,)
        assert np.any(emb != 0)

    def test_mood_encode_vad(self):
        from models.music_foundation.conditioning import MoodEncoder
        enc = MoodEncoder(16, 32)
        emb = enc.encode_vad(0.8, 0.6, 0.5)
        assert emb.shape == (32,)
        assert abs(emb[0] - 0.8) < 1e-10

    def test_key_encode(self):
        from models.music_foundation.conditioning import KeyEncoder
        enc = KeyEncoder(12, 2, 16)
        c_major = enc.encode_name("C major")
        a_minor = enc.encode_name("A minor")
        assert c_major.shape == (16,)
        assert not np.allclose(c_major, a_minor)

    def test_tempo_encode(self):
        from models.music_foundation.conditioning import TempoEncoder
        enc = TempoEncoder(16)
        slow = enc.encode(60.0)
        fast = enc.encode(180.0)
        assert slow.shape == (16,)
        assert not np.allclose(slow, fast)

    def test_tag_encode_hash(self):
        from models.music_foundation.conditioning import TagEncoder
        enc = TagEncoder(1000, 32, 10)
        emb = enc.encode_hash(["acoustic", "guitar", "fingerpicking"])
        assert emb.shape == (32,)

    def test_conditioning_module_full(self):
        from models.music_foundation.conditioning import ConditioningModule
        mod = ConditioningModule()
        cond = mod.encode(genre="rock", mood="energetic", key="E minor", tempo_bpm=120.0, tags=["electric"])
        assert cond.shape == (128,)
        assert np.all(np.isfinite(cond))

    def test_null_conditioning(self):
        from models.music_foundation.conditioning import ConditioningModule
        mod = ConditioningModule()
        null = mod.null_conditioning()
        assert np.all(null == 0)
        assert null.shape == (128,)

    def test_film_modulate(self):
        from models.music_foundation.conditioning import FiLMLayer
        film = FiLMLayer(cond_dim=128, feature_dim=64)
        x = np.random.randn(5, 64)
        cond = np.random.randn(128)
        out = film.modulate(x, cond)
        assert out.shape == (5, 64)

    def test_genre_integer_input(self):
        from models.music_foundation.conditioning import ConditioningModule
        mod = ConditioningModule()
        cond = mod.encode(genre=3)
        assert cond.shape == (128,)


class TestDiffusion:
    def test_schedule_linear(self):
        from models.music_foundation.diffusion import DiffusionConfig, _make_schedule
        cfg = DiffusionConfig(n_timesteps=100)
        sched = _make_schedule(cfg)
        assert sched["betas"].shape == (100,)
        assert sched["alpha_cumprod"][-1] < sched["alpha_cumprod"][0]
        assert np.all(sched["alpha_cumprod"] > 0)

    def test_schedule_cosine(self):
        from models.music_foundation.diffusion import DiffusionConfig, _make_schedule
        cfg = DiffusionConfig(n_timesteps=100, schedule="cosine")
        sched = _make_schedule(cfg)
        assert sched["alpha_cumprod"][0] > sched["alpha_cumprod"][-1]

    def test_q_sample(self):
        from models.music_foundation.diffusion import LatentDiffusionModel, DiffusionConfig
        cfg = DiffusionConfig(latent_dim=32, hidden_dim=64, n_layers=1, n_heads=4, n_timesteps=50)
        model = LatentDiffusionModel(cfg, seed=42)
        x_0 = np.random.randn(4, 32)
        x_t = model.q_sample(x_0, t=25)
        assert x_t.shape == x_0.shape
        assert not np.allclose(x_t, x_0)

    def test_training_loss_finite(self):
        from models.music_foundation.diffusion import LatentDiffusionModel, DiffusionConfig
        cfg = DiffusionConfig(latent_dim=32, hidden_dim=64, n_layers=1, n_heads=4, n_timesteps=50)
        model = LatentDiffusionModel(cfg, seed=42)
        x_0 = np.random.randn(4, 32)
        loss = model.training_loss(x_0)
        assert np.isfinite(loss)
        assert loss > 0

    def test_training_loss_with_cond(self):
        from models.music_foundation.diffusion import LatentDiffusionModel, DiffusionConfig
        cfg = DiffusionConfig(latent_dim=32, hidden_dim=64, n_layers=1, n_heads=4, n_timesteps=50, cond_dim=64)
        model = LatentDiffusionModel(cfg, seed=42)
        x_0 = np.random.randn(4, 32)
        cond = np.random.randn(64)
        loss = model.training_loss(x_0, cond)
        assert np.isfinite(loss)

    def test_sample_shape(self):
        from models.music_foundation.diffusion import LatentDiffusionModel, DiffusionConfig
        cfg = DiffusionConfig(latent_dim=32, hidden_dim=64, n_layers=1, n_heads=4, n_timesteps=20)
        model = LatentDiffusionModel(cfg, seed=42)
        sample = model.sample(shape=(8, 32), n_steps=5)
        assert sample.shape == (8, 32)
        assert np.all(np.isfinite(sample))

    def test_sample_with_cond(self):
        from models.music_foundation.diffusion import LatentDiffusionModel, DiffusionConfig
        cfg = DiffusionConfig(latent_dim=32, hidden_dim=64, n_layers=1, n_heads=4, n_timesteps=20, cond_dim=64, cfg_scale=2.0)
        model = LatentDiffusionModel(cfg, seed=42)
        cond = np.random.randn(64)
        sample = model.sample(shape=(4, 32), cond=cond, n_steps=5)
        assert sample.shape == (4, 32)

    def test_sample_progressive(self):
        from models.music_foundation.diffusion import LatentDiffusionModel, DiffusionConfig
        cfg = DiffusionConfig(latent_dim=16, hidden_dim=32, n_layers=1, n_heads=4, n_timesteps=20)
        model = LatentDiffusionModel(cfg, seed=42)
        intermediates = model.sample_progressive(shape=(2, 16), save_every=5)
        assert len(intermediates) >= 2
        assert intermediates[-1].shape == (2, 16)

    def test_timestep_embedding(self):
        from models.music_foundation.diffusion import TimestepEmbedding
        emb = TimestepEmbedding(dim=64)
        t0 = emb.forward(0)
        t50 = emb.forward(50)
        assert t0.shape[-1] == 64
        assert not np.allclose(t0, t50)

    def test_noise_predictor_params(self):
        from models.music_foundation.diffusion import NoisePredictor, DiffusionConfig
        cfg = DiffusionConfig(latent_dim=32, hidden_dim=64, n_layers=1, n_heads=4)
        pred = NoisePredictor(cfg)
        p = pred.params()
        assert len(p) > 0


class TestFrontierModel:
    def _make_model(self):
        from models.music_foundation.frontier import MusicFrontierModel, FrontierConfig
        cfg = FrontierConfig(
            n_mels=32, n_fft=512, hop_length=128, latent_dim=32,
            hidden_dim=64, n_heads=4, spec_layers=1, dec_layers=1,
            diff_layers=1, n_timesteps=20, codebook_size=64, n_codebooks=2,
            max_gen_frames=32, griffin_lim_iters=4,
        )
        return MusicFrontierModel(cfg, seed=42)

    def test_encode_shape(self):
        model = self._make_model()
        audio = np.random.randn(4410).astype(np.float64)
        frames = model.encode(audio, sr=22050)
        assert frames.ndim == 2
        assert frames.shape[1] == 32

    def test_encode_to_tokens(self):
        model = self._make_model()
        audio = np.random.randn(4410).astype(np.float64)
        tokens = model.encode_to_tokens(audio, sr=22050)
        assert len(tokens) == 2
        assert all(isinstance(t, np.ndarray) for t in tokens)

    def test_generate_latents(self):
        model = self._make_model()
        latents = model.generate_latents(n_frames=8, genre="jazz", mood="calm", seed=42)
        assert latents.shape == (8, 32)
        assert np.all(np.isfinite(latents))

    def test_generate_mel(self):
        model = self._make_model()
        mel = model.generate_mel(n_frames=8, genre="rock", seed=42)
        assert mel.ndim == 2
        assert mel.shape[0] == 8
        assert np.all(np.isfinite(mel))

    def test_generate_audio_shape(self):
        model = self._make_model()
        audio = model.generate_audio(duration_s=0.1, genre="electronic", seed=42)
        assert audio.ndim == 1
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_training_loss(self):
        model = self._make_model()
        audio = np.random.randn(4410).astype(np.float64)
        losses = model.training_loss(audio, sr=22050)
        assert "total" in losses
        assert "reconstruction" in losses
        assert "commitment" in losses
        assert "diffusion" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_reconstruct(self):
        model = self._make_model()
        audio = np.sin(2 * np.pi * 440 * np.arange(4410) / 22050)
        recon = model.reconstruct(audio, sr=22050)
        assert recon.ndim == 1
        assert len(recon) > 0

    def test_interpolate(self):
        model = self._make_model()
        a = np.sin(2 * np.pi * 440 * np.arange(4410) / 22050)
        b = np.sin(2 * np.pi * 880 * np.arange(4410) / 22050)
        mixed = model.interpolate(a, b, alpha=0.5, sr=22050)
        assert mixed.ndim == 1
        assert len(mixed) > 0

    def test_style_transfer(self):
        model = self._make_model()
        audio = np.sin(2 * np.pi * 440 * np.arange(4410) / 22050)
        transferred = model.style_transfer(audio, style_genre="jazz", strength=0.7, sr=22050)
        assert transferred.ndim == 1
        assert len(transferred) > 0

    def test_params(self):
        model = self._make_model()
        p = model.params()
        assert len(p) > 0

    def test_conditioning_affects_generation(self):
        model = self._make_model()
        rock = model.generate_latents(n_frames=8, genre="rock", seed=0)
        jazz = model.generate_latents(n_frames=8, genre="jazz", seed=0)
        assert not np.allclose(rock, jazz, atol=1e-6)


class TestGriffinLim:
    def test_roundtrip_energy(self):
        from models.music_foundation.frontier import _griffin_lim
        rng = np.random.RandomState(42)
        mel = rng.rand(16, 32) * 10.0
        audio = _griffin_lim(mel, n_fft=512, hop_length=128, sr=22050, n_iters=8, n_mels=32)
        assert len(audio) > 0
        assert np.all(np.isfinite(audio))

    def test_longer_mel(self):
        from models.music_foundation.frontier import _griffin_lim
        rng = np.random.RandomState(0)
        mel = rng.rand(64, 64) * 5.0
        audio = _griffin_lim(mel, n_fft=1024, hop_length=256, sr=22050, n_iters=4, n_mels=64)
        assert len(audio) > 100
