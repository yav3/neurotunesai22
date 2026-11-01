"""Tests for music_foundation module: encoder, bridge, pre-training objectives."""
import pytest
import numpy as np


class TestLinearLayer:
    def test_forward_shape(self):
        from models.music_foundation.encoder import LinearLayer
        layer = LinearLayer(10, 5)
        x = np.random.randn(3, 10)
        out = layer.forward(x)
        assert out.shape == (3, 5)

    def test_params_roundtrip(self):
        from models.music_foundation.encoder import LinearLayer
        layer = LinearLayer(8, 4, seed=42)
        params = layer.params
        layer2 = LinearLayer(8, 4, seed=99)
        layer2.load_params(params)
        np.testing.assert_array_equal(layer.W, layer2.W)
        np.testing.assert_array_equal(layer.b, layer2.b)


class TestActivations:
    def test_gelu(self):
        from models.music_foundation.encoder import gelu
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        out = gelu(x)
        assert abs(out[1]) < 1e-6
        assert out[2] > 0.8
        assert out[3] > 1.5

    def test_layer_norm(self):
        from models.music_foundation.encoder import layer_norm
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = layer_norm(x)
        assert abs(out.mean()) < 1e-6
        assert abs(out.std() - 1.0) < 0.2


class TestMultiHeadAttention:
    def test_output_shape(self):
        from models.music_foundation.encoder import MultiHeadAttention
        attn = MultiHeadAttention(dim=16, n_heads=4)
        x = np.random.randn(3, 16)
        out = attn.forward(x)
        assert out.shape == (3, 16)

    def test_single_input(self):
        from models.music_foundation.encoder import MultiHeadAttention
        attn = MultiHeadAttention(dim=8, n_heads=2)
        x = np.random.randn(1, 8)
        out = attn.forward(x)
        assert out.shape == (1, 8)


class TestAudioEncoder:
    def test_encode_shape(self):
        from models.music_foundation.encoder import AudioEncoder, EncoderConfig
        cfg = EncoderConfig(audio_dim=32, latent_dim=16, hidden_dim=32, n_heads=4)
        enc = AudioEncoder(cfg)
        x = np.random.randn(4, 32)
        out = enc.encode(x)
        assert out.shape == (4, 16)

    def test_encode_single(self):
        from models.music_foundation.encoder import AudioEncoder, EncoderConfig
        cfg = EncoderConfig(audio_dim=16, latent_dim=8, hidden_dim=16, n_heads=4)
        enc = AudioEncoder(cfg)
        x = np.random.randn(16)
        out = enc.encode(x)
        assert out.shape == (1, 8)

    def test_dimension_mismatch_padding(self):
        from models.music_foundation.encoder import AudioEncoder, EncoderConfig
        cfg = EncoderConfig(audio_dim=32, latent_dim=16, hidden_dim=32, n_heads=4)
        enc = AudioEncoder(cfg)
        x = np.random.randn(2, 10)
        out = enc.encode(x)
        assert out.shape == (2, 16)


class TestSemanticEncoder:
    def test_encode_shape(self):
        from models.music_foundation.encoder import SemanticEncoder, EncoderConfig
        cfg = EncoderConfig(semantic_dim=16, latent_dim=8, hidden_dim=32, n_heads=4)
        enc = SemanticEncoder(cfg)
        x = np.random.randn(3, 16)
        out = enc.encode(x)
        assert out.shape == (3, 8)


class TestGovernedBridge:
    def test_alpha_zero(self):
        from models.music_foundation.encoder import GovernedBridge, EncoderConfig
        cfg = EncoderConfig(latent_dim=8)
        bridge = GovernedBridge(cfg)
        bridge.set_alpha(0.0)
        audio = np.ones((1, 8))
        semantic = np.ones((1, 8)) * 2
        fused = bridge.fuse(audio, semantic)
        assert fused.shape == (1, 8)

    def test_alpha_one(self):
        from models.music_foundation.encoder import GovernedBridge, EncoderConfig
        cfg = EncoderConfig(latent_dim=4)
        bridge = GovernedBridge(cfg)
        bridge.set_alpha(1.0)
        audio = np.array([[1.0, 2.0, 3.0, 4.0]])
        semantic = np.array([[10.0, 20.0, 30.0, 40.0]])
        fused = bridge.fuse(audio, semantic)
        assert fused.shape == (1, 4)

    def test_alpha_clamp(self):
        from models.music_foundation.encoder import GovernedBridge, EncoderConfig
        bridge = GovernedBridge(EncoderConfig())
        bridge.set_alpha(2.0)
        assert bridge.alpha == 1.0
        bridge.set_alpha(-1.0)
        assert bridge.alpha == 0.0


class TestMusicFoundationModel:
    def test_encode_joint(self):
        from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
        cfg = EncoderConfig(audio_dim=32, semantic_dim=16, hidden_dim=32, latent_dim=16, n_heads=4)
        model = MusicFoundationModel(cfg)
        audio = np.random.randn(2, 32)
        semantic = np.random.randn(2, 16)
        emb = model.encode(audio, semantic)
        assert emb.shape == (2, 16)
        assert np.all(np.isfinite(emb))

    def test_similarity(self):
        from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
        cfg = EncoderConfig(audio_dim=16, semantic_dim=8, hidden_dim=16, latent_dim=8, n_heads=4)
        model = MusicFoundationModel(cfg)
        a = np.random.randn(1, 16)
        s = np.random.randn(1, 8)
        emb1 = model.encode(a, s)
        emb2 = model.encode(a, s)
        sim = model.similarity(emb1, emb2)
        assert abs(sim - 1.0) < 1e-6

    def test_params_dict(self):
        from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
        cfg = EncoderConfig(audio_dim=8, semantic_dim=4, hidden_dim=8, latent_dim=4, n_heads=2)
        model = MusicFoundationModel(cfg)
        params = model.params()
        assert len(params) > 0
        assert all(isinstance(v, np.ndarray) for v in params.values())

    def test_deterministic(self):
        from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
        cfg = EncoderConfig(audio_dim=16, semantic_dim=8, hidden_dim=16, latent_dim=8, n_heads=4)
        m1 = MusicFoundationModel(cfg)
        m2 = MusicFoundationModel(cfg)
        audio = np.random.randn(1, 16)
        sem = np.random.randn(1, 8)
        e1 = m1.encode(audio, sem)
        e2 = m2.encode(audio, sem)
        np.testing.assert_allclose(e1, e2, atol=1e-10)


class TestInfoNCELoss:
    def test_positive_loss(self):
        from models.music_foundation.pretraining import InfoNCELoss
        loss = InfoNCELoss(temperature=0.07)
        anchors = np.random.randn(8, 16)
        positives = np.random.randn(8, 16)
        val = loss(anchors, positives)
        assert isinstance(val, float)
        assert val >= 0

    def test_identical_pairs(self):
        from models.music_foundation.pretraining import InfoNCELoss
        loss = InfoNCELoss(temperature=0.5)
        x = np.random.randn(4, 8)
        val = loss(x, x)
        assert val >= 0


class TestMaskedFeaturePrediction:
    def test_mask_shape(self):
        from models.music_foundation.pretraining import MaskedFeaturePrediction
        mfp = MaskedFeaturePrediction(mask_ratio=0.3, seed=42)
        features = np.random.randn(4, 16)
        masked, mask = mfp.mask_features(features)
        assert masked.shape == features.shape
        assert mask.shape == features.shape
        assert masked[mask].sum() == 0.0

    def test_loss_zero_for_perfect(self):
        from models.music_foundation.pretraining import MaskedFeaturePrediction
        mfp = MaskedFeaturePrediction(seed=42)
        features = np.random.randn(4, 16)
        _, mask = mfp.mask_features(features)
        loss = mfp.loss(features, features, mask)
        assert loss < 1e-10


class TestCrossModalAlignment:
    def test_symmetric(self):
        from models.music_foundation.pretraining import CrossModalAlignment
        cma = CrossModalAlignment(temperature=0.1)
        a = np.random.randn(8, 16)
        b = np.random.randn(8, 16)
        loss = cma(a, b)
        assert isinstance(loss, float)
        assert loss >= 0


class TestPretrainingObjective:
    def test_full_objective(self):
        from models.music_foundation.pretraining import PretrainingObjective
        obj = PretrainingObjective()
        audio_embs = np.random.randn(4, 16)
        semantic_embs = np.random.randn(4, 16)
        fused = np.random.randn(4, 16)
        original = np.random.randn(4, 16)
        predicted = np.random.randn(4, 16)
        mask = np.random.random((4, 16)) < 0.15
        result = obj(audio_embs, semantic_embs, fused, original, predicted, mask)
        assert "total" in result
        assert "contrastive" in result
        assert "masked" in result
        assert "alignment" in result
        assert result["total"] >= 0

    def test_without_masked(self):
        from models.music_foundation.pretraining import PretrainingObjective
        obj = PretrainingObjective()
        audio_embs = np.random.randn(4, 8)
        semantic_embs = np.random.randn(4, 8)
        fused = np.random.randn(4, 8)
        result = obj(audio_embs, semantic_embs, fused)
        assert result["masked"] == 0.0
        assert result["total"] >= 0
