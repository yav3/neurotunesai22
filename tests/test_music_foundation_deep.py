"""Deep tests for the music foundation model: encoder, decoder, tokenizer, optimizer, training."""
import pytest
import numpy as np


class TestLinearLayer:
    def test_forward_shape(self):
        from models.music_foundation.encoder import LinearLayer
        ll = LinearLayer(10, 20, seed=0)
        x = np.random.randn(5, 10)
        out = ll.forward(x)
        assert out.shape == (5, 20)

    def test_params_roundtrip(self):
        from models.music_foundation.encoder import LinearLayer
        ll = LinearLayer(8, 4, seed=1)
        p = ll.params
        ll2 = LinearLayer(8, 4, seed=99)
        ll2.load_params(p)
        np.testing.assert_array_equal(ll.W, ll2.W)
        np.testing.assert_array_equal(ll.b, ll2.b)


class TestActivations:
    def test_gelu_zero(self):
        from models.music_foundation.encoder import gelu
        assert gelu(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_gelu_positive(self):
        from models.music_foundation.encoder import gelu
        out = gelu(np.array([10.0]))[0]
        assert out == pytest.approx(10.0, abs=0.1)

    def test_layer_norm_stats(self):
        from models.music_foundation.encoder import layer_norm
        x = np.random.randn(3, 16)
        normed = layer_norm(x)
        for i in range(3):
            assert abs(normed[i].mean()) < 0.01
            assert abs(normed[i].var() - 1.0) < 0.1


class TestMultiHeadAttention:
    def test_output_shape(self):
        from models.music_foundation.encoder import MultiHeadAttention
        mha = MultiHeadAttention(32, 4, seed=0)
        x = np.random.randn(5, 32)
        out = mha.forward(x)
        assert out.shape == (5, 32)

    def test_1d_input(self):
        from models.music_foundation.encoder import MultiHeadAttention
        mha = MultiHeadAttention(16, 2, seed=0)
        out = mha.forward(np.random.randn(16))
        assert out.shape == (1, 16)

    def test_deterministic(self):
        from models.music_foundation.encoder import MultiHeadAttention
        mha = MultiHeadAttention(16, 2, seed=0)
        x = np.random.randn(3, 16)
        np.testing.assert_array_equal(mha.forward(x), mha.forward(x))


class TestAudioEncoder:
    def test_encode_shape(self):
        from models.music_foundation.encoder import AudioEncoder, EncoderConfig
        cfg = EncoderConfig(audio_dim=64, hidden_dim=32, latent_dim=16, n_heads=2)
        enc = AudioEncoder(cfg)
        out = enc.encode(np.random.randn(4, 64))
        assert out.shape == (4, 16)

    def test_pad_short_input(self):
        from models.music_foundation.encoder import AudioEncoder, EncoderConfig
        cfg = EncoderConfig(audio_dim=64, hidden_dim=32, latent_dim=16, n_heads=2)
        enc = AudioEncoder(cfg)
        out = enc.encode(np.random.randn(2, 10))
        assert out.shape == (2, 16)

    def test_params_nonempty(self):
        from models.music_foundation.encoder import AudioEncoder, EncoderConfig
        cfg = EncoderConfig(audio_dim=32, hidden_dim=16, latent_dim=8, n_heads=2)
        enc = AudioEncoder(cfg)
        assert len(enc.params()) > 0


class TestSemanticEncoder:
    def test_encode_shape(self):
        from models.music_foundation.encoder import SemanticEncoder, EncoderConfig
        cfg = EncoderConfig(semantic_dim=32, hidden_dim=16, latent_dim=8, n_heads=2)
        enc = SemanticEncoder(cfg)
        out = enc.encode(np.random.randn(3, 32))
        assert out.shape == (3, 8)


class TestGovernedBridge:
    def test_alpha_zero(self):
        from models.music_foundation.encoder import GovernedBridge, EncoderConfig
        cfg = EncoderConfig(latent_dim=8, alpha=0.0)
        bridge = GovernedBridge(cfg)
        audio = np.ones((1, 8))
        semantic = np.ones((1, 8)) * 2
        fused = bridge.fuse(audio, semantic)
        assert fused.shape == (1, 8)

    def test_alpha_one(self):
        from models.music_foundation.encoder import GovernedBridge, EncoderConfig
        cfg = EncoderConfig(latent_dim=8, alpha=1.0)
        bridge = GovernedBridge(cfg)
        audio = np.array([[3.0, 0, 0, 0, 0, 0, 0, 0]])
        semantic = np.array([[0.0, 3, 0, 0, 0, 0, 0, 0]])
        fused = bridge.fuse(audio, semantic)
        assert abs(fused[0, 0]) > abs(fused[0, 1])


class TestMusicFoundationModel:
    def test_full_encode(self):
        from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
        cfg = EncoderConfig(audio_dim=32, semantic_dim=16, hidden_dim=32, latent_dim=16, n_heads=2)
        model = MusicFoundationModel(cfg)
        out = model.encode(np.random.randn(2, 32), np.random.randn(2, 16))
        assert out.shape == (2, 16)

    def test_similarity_bounded(self):
        from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
        cfg = EncoderConfig(audio_dim=16, semantic_dim=8, hidden_dim=16, latent_dim=8, n_heads=2)
        model = MusicFoundationModel(cfg)
        a = model.encode(np.random.randn(1, 16), np.random.randn(1, 8))
        b = model.encode(np.random.randn(1, 16), np.random.randn(1, 8))
        sim = model.similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_self_similarity_one(self):
        from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
        cfg = EncoderConfig(audio_dim=16, semantic_dim=8, hidden_dim=16, latent_dim=8, n_heads=2)
        model = MusicFoundationModel(cfg)
        emb = model.encode(np.random.randn(1, 16), np.random.randn(1, 8))
        assert model.similarity(emb, emb) == pytest.approx(1.0, abs=1e-6)


class TestPretraining:
    def test_info_nce_positive(self):
        from models.music_foundation.pretraining import InfoNCELoss
        loss_fn = InfoNCELoss(temperature=0.07)
        embs = np.random.randn(8, 16)
        loss = loss_fn(embs, embs)
        assert isinstance(loss, float)

    def test_masked_feature_pred(self):
        from models.music_foundation.pretraining import MaskedFeaturePrediction
        mfp = MaskedFeaturePrediction(mask_ratio=0.3, seed=0)
        features = np.random.randn(4, 16)
        masked, mask = mfp.mask_features(features)
        assert masked.shape == features.shape
        assert mask.shape == features.shape
        assert mfp.loss(features, features, mask) == pytest.approx(0.0, abs=1e-10)

    def test_cross_modal_symmetric(self):
        from models.music_foundation.pretraining import CrossModalAlignment
        cma = CrossModalAlignment()
        a = np.random.randn(5, 16)
        b = np.random.randn(5, 16)
        loss_ab = cma(a, b)
        loss_ba = cma(b, a)
        assert loss_ab == pytest.approx(loss_ba, abs=1e-6)

    def test_objective_all_keys(self):
        from models.music_foundation.pretraining import PretrainingObjective
        obj = PretrainingObjective()
        a = np.random.randn(4, 16)
        s = np.random.randn(4, 16)
        f = np.random.randn(4, 16)
        result = obj(a, s, f)
        assert "total" in result
        assert "contrastive" in result
        assert "alignment" in result


class TestOptimizer:
    def test_adam_step(self):
        from models.music_foundation.optimizer import Adam
        opt = Adam(lr=0.01)
        params = {"w": np.array([1.0, 2.0, 3.0])}
        grads = {"w": np.array([0.1, 0.2, 0.3])}
        updated = opt.step(params, grads)
        assert not np.array_equal(updated["w"], np.array([1.0, 2.0, 3.0]))

    def test_adam_state_roundtrip(self):
        from models.music_foundation.optimizer import Adam
        opt = Adam()
        params = {"w": np.zeros(4)}
        grads = {"w": np.ones(4)}
        opt.step(params, grads)
        state = opt.state_dict()
        opt2 = Adam()
        opt2.load_state_dict(state)
        assert opt2._step_count == 1

    def test_sgd_momentum(self):
        from models.music_foundation.optimizer import SGD
        opt = SGD(lr=0.1, momentum=0.9)
        params = {"w": np.array([0.0])}
        grads = {"w": np.array([1.0])}
        opt.step(params, grads)
        v1 = params["w"][0]
        opt.step(params, grads)
        v2 = params["w"][0]
        assert v2 < v1

    def test_adamw_weight_decay(self):
        from models.music_foundation.optimizer import AdamW
        opt = AdamW(lr=0.01, weight_decay=0.1)
        params = {"w": np.array([10.0])}
        grads = {"w": np.array([0.0])}
        opt.step(params, grads)
        assert params["w"][0] < 10.0

    def test_gradient_clipper_norm(self):
        from models.music_foundation.optimizer import GradientClipper
        clipper = GradientClipper(max_norm=1.0)
        grads = {"a": np.array([10.0, 0.0])}
        clipped = clipper.clip(grads)
        norm = np.linalg.norm(clipped["a"])
        assert norm <= 1.0 + 1e-6

    def test_gradient_clipper_value(self):
        from models.music_foundation.optimizer import GradientClipper
        clipper = GradientClipper(max_value=0.5)
        grads = {"a": np.array([10.0, -10.0])}
        clipped = clipper.clip(grads)
        assert np.all(clipped["a"] <= 0.5)
        assert np.all(clipped["a"] >= -0.5)


class TestGradient:
    def test_numerical_gradient(self):
        from models.music_foundation.gradient import numerical_gradient
        params = {"w": np.array([2.0, 3.0])}
        def loss_fn(p):
            return float(np.sum(p["w"] ** 2))
        grads = numerical_gradient(loss_fn, params, eps=1e-5)
        np.testing.assert_allclose(grads["w"], 2 * params["w"], atol=1e-4)

    def test_gradient_accumulator(self):
        from models.music_foundation.gradient import GradientAccumulator
        acc = GradientAccumulator()
        acc.accumulate({"w": np.array([1.0, 2.0])})
        acc.accumulate({"w": np.array([3.0, 4.0])})
        avg = acc.get(normalize=True)
        np.testing.assert_allclose(avg["w"], [2.0, 3.0])
        assert acc.count == 2


class TestPositional:
    def test_sinusoidal_shape(self):
        from models.music_foundation.positional import sinusoidal_encoding
        pe = sinusoidal_encoding(50, 32)
        assert pe.shape == (50, 32)

    def test_sinusoidal_bounded(self):
        from models.music_foundation.positional import sinusoidal_encoding
        pe = sinusoidal_encoding(100, 64)
        assert np.all(pe >= -1.0)
        assert np.all(pe <= 1.0)

    def test_rope_shape(self):
        from models.music_foundation.positional import RotaryPositionalEncoding
        rope = RotaryPositionalEncoding(32)
        x = np.random.randn(5, 32)
        out = rope.apply(x)
        assert out.shape == (5, 32)

    def test_temporal_encoding(self):
        from models.music_foundation.positional import TemporalEncoding
        te = TemporalEncoding(64, max_bars=100, max_beats=8)
        bars = np.array([0, 1, 2, 3])
        beats = np.array([0, 1, 2, 3])
        out = te.encode(bars, beats)
        assert out.shape == (4, 64)

    def test_relative_position_bias(self):
        from models.music_foundation.positional import RelativePositionBias
        rpb = RelativePositionBias(n_heads=4, max_distance=32)
        bias = rpb.get_bias(10)
        assert bias.shape == (4, 10, 10)


class TestTokenizer:
    def test_codebook_quantize(self):
        from models.music_foundation.tokenizer import MusicCodebook
        cb = MusicCodebook(n_codes=64, dim=16, seed=0)
        x = np.random.randn(5, 16)
        quantized, indices, loss = cb.quantize(x)
        assert quantized.shape == (5, 16)
        assert indices.shape == (5,)
        assert loss >= 0

    def test_codebook_encode_decode(self):
        from models.music_foundation.tokenizer import MusicCodebook
        cb = MusicCodebook(n_codes=32, dim=8, seed=0)
        x = cb.embeddings[:3]
        indices = cb.encode(x)
        decoded = cb.decode(indices)
        np.testing.assert_allclose(x, decoded, atol=1e-10)

    def test_codebook_perplexity(self):
        from models.music_foundation.tokenizer import MusicCodebook
        cb = MusicCodebook(n_codes=16, dim=8, seed=0)
        x = np.random.randn(100, 8)
        cb.quantize(x)
        perp = cb.perplexity()
        assert 1.0 <= perp <= 16.0

    def test_tokenizer_roundtrip(self):
        from models.music_foundation.tokenizer import MusicTokenizer
        tok = MusicTokenizer(n_codes=64, dim=16, n_quantizers=4, seed=0)
        x = np.random.randn(3, 16)
        token_ids = tok.tokenize(x)
        assert len(token_ids) == 4
        recon = tok.detokenize(token_ids)
        assert recon.shape == (3, 16)

    def test_tokenizer_vocab_size(self):
        from models.music_foundation.tokenizer import MusicTokenizer
        tok = MusicTokenizer(n_codes=256, dim=32, n_quantizers=2)
        assert tok.vocab_size == 256

    def test_tokens_to_sequence(self):
        from models.music_foundation.tokenizer import tokens_to_sequence
        ids = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        flat = tokens_to_sequence(ids)
        assert len(flat) == 6
        assert flat[0] == 0
        assert flat[1] == 3


class TestDecoder:
    def test_decoder_shape(self):
        from models.music_foundation.decoder import MusicDecoder, DecoderConfig
        cfg = DecoderConfig(latent_dim=16, hidden_dim=32, output_dim=16, n_heads=2, n_layers=1)
        dec = MusicDecoder(cfg)
        z = np.random.randn(3, 16)
        out = dec.decode(z)
        assert out.shape == (3, 16)

    def test_decoder_with_context(self):
        from models.music_foundation.decoder import MusicDecoder, DecoderConfig
        cfg = DecoderConfig(latent_dim=16, hidden_dim=32, output_dim=16, n_heads=2, n_layers=1)
        dec = MusicDecoder(cfg)
        z = np.random.randn(3, 16)
        ctx = np.random.randn(5, 32)
        out = dec.decode(z, context=ctx)
        assert out.shape == (3, 16)

    def test_generate_length(self):
        from models.music_foundation.decoder import MusicDecoder, DecoderConfig
        cfg = DecoderConfig(latent_dim=8, hidden_dim=16, output_dim=8, n_heads=2, n_layers=1)
        dec = MusicDecoder(cfg)
        z = np.random.randn(1, 8)
        seq = dec.generate(z, max_steps=5)
        assert seq.shape[0] == 5

    def test_reconstruction_head(self):
        from models.music_foundation.decoder import ReconstructionHead
        head = ReconstructionHead(16, 32)
        z = np.random.randn(4, 16)
        out = head.forward(z)
        assert out.shape == (4, 32)
        loss = head.loss(z, np.random.randn(4, 32))
        assert loss >= 0

    def test_generation_head_predict(self):
        from models.music_foundation.decoder import GenerationHead
        head = GenerationHead(16, 64)
        z = np.random.randn(3, 16)
        probs = head.predict(z)
        assert probs.shape == (3, 64)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_generation_head_sample(self):
        from models.music_foundation.decoder import GenerationHead
        head = GenerationHead(16, 64)
        z = np.random.randn(3, 16)
        samples = head.sample(z, seed=42)
        assert samples.shape == (3,)
        assert all(0 <= s < 64 for s in samples)

    def test_generation_head_loss(self):
        from models.music_foundation.decoder import GenerationHead
        head = GenerationHead(16, 10)
        z = np.random.randn(5, 16)
        targets = np.array([0, 3, 5, 7, 9])
        loss = head.loss(z, targets)
        assert loss > 0


class TestTrainer:
    def test_early_stopping(self):
        from models.music_foundation.trainer import EarlyStopping
        es = EarlyStopping(patience=3, min_delta=0.01)
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.95)
        assert not es(0.96)
        assert not es(0.97)
        assert es(0.98)

    def test_training_history(self):
        from models.music_foundation.trainer import TrainingHistory
        h = TrainingHistory()
        h.log_step({"total": 1.0})
        h.log_step({"total": 0.8})
        h.log_epoch({"total": 0.9})
        h.log_val({"total": 0.85})
        s = h.summary()
        assert s["total_steps"] == 2
        assert s["total_epochs"] == 1

    def test_trainer_single_step(self):
        from models.music_foundation.trainer import MusicTrainer, TrainerConfig
        from models.music_foundation.encoder import EncoderConfig, MusicFoundationModel
        cfg = EncoderConfig(audio_dim=8, semantic_dim=4, hidden_dim=8, latent_dim=4, n_heads=2)
        model = MusicFoundationModel(cfg)
        tcfg = TrainerConfig(lr=1e-3, batch_size=2)
        trainer = MusicTrainer(model=model, config=tcfg)
        audio = np.random.randn(2, 8)
        semantic = np.random.randn(2, 4)
        losses = trainer.train_step(audio, semantic)
        assert "total" in losses
        assert isinstance(losses["total"], float)

    def test_trainer_eval_step(self):
        from models.music_foundation.trainer import MusicTrainer, TrainerConfig
        from models.music_foundation.encoder import EncoderConfig, MusicFoundationModel
        cfg = EncoderConfig(audio_dim=8, semantic_dim=4, hidden_dim=8, latent_dim=4, n_heads=2)
        model = MusicFoundationModel(cfg)
        trainer = MusicTrainer(model=model)
        metrics = trainer.eval_step(np.random.randn(2, 8), np.random.randn(2, 4))
        assert "total" in metrics


class TestTrainingInfra:
    def test_dataset_split(self):
        from models.music_training.dataset import MusicDataset
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        ds = MusicDataset(X, y)
        train, val, test = ds.split()
        assert len(train) + len(val) + len(test) == 100

    def test_normalizer_roundtrip(self):
        from models.music_training.dataset import FeatureNormalizer
        fn = FeatureNormalizer()
        X = np.random.randn(50, 5) * 10 + 5
        Xn = fn.fit_transform(X)
        Xr = fn.inverse_transform(Xn)
        np.testing.assert_allclose(X, Xr, atol=1e-10)

    def test_batches(self):
        from models.music_training.dataset import MusicDataset
        X = np.random.randn(20, 4)
        ds = MusicDataset(X)
        batches = ds.batches(batch_size=8)
        assert len(batches) == 3
        assert batches[0][0].shape[0] == 8

    def test_cosine_scheduler(self):
        from models.music_training.scheduler import CosineAnnealingScheduler
        sched = CosineAnnealingScheduler(base_lr=0.01, min_lr=0.0001, total_steps=100)
        assert sched.get_lr(0) == pytest.approx(0.01, abs=1e-4)
        assert sched.get_lr(100) == pytest.approx(0.0001, abs=1e-4)

    def test_warmup_scheduler(self):
        from models.music_training.scheduler import WarmupScheduler
        sched = WarmupScheduler(base_lr=0.01, warmup_steps=10, total_steps=100)
        assert sched.get_lr(0) < sched.get_lr(9)
        assert sched.get_lr(10) == pytest.approx(0.01, abs=1e-3)

    def test_one_cycle_scheduler(self):
        from models.music_training.scheduler import OneCycleScheduler
        sched = OneCycleScheduler(max_lr=0.01, total_steps=100)
        lrs = sched.schedule()
        assert len(lrs) == 100
        assert max(lrs) <= 0.01 + 1e-6

    def test_checkpoint_manager(self, tmp_path):
        from models.music_training.checkpoint import CheckpointManager
        cm = CheckpointManager(tmp_path, metric_name="loss", mode="min", top_k=2)
        w1 = {"a": np.array([1.0, 2.0])}
        cm.save(w1, epoch=0, metrics={"loss": 1.0})
        w2 = {"a": np.array([3.0, 4.0])}
        cm.save(w2, epoch=1, metrics={"loss": 0.5})
        best = cm.best()
        assert best.epoch == 1
        loaded = cm.load(best)
        np.testing.assert_array_equal(loaded["a"], w2["a"])

    def test_augmentation_pitch_shift(self):
        from models.music_training.augmentation import pitch_shift_features
        x = np.arange(12, dtype=float)
        shifted = pitch_shift_features(x, semitones=1)
        assert shifted.shape == x.shape
        assert not np.array_equal(shifted[:12], x[:12])

    def test_augmentation_mixup(self):
        from models.music_training.augmentation import mixup
        a = np.ones(10)
        b = np.zeros(10)
        mixed, lam = mixup(a, b, alpha=1.0, seed=0)
        assert 0 <= lam <= 1
        assert mixed.shape == a.shape

    def test_evaluator_genre_accuracy(self):
        from models.music_training.evaluator import genre_accuracy
        pred = np.array([0, 1, 2, 0])
        true = np.array([0, 1, 2, 1])
        assert genre_accuracy(pred, true) == 0.75

    def test_evaluator_ndcg(self):
        from models.music_training.evaluator import recommendation_ndcg
        pred = [["a", "b", "c"]]
        rel = [{"a", "c"}]
        score = recommendation_ndcg(pred, rel, k=3)
        assert 0 <= score <= 1.0

    def test_evaluator_embedding_quality(self):
        from models.music_training.evaluator import embedding_quality
        rng = np.random.RandomState(42)
        embs = np.vstack([rng.randn(10, 4) + [2, 0, 0, 0], rng.randn(10, 4) + [-2, 0, 0, 0]])
        labels = np.array([0] * 10 + [1] * 10)
        score = embedding_quality(embs, labels)
        assert score > 0.3

    def test_contrastive_loss(self):
        from models.music_training.objectives import ContrastiveLoss
        cl = ContrastiveLoss()
        embs = np.random.randn(8, 16)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        loss = cl(embs, labels)
        assert loss >= 0

    def test_multi_task_head(self):
        from models.music_training.objectives import MultiTaskHead
        mth = MultiTaskHead(n_classes=5)
        logits = np.random.randn(4, 5)
        labels = np.array([0, 1, 3, 4])
        reg_pred = np.random.randn(4, 2)
        reg_target = np.random.randn(4, 2)
        total, components = mth(logits, labels, reg_pred, reg_target)
        assert total > 0
        assert "classification" in components
        assert "regression" in components
