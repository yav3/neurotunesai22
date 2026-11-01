"""Tests for music_training module: dataset, objectives, schedulers, evaluator, checkpoint, augmentation."""
import pytest
import numpy as np
import json


class TestFeatureNormalizer:
    def test_fit_transform(self):
        from models.music_training.dataset import FeatureNormalizer
        fn = FeatureNormalizer()
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normed = fn.fit_transform(data)
        assert normed.shape == data.shape
        assert abs(normed.mean(axis=0)[0]) < 1e-6
        assert abs(normed.std(axis=0)[0] - 1.0) < 0.1

    def test_inverse_transform(self):
        from models.music_training.dataset import FeatureNormalizer
        fn = FeatureNormalizer()
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        normed = fn.fit_transform(data)
        recovered = fn.inverse_transform(normed)
        np.testing.assert_allclose(recovered, data, atol=1e-6)

    def test_constant_feature(self):
        from models.music_training.dataset import FeatureNormalizer
        fn = FeatureNormalizer()
        data = np.array([[5.0], [5.0], [5.0]])
        normed = fn.fit_transform(data)
        assert np.all(np.isfinite(normed))


class TestMusicDataset:
    def test_create(self):
        from models.music_training.dataset import MusicDataset
        feats = np.random.randn(100, 10)
        labels = np.random.randint(0, 5, 100)
        ds = MusicDataset(feats, labels)
        assert ds.feature_dim() == 10
        assert len(ds) == 100

    def test_split(self):
        from models.music_training.dataset import MusicDataset
        feats = np.random.randn(100, 10)
        labels = np.random.randint(0, 3, 100)
        ds = MusicDataset(feats, labels, normalize=False)
        train, val, test = ds.split(train_frac=0.7, val_frac=0.15)
        assert len(train) + len(val) + len(test) == 100

    def test_batches(self):
        from models.music_training.dataset import MusicDataset
        feats = np.random.randn(50, 5)
        labels = np.zeros(50)
        ds = MusicDataset(feats, labels, normalize=False)
        batches = list(ds.batches(batch_size=16))
        assert len(batches) == 4
        assert batches[0][0].shape[0] == 16
        assert batches[-1][0].shape[0] == 2

    def test_class_counts(self):
        from models.music_training.dataset import MusicDataset
        labels = np.array([0, 0, 1, 1, 1, 2])
        ds = MusicDataset(np.random.randn(6, 3), labels, normalize=False)
        counts = ds.class_counts()
        assert counts[0] == 2
        assert counts[1] == 3
        assert counts[2] == 1

    def test_metadata(self):
        from models.music_training.dataset import MusicDataset
        feats = np.random.randn(10, 4)
        labels = np.zeros(10)
        meta = [{"title": f"song_{i}"} for i in range(10)]
        ds = MusicDataset(feats, labels, metadata=meta, normalize=False)
        assert ds.metadata[0]["title"] == "song_0"


class TestContrastiveLoss:
    def test_losses_nonnegative(self):
        from models.music_training.objectives import ContrastiveLoss
        cl = ContrastiveLoss(temperature=0.1)
        close = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]])
        spread = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        labels = np.array([0, 0, 1, 1])
        loss_close = cl(close, labels)
        loss_spread = cl(spread, labels)
        assert loss_close >= 0
        assert loss_spread >= 0

    def test_output_positive(self):
        from models.music_training.objectives import ContrastiveLoss
        cl = ContrastiveLoss()
        a = np.random.randn(8, 16)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        loss = cl(a, labels)
        assert isinstance(loss, float)
        assert loss >= 0


class TestReconstructionLoss:
    def test_zero_loss(self):
        from models.music_training.objectives import ReconstructionLoss
        rl = ReconstructionLoss()
        x = np.random.randn(10, 5)
        assert rl(x, x) < 1e-10

    def test_per_feature(self):
        from models.music_training.objectives import ReconstructionLoss
        rl = ReconstructionLoss()
        x = np.ones((10, 3))
        y = np.zeros((10, 3))
        pf = rl.per_feature(x, y)
        assert len(pf) == 3
        assert all(abs(v - 1.0) < 1e-6 for v in pf)


class TestMultiTaskHead:
    def test_combined(self):
        from models.music_training.objectives import MultiTaskHead
        mt = MultiTaskHead(classification_weight=0.5, regression_weight=0.5)
        logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
        true_cls = np.array([0, 1])
        pred_reg = np.array([[0.5], [0.8]])
        true_reg = np.array([[0.5], [0.8]])
        total, details = mt(logits, true_cls, pred_reg, true_reg)
        assert details["regression"] < 1e-10
        assert details["classification"] > 0
        assert total > 0


class TestSchedulers:
    def test_cosine_boundaries(self):
        from models.music_training.scheduler import CosineAnnealingScheduler
        s = CosineAnnealingScheduler(base_lr=1e-3, min_lr=1e-6, total_steps=100)
        assert abs(s.get_lr(0) - 1e-3) < 1e-8
        assert abs(s.get_lr(100) - 1e-6) < 1e-8

    def test_warmup_ramp(self):
        from models.music_training.scheduler import WarmupScheduler
        s = WarmupScheduler(base_lr=1e-3, warmup_steps=10, total_steps=100)
        assert s.get_lr(0) < s.get_lr(5)
        assert s.get_lr(5) < s.get_lr(10)
        assert abs(s.get_lr(9) - 1e-3) < 2e-4

    def test_one_cycle_peak(self):
        from models.music_training.scheduler import OneCycleScheduler
        s = OneCycleScheduler(max_lr=0.01, total_steps=1000, pct_start=0.3)
        schedule = s.schedule()
        peak_idx = np.argmax(schedule)
        assert 250 <= peak_idx <= 350

    def test_schedule_length(self):
        from models.music_training.scheduler import CosineAnnealingScheduler
        s = CosineAnnealingScheduler(total_steps=50)
        assert len(s.schedule()) == 50


class TestEvaluator:
    def test_genre_accuracy_perfect(self):
        from models.music_training.evaluator import genre_accuracy
        pred = np.array([0, 1, 2, 3])
        assert genre_accuracy(pred, pred) == 1.0

    def test_genre_accuracy_empty(self):
        from models.music_training.evaluator import genre_accuracy
        assert genre_accuracy(np.array([]), np.array([])) == 0.0

    def test_mood_quadrant(self):
        from models.music_training.evaluator import mood_quadrant_accuracy
        pred = np.array([[0.8, 0.8], [0.2, 0.2]])
        true = np.array([[0.9, 0.9], [0.1, 0.1]])
        assert mood_quadrant_accuracy(pred, true) == 1.0

    def test_key_detection_exact(self):
        from models.music_training.evaluator import key_detection_accuracy
        assert key_detection_accuracy(["C", "D", "E"], ["C", "D", "E"]) == 1.0

    def test_key_detection_relative(self):
        from models.music_training.evaluator import key_detection_accuracy
        acc = key_detection_accuracy(["A"], ["C"], allow_relative=True)
        assert acc == 1.0

    def test_tempo_accuracy_octave(self):
        from models.music_training.evaluator import tempo_estimation_accuracy
        pred = np.array([240.0])
        true = np.array([120.0])
        assert tempo_estimation_accuracy(pred, true) == 1.0

    def test_embedding_quality(self):
        from models.music_training.evaluator import embedding_quality
        rng = np.random.RandomState(42)
        c1 = rng.randn(20, 5) + np.array([5, 0, 0, 0, 0])
        c2 = rng.randn(20, 5) + np.array([0, 5, 0, 0, 0])
        emb = np.vstack([c1, c2])
        labels = np.array([0]*20 + [1]*20)
        score = embedding_quality(emb, labels)
        assert score > 0.5

    def test_ndcg_perfect(self):
        from models.music_training.evaluator import recommendation_ndcg
        pred = [["a", "b", "c"]]
        rel = [{"a", "b", "c"}]
        assert abs(recommendation_ndcg(pred, rel, k=3) - 1.0) < 1e-6


class TestCheckpointManager:
    def test_save_and_load(self, tmp_path):
        from models.music_training.checkpoint import CheckpointManager
        cm = CheckpointManager(tmp_path / "ckpts", metric_name="loss", mode="min")
        w = {"layer1": np.array([1.0, 2.0]), "layer2": np.array([3.0])}
        cm.save(w, epoch=1, metrics={"loss": 0.5})
        loaded = cm.load()
        np.testing.assert_array_equal(loaded["layer1"], w["layer1"])

    def test_best_selection(self, tmp_path):
        from models.music_training.checkpoint import CheckpointManager
        cm = CheckpointManager(tmp_path / "ckpts", metric_name="acc", mode="max", top_k=2)
        cm.save({"w": np.array([1.0])}, epoch=1, metrics={"acc": 0.7})
        cm.save({"w": np.array([2.0])}, epoch=2, metrics={"acc": 0.9})
        cm.save({"w": np.array([3.0])}, epoch=3, metrics={"acc": 0.8})
        best = cm.best()
        assert best.epoch == 2
        assert len(cm.checkpoints) == 2

    def test_manifest_persistence(self, tmp_path):
        from models.music_training.checkpoint import CheckpointManager
        d = tmp_path / "ckpts2"
        cm1 = CheckpointManager(d)
        cm1.save({"w": np.array([1.0])}, epoch=1, metrics={"loss": 0.3})
        cm2 = CheckpointManager(d)
        assert len(cm2.checkpoints) == 1
        assert cm2.checkpoints[0].epoch == 1

    def test_config_hash(self):
        from models.music_training.checkpoint import CheckpointManager
        h1 = CheckpointManager.config_hash({"lr": 0.001, "batch": 32})
        h2 = CheckpointManager.config_hash({"batch": 32, "lr": 0.001})
        assert h1 == h2


class TestAugmentation:
    def test_pitch_shift(self):
        from models.music_training.augmentation import pitch_shift_features
        f = np.zeros(20)
        f[0] = 1.0
        shifted = pitch_shift_features(f, semitones=3)
        assert shifted[3] == 1.0
        assert shifted[0] == 0.0

    def test_pitch_shift_2d(self):
        from models.music_training.augmentation import pitch_shift_features
        f = np.zeros((4, 20))
        f[:, 0] = 1.0
        shifted = pitch_shift_features(f, semitones=7)
        assert np.all(shifted[:, 7] == 1.0)

    def test_tempo_scale(self):
        from models.music_training.augmentation import tempo_scale_features
        f = np.array([120.0, 0.5, 0.3])
        scaled = tempo_scale_features(f, scale_factor=2.0, bpm_index=0)
        assert scaled[0] == 240.0
        assert scaled[1] == 0.5

    def test_noise_injection_deterministic(self):
        from models.music_training.augmentation import noise_injection
        f = np.ones(10)
        n1 = noise_injection(f, noise_scale=0.01, seed=42)
        n2 = noise_injection(f, noise_scale=0.01, seed=42)
        np.testing.assert_array_equal(n1, n2)

    def test_feature_dropout(self):
        from models.music_training.augmentation import feature_dropout
        f = np.ones(1000)
        dropped = feature_dropout(f, dropout_rate=0.5, seed=42)
        zero_count = np.sum(dropped == 0)
        assert 350 < zero_count < 650

    def test_mixup(self):
        from models.music_training.augmentation import mixup
        a = np.ones(10)
        b = np.zeros(10)
        mixed, lam = mixup(a, b, alpha=1.0, seed=42)
        assert 0 <= lam <= 1
        np.testing.assert_allclose(mixed, lam * a + (1 - lam) * b)


class TestContrastiveLearning:
    def test_cosine_similarity_identical(self):
        from models.music_training.contrastive_learning import cosine_similarity
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        from models.music_training.contrastive_learning import cosine_similarity
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_pairwise_cosine(self):
        from models.music_training.contrastive_learning import pairwise_cosine_similarity
        emb = np.eye(3)
        sim = pairwise_cosine_similarity(emb)
        assert sim.shape == (3, 3)
        np.testing.assert_allclose(np.diag(sim), 1.0)

    def test_nt_xent_loss_positive(self):
        from models.music_training.contrastive_learning import nt_xent_loss
        z_i = np.array([1.0, 0.0, 0.0])
        z_j = np.array([0.9, 0.1, 0.0])
        negatives = np.random.RandomState(0).randn(5, 3)
        loss = nt_xent_loss(z_i, z_j, negatives)
        assert loss > 0

    def test_triplet_loss_correct_order(self):
        from models.music_training.contrastive_learning import triplet_loss
        anchor = np.array([1.0, 0.0])
        positive = np.array([0.9, 0.1])
        negative = np.array([-1.0, 0.0])
        loss = triplet_loss(anchor, positive, negative, margin=0.5)
        assert loss >= 0.0

    def test_hard_negative_mining(self):
        from models.music_training.contrastive_learning import hard_negative_mining
        anchor = np.array([1.0, 0.0, 0.0])
        negatives = np.array([
            [0.9, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        hard = hard_negative_mining(anchor, negatives, k=1)
        assert len(hard) == 1

    def test_music_augmentor(self):
        from models.music_training.contrastive_learning import MusicAugmentor, AugmentationType
        aug = MusicAugmentor(augmentations=[AugmentationType.NOISE, AugmentationType.TIME_STRETCH])
        features = np.ones(64)
        augmented = aug.apply(features, seed=42)
        assert augmented.shape == features.shape
        assert not np.allclose(augmented, features)

    def test_alignment_metric(self):
        from models.music_training.contrastive_learning import alignment_metric
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        align = alignment_metric(a, b)
        assert align == pytest.approx(0.0, abs=1e-5)

    def test_uniformity_metric(self):
        from models.music_training.contrastive_learning import uniformity_metric
        z = np.random.RandomState(42).randn(20, 8)
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        unif = uniformity_metric(z)
        assert unif < 0

    def test_contrastive_training_state(self):
        from models.music_training.contrastive_learning import ContrastiveTrainingState
        state = ContrastiveTrainingState(embed_dim=16, temperature=0.1)
        assert state.embed_dim == 16
        assert state.temperature == 0.1


class TestTrainingPackageExports:
    def test_new_exports(self):
        import models.music_training as mt
        assert hasattr(mt, "AugmentationType")
        assert hasattr(mt, "ContrastivePair")
        assert hasattr(mt, "nt_xent_loss")
        assert hasattr(mt, "MusicAugmentor")
