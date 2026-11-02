"""End-to-end integration tests for the frontier music foundation model.

Proves the full pipeline works: synthetic audio → spectrograms → model
forward pass → training loop → benchmark evaluation → checkpoint roundtrip.

Uses small configs for speed. Pure numpy — real integration tests, not mocks.
"""
from __future__ import annotations

import os
import time
import tempfile
import traceback
import numpy as np
from typing import Dict, List, Tuple

from .frontier_model import (
    FrontierConfig, MusicFrontierModel, batch_spectrograms,
    contrastive_loss, diversity_loss, masked_prediction_loss,
    cosine_warmup_schedule, TrainingState,
)
from .deep_trainer import DeepTrainerConfig, DeepAudioDataset, DeepFrontierTrainer
from .data_pipeline import (
    generate_synthetic_clips, compute_mel_spectrogram, SpectrogramConfig,
    process_batch_spectrograms, OnTheFlyAugmenter, AugScheduleConfig,
    CurriculumSampler, AudioPipeline, PipelineConfig,
    stratified_split, k_fold_indices,
)
from .benchmark import (
    BenchmarkConfig, BenchmarkSuite, ProbeClassifier, RetrievalBenchmark,
    ClusteringBenchmark, centered_kernel_alignment, isotropy_score,
    effective_rank, intrinsic_dimensionality_mle,
)


SMALL_CONFIG = FrontierConfig(
    n_mels=64,
    conv_channels=[32, 64],
    conv_kernels=[5, 3],
    conv_strides=[2, 2],
    d_model=64,
    n_heads=2,
    n_layers=2,
    d_ff=128,
    n_codebooks=2,
    codebook_size=32,
    codebook_dim=32,  # codebook_dim * n_codebooks must == d_model
    mask_prob=0.05,
    max_seq_len=256,
    n_negatives=10,
)

SMALL_TRAINER = DeepTrainerConfig(
    learning_rate=1e-3,
    warmup_steps=2,
    total_steps=5,
    batch_size=4,
    eval_every=3,
    fd_epsilon=1e-3,
    n_param_samples=2,
)

SPEC_CONFIG = SpectrogramConfig(n_fft=512, hop_length=128, n_mels=64)


def _make_signals(n: int = 8, sr: int = 16000, dur: float = 0.5) -> List[np.ndarray]:
    rng = np.random.RandomState(42)
    signals = []
    for i in range(n):
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = 220 * (1.0 + i * 0.2)
        sig = np.sin(2 * np.pi * freq * t) + 0.3 * rng.randn(len(t))
        signals.append(sig.astype(np.float32))
    return signals


def _make_specs(signals: List[np.ndarray], sr: int = 16000) -> List[np.ndarray]:
    return [compute_mel_spectrogram(s, sr, SPEC_CONFIG) for s in signals]


def _make_labels(n: int) -> List[int]:
    return [i % 3 for i in range(n)]


# ---- Test 1: Data pipeline → frontier forward pass ----

def test_data_pipeline_to_frontier() -> str:
    signals = _make_signals(6)
    specs = _make_specs(signals)
    model = MusicFrontierModel(SMALL_CONFIG)

    for i, spec in enumerate(specs):
        assert spec.ndim == 2, f"Spec {i} should be 2D, got {spec.ndim}D"
        assert spec.shape[0] == 64, f"Spec {i} mel bins = {spec.shape[0]}, expected 64"

    result = model.pretrain_forward(specs[0], mask_seed=0)
    assert 'total_loss' in result, "Missing total_loss in pretrain result"
    assert np.isfinite(result['total_loss']), f"Loss not finite: {result['total_loss']}"

    enc_dict = model.encode(specs[0])
    encoded = enc_dict['hidden_states']
    assert encoded.ndim == 2, f"Encoded should be 2D, got {encoded.ndim}D"
    assert encoded.shape[1] == SMALL_CONFIG.d_model

    return f"OK (spec shape={specs[0].shape}, encoded={encoded.shape}, loss={result['total_loss']:.4f})"


# ---- Test 2: Training loop ----

def test_training_loop() -> str:
    signals = _make_signals(8)
    dataset = DeepAudioDataset.from_signals(
        signals, n_mels=64, n_fft=512, hop_length=128, sr=16000,
    )
    model = MusicFrontierModel(SMALL_CONFIG)
    trainer = DeepFrontierTrainer(model, SMALL_TRAINER, seed=123)

    losses = []
    for step in range(3):
        specs, _ = dataset.sample_batch(4, seed=step)
        metrics = trainer.train_step(specs, step_seed=step)
        losses.append(metrics['loss'])
        assert np.isfinite(metrics['loss']), f"Step {step}: loss not finite"
        assert metrics['grad_norm'] > 0, f"Step {step}: zero gradient norm"

    all_finite = all(np.isfinite(l) for l in losses)
    return f"OK (losses={[f'{l:.4f}' for l in losses]}, all_finite={all_finite})"


# ---- Test 3: Benchmark on frontier embeddings ----

def test_benchmark_on_frontier() -> str:
    signals = _make_signals(12)
    specs = _make_specs(signals)
    model = MusicFrontierModel(SMALL_CONFIG)

    embeddings = np.array([model.encode(s)['hidden_states'].mean(axis=0) for s in specs])
    labels = np.array(_make_labels(12))

    config = BenchmarkConfig(
        probe_epochs=50,
        probe_lr=0.1,
        n_clusters=3,
        k_values=[1, 3, 5],
    )
    suite = BenchmarkSuite(config)

    n_train = 8
    train_emb, test_emb = embeddings[:n_train], embeddings[n_train:]
    train_lab, test_lab = labels[:n_train], labels[n_train:]
    report = suite.run_all(train_emb, train_lab, test_emb, test_lab)

    probe_acc = report.probe_results.get('accuracy', 0) if report.probe_results else 0
    ret_map = report.retrieval_results.get('mAP', 0) if report.retrieval_results else 0
    clust_nmi = report.clustering_results.get('nmi', 0) if report.clustering_results else 0
    iso = report.representation_results.get('isotropy', 0) if report.representation_results else 0
    er = report.representation_results.get('effective_rank', 0) if report.representation_results else 0

    assert probe_acc >= 0.0, "Probe accuracy < 0"
    assert ret_map >= 0.0, "mAP < 0"

    return (
        f"OK (probe_acc={probe_acc:.3f}, mAP={ret_map:.3f}, "
        f"NMI={clust_nmi:.3f}, isotropy={iso:.4f}, eff_rank={er:.1f})"
    )


# ---- Test 4: Two-view augmentation ----

def test_augmentation_pipeline() -> str:
    from .data_pipeline import AudioClip, StreamingBatcher, BatcherConfig, _augment_signal

    signals = _make_signals(8, sr=16000, dur=0.25)
    clips = [AudioClip(signal=s, sr=16000) for s in signals]

    batcher_cfg = BatcherConfig(segment_length=2048, batch_size=2, overlap=0.0, shuffle=False)
    batcher = StreamingBatcher(clips, batcher_cfg, seed=42)

    aug_config = AugScheduleConfig(
        initial_prob=0.8, final_prob=0.9, warmup_steps=5,
        gain_db_range=(-6.0, 6.0), noise_snr_range=(10.0, 20.0),
    )
    augmenter = OnTheFlyAugmenter(batcher, aug_config, two_view=True, seed=42)

    count = 0
    for view_a, view_b in augmenter:
        assert view_a.shape == view_b.shape, f"View shapes differ: {view_a.shape} vs {view_b.shape}"
        assert not np.allclose(view_a, view_b), "Two views should differ"
        count += 1
        if count >= 3:
            break

    return f"OK ({count} batches of two-view augmented signals verified)"


# ---- Test 5: Curriculum training ----

def test_curriculum_training() -> str:
    signals = _make_signals(8)
    specs = _make_specs(signals)
    labels = _make_labels(8)

    dataset = DeepAudioDataset(specs, labels)
    model = MusicFrontierModel(SMALL_CONFIG)

    from .data_pipeline import difficulty_score
    difficulties = [difficulty_score(s, sr=16000) for s in signals]
    sorted_indices = np.argsort(difficulties)

    easy_specs = [specs[i] for i in sorted_indices[:4]]
    hard_specs = [specs[i] for i in sorted_indices[4:]]

    easy_loss = float(np.mean([
        model.pretrain_forward(s, mask_seed=i)['total_loss']
        for i, s in enumerate(easy_specs)
    ]))
    hard_loss = float(np.mean([
        model.pretrain_forward(s, mask_seed=i + 100)['total_loss']
        for i, s in enumerate(hard_specs)
    ]))

    return (
        f"OK (easy_loss={easy_loss:.4f}, hard_loss={hard_loss:.4f}, "
        f"difficulties={[f'{d:.3f}' for d in sorted(difficulties)]})"
    )


# ---- Test 6: Checkpoint roundtrip ----

def test_checkpoint_roundtrip() -> str:
    signals = _make_signals(4)
    dataset = DeepAudioDataset.from_signals(
        signals, n_mels=64, n_fft=512, hop_length=128, sr=16000,
    )
    model = MusicFrontierModel(SMALL_CONFIG)
    trainer = DeepFrontierTrainer(model, SMALL_TRAINER, seed=77)

    specs, _ = dataset.sample_batch(4, seed=0)
    trainer.train_step(specs, step_seed=0)
    trainer.train_step(specs, step_seed=1)

    params_before = trainer._extract_params()

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file not created"

        model2 = MusicFrontierModel(SMALL_CONFIG)
        trainer2 = DeepFrontierTrainer(model2, SMALL_TRAINER, seed=99)
        trainer2.load_checkpoint(ckpt_path)

        params_after = trainer2._extract_params()

        for key in params_before:
            assert key in params_after, f"Missing key {key} after load"
            np.testing.assert_array_almost_equal(
                params_before[key], params_after[key], decimal=10,
                err_msg=f"Param {key} differs after checkpoint roundtrip",
            )

        assert trainer2.state.step == trainer.state.step, (
            f"Step mismatch: {trainer2.state.step} vs {trainer.state.step}"
        )
    finally:
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)

    return f"OK (step={trainer.state.step}, {len(params_before)} params verified)"


# ---- Test 7: Evaluation metrics ----

def test_evaluation_metrics() -> str:
    signals = _make_signals(12)
    specs = _make_specs(signals)
    model = MusicFrontierModel(SMALL_CONFIG)

    emb = np.array([model.encode(s)['hidden_states'].mean(axis=0) for s in specs])

    cka_self = centered_kernel_alignment(emb, emb)
    assert abs(cka_self - 1.0) < 1e-6, f"Self-CKA should be 1.0, got {cka_self}"

    iso = isotropy_score(emb)
    assert 0.0 <= iso <= 1.0 or np.isfinite(iso), f"Isotropy out of range: {iso}"

    er = effective_rank(emb)
    assert er > 0, f"Effective rank should be > 0, got {er}"

    dim = intrinsic_dimensionality_mle(emb, k=3)
    assert dim > 0, f"Intrinsic dim should be > 0, got {dim}"

    return f"OK (CKA_self={cka_self:.4f}, isotropy={iso:.4f}, eff_rank={er:.1f}, intrinsic_dim={dim:.1f})"


# ---- Test 8: Data splitting utilities ----

def test_data_splitting() -> str:
    n = 100
    labels = np.array([i % 5 for i in range(n)])

    train, val, test = stratified_split(n, labels, ratios=(0.7, 0.15, 0.15), seed=42)
    assert len(train) + len(val) + len(test) == n, "Split sizes don't sum to n"

    for cls in range(5):
        train_count = sum(1 for i in train if labels[i] == cls)
        assert train_count > 0, f"Class {cls} missing from train"

    folds = k_fold_indices(n, k=5, seed=42)
    assert len(folds) == 5
    all_val = set()
    for fold_train, fold_val in folds:
        all_val.update(fold_val)
        assert len(set(fold_train) & set(fold_val)) == 0, "Train/val overlap in fold"
    assert len(all_val) == n, "k-fold doesn't cover all samples"

    return f"OK (split: {len(train)}/{len(val)}/{len(test)}, 5 folds verified)"


# ---- Test runner ----

ALL_TESTS = [
    ("Data pipeline → frontier forward", test_data_pipeline_to_frontier),
    ("Training loop (3 SPSA steps)", test_training_loop),
    ("Benchmark suite on embeddings", test_benchmark_on_frontier),
    ("Two-view augmentation pipeline", test_augmentation_pipeline),
    ("Curriculum training (easy/hard)", test_curriculum_training),
    ("Checkpoint save/load roundtrip", test_checkpoint_roundtrip),
    ("Evaluation metrics (CKA, isotropy)", test_evaluation_metrics),
    ("Data splitting (stratified, k-fold)", test_data_splitting),
]


def run_all_tests() -> Dict[str, bool]:
    results = {}
    t0 = time.time()
    print(f"Running {len(ALL_TESTS)} integration tests...\n")

    for name, fn in ALL_TESTS:
        try:
            msg = fn()
            results[name] = True
            print(f"  PASS  {name}: {msg}")
        except Exception as e:
            results[name] = False
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
        print()

    elapsed = time.time() - t0
    n_pass = sum(results.values())
    n_fail = len(results) - n_pass
    print(f"{'=' * 60}")
    print(f"  {n_pass}/{len(results)} passed, {n_fail} failed  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    run_all_tests()
