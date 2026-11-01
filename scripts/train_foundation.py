#!/usr/bin/env python3
"""Train the music foundation model end-to-end.

Generates synthetic audio data across genres, trains the two-tower model
with the PretrainingObjective, evaluates retrieval/classification/embedding
metrics, and saves a checkpoint.

Usage:
    python -m scripts.train_foundation [--epochs 5] [--batch-size 16] [--output checkpoints/]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from models.music_foundation.encoder import MusicFoundationModel, EncoderConfig
from models.music_foundation.training import Trainer, TrainingConfig
from models.music_foundation.evaluation import (
    ModelEvaluator,
    evaluate_retrieval,
    evaluate_classification,
    evaluate_embeddings,
)
from models.music_foundation.inference import MusicInferencePipeline, InferenceConfig
from models.music_foundation.checkpoint import save_checkpoint


GENRE_PROFILES = {
    0: {"freq_center": 200, "bandwidth": 50, "rhythm_rate": 2.0},
    1: {"freq_center": 500, "bandwidth": 100, "rhythm_rate": 4.0},
    2: {"freq_center": 800, "bandwidth": 200, "rhythm_rate": 1.5},
    3: {"freq_center": 1200, "bandwidth": 300, "rhythm_rate": 3.0},
    4: {"freq_center": 350, "bandwidth": 80, "rhythm_rate": 5.0},
}


def generate_synthetic_audio(
    n_samples: int,
    n_genres: int = 5,
    sr: int = 22050,
    duration: float = 1.0,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    audios = []
    labels = []
    for i in range(n_samples):
        genre = i % n_genres
        profile = GENRE_PROFILES[genre]
        freq = profile["freq_center"] + rng.randn() * profile["bandwidth"]
        freq = max(50, freq)
        rate = profile["rhythm_rate"] + rng.randn() * 0.5

        signal = 0.3 * np.sin(2 * np.pi * freq * t)
        signal += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
        signal += 0.1 * np.sin(2 * np.pi * freq * 3 * t)

        envelope = 1.0 + 0.3 * np.sin(2 * np.pi * rate * t)
        signal *= envelope
        signal += rng.randn(len(t)) * 0.02
        signal = signal.astype(np.float32)

        audios.append(signal)
        labels.append(genre)

    return audios, np.array(labels)


def train_and_evaluate(args: argparse.Namespace) -> dict:
    print(f"Generating {args.n_train + args.n_eval} synthetic audio samples...")
    audios, labels = generate_synthetic_audio(
        args.n_train + args.n_eval,
        n_genres=args.n_genres,
        seed=args.seed,
    )

    train_audios = audios[: args.n_train]
    train_labels = labels[: args.n_train]
    eval_audios = audios[args.n_train :]
    eval_labels = labels[args.n_train :]

    enc_cfg = EncoderConfig(
        audio_dim=args.audio_dim,
        semantic_dim=args.semantic_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_heads=args.n_heads,
    )
    inf_cfg = InferenceConfig(
        encoder=enc_cfg,
        n_genres=args.n_genres,
        n_moods=4,
        enable_domain=False,
    )
    pipeline = MusicInferencePipeline(inf_cfg)

    print("Computing pre-training embeddings...")
    pre_results = pipeline.infer_batch(eval_audios, 22050)
    pre_embeddings = np.vstack([r.embedding.reshape(1, -1) for r in pre_results])
    pre_logits = np.vstack([r.genre_logits.reshape(1, -1) for r in pre_results])

    pre_retrieval = evaluate_retrieval(pre_embeddings, eval_labels)
    pre_classification = evaluate_classification(pre_logits, eval_labels)
    pre_embedding = evaluate_embeddings(pre_embeddings, eval_labels)
    print(f"  Pre-train MRR: {pre_retrieval.mrr:.4f}  Acc: {pre_classification.accuracy:.4f}  "
          f"Silhouette: {pre_embedding.silhouette:.4f}")

    model = pipeline.foundation

    print(f"\nExtracting features for training ({args.n_train} samples)...")
    from models.music_pipeline.feature_pipeline import FeaturePipeline
    fp = FeaturePipeline()
    train_features = []
    for audio in train_audios:
        feats = fp.extract(audio, 22050)
        vec = feats.to_vector()
        if len(vec) < enc_cfg.audio_dim:
            vec = np.pad(vec, (0, enc_cfg.audio_dim - len(vec)))
        else:
            vec = vec[: enc_cfg.audio_dim]
        train_features.append(vec)
    audio_data = np.vstack(train_features)

    rng = np.random.RandomState(args.seed + 1)
    semantic_data = rng.randn(args.n_train, enc_cfg.semantic_dim)
    for i in range(args.n_train):
        genre = train_labels[i]
        semantic_data[i, : min(5, enc_cfg.semantic_dim)] += genre * 0.5

    train_cfg = TrainingConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=max(1, args.n_train // args.batch_size),
        total_steps=args.epochs * (args.n_train // args.batch_size + 1),
        clip_norm=1.0,
        seed=args.seed,
    )
    trainer = Trainer(model, train_cfg)

    print(f"Training for {args.epochs} epochs (batch_size={args.batch_size}, lr={args.lr})...")
    t0 = time.time()
    metrics = trainer.fit(audio_data, semantic_data)
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Final loss: {metrics.loss_history[-1]:.4f}  "
          f"Best: {min(metrics.loss_history):.4f}  "
          f"Steps: {metrics.step}")

    print("\nComputing post-training embeddings...")
    post_results = pipeline.infer_batch(eval_audios, 22050)
    post_embeddings = np.vstack([r.embedding.reshape(1, -1) for r in post_results])
    post_logits = np.vstack([r.genre_logits.reshape(1, -1) for r in post_results])

    post_retrieval = evaluate_retrieval(post_embeddings, eval_labels)
    post_classification = evaluate_classification(post_logits, eval_labels)
    post_embedding = evaluate_embeddings(post_embeddings, eval_labels)
    print(f"  Post-train MRR: {post_retrieval.mrr:.4f}  Acc: {post_classification.accuracy:.4f}  "
          f"Silhouette: {post_embedding.silhouette:.4f}")

    emb_changed = not np.allclose(pre_embeddings, post_embeddings, atol=1e-6)
    print(f"\n  Embeddings changed after training: {emb_changed}")
    print(f"  MRR delta: {post_retrieval.mrr - pre_retrieval.mrr:+.4f}")
    print(f"  Accuracy delta: {post_classification.accuracy - pre_classification.accuracy:+.4f}")
    print(f"  Silhouette delta: {post_embedding.silhouette - pre_embedding.silhouette:+.4f}")

    if args.output:
        output_dir = Path(args.output)
        ckpt_path = save_checkpoint(model, output_dir / "foundation_model.json", metadata={
            "epochs": args.epochs,
            "final_loss": metrics.loss_history[-1],
            "best_loss": min(metrics.loss_history),
            "steps": metrics.step,
            "n_train": args.n_train,
            "post_mrr": post_retrieval.mrr,
            "post_accuracy": post_classification.accuracy,
            "post_silhouette": post_embedding.silhouette,
        })
        print(f"\nCheckpoint saved to {ckpt_path}")

    return {
        "pre": {"mrr": pre_retrieval.mrr, "acc": pre_classification.accuracy, "sil": pre_embedding.silhouette},
        "post": {"mrr": post_retrieval.mrr, "acc": post_classification.accuracy, "sil": post_embedding.silhouette},
        "training": metrics.summary(),
        "embeddings_changed": emb_changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train music foundation model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=25)
    parser.add_argument("--n-genres", type=int, default=5)
    parser.add_argument("--audio-dim", type=int, default=64)
    parser.add_argument("--semantic-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="checkpoints")
    args = parser.parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
