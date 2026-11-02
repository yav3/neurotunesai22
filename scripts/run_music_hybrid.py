#!/usr/bin/env python3
"""CLI for the Music Foundation–Frontier Hybrid Model.

Usage:
    # Full pipeline: pretrain + finetune + inference
    python scripts/run_music_hybrid.py

    # With real frontier (Claude API)
    python scripts/run_music_hybrid.py --frontier real

    # Inference only from checkpoint
    python scripts/run_music_hybrid.py --infer-only checkpoints/music/MUSIC-001_*.pt

    # Custom hypothesis
    python scripts/run_music_hybrid.py --hypothesis "High tempo tracks increase arousal"

    # Skip pretraining
    python scripts/run_music_hybrid.py --no-pretrain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from models.foundation.music_foundation import MusicFoundationConfig
from models.foundation.music_hybrid_model import MusicHybridModel, MusicHybridConfig
from models.foundation.music_frontier_bridge import (
    AUDIO_FEATURE_NAMES,
    MusicBridgeConfig,
    MusicQueryType,
    MockMusicFrontierCaller,
)
from models.training.music_trainer import (
    MusicTrainConfig,
    MusicTrainingPipeline,
)
from models.inference.music_inference import (
    MusicInferenceConfig,
    MusicInferenceService,
)
from models.data.music_dataset import generate_synthetic_music_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Music Foundation–Frontier Hybrid Model")
    p.add_argument("--frontier", choices=["mock", "real"], default="mock")
    p.add_argument("--n-features", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--study-id", default="MUSIC-001")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--infer-only", type=str, default=None,
                   help="Path to checkpoint for inference-only mode")
    p.add_argument("--hypothesis", default="Analyse the audio feature profile and predict mood shift")
    p.add_argument("--checkpoint-dir", default="checkpoints/music")
    p.add_argument("--query-type", default="mood_shift_hypothesis",
                   choices=[qt.value for qt in MusicQueryType])
    p.add_argument("--domain", default="music",
                   choices=["music", "therapeutic_audio"])
    p.add_argument("--regime", default="data-driven",
                   choices=["data-driven", "prior-contrast", "therapeutic"])
    return p.parse_args()


def run_training(args: argparse.Namespace) -> None:
    """Run the full training pipeline."""
    print(f"\n{'='*60}")
    print(f"  Music Foundation–Frontier Hybrid Model")
    print(f"  Study: {args.study_id}")
    print(f"  Frontier: {args.frontier}")
    print(f"  Features: {args.n_features}, Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    music_config = MusicFoundationConfig(
        max_features=max(args.n_features, 64),
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ffn_dim=512,
    )

    hybrid_config = MusicHybridConfig(
        music_config=music_config,
        use_frontier=(args.frontier == "real"),
    )

    train_config = MusicTrainConfig(
        pretrain_epochs=0 if args.no_pretrain else args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        n_synthetic_samples=args.n_samples,
        n_features=args.n_features,
        checkpoint_dir=args.checkpoint_dir,
    )

    pipeline = MusicTrainingPipeline(train_config, hybrid_config)

    print("[1/3] Training...")
    t0 = time.time()
    record, service = pipeline.run_and_serve(
        study_id=args.study_id,
        checkpoint_dir=args.checkpoint_dir,
    )
    train_time = time.time() - t0

    print(f"\n  Pretrain loss:  {record.pretrain_loss:.4f}")
    print(f"  Finetune loss:  {record.finetune_loss:.4f}")
    print(f"  Val VAD MSE:    {record.val_vad_mse:.4f}")
    print(f"  Val Mood MSE:   {record.val_mood_mse:.4f}")
    print(f"  Training time:  {train_time:.1f}s")
    print(f"  Content hash:   {record.content_hash[:16]}...")

    run_inference(service, args)


def run_inference_only(args: argparse.Namespace) -> None:
    """Load checkpoint and run inference."""
    print(f"\n{'='*60}")
    print(f"  Music Hybrid — Inference Only")
    print(f"  Checkpoint: {args.infer_only}")
    print(f"{'='*60}\n")

    infer_config = MusicInferenceConfig(
        use_real_frontier=(args.frontier == "real"),
        default_domain=args.domain,
        default_regime=args.regime,
    )

    service = MusicInferenceService.from_checkpoint(
        args.infer_only, config=infer_config,
    )

    run_inference(service, args)


def run_inference(service: MusicInferenceService, args: argparse.Namespace) -> None:
    """Run inference on a synthetic sample and display results."""
    print(f"\n[2/3] Running governed inference...")

    data = generate_synthetic_music_data(n_samples=5, n_features=args.n_features, seed=42)
    sample = data["features"][0]
    feature_names = data["feature_names"]

    query_type = MusicQueryType(args.query_type)

    answer = service.infer(
        features=sample,
        hypothesis=args.hypothesis,
        domain=args.domain,
        regime=args.regime,
        query_type=query_type,
    )

    print(f"\n[3/3] Results")
    print(f"{'─'*60}")
    print(f"  Domain:    {answer.domain}")
    print(f"  Regime:    {answer.regime}")
    print(f"  Governed:  {answer.is_governed}")
    print(f"  Hash:      {answer.content_hash[:16]}...")
    print(f"  Lineage:   {len(answer.lineage_chain)} entries")

    pred = answer.foundation_prediction
    if "predicted_vad" in pred:
        vad = pred["predicted_vad"]
        print(f"\n  Predicted VAD:")
        print(f"    Valence:   {vad['valence']:.3f}")
        print(f"    Arousal:   {vad['arousal']:.3f}")
        print(f"    Dominance: {vad['dominance']:.3f}")

    if "predicted_mood" in pred:
        print(f"\n  Predicted Mood:")
        for mood, score in pred["predicted_mood"].items():
            bar = "█" * int(score * 20)
            print(f"    {mood:12s} {score:.3f} {bar}")

    if "feature_delta" in pred:
        delta = pred["feature_delta"]
        print(f"\n  Top feature changes:")
        for f in delta.get("top_features", [])[:8]:
            print(f"    {f['feature']:25s} {f['delta']:+.3f}  "
                  f"({f['baseline']:.2f} → {f['predicted']:.2f})")

    if answer.frontier_reasoning:
        fr = answer.frontier_reasoning
        print(f"\n  Frontier Reasoning (confidence={fr.confidence:.2f}):")
        print(f"    {fr.reasoning}")
        if fr.suggested_tracks:
            print(f"\n  Suggested tracks:")
            for t in fr.suggested_tracks:
                print(f"    • {t}")
        if fr.mood_trajectory:
            print(f"\n  Mood trajectory:")
            for m in fr.mood_trajectory:
                print(f"    t={m['time_pct']:.1f}: V={m['valence']:.2f}, A={m['arousal']:.2f}")
        if fr.caveats:
            print(f"\n  Caveats:")
            for c in fr.caveats:
                print(f"    ⚠ {c}")

    params = service.model.param_count()
    print(f"\n  Model parameters:")
    for k, v in params.items():
        print(f"    {k:20s} {v:>10,}")

    print(f"\n{'='*60}")
    print(f"  Music Foundation–Frontier pipeline complete.")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    if args.infer_only:
        run_inference_only(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
