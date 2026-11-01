#!/usr/bin/env python3
"""Run the foundational-frontier hybrid model end-to-end.

Usage:
    python scripts/run_hybrid.py                  # train + infer (mock frontier)
    python scripts/run_hybrid.py --frontier real   # train + infer (real Claude)
    python scripts/run_hybrid.py --infer-only PATH # load checkpoint, infer
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from models.foundation.gene_foundation import GeneFoundationConfig
from models.foundation.hybrid_model import HybridFoundationModel, HybridModelConfig
from models.foundation.frontier_bridge import QueryType
from models.foundation.frontier_caller import (
    ClaudeFrontierCaller,
    MockFrontierCaller,
)
from models.training.hybrid_trainer import (
    HybridTrainConfig,
    HybridTrainingPipeline,
)
from models.training.checkpoint import load_checkpoint
from models.inference.hybrid_inference import (
    HybridInferenceService,
    InferenceConfig,
)


def print_record(record):
    print(f"\n{'='*60}")
    print(f"  GOVERNED TRIAL: {record.study_id}")
    print(f"{'='*60}")
    print(f"  Samples:     {record.n_samples}")
    print(f"  Genes:       {record.n_genes}")
    print(f"  Parameters:  {record.model_params:,}")
    print(f"  Regime:      {record.regime}")
    print(f"  Species:     {record.species}")
    print(f"  Elapsed:     {record.elapsed_seconds:.1f}s")
    print(f"{'='*60}")
    print(f"  Train MSE:   {record.train_mse:.6f}   r={record.train_r:.4f}")
    print(f"  Val MSE:     {record.val_mse:.6f}   r={record.val_r:.4f}")
    print(f"  Test MSE:    {record.test_mse:.6f}   r={record.test_r:.4f}")
    print(f"  Baseline:    MSE={record.baseline_mse:.6f}   r={record.baseline_r:.4f}")
    print(f"{'='*60}")
    print(f"  Criteria:    {record.criteria_met}/{record.criteria_total}")
    print(f"  Verdict:     {record.verdict}")
    print(f"  Frontier:    {record.extra.get('frontier_used', False)}")
    print(f"  Hash:        {record.content_hash[:16]}...")
    if "checkpoint_path" in record.extra:
        print(f"  Checkpoint:  {record.extra['checkpoint_path']}")
    print(f"{'='*60}\n")


def print_answer(answer, elapsed: float):
    print(f"\n{'='*60}")
    print(f"  GOVERNED HYBRID ANSWER")
    print(f"{'='*60}")
    print(f"  Species:     {answer.species}")
    print(f"  Regime:      {answer.regime}")
    print(f"  Study:       {answer.study_id}")
    print(f"  Governed:    {answer.is_governed}")
    print(f"  Lineage:     {len(answer.lineage_chain)} hashes")
    print(f"  Hash:        {answer.content_hash[:16]}...")

    pred = answer.foundation_prediction
    if "confidence" in pred:
        print(f"  Confidence:  {pred['confidence']:.4f}")

    delta = pred.get("expression_delta", {})
    if delta:
        print(f"  Genes:       {delta.get('total_genes', '?')}")
        print(f"  Upregulated: {delta.get('n_upregulated', '?')}")
        print(f"  Downregulated: {delta.get('n_downregulated', '?')}")
        top = delta.get("top_genes", [])[:5]
        if top:
            print(f"  Top genes:")
            for g in top:
                print(f"    {g['gene']:>10s}  delta={g['delta']:+.4f}")

    if answer.frontier_reasoning is not None:
        fr = answer.frontier_reasoning
        print(f"\n  --- Frontier Reasoning ---")
        print(f"  Confidence:  {fr.confidence:.2f}")
        print(f"  Reasoning:   {fr.reasoning[:200]}")
        if fr.evidence_citations:
            print(f"  Citations:   {', '.join(fr.evidence_citations[:3])}")
        if fr.suggested_experiments:
            print(f"  Experiments: {fr.suggested_experiments[0]}")
        if fr.caveats:
            print(f"  Caveats:     {fr.caveats[0]}")
    else:
        print(f"\n  No frontier reasoning (foundation-only)")

    print(f"\n  Inference time: {elapsed:.2f}s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the foundational-frontier hybrid model"
    )
    parser.add_argument(
        "--frontier", choices=["mock", "real"], default="mock",
        help="Frontier LLM mode: mock (default) or real (needs ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--max-genes", type=int, default=2000,
        help="Maximum genes to use (default: 2000)",
    )
    parser.add_argument(
        "--study-id", default="NT-006",
        help="Study ID for the governed trial",
    )
    parser.add_argument(
        "--no-pretrain", action="store_true",
        help="Skip self-supervised pretraining",
    )
    parser.add_argument(
        "--infer-only", type=str, default=None,
        help="Path to checkpoint -- skip training, run inference only",
    )
    parser.add_argument(
        "--hypothesis", type=str,
        default="Perturbation alters gene regulatory network topology",
        help="Pre-registered hypothesis for frontier reasoning",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for saved checkpoints (default: checkpoints/)",
    )
    args = parser.parse_args()

    print(f"\n  Foundational-Frontier Hybrid Model")
    print(f"  {'='*40}")
    print(f"  Frontier:    {args.frontier}")
    print(f"  Max genes:   {args.max_genes}")
    print(f"  Study ID:    {args.study_id}")
    print(f"  Pretrain:    {not args.no_pretrain}")
    print()

    use_real = args.frontier == "real"
    if use_real:
        caller = ClaudeFrontierCaller()
    else:
        caller = MockFrontierCaller(confidence=0.75)

    if args.infer_only:
        ckpt_path = Path(args.infer_only)
        print(f"  Loading checkpoint: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = payload.get("config", {})
        gc_cfg = cfg.get("gene_config", {})

        gc = GeneFoundationConfig(
            max_genes=gc_cfg.get("max_genes", args.max_genes),
            hidden_dim=gc_cfg.get("hidden_dim", 256),
            num_layers=gc_cfg.get("num_layers", 6),
            num_heads=gc_cfg.get("num_heads", 8),
        )
        model_config = HybridModelConfig(
            gene_config=gc,
            use_frontier=True,
            prediction_hidden=cfg.get("prediction_hidden", 512),
            prediction_dropout=cfg.get("prediction_dropout", 0.1),
        )
        gene_names = [f"GENE{i}" for i in range(gc.max_genes)]

        inf_config = InferenceConfig(use_real_frontier=use_real)
        service = HybridInferenceService.from_checkpoint(
            ckpt_path,
            gene_names=gene_names,
            model_config=model_config,
            inference_config=inf_config,
        )

        expression = np.random.randn(gc.max_genes).astype(np.float32)
        t0 = time.time()
        answer = service.infer(
            expression,
            hypothesis=args.hypothesis,
            study_id=f"{args.study_id}-infer",
        )
        print_answer(answer, time.time() - t0)

    else:
        config = HybridTrainConfig(
            pretrain_epochs=50,
            finetune_epochs=30,
            pretrain_lr=1e-4,
            finetune_lr=5e-5,
            patience=8,
        )
        pipeline = HybridTrainingPipeline(
            config=config, frontier_caller=caller,
        )

        print("  Training via run_and_serve()...")
        record, service = pipeline.run_and_serve(
            study_id=args.study_id,
            max_genes=args.max_genes,
            species="human",
            pretrain_first=not args.no_pretrain,
            checkpoint_dir=args.checkpoint_dir,
        )
        print_record(record)

        pipeline2 = HybridTrainingPipeline(config=config)
        dataset = pipeline2.load_pooled_data(
            max_genes=args.max_genes, species="human",
        )
        sample_expr = np.array(dataset.pairs[0][0], dtype=np.float32)

        t0 = time.time()
        answer = service.infer(
            sample_expr,
            hypothesis=args.hypothesis,
            study_id=f"{args.study_id}-infer",
        )
        print_answer(answer, time.time() - t0)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"governed_answer_{ts}.json"

    pred = answer.foundation_prediction
    fr = answer.frontier_reasoning
    result_json = {
        "species": answer.species,
        "regime": answer.regime,
        "study_id": answer.study_id,
        "is_governed": answer.is_governed,
        "content_hash": answer.content_hash,
        "lineage_chain": answer.lineage_chain,
        "foundation_prediction": {
            k: v for k, v in pred.items()
            if k != "predicted_expression"
        },
        "frontier_reasoning": {
            "reasoning": fr.reasoning,
            "confidence": fr.confidence,
            "evidence_citations": fr.evidence_citations,
            "suggested_experiments": fr.suggested_experiments,
            "caveats": fr.caveats,
        } if fr else None,
    }
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    print(f"\n  Result saved: {result_path}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
