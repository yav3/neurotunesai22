"""Batch processing pipeline for music collections.

Sequential multi-stage pipeline for processing music file collections
with timing, retry, and progress reporting.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class ProcessingStage(enum.Enum):
    LOAD = "load"
    PREPROCESS = "preprocess"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    EXPORT = "export"


class ProgressCallback(Protocol):
    def __call__(self, completed: int, total: int, stage: ProcessingStage) -> None: ...


@dataclass
class BatchConfig:
    stages: list[ProcessingStage] = field(default_factory=lambda: list(ProcessingStage))
    n_workers: int = 1
    chunk_size: int = 16
    output_format: str = "json"
    timeout_per_track: float = 60.0


@dataclass
class TrackResult:
    path: str
    success: bool
    stage_timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    output: dict | None = None


@dataclass
class BatchResult:
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    timing_per_stage: dict[str, float] = field(default_factory=dict)
    results: list[TrackResult] = field(default_factory=list)


def validate_batch_config(config: BatchConfig) -> list[str]:
    errors: list[str] = []
    if config.n_workers < 1:
        errors.append("n_workers must be >= 1")
    if config.chunk_size < 1:
        errors.append("chunk_size must be >= 1")
    if not config.stages:
        errors.append("at least one stage required")
    valid_order = list(ProcessingStage)
    prev_idx = -1
    for s in config.stages:
        idx = valid_order.index(s)
        if idx <= prev_idx:
            errors.append(f"stage {s.value} out of order")
        prev_idx = idx
    return errors


def _simulate_stage(stage: ProcessingStage, data: dict) -> dict:
    if stage == ProcessingStage.LOAD:
        data["loaded"] = True
        data["samples"] = int(np.random.uniform(22050, 22050 * 300))
    elif stage == ProcessingStage.PREPROCESS:
        data["preprocessed"] = True
        data["sample_rate"] = 22050
    elif stage == ProcessingStage.EXTRACT:
        data["features"] = {"mfcc_mean": float(np.random.randn()), "rms": float(np.random.uniform(0, 1))}
    elif stage == ProcessingStage.ANALYZE:
        data["analysis"] = {"tempo": float(np.random.uniform(60, 200)), "key": "C major"}
    elif stage == ProcessingStage.EXPORT:
        data["exported"] = True
    return data


def process_single(
    path: str,
    config: BatchConfig,
    stage_fn: Callable[[ProcessingStage, dict], dict] | None = None,
) -> TrackResult:
    fn = stage_fn or _simulate_stage
    data: dict = {"path": path}
    timings: dict[str, float] = {}
    for stage in config.stages:
        t0 = time.monotonic()
        try:
            data = fn(stage, data)
        except Exception as exc:
            return TrackResult(path=path, success=False, stage_timings=timings, error=f"{stage.value}: {exc}")
        timings[stage.value] = time.monotonic() - t0
    return TrackResult(path=path, success=True, stage_timings=timings, output=data)


def process_batch(
    paths: list[str],
    config: BatchConfig,
    stage_fn: Callable[[ProcessingStage, dict], dict] | None = None,
    progress: ProgressCallback | None = None,
) -> BatchResult:
    result = BatchResult()
    stage_totals: dict[str, float] = {s.value: 0.0 for s in config.stages}
    for i, path in enumerate(paths):
        tr = process_single(path, config, stage_fn)
        result.results.append(tr)
        if tr.success:
            result.processed += 1
        else:
            result.failed += 1
        for k, v in tr.stage_timings.items():
            stage_totals[k] = stage_totals.get(k, 0.0) + v
        if progress and config.stages:
            progress(i + 1, len(paths), config.stages[-1])
    result.timing_per_stage = stage_totals
    return result


def estimate_time(
    n_tracks: int,
    avg_seconds_per_track: float = 2.0,
    n_workers: int = 1,
) -> float:
    return (n_tracks * avg_seconds_per_track) / max(n_workers, 1)


def retry_failed(
    batch_result: BatchResult,
    config: BatchConfig,
    max_retries: int = 3,
    stage_fn: Callable[[ProcessingStage, dict], dict] | None = None,
) -> BatchResult:
    failed_paths = [r.path for r in batch_result.results if not r.success]
    if not failed_paths:
        return batch_result
    for attempt in range(max_retries):
        retry_result = process_batch(failed_paths, config, stage_fn)
        still_failed = [r.path for r in retry_result.results if not r.success]
        for r in retry_result.results:
            if r.success:
                batch_result.processed += 1
                batch_result.failed -= 1
        if not still_failed:
            break
        failed_paths = still_failed
    return batch_result
