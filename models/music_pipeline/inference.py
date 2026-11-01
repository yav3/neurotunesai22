"""Music model inference pipeline: batch/streaming prediction, model registry.

Implements the inference serving layer for music models including
batch prediction, streaming prediction, model versioning, and
ensemble methods for production deployment.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np


class ModelType(Enum):
    GENRE_CLASSIFIER = "genre_classifier"
    MOOD_REGRESSOR = "mood_regressor"
    KEY_DETECTOR = "key_detector"
    TEMPO_ESTIMATOR = "tempo_estimator"
    INSTRUMENT_RECOGNIZER = "instrument_recognizer"
    EMBEDDING_MODEL = "embedding_model"
    SEPARATOR = "separator"
    TRANSCRIBER = "transcriber"


class PredictionStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ModelVersion:
    model_type: ModelType
    version: str
    created_at: float
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return f"{self.model_type.value}:{self.version}"

    def content_hash(self) -> str:
        s = f"{self.model_type.value}:{self.version}:{self.input_shape}:{self.output_shape}"
        return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class PredictionResult:
    status: PredictionStatus
    model_id: str
    output: np.ndarray | None
    latency_ms: float
    confidence: float = 0.0
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceMetrics:
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.successful == 0:
            return 0.0
        return self.total_latency_ms / self.successful

    def record(self, result: PredictionResult) -> None:
        self.total_requests += 1
        if result.status == PredictionStatus.SUCCESS:
            self.successful += 1
            self.total_latency_ms += result.latency_ms
            self.max_latency_ms = max(self.max_latency_ms, result.latency_ms)
        else:
            self.failed += 1


class ModelRegistry:
    """Registry of available model versions."""

    def __init__(self) -> None:
        self._models: dict[str, ModelVersion] = {}
        self._active: dict[ModelType, str] = {}

    def register(self, version: ModelVersion) -> None:
        self._models[version.model_id] = version
        self._active[version.model_type] = version.model_id

    def get(self, model_id: str) -> ModelVersion | None:
        return self._models.get(model_id)

    def get_active(self, model_type: ModelType) -> ModelVersion | None:
        mid = self._active.get(model_type)
        return self._models.get(mid) if mid else None

    def set_active(self, model_type: ModelType, version: str) -> bool:
        mid = f"{model_type.value}:{version}"
        if mid in self._models:
            self._active[model_type] = mid
            return True
        return False

    def list_versions(self, model_type: ModelType) -> list[ModelVersion]:
        return [v for v in self._models.values() if v.model_type == model_type]

    @property
    def n_models(self) -> int:
        return len(self._models)


PredictFn = Callable[[np.ndarray], np.ndarray]


def predict_with_timeout(
    fn: PredictFn,
    input_data: np.ndarray,
    model_id: str,
    timeout_ms: float = 5000.0,
) -> PredictionResult:
    """Run a prediction function with timing."""
    t0 = time.monotonic()
    try:
        output = fn(input_data)
        latency = (time.monotonic() - t0) * 1000
        if latency > timeout_ms:
            return PredictionResult(
                status=PredictionStatus.TIMEOUT,
                model_id=model_id,
                output=output,
                latency_ms=latency,
            )
        confidence = float(np.max(output)) if output.ndim >= 1 else 0.0
        return PredictionResult(
            status=PredictionStatus.SUCCESS,
            model_id=model_id,
            output=output,
            latency_ms=latency,
            confidence=confidence,
        )
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return PredictionResult(
            status=PredictionStatus.ERROR,
            model_id=model_id,
            output=None,
            latency_ms=latency,
            metadata={"error": str(e)},
        )


def batch_predict(
    fn: PredictFn,
    inputs: list[np.ndarray],
    model_id: str,
    batch_size: int = 32,
) -> list[PredictionResult]:
    """Run batch prediction with configurable batch size."""
    results: list[PredictionResult] = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_array = np.stack(batch)
        t0 = time.monotonic()
        try:
            outputs = fn(batch_array)
            latency = (time.monotonic() - t0) * 1000 / len(batch)
            for j, output in enumerate(outputs):
                confidence = float(np.max(output)) if output.ndim >= 1 else 0.0
                results.append(PredictionResult(
                    status=PredictionStatus.SUCCESS,
                    model_id=model_id,
                    output=output,
                    latency_ms=latency,
                    confidence=confidence,
                ))
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000 / len(batch)
            for _ in batch:
                results.append(PredictionResult(
                    status=PredictionStatus.ERROR,
                    model_id=model_id,
                    output=None,
                    latency_ms=latency,
                    metadata={"error": str(e)},
                ))
    return results


def ensemble_predict(
    fns: list[PredictFn],
    input_data: np.ndarray,
    model_ids: list[str],
    weights: np.ndarray | None = None,
    strategy: str = "mean",
) -> PredictionResult:
    """Ensemble prediction from multiple models.

    strategy: "mean" (weighted average), "vote" (majority vote), "max" (max confidence).
    """
    if weights is None:
        weights = np.ones(len(fns)) / len(fns)
    outputs: list[np.ndarray] = []
    total_latency = 0.0
    for fn, mid in zip(fns, model_ids):
        result = predict_with_timeout(fn, input_data, mid)
        if result.status == PredictionStatus.SUCCESS and result.output is not None:
            outputs.append(result.output)
            total_latency += result.latency_ms
    if not outputs:
        return PredictionResult(
            status=PredictionStatus.ERROR,
            model_id="ensemble",
            output=None,
            latency_ms=total_latency,
        )
    if strategy == "mean":
        stacked = np.stack(outputs)
        w = weights[:len(outputs), np.newaxis] if stacked.ndim > 1 else weights[:len(outputs)]
        w = w / w.sum()
        combined = np.sum(stacked * w.reshape(-1, *([1] * (stacked.ndim - 1))), axis=0)
    elif strategy == "vote":
        votes = np.array([np.argmax(o) for o in outputs])
        combined = np.zeros_like(outputs[0])
        for v, w in zip(votes, weights[:len(outputs)]):
            combined[v] += w
    elif strategy == "max":
        confidences = [float(np.max(o)) for o in outputs]
        best_idx = int(np.argmax(confidences))
        combined = outputs[best_idx]
    else:
        combined = np.mean(np.stack(outputs), axis=0)
    confidence = float(np.max(combined)) if combined.ndim >= 1 else 0.0
    return PredictionResult(
        status=PredictionStatus.SUCCESS,
        model_id="ensemble",
        output=combined,
        latency_ms=total_latency,
        confidence=confidence,
    )


@dataclass
class StreamingPredictor:
    """Sliding-window streaming predictor for real-time audio."""
    predict_fn: PredictFn
    model_id: str
    window_size: int
    hop_size: int
    _buffer: np.ndarray = field(init=False)
    _results: list[PredictionResult] = field(default_factory=list)
    _position: int = 0

    def __post_init__(self) -> None:
        self._buffer = np.zeros(self.window_size)

    def feed(self, chunk: np.ndarray) -> list[PredictionResult]:
        new_results: list[PredictionResult] = []
        for sample in chunk:
            self._buffer[self._position % self.window_size] = sample
            self._position += 1
            if self._position >= self.window_size and (self._position - self.window_size) % self.hop_size == 0:
                start = (self._position - self.window_size) % self.window_size
                window = np.roll(self._buffer, -start)
                result = predict_with_timeout(self.predict_fn, window, self.model_id)
                new_results.append(result)
                self._results.append(result)
        return new_results

    @property
    def total_predictions(self) -> int:
        return len(self._results)
