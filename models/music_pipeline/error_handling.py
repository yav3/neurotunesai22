"""Pipeline error handling: retry, circuit breaker, fallback strategies.

Pure numpy (no I/O — models error-handling logic only).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any

import numpy as np


class ErrorSeverity(Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class PipelineError:
    stage: str
    message: str
    severity: ErrorSeverity
    timestamp: float = 0.0
    retryable: bool = True


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0

    def record_success(self) -> None:
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True


@dataclass
class HealthMetrics:
    total_calls: int = 0
    total_failures: int = 0
    total_retries: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    uptime_ratio: float = 1.0


def classify_error(error_msg: str) -> ErrorSeverity:
    """Classify an error as transient, permanent, or degraded."""
    transient_keywords = {"timeout", "temporary", "retry", "unavailable", "connection", "reset"}
    permanent_keywords = {"invalid", "corrupt", "format", "schema", "permission", "denied"}
    degraded_keywords = {"partial", "incomplete", "truncated", "fallback"}

    lower = error_msg.lower()
    for kw in permanent_keywords:
        if kw in lower:
            return ErrorSeverity.PERMANENT
    for kw in transient_keywords:
        if kw in lower:
            return ErrorSeverity.TRANSIENT
    for kw in degraded_keywords:
        if kw in lower:
            return ErrorSeverity.DEGRADED
    return ErrorSeverity.UNKNOWN


def compute_backoff_delay(attempt: int, config: RetryConfig, rng_seed: int = 42) -> float:
    """Exponential backoff with optional jitter."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)
    if config.jitter:
        rng = np.random.RandomState(rng_seed + attempt)
        delay *= (0.5 + rng.random())
    return delay


def should_retry(error: PipelineError, attempt: int, config: RetryConfig) -> bool:
    """Decide whether to retry based on error type and attempt count."""
    if not error.retryable:
        return False
    if error.severity == ErrorSeverity.PERMANENT:
        return False
    return attempt < config.max_retries


def update_health_metrics(
    metrics: HealthMetrics, success: bool, latency_ms: float,
) -> HealthMetrics:
    """Update pipeline health metrics after a call."""
    metrics.total_calls += 1
    if not success:
        metrics.total_failures += 1
    n = metrics.total_calls
    metrics.avg_latency_ms = ((n - 1) * metrics.avg_latency_ms + latency_ms) / n
    metrics.error_rate = metrics.total_failures / n
    metrics.uptime_ratio = 1.0 - metrics.error_rate
    return metrics


def fallback_chain(
    primary_result: Any | None, fallbacks: list[Callable[[], Any]],
) -> Any | None:
    """Try primary result, then fallback callables in order."""
    if primary_result is not None:
        return primary_result
    for fb in fallbacks:
        try:
            result = fb()
            if result is not None:
                return result
        except Exception:
            continue
    return None
