"""Numerical gradient computation for pure-numpy training."""
from __future__ import annotations

from typing import Callable
import numpy as np


def numerical_gradient(
    loss_fn: Callable[[dict[str, np.ndarray]], float],
    params: dict[str, np.ndarray],
    eps: float = 1e-5,
) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}
    for name, p in params.items():
        grad = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = p[idx]
            p[idx] = old_val + eps
            loss_plus = loss_fn(params)
            p[idx] = old_val - eps
            loss_minus = loss_fn(params)
            p[idx] = old_val
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        grads[name] = grad
    return grads


def check_gradient(
    analytical: dict[str, np.ndarray],
    numerical: dict[str, np.ndarray],
    rtol: float = 1e-4,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for name in analytical:
        if name not in numerical:
            continue
        a = analytical[name].flatten()
        n = numerical[name].flatten()
        denom = np.maximum(np.abs(a) + np.abs(n), 1e-12)
        rel_err = np.abs(a - n) / denom
        results[name] = float(np.max(rel_err))
    return results


class GradientAccumulator:
    def __init__(self) -> None:
        self._grads: dict[str, np.ndarray] = {}
        self._count = 0

    def accumulate(self, grads: dict[str, np.ndarray]) -> None:
        for name, g in grads.items():
            if name not in self._grads:
                self._grads[name] = np.zeros_like(g)
            self._grads[name] += g
        self._count += 1

    def get(self, normalize: bool = True) -> dict[str, np.ndarray]:
        if self._count == 0:
            return {}
        if normalize:
            return {k: v / self._count for k, v in self._grads.items()}
        return dict(self._grads)

    def reset(self) -> None:
        self._grads.clear()
        self._count = 0

    @property
    def count(self) -> int:
        return self._count
