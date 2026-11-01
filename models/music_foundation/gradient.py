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


# ---------------------------------------------------------------------------
# SPSA gradient estimator (Spall, 1992)
# ---------------------------------------------------------------------------

def spsa_gradient(
    loss_fn: Callable[[dict[str, np.ndarray]], float],
    params: dict[str, np.ndarray],
    eps: float = 1e-4,
    rng: np.random.RandomState | None = None,
) -> dict[str, np.ndarray]:
    """Simultaneous Perturbation Stochastic Approximation.

    Two forward passes regardless of parameter count.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    deltas = {k: rng.choice([-1.0, 1.0], size=v.shape) for k, v in params.items()}
    plus = {k: params[k] + eps * deltas[k] for k in params}
    minus = {k: params[k] - eps * deltas[k] for k in params}
    loss_plus = loss_fn(plus)
    loss_minus = loss_fn(minus)
    diff = loss_plus - loss_minus
    return {k: diff / (2 * eps * deltas[k]) for k in params}


# ---------------------------------------------------------------------------
# Natural Evolution Strategies (Salimans et al., 2017)
# ---------------------------------------------------------------------------

def nes_gradient(
    loss_fn: Callable[[dict[str, np.ndarray]], float],
    params: dict[str, np.ndarray],
    sigma: float = 0.02,
    n_samples: int = 20,
    rng: np.random.RandomState | None = None,
) -> dict[str, np.ndarray]:
    """Antithetic NES gradient estimate via Gaussian smoothing."""
    if rng is None:
        rng = np.random.RandomState(42)
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    for _ in range(n_samples // 2):
        noise = {k: rng.randn(*v.shape) for k, v in params.items()}
        plus = {k: params[k] + sigma * noise[k] for k in params}
        minus = {k: params[k] - sigma * noise[k] for k in params}
        fp = loss_fn(plus)
        fm = loss_fn(minus)
        for k in grads:
            grads[k] += (fp - fm) * noise[k]
    for k in grads:
        grads[k] /= (n_samples * sigma)
    return grads


# ---------------------------------------------------------------------------
# Gradient noise injection (Neelakantan et al., 2015)
# ---------------------------------------------------------------------------

def add_gradient_noise(
    grads: dict[str, np.ndarray],
    step: int,
    eta: float = 0.01,
    gamma: float = 0.55,
    rng: np.random.RandomState | None = None,
) -> dict[str, np.ndarray]:
    """Annealed Gaussian noise: σ² = η / (1 + t)^γ."""
    if rng is None:
        rng = np.random.RandomState(step)
    variance = eta / (1.0 + step) ** gamma
    std = np.sqrt(variance)
    return {k: v + std * rng.randn(*v.shape) for k, v in grads.items()}


# ---------------------------------------------------------------------------
# Gradient centralization (Yong et al., 2020)
# ---------------------------------------------------------------------------

def centralize_gradients(grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Subtract per-tensor mean from gradients (skip 1-D / bias terms)."""
    result = {}
    for k, g in grads.items():
        if g.ndim <= 1:
            result[k] = g
        else:
            result[k] = g - g.mean(axis=tuple(range(1, g.ndim)), keepdims=True)
    return result


# ---------------------------------------------------------------------------
# Gradient variance tracker (Welford's online algorithm)
# ---------------------------------------------------------------------------

class GradientVarianceTracker:
    """Tracks running mean and variance of gradients per parameter."""

    def __init__(self) -> None:
        self._mean: dict[str, np.ndarray] = {}
        self._m2: dict[str, np.ndarray] = {}
        self._count = 0

    def update(self, grads: dict[str, np.ndarray]) -> None:
        self._count += 1
        for k, g in grads.items():
            if k not in self._mean:
                self._mean[k] = np.zeros_like(g)
                self._m2[k] = np.zeros_like(g)
            delta = g - self._mean[k]
            self._mean[k] += delta / self._count
            delta2 = g - self._mean[k]
            self._m2[k] += delta * delta2

    def variance(self) -> dict[str, np.ndarray]:
        if self._count < 2:
            return {k: np.zeros_like(v) for k, v in self._mean.items()}
        return {k: self._m2[k] / (self._count - 1) for k in self._m2}

    def signal_to_noise(self) -> dict[str, float]:
        var = self.variance()
        return {
            k: float(np.abs(self._mean[k]).mean() / (np.sqrt(var[k]).mean() + 1e-12))
            for k in self._mean
        }

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Lookahead gradient (Zhang et al., 2019)
# ---------------------------------------------------------------------------

class LookaheadGradient:
    """Slow–fast weight interpolation every k steps."""

    def __init__(self, slow_step_size: float = 0.5, k: int = 5) -> None:
        self.alpha = slow_step_size
        self.k = k
        self._slow: dict[str, np.ndarray] | None = None
        self._step = 0

    def step(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._slow is None:
            self._slow = {k: v.copy() for k, v in params.items()}
        self._step += 1
        if self._step % self.k == 0:
            for k in params:
                if k in self._slow:
                    self._slow[k] = self._slow[k] + self.alpha * (params[k] - self._slow[k])
                    params[k] = self._slow[k].copy()
        return params


# ---------------------------------------------------------------------------
# Gradient projection to subspace
# ---------------------------------------------------------------------------

def project_gradient_to_subspace(
    grads: dict[str, np.ndarray],
    basis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project each gradient tensor onto a column subspace.

    basis: (d, r) orthonormal matrix. Gradient is flattened, projected,
    then reshaped. Useful for continual-learning constraints.
    """
    Q = basis
    result = {}
    for k, g in grads.items():
        shape = g.shape
        g_flat = g.flatten()
        if g_flat.shape[0] != Q.shape[0]:
            result[k] = g
            continue
        proj = Q @ (Q.T @ g_flat)
        result[k] = proj.reshape(shape)
    return result


# ---------------------------------------------------------------------------
# Empirical Fisher Information diagonal
# ---------------------------------------------------------------------------

def empirical_fisher_diagonal(
    loss_fn: Callable[[dict[str, np.ndarray]], float],
    params: dict[str, np.ndarray],
    data_samples: list,
    eps: float = 1e-5,
    rng: np.random.RandomState | None = None,
) -> dict[str, np.ndarray]:
    """Diagonal of the empirical FIM via squared SPSA gradients."""
    if rng is None:
        rng = np.random.RandomState(42)
    fisher = {k: np.zeros_like(v) for k, v in params.items()}
    for _ in data_samples:
        g = spsa_gradient(loss_fn, params, eps, rng)
        for k in fisher:
            fisher[k] += g[k] ** 2
    n = max(len(data_samples), 1)
    return {k: v / n for k, v in fisher.items()}


# ---------------------------------------------------------------------------
# PCGrad — Projecting Conflicting Gradients (Yu et al., 2020)
# ---------------------------------------------------------------------------

def pcgrad(grads_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Gradient surgery for multi-task learning.

    For each task gradient g_i, project out components conflicting
    with other tasks' gradients.
    """
    n_tasks = len(grads_list)
    if n_tasks == 0:
        return {}
    keys = list(grads_list[0].keys())
    projected = [{k: g[k].copy() for k in keys} for g in grads_list]

    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                continue
            for k in keys:
                gi = projected[i][k].flatten()
                gj = grads_list[j][k].flatten()
                dot = np.dot(gi, gj)
                if dot < 0:
                    proj = dot / (np.dot(gj, gj) + 1e-12) * gj
                    projected[i][k] = (gi - proj).reshape(projected[i][k].shape)

    result = {k: np.zeros_like(grads_list[0][k]) for k in keys}
    for i in range(n_tasks):
        for k in keys:
            result[k] += projected[i][k]
    return {k: v / n_tasks for k, v in result.items()}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    rng = np.random.RandomState(0)
    params = {"w": rng.randn(4, 3), "b": rng.randn(3)}
    target = rng.randn(3)

    def loss_fn(p: dict[str, np.ndarray]) -> float:
        return float(np.sum((p["w"].mean(axis=0) + p["b"] - target) ** 2))

    # SPSA
    g = spsa_gradient(loss_fn, params, rng=rng)
    assert set(g.keys()) == {"w", "b"}, "SPSA keys"
    assert g["w"].shape == (4, 3), "SPSA w shape"

    # NES
    g = nes_gradient(loss_fn, params, n_samples=10, rng=rng)
    assert g["b"].shape == (3,), "NES b shape"

    # Gradient noise
    grads = {"w": np.ones((4, 3)), "b": np.ones(3)}
    noisy = add_gradient_noise(grads, step=5)
    assert not np.allclose(noisy["w"], grads["w"]), "Noise should modify"

    # Centralize
    centered = centralize_gradients(grads)
    assert centered["b"].shape == (3,), "Bias unchanged"
    # 2-D gradient should have zero mean per row
    assert np.allclose(centered["w"].mean(axis=1), 0.0, atol=1e-12), "Centralized mean"

    # Variance tracker
    tracker = GradientVarianceTracker()
    for _ in range(20):
        tracker.update({"w": rng.randn(4, 3)})
    snr = tracker.signal_to_noise()
    assert "w" in snr, "SNR key"
    assert snr["w"] >= 0, "SNR non-negative"

    # Lookahead
    la = LookaheadGradient(k=3)
    p = {"w": np.ones((2, 2))}
    for i in range(6):
        p["w"] = p["w"] + 0.1
        p = la.step(p)
    assert p["w"].shape == (2, 2), "Lookahead shape"

    # Projection
    basis = np.eye(12, 2)  # project onto first 2 dims
    g = {"w": rng.randn(4, 3)}
    pg = project_gradient_to_subspace(g, basis)
    flat = pg["w"].flatten()
    assert np.allclose(flat[2:], 0.0), "Projected out higher dims"

    # Fisher
    f = empirical_fisher_diagonal(loss_fn, params, [None, None, None], rng=rng)
    assert np.all(f["w"] >= 0), "Fisher non-negative"

    # PCGrad
    g1 = {"w": np.array([1.0, -1.0])}
    g2 = {"w": np.array([-1.0, -1.0])}
    pc = pcgrad([g1, g2])
    assert pc["w"].shape == (2,), "PCGrad shape"

    parts = [
        "SPSA", "NES", "GradNoise", "Centralize", "VarianceTracker",
        "Lookahead", "Projection", "Fisher", "PCGrad",
    ]
    print(f"Self-test passed: {', '.join(parts)}.")
