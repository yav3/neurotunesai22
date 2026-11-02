"""Pure-numpy optimizers: SGD, Adam, AdamW, gradient clipping."""
from __future__ import annotations

import numpy as np


class GradientClipper:
    def __init__(self, max_norm: float | None = None, max_value: float | None = None) -> None:
        self.max_norm = max_norm
        self.max_value = max_value

    def clip(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.max_value is not None:
            grads = {k: np.clip(v, -self.max_value, self.max_value) for k, v in grads.items()}
        if self.max_norm is not None:
            total_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
            if total_norm > self.max_norm:
                scale = self.max_norm / (total_norm + 1e-12)
                grads = {k: v * scale for k, v in grads.items()}
        return grads


class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: dict[str, np.ndarray] = {}
        self._step_count = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._step_count += 1
        for name in grads:
            if name not in params:
                continue
            g = grads[name]
            if self.weight_decay > 0:
                g = g + self.weight_decay * params[name]
            if self.momentum > 0:
                if name not in self._velocity:
                    self._velocity[name] = np.zeros_like(g)
                self._velocity[name] = self.momentum * self._velocity[name] + g
                params[name] = params[name] - self.lr * self._velocity[name]
            else:
                params[name] = params[name] - self.lr * g
        return params

    def state_dict(self) -> dict:
        return {"velocity": {k: v.copy() for k, v in self._velocity.items()}, "step": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._velocity = {k: v.copy() for k, v in state.get("velocity", {}).items()}
        self._step_count = state.get("step", 0)


class Adam:
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._step_count = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._step_count += 1
        t = self._step_count
        for name in grads:
            if name not in params:
                continue
            g = grads[name]
            if name not in self._m:
                self._m[name] = np.zeros_like(g)
                self._v[name] = np.zeros_like(g)
            self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * g
            self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g ** 2
            m_hat = self._m[name] / (1 - self.beta1 ** t)
            v_hat = self._v[name] / (1 - self.beta2 ** t)
            params[name] = params[name] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params

    def state_dict(self) -> dict:
        return {
            "m": {k: v.copy() for k, v in self._m.items()},
            "v": {k: v.copy() for k, v in self._v.items()},
            "step": self._step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._m = {k: v.copy() for k, v in state.get("m", {}).items()}
        self._v = {k: v.copy() for k, v in state.get("v", {}).items()}
        self._step_count = state.get("step", 0)


class AdamW(Adam):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for name in grads:
            if name in params:
                params[name] = params[name] * (1 - self.lr * self.weight_decay)
        return super().step(params, grads)


# ---------------------------------------------------------------------------
# LAMB optimizer (You et al., 2020)
# ---------------------------------------------------------------------------

class LAMB:
    """Layer-wise Adaptive Moments for Batch training."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        trust_clip: float = 10.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.trust_clip = trust_clip
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._step_count = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._step_count += 1
        t = self._step_count
        for name in grads:
            if name not in params:
                continue
            g = grads[name]
            if name not in self._m:
                self._m[name] = np.zeros_like(g)
                self._v[name] = np.zeros_like(g)
            self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * g
            self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * g ** 2
            m_hat = self._m[name] / (1 - self.beta1 ** t)
            v_hat = self._v[name] / (1 - self.beta2 ** t)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                update = update + self.weight_decay * params[name]
            w_norm = float(np.linalg.norm(params[name]))
            u_norm = float(np.linalg.norm(update))
            trust = 1.0
            if w_norm > 0 and u_norm > 0:
                trust = min(w_norm / u_norm, self.trust_clip)
            params[name] = params[name] - self.lr * trust * update
        return params

    def state_dict(self) -> dict:
        return {"m": {k: v.copy() for k, v in self._m.items()},
                "v": {k: v.copy() for k, v in self._v.items()},
                "step": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._m = {k: v.copy() for k, v in state.get("m", {}).items()}
        self._v = {k: v.copy() for k, v in state.get("v", {}).items()}
        self._step_count = state.get("step", 0)


# ---------------------------------------------------------------------------
# Adafactor (Shazeer & Stern, 2018)
# ---------------------------------------------------------------------------

class Adafactor:
    """Memory-efficient optimizer: factored second moments for matrices."""

    def __init__(
        self,
        lr: float = 1e-3,
        eps: tuple[float, float] = (1e-30, 1e-3),
        decay_rate: float = -0.8,
        beta1: float = 0.0,
    ) -> None:
        self.lr = lr
        self.eps1 = eps[0]
        self.eps2 = eps[1]
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self._row_factor: dict[str, np.ndarray] = {}
        self._col_factor: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._m: dict[str, np.ndarray] = {}
        self._step_count = 0

    def _rms(self, x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2)))

    def _decay(self) -> float:
        return min(1 - (self._step_count + 1) ** self.decay_rate, 1.0 - self.eps1)

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._step_count += 1
        rho = self._decay()
        for name in grads:
            if name not in params:
                continue
            g = grads[name]
            p = params[name]
            if g.ndim >= 2:
                if name not in self._row_factor:
                    self._row_factor[name] = np.zeros(g.shape[0])
                    self._col_factor[name] = np.zeros(g.shape[1])
                g2 = g.reshape(g.shape[0], -1) ** 2
                self._row_factor[name] = rho * self._row_factor[name] + (1 - rho) * g2.mean(axis=1)
                self._col_factor[name] = rho * self._col_factor[name] + (1 - rho) * g2.mean(axis=0)
                rf = self._row_factor[name]
                cf = self._col_factor[name]
                mean_rf = rf.mean() + self.eps1
                v_est = np.outer(rf, cf) / mean_rf
                update = g / (np.sqrt(v_est.reshape(g.shape)) + self.eps1)
            else:
                if name not in self._v:
                    self._v[name] = np.zeros_like(g)
                self._v[name] = rho * self._v[name] + (1 - rho) * g ** 2
                update = g / (np.sqrt(self._v[name]) + self.eps1)
            rms_u = self._rms(update)
            update = update / max(rms_u / self.eps2, 1.0)
            if self.beta1 > 0:
                if name not in self._m:
                    self._m[name] = np.zeros_like(g)
                self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * update
                update = self._m[name]
            params[name] = p - self.lr * update
        return params


# ---------------------------------------------------------------------------
# Lion optimizer (Chen et al., 2024)
# ---------------------------------------------------------------------------

class Lion:
    """Evolved Sign Momentum: update = sign(β₁·m + (1−β₁)·g)."""

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self._m: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for name in grads:
            if name not in params:
                continue
            g = grads[name]
            if name not in self._m:
                self._m[name] = np.zeros_like(g)
            update = np.sign(self.beta1 * self._m[name] + (1 - self.beta1) * g)
            if self.weight_decay > 0:
                params[name] = params[name] * (1 - self.lr * self.weight_decay)
            params[name] = params[name] - self.lr * update
            self._m[name] = self.beta2 * self._m[name] + (1 - self.beta2) * g
        return params


# ---------------------------------------------------------------------------
# LR Schedulers
# ---------------------------------------------------------------------------

def cosine_annealing(step: int, total_steps: int, lr_max: float, lr_min: float = 0.0) -> float:
    if total_steps <= 0:
        return lr_max
    t = min(step / total_steps, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t))


def cosine_warmup(
    step: int, warmup_steps: int, total_steps: int,
    lr_max: float, lr_min: float = 0.0,
) -> float:
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    return cosine_annealing(step - warmup_steps, total_steps - warmup_steps, lr_max, lr_min)


def linear_warmup_decay(
    step: int, warmup_steps: int, total_steps: int, lr_max: float,
) -> float:
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    remaining = total_steps - warmup_steps
    if remaining <= 0:
        return 0.0
    return lr_max * max(1.0 - (step - warmup_steps) / remaining, 0.0)


def one_cycle(
    step: int, total_steps: int, lr_max: float,
    div_factor: float = 25.0, final_div: float = 1e4,
) -> float:
    lr_init = lr_max / div_factor
    lr_final = lr_max / final_div
    mid = total_steps // 2
    if step <= mid:
        t = step / max(mid, 1)
        return lr_init + 0.5 * (lr_max - lr_init) * (1 - np.cos(np.pi * t))
    else:
        t = (step - mid) / max(total_steps - mid, 1)
        return lr_final + 0.5 * (lr_max - lr_final) * (1 + np.cos(np.pi * t))


def inverse_sqrt(step: int, warmup_steps: int, lr_max: float) -> float:
    if step == 0:
        return 0.0
    return lr_max * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def exponential_decay(step: int, lr_init: float, decay_rate: float, decay_steps: int) -> float:
    return lr_init * decay_rate ** (step / max(decay_steps, 1))


# ---------------------------------------------------------------------------
# Exponential Moving Average of parameters
# ---------------------------------------------------------------------------

class ExponentialMovingAverage:
    """Shadow parameter copy for Polyak averaging."""

    def __init__(self, params: dict[str, np.ndarray], decay: float = 0.999) -> None:
        self.decay = decay
        self._shadow = {k: v.copy() for k, v in params.items()}
        self._backup: dict[str, np.ndarray] = {}

    def update(self, params: dict[str, np.ndarray]) -> None:
        for k in self._shadow:
            if k in params:
                self._shadow[k] = self.decay * self._shadow[k] + (1 - self.decay) * params[k]

    def get(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._shadow.items()}

    def apply(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._backup = {k: v.copy() for k, v in params.items()}
        for k in params:
            if k in self._shadow:
                params[k] = self._shadow[k].copy()
        return params

    def restore(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for k in params:
            if k in self._backup:
                params[k] = self._backup[k].copy()
        self._backup.clear()
        return params


# ---------------------------------------------------------------------------
# LR range test (Smith, 2017)
# ---------------------------------------------------------------------------

def lr_range_test(
    loss_fn,
    params: dict[str, np.ndarray],
    optimizer,
    lr_min: float = 1e-7,
    lr_max: float = 10.0,
    n_steps: int = 100,
) -> tuple[list[float], list[float], float]:
    """Sweep LR exponentially, return (lr_values, losses, suggested_lr)."""
    factor = (lr_max / lr_min) ** (1.0 / n_steps)
    lr_values, losses = [], []
    test_params = {k: v.copy() for k, v in params.items()}
    best_loss = float("inf")
    suggested_lr = lr_min
    for i in range(n_steps):
        lr = lr_min * factor ** i
        optimizer.lr = lr
        loss = loss_fn(test_params)
        lr_values.append(lr)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
        if loss > 4 * best_loss and i > 10:
            break
        grads = {k: np.zeros_like(v) for k, v in test_params.items()}
        for k in test_params:
            eps = 1e-5
            for idx in np.ndindex(test_params[k].shape):
                old = test_params[k][idx]
                test_params[k][idx] = old + eps
                lp = loss_fn(test_params)
                test_params[k][idx] = old - eps
                lm = loss_fn(test_params)
                test_params[k][idx] = old
                grads[k][idx] = (lp - lm) / (2 * eps)
                if np.random.random() > 0.1:
                    break
            break
        test_params = optimizer.step(test_params, grads)
    if len(losses) >= 3:
        smoothed = np.convolve(losses, np.ones(3) / 3, mode="valid")
        diffs = np.diff(smoothed)
        steepest = int(np.argmin(diffs))
        suggested_lr = lr_values[steepest + 1]
    return lr_values, losses, suggested_lr


# ---------------------------------------------------------------------------
# Parameter group optimizer
# ---------------------------------------------------------------------------

class ParameterGroupOptimizer:
    """Wraps a base optimizer class with per-group hyperparameters."""

    def __init__(
        self,
        groups: list[dict],
        base_class=None,
    ) -> None:
        if base_class is None:
            base_class = AdamW
        self.groups = groups
        self.optimizers = []
        for g in groups:
            kw = {k: v for k, v in g.items() if k != "params"}
            self.optimizers.append(base_class(**kw))
        self._param_to_group: dict[str, int] = {}
        for i, g in enumerate(groups):
            for name in g["params"]:
                self._param_to_group[name] = i

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        group_grads: list[dict[str, np.ndarray]] = [{} for _ in self.groups]
        group_params: list[dict[str, np.ndarray]] = [{} for _ in self.groups]
        for name in grads:
            gi = self._param_to_group.get(name, 0)
            group_grads[gi][name] = grads[name]
            group_params[gi][name] = params[name]
        for i, opt in enumerate(self.optimizers):
            if group_grads[i]:
                updated = opt.step(group_params[i], group_grads[i])
                params.update(updated)
        return params


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    rng = np.random.RandomState(0)
    params = {"w": rng.randn(4, 3), "b": rng.randn(3)}
    target = rng.randn(3)

    def loss_fn(p: dict[str, np.ndarray]) -> float:
        return float(np.sum((p["w"].mean(axis=0) + p["b"] - target) ** 2))

    grads = {"w": rng.randn(4, 3) * 0.1, "b": rng.randn(3) * 0.1}

    # LAMB
    lamb = LAMB()
    p = {k: v.copy() for k, v in params.items()}
    p = lamb.step(p, grads)
    assert p["w"].shape == (4, 3), "LAMB shape"

    # Adafactor
    af = Adafactor()
    p = {k: v.copy() for k, v in params.items()}
    p = af.step(p, grads)
    assert p["w"].shape == (4, 3), "Adafactor shape"

    # Lion
    lion = Lion()
    p = {k: v.copy() for k, v in params.items()}
    p = lion.step(p, grads)
    assert p["w"].shape == (4, 3), "Lion shape"

    # LR Schedulers
    assert cosine_annealing(0, 100, 1.0) == 1.0, "Cosine start"
    assert abs(cosine_annealing(100, 100, 1.0)) < 1e-10, "Cosine end"
    assert cosine_warmup(0, 10, 100, 1.0) == 0.0, "Warmup start"
    assert abs(cosine_warmup(10, 10, 100, 1.0) - 1.0) < 1e-10, "Warmup peak"
    assert linear_warmup_decay(50, 10, 100, 1.0) > 0, "Linear mid"
    assert one_cycle(0, 100, 1.0) < 1.0, "OneCycle start"
    assert inverse_sqrt(1000, 100, 1.0) > 0, "InvSqrt positive"
    assert exponential_decay(100, 1.0, 0.5, 100) < 1.0, "ExpDecay"

    # EMA
    ema = ExponentialMovingAverage(params, decay=0.9)
    new_params = {k: v + 1.0 for k, v in params.items()}
    ema.update(new_params)
    shadow = ema.get()
    assert not np.allclose(shadow["w"], params["w"]), "EMA should move"
    applied = ema.apply({k: v.copy() for k, v in new_params.items()})
    assert np.allclose(applied["w"], shadow["w"]), "EMA apply"

    # ParameterGroupOptimizer
    groups = [
        {"params": ["w"], "lr": 0.01, "weight_decay": 0.1},
        {"params": ["b"], "lr": 0.001, "weight_decay": 0.0},
    ]
    pgo = ParameterGroupOptimizer(groups)
    p = {k: v.copy() for k, v in params.items()}
    p = pgo.step(p, grads)
    assert p["w"].shape == (4, 3), "PGO shape"

    parts = [
        "LAMB", "Adafactor", "Lion", "CosineAnnealing", "CosineWarmup",
        "LinearWarmupDecay", "OneCycle", "InvSqrt", "ExpDecay",
        "EMA", "ParameterGroupOptimizer",
    ]
    print(f"Self-test passed: {', '.join(parts)}.")
