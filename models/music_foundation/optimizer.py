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
