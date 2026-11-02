"""Learning rate schedulers: cosine annealing, warmup, one-cycle."""
from __future__ import annotations

import math


class CosineAnnealingScheduler:
    def __init__(self, base_lr: float = 1e-3, min_lr: float = 1e-6, total_steps: int = 1000) -> None:
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = max(total_steps, 1)

    def get_lr(self, step: int) -> float:
        t = min(step / self.total_steps, 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))

    def schedule(self) -> list[float]:
        return [self.get_lr(s) for s in range(self.total_steps)]


class WarmupScheduler:
    def __init__(self, base_lr: float = 1e-3, warmup_steps: int = 100, total_steps: int = 1000) -> None:
        self.base_lr = base_lr
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, 1)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        remaining = self.total_steps - self.warmup_steps
        if remaining <= 0:
            return self.base_lr
        decay = 1.0 - (step - self.warmup_steps) / remaining
        return self.base_lr * max(decay, 0.0)

    def schedule(self) -> list[float]:
        return [self.get_lr(s) for s in range(self.total_steps)]


class OneCycleScheduler:
    def __init__(
        self,
        max_lr: float = 1e-2,
        total_steps: int = 1000,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ) -> None:
        self.max_lr = max_lr
        self.total_steps = max(total_steps, 1)
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor
        self.up_steps = int(total_steps * pct_start)
        self.down_steps = total_steps - self.up_steps

    def get_lr(self, step: int) -> float:
        if step < self.up_steps:
            t = step / max(self.up_steps, 1)
            return self.initial_lr + (self.max_lr - self.initial_lr) * 0.5 * (1 - math.cos(math.pi * t))
        else:
            t = (step - self.up_steps) / max(self.down_steps, 1)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * t))

    def schedule(self) -> list[float]:
        return [self.get_lr(s) for s in range(self.total_steps)]
