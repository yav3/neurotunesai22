"""Curriculum learning, data augmentation, and LR schedules for music model training."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


class DifficultyScorer:
    def score(self, losses: np.ndarray) -> float:
        if len(losses) == 0:
            return 0.0
        variance = float(np.var(losses))
        if len(losses) < 2:
            return variance
        x = np.arange(len(losses), dtype=float)
        slope = float(np.polyfit(x, losses, 1)[0])
        return variance + max(0, -slope)


class CurriculumScheduler:
    def __init__(self, n_samples: int, strategy: str = "linear"):
        self.n_samples = n_samples
        self.strategy = strategy

    def competence(self, epoch: int, max_epochs: int) -> float:
        if max_epochs <= 0:
            return 1.0
        t = epoch / max_epochs
        if self.strategy == "exponential":
            return float(min(1.0, 0.01 * np.exp(np.log(100) * t)))
        return float(min(1.0, 0.1 + 0.9 * t))

    def select_batch(self, difficulties: np.ndarray, competence: float, batch_size: int) -> np.ndarray:
        threshold = np.percentile(difficulties, competence * 100)
        eligible = np.where(difficulties <= threshold)[0]
        if len(eligible) == 0:
            eligible = np.arange(len(difficulties))
        rng = np.random.default_rng()
        chosen = rng.choice(eligible, size=min(batch_size, len(eligible)), replace=False)
        return chosen


class AntiCurriculumScheduler(CurriculumScheduler):
    def select_batch(self, difficulties: np.ndarray, competence: float, batch_size: int) -> np.ndarray:
        threshold = np.percentile(difficulties, (1 - competence) * 100)
        eligible = np.where(difficulties >= threshold)[0]
        if len(eligible) == 0:
            eligible = np.arange(len(difficulties))
        rng = np.random.default_rng()
        chosen = rng.choice(eligible, size=min(batch_size, len(eligible)), replace=False)
        return chosen


def self_paced_weight(loss: float, threshold: float, soft: bool = False) -> float:
    if soft:
        return float(1.0 / (1.0 + np.exp(5 * (loss - threshold))))
    return 1.0 if loss <= threshold else 0.0


class MixupAugmentor:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def mixup(self, x1: np.ndarray, x2: np.ndarray, alpha: float = 0.2) -> tuple[np.ndarray, float]:
        lam = float(self.rng.beta(alpha, alpha)) if alpha > 0 else 0.5
        mixed = lam * x1 + (1 - lam) * x2
        return mixed, lam

    def cutmix_1d(self, x1: np.ndarray, x2: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
        lam = float(self.rng.beta(alpha, alpha)) if alpha > 0 else 0.5
        n = len(x1)
        cut_len = int(n * (1 - lam))
        start = int(self.rng.integers(0, max(n - cut_len, 1)))
        result = x1.copy()
        result[start:start + cut_len] = x2[start:start + cut_len]
        actual_lam = 1.0 - cut_len / n
        return result, actual_lam

    def specaugment(
        self,
        spec: np.ndarray,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        freq_width: int = 10,
        time_width: int = 10,
    ) -> np.ndarray:
        result = spec.copy()
        n_freq, n_time = result.shape
        for _ in range(n_freq_masks):
            f = int(self.rng.integers(0, max(n_freq - freq_width, 1)))
            w = int(self.rng.integers(1, freq_width + 1))
            result[f:f + w, :] = 0.0
        for _ in range(n_time_masks):
            t = int(self.rng.integers(0, max(n_time - time_width, 1)))
            w = int(self.rng.integers(1, time_width + 1))
            result[:, t:t + w] = 0.0
        return result


class LearningRateSchedule:
    @staticmethod
    def warmup_cosine(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
        if step < warmup_steps:
            return base_lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        return float(min_lr + (base_lr - min_lr) * cosine_decay)

    @staticmethod
    def cyclic(step: int, cycle_length: int, base_lr: float, max_lr: float) -> float:
        cycle = step % cycle_length
        half = cycle_length / 2
        if cycle < half:
            return float(base_lr + (max_lr - base_lr) * cycle / half)
        return float(max_lr - (max_lr - base_lr) * (cycle - half) / half)

    @staticmethod
    def one_cycle(step: int, total_steps: int, max_lr: float, div_factor: float = 25.0, final_div: float = 1e4) -> float:
        initial_lr = max_lr / div_factor
        min_lr = initial_lr / final_div
        mid = total_steps * 0.3
        if step < mid:
            t = step / max(mid, 1)
            return float(initial_lr + (max_lr - initial_lr) * t)
        t = (step - mid) / max(total_steps - mid, 1)
        return float(max_lr + (min_lr - max_lr) * t)
