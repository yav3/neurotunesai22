"""Curriculum learning for music model training: difficulty scoring, pacing, self-paced weights."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class DifficultyMetric(Enum):
    LOSS = "loss"
    GRADIENT_NORM = "gradient_norm"
    PREDICTION_ENTROPY = "prediction_entropy"
    DATA_COMPLEXITY = "data_complexity"


class CurriculumStrategy(Enum):
    EASY_TO_HARD = "easy_to_hard"
    HARD_TO_EASY = "hard_to_easy"
    ANTI_CURRICULUM = "anti_curriculum"
    SELF_PACED = "self_paced"
    COMPETENCE_BASED = "competence_based"


@dataclass(frozen=True)
class SampleDifficulty:
    sample_id: str
    score: float
    metric: DifficultyMetric


def score_by_loss(losses: dict[str, float]) -> list[SampleDifficulty]:
    return sorted(
        [SampleDifficulty(sid, loss, DifficultyMetric.LOSS) for sid, loss in losses.items()],
        key=lambda s: s.score,
    )


def score_by_entropy(predictions: dict[str, np.ndarray]) -> list[SampleDifficulty]:
    results = []
    for sid, probs in predictions.items():
        p = np.clip(probs, 1e-10, 1.0)
        p = p / p.sum()
        entropy = float(-np.sum(p * np.log(p)))
        results.append(SampleDifficulty(sid, entropy, DifficultyMetric.PREDICTION_ENTROPY))
    return sorted(results, key=lambda s: s.score)


def score_by_complexity(features: dict[str, np.ndarray]) -> list[SampleDifficulty]:
    results = []
    for sid, spec in features.items():
        power = np.abs(spec) ** 2 + 1e-10
        geo_mean = np.exp(np.mean(np.log(power)))
        arith_mean = np.mean(power)
        flatness = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0
        results.append(SampleDifficulty(sid, flatness, DifficultyMetric.DATA_COMPLEXITY))
    return sorted(results, key=lambda s: s.score)


class CurriculumScheduler:
    def __init__(self, strategy: CurriculumStrategy = CurriculumStrategy.EASY_TO_HARD,
                 total_epochs: int = 100, warmup_fraction: float = 0.2):
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.warmup_fraction = warmup_fraction
        self._warmup_epochs = int(total_epochs * warmup_fraction)

    def competence(self, epoch: int) -> float:
        if epoch >= self._warmup_epochs:
            return 1.0
        return epoch / max(self._warmup_epochs, 1)

    def pacing_function(self, epoch: int) -> float:
        c = self.competence(epoch)
        if self.strategy == CurriculumStrategy.EASY_TO_HARD:
            return c
        elif self.strategy == CurriculumStrategy.HARD_TO_EASY:
            return 1.0 - c + c * c
        elif self.strategy == CurriculumStrategy.ANTI_CURRICULUM:
            return 1.0 - c
        return c

    def select_batch(self, difficulties: list[SampleDifficulty],
                     batch_size: int, epoch: int) -> list[str]:
        if not difficulties:
            return []
        threshold = self.pacing_function(epoch)
        max_score = max(d.score for d in difficulties)
        min_score = min(d.score for d in difficulties)
        score_range = max_score - min_score if max_score > min_score else 1.0

        if self.strategy in (CurriculumStrategy.EASY_TO_HARD,
                             CurriculumStrategy.COMPETENCE_BASED,
                             CurriculumStrategy.SELF_PACED):
            cutoff = min_score + threshold * score_range
            eligible = [d for d in difficulties if d.score <= cutoff]
        elif self.strategy == CurriculumStrategy.HARD_TO_EASY:
            cutoff = max_score - threshold * score_range
            eligible = [d for d in difficulties if d.score >= cutoff]
        else:
            cutoff = max_score - threshold * score_range
            eligible = [d for d in difficulties if d.score >= cutoff]

        if not eligible:
            eligible = difficulties
        rng = np.random.default_rng()
        indices = rng.choice(len(eligible), size=min(batch_size, len(eligible)), replace=False)
        return [eligible[i].sample_id for i in indices]


@dataclass
class SelfPacedWeight:
    lambda_sp: float = 1.0
    soft: bool = False
    gamma: float = 1.0

    def compute_weights(self, losses: np.ndarray) -> np.ndarray:
        if self.soft:
            return np.clip(1.0 - losses / (self.lambda_sp * self.gamma), 0.0, 1.0)
        return (losses < self.lambda_sp).astype(float)

    def update_lambda(self, factor: float = 1.1) -> None:
        self.lambda_sp *= factor
