"""Hierarchical structure analysis: multi-level decomposition and periodicity."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.music_structure.segmentation import (
    SelfSimilarityMatrix,
    novelty_curve,
    segment_boundaries,
)


@dataclass
class PeriodicPattern:
    period: int
    strength: float
    start_offset: int


@dataclass
class StructureLevel:
    level: int
    boundaries: list[int]
    kernel_size: int


@dataclass
class StructureTree:
    levels: list[StructureLevel]

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def boundaries_at_level(self, level: int) -> list[int]:
        if level < 0 or level >= len(self.levels):
            return []
        return self.levels[level].boundaries

    def n_segments_at_level(self, level: int) -> int:
        b = self.boundaries_at_level(level)
        return max(len(b), 1)


class LagMatrix:
    """Time-lag representation of a self-similarity matrix.

    L[lag, t] = SSM[t, t + lag] for detecting periodic structure.
    """

    def compute_lag(self, ssm: np.ndarray) -> np.ndarray:
        n = ssm.shape[0]
        L = np.zeros((n, n))
        for lag in range(n):
            for t in range(n - lag):
                L[lag, t] = ssm[t, t + lag]
        return L

    def find_periodic_patterns(
        self,
        lag_matrix: np.ndarray,
        min_period: int = 4,
        max_period: int | None = None,
    ) -> list[PeriodicPattern]:
        n_lags = lag_matrix.shape[0]
        if max_period is None:
            max_period = n_lags // 2

        patterns = []
        for lag in range(min_period, min(max_period + 1, n_lags)):
            row = lag_matrix[lag, :]
            valid = row[row > 0]
            if len(valid) == 0:
                continue
            strength = float(np.mean(valid))
            if strength > 0.3:
                best_offset = int(np.argmax(row))
                patterns.append(
                    PeriodicPattern(
                        period=lag,
                        strength=strength,
                        start_offset=best_offset,
                    )
                )

        patterns.sort(key=lambda p: -p.strength)
        return patterns


def build_hierarchy(
    features: np.ndarray,
    levels: int = 3,
) -> StructureTree:
    ssm_builder = SelfSimilarityMatrix()
    ssm = ssm_builder.compute(features, metric="cosine")
    n = features.shape[0]

    structure_levels = []
    base_kernel = max(4, n // 8)

    for level in range(levels):
        scale = 2 ** level
        kernel_size = base_kernel * scale
        kernel_size = min(kernel_size, n // 2)
        kernel_size = max(kernel_size, 4)

        nov = novelty_curve(ssm, kernel_size=kernel_size)
        threshold = 0.3 / scale
        min_seg = max(2, kernel_size // 2)
        bounds = segment_boundaries(nov, threshold=threshold, min_segment_length=min_seg)

        structure_levels.append(
            StructureLevel(
                level=level,
                boundaries=bounds,
                kernel_size=kernel_size,
            )
        )

    return StructureTree(levels=structure_levels)
