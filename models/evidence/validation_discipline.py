"""Validation discipline — detect degenerate solver outputs.

Ported from merlinaineuralplus2/backend/methods/validation_discipline.py.
Born from a real bug: a scoring function silently collapsed to all-zeros under
z-score preprocessing, and nothing caught it. This module catches that class of
failure generically: probe a (transform, scoring_fn) pair and flag if the output
is all-zero, all-equal, or non-finite.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class DegeneracyCheck:
    transform_name: str
    is_degenerate: bool
    reason: str
    values: NDArray[np.float64] | None = None


def is_degenerate(
    values: NDArray[np.float64],
    zero_atol: float = 1e-9,
    equal_atol: float = 1e-9,
) -> tuple[bool, str]:
    """Check if a 1-D score array is degenerate.

    Degenerate means: not finite anywhere, all within zero_atol of 0,
    or all within equal_atol of each other (no discriminative signal).
    """
    if values.size == 0:
        return True, "empty"
    if not np.all(np.isfinite(values)):
        n_bad = int(np.sum(~np.isfinite(values)))
        return True, f"{n_bad}/{values.size} non-finite values"
    if np.all(np.abs(values) < zero_atol):
        return True, f"all values within {zero_atol} of zero"
    if np.ptp(values) < equal_atol:
        return True, f"range {np.ptp(values):.2e} < {equal_atol}: no discriminative signal"
    return False, "ok"


def assert_signal_survives_transform(
    scoring_fn: Callable[..., NDArray[np.float64]],
    input_data: NDArray[np.float64],
    transforms: dict[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] | None = None,
    zero_atol: float = 1e-9,
    equal_atol: float = 1e-9,
) -> list[DegeneracyCheck]:
    """Run scoring_fn under each transform and check for collapse.

    Default transforms: identity, z-score (row), z-score (col), log1p, rank.
    """
    if transforms is None:
        transforms = _default_transforms()

    checks = []
    for name, xform in transforms.items():
        try:
            transformed = xform(input_data.copy())
            scores = scoring_fn(transformed)
            scores = np.asarray(scores, dtype=np.float64).ravel()
            degen, reason = is_degenerate(scores, zero_atol, equal_atol)
            checks.append(DegeneracyCheck(name, degen, reason, scores))
        except Exception as exc:
            checks.append(DegeneracyCheck(name, True, f"exception: {exc}"))
    return checks


def _default_transforms() -> dict[str, Callable]:
    def identity(x: NDArray) -> NDArray:
        return x

    def zscore_row(x: NDArray) -> NDArray:
        mu = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return (x - mu) / std

    def zscore_col(x: NDArray) -> NDArray:
        mu = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return (x - mu) / std

    def log1p(x: NDArray) -> NDArray:
        return np.log1p(np.abs(x))

    def rank(x: NDArray) -> NDArray:
        from scipy.stats import rankdata
        return np.apply_along_axis(rankdata, 1, x)

    return {
        "identity": identity,
        "zscore_row": zscore_row,
        "zscore_col": zscore_col,
        "log1p": log1p,
        "rank": rank,
    }
