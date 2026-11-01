"""Expression contours: continuous dynamic shaping over musical phrases."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ExpressionContour:
    """A continuous expression curve over time."""
    values: np.ndarray
    times: np.ndarray
    min_val: float
    max_val: float


def phrase_arc(n_points: int, peak_position: float = 0.6, peak_height: float = 1.0) -> np.ndarray:
    """Generate a phrase-arc contour (rise-fall) with adjustable peak position.

    Typical musical phrasing peaks around 60% through the phrase.
    """
    t = np.linspace(0.0, 1.0, n_points)
    left = t / peak_position
    right = (1.0 - t) / (1.0 - peak_position)
    raw = np.where(t <= peak_position, left, right)
    return raw * peak_height


def crescendo_contour(n_points: int, start: float = 0.3, end: float = 1.0) -> np.ndarray:
    """Linear crescendo from start to end level."""
    return np.linspace(start, end, n_points)


def diminuendo_contour(n_points: int, start: float = 1.0, end: float = 0.3) -> np.ndarray:
    """Linear diminuendo from start to end level."""
    return np.linspace(start, end, n_points)


def swell(n_points: int, base: float = 0.4, peak: float = 1.0) -> np.ndarray:
    """Sine-shaped swell: rise then fall symmetrically."""
    t = np.linspace(0.0, np.pi, n_points)
    amplitude = (peak - base) * np.sin(t)
    return base + amplitude


def terraced_dynamics(
    levels: list[float],
    durations: list[int],
) -> np.ndarray:
    """Step-wise (terraced/Baroque) dynamics: constant level per segment."""
    segments = []
    for level, dur in zip(levels, durations):
        segments.append(np.full(dur, level))
    return np.concatenate(segments)


def smooth_transition(
    contour: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Smooth a contour with a simple moving average."""
    if kernel_size < 2 or len(contour) < kernel_size:
        return contour.copy()
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.pad(contour, kernel_size // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(contour)]


def rubato_warp(
    n_points: int,
    stretch_center: float = 0.5,
    stretch_amount: float = 0.3,
    sigma: float = 0.15,
) -> np.ndarray:
    """Generate a tempo-warp curve for rubato (expressive timing).

    Returns a multiplicative factor: >1 = slower, <1 = faster.
    The integral is normalised so total duration is preserved.
    """
    t = np.linspace(0.0, 1.0, n_points)
    warp = 1.0 + stretch_amount * np.exp(-0.5 * ((t - stretch_center) / sigma) ** 2)
    warp *= n_points / np.sum(warp)
    return warp


def agogic_accent(
    durations: list[float],
    accent_indices: list[int],
    stretch: float = 1.2,
) -> list[float]:
    """Agogic accent: lengthen specific notes for expressive stress."""
    result = list(durations)
    for idx in accent_indices:
        if 0 <= idx < len(result):
            result[idx] *= stretch
    return result


def combine_contours(
    contours: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Weighted combination of multiple expression contours."""
    if not contours:
        return np.array([])
    n = min(len(c) for c in contours)
    if weights is None:
        weights = [1.0 / len(contours)] * len(contours)
    result = np.zeros(n)
    for c, w in zip(contours, weights):
        result += w * c[:n]
    return result


def build_expression_contour(
    values: np.ndarray,
    total_duration: float,
) -> ExpressionContour:
    """Wrap raw values into an ExpressionContour with time axis."""
    n = len(values)
    times = np.linspace(0.0, total_duration, n)
    return ExpressionContour(
        values=values,
        times=times,
        min_val=float(np.min(values)),
        max_val=float(np.max(values)),
    )
