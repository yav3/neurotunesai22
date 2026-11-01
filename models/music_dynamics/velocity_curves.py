"""Velocity curves: MIDI velocity to amplitude mappings and humanization."""
from __future__ import annotations

import enum
import numpy as np
from dataclasses import dataclass


class VelocityCurve(enum.Enum):
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    S_CURVE = "s_curve"
    FIXED = "fixed"


def velocity_to_amplitude(velocity: int, curve: VelocityCurve = VelocityCurve.LINEAR) -> float:
    """Convert MIDI velocity (0-127) to amplitude [0, 1]."""
    v = max(0, min(127, velocity))
    t = v / 127.0

    if curve == VelocityCurve.FIXED:
        return 1.0 if v > 0 else 0.0
    elif curve == VelocityCurve.LINEAR:
        return t
    elif curve == VelocityCurve.LOGARITHMIC:
        return np.log1p(t * (np.e - 1)) / 1.0 if t > 0 else 0.0
    elif curve == VelocityCurve.EXPONENTIAL:
        return (np.exp(t) - 1.0) / (np.e - 1.0)
    elif curve == VelocityCurve.S_CURVE:
        return 1.0 / (1.0 + np.exp(-10.0 * (t - 0.5)))
    return t


def amplitude_to_velocity(amplitude: float, curve: VelocityCurve = VelocityCurve.LINEAR) -> int:
    """Convert amplitude [0, 1] to MIDI velocity (0-127). Inverse of velocity_to_amplitude."""
    a = max(0.0, min(1.0, amplitude))

    if curve == VelocityCurve.FIXED:
        return 127 if a > 0.5 else 0
    elif curve == VelocityCurve.LINEAR:
        return int(round(a * 127))
    elif curve == VelocityCurve.LOGARITHMIC:
        if a <= 0:
            return 0
        t = (np.exp(a) - 1.0) / (np.e - 1.0)
        return int(round(t * 127))
    elif curve == VelocityCurve.EXPONENTIAL:
        t = np.log1p(a * (np.e - 1.0))
        return int(round(t * 127))
    elif curve == VelocityCurve.S_CURVE:
        if a <= 0.001:
            return 0
        if a >= 0.999:
            return 127
        t = 0.5 + np.log(a / (1.0 - a)) / 10.0
        return int(round(max(0.0, min(1.0, t)) * 127))
    return int(round(a * 127))


def apply_velocity_curve(velocities: list[int], curve: VelocityCurve) -> list[float]:
    """Apply a velocity curve to a list of MIDI velocities."""
    return [velocity_to_amplitude(v, curve) for v in velocities]


def humanize_velocities(
    velocities: list[int],
    amount: float = 0.1,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Add controlled random variation to velocities.

    Args:
        velocities: input MIDI velocities.
        amount: variation strength (0 = none, 1 = full range noise).
        rng: random generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_range = int(amount * 20)
    result = []
    for v in velocities:
        delta = rng.integers(-noise_range, noise_range + 1)
        result.append(max(1, min(127, v + delta)))
    return result


def accent_pattern(velocities: list[int], pattern: list[bool], accent_boost: int = 20) -> list[int]:
    """Apply accent pattern to velocities.

    Args:
        velocities: base velocities.
        pattern: boolean mask (True = accented). Cycled to match length.
        accent_boost: velocity increase for accented notes.
    """
    result = []
    for i, v in enumerate(velocities):
        if pattern[i % len(pattern)]:
            result.append(min(127, v + accent_boost))
        else:
            result.append(v)
    return result


def velocity_gradient(
    start: int,
    end: int,
    n_notes: int,
    curve: VelocityCurve = VelocityCurve.LINEAR,
) -> list[int]:
    """Smooth velocity gradient (crescendo/diminuendo).

    Args:
        start: starting velocity.
        end: ending velocity.
        n_notes: number of notes.
        curve: interpolation curve shape.
    """
    if n_notes <= 1:
        return [start]

    result = []
    for i in range(n_notes):
        t = i / (n_notes - 1)
        if curve == VelocityCurve.EXPONENTIAL:
            t = (np.exp(t) - 1.0) / (np.e - 1.0)
        elif curve == VelocityCurve.LOGARITHMIC:
            t = np.log1p(t * (np.e - 1)) if t > 0 else 0.0
        elif curve == VelocityCurve.S_CURVE:
            t = 1.0 / (1.0 + np.exp(-10.0 * (t - 0.5)))
        v = start + (end - start) * t
        result.append(max(0, min(127, int(round(v)))))
    return result
