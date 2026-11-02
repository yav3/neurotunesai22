"""Dynamic planning: phrase-level dynamic arcs, crescendo/decrescendo curves, dynamic form.

Plans dynamics over musical structure: phrase-level arc shapes (arch, ramp,
plateau), structural dynamic contours (sonata-form dynamic expectations),
performer-style dynamic deviations, and MIDI velocity mapping.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


class DynamicLevel(Enum):
    PPP = 16
    PP = 32
    P = 48
    MP = 64
    MF = 80
    F = 96
    FF = 112
    FFF = 127

    @property
    def velocity(self) -> int:
        return self.value


class ArcShape(Enum):
    ARCH = "arch"           # soft → loud → soft
    RAMP_UP = "ramp_up"     # soft → loud
    RAMP_DOWN = "ramp_down" # loud → soft
    PLATEAU = "plateau"     # soft → loud (hold) → soft
    WEDGE = "wedge"         # loud → soft → loud
    TERRACED = "terraced"   # discrete levels, no gradual change
    FLAT = "flat"           # constant


@dataclass
class DynamicArc:
    shape: ArcShape
    start_level: float    # 0-1
    peak_level: float     # 0-1
    end_level: float      # 0-1
    peak_position: float  # 0-1, where the peak occurs


def generate_arc(arc: DynamicArc, n_points: int = 128) -> np.ndarray:
    """Generate a dynamic curve from an arc specification."""
    t = np.linspace(0, 1, n_points)
    pp = arc.peak_position

    if arc.shape == ArcShape.FLAT:
        return np.full(n_points, arc.start_level)

    elif arc.shape == ArcShape.ARCH:
        left = np.where(t <= pp,
                        arc.start_level + (arc.peak_level - arc.start_level) * (t / max(pp, 1e-10)),
                        0.0)
        right = np.where(t > pp,
                         arc.peak_level + (arc.end_level - arc.peak_level) * ((t - pp) / max(1 - pp, 1e-10)),
                         0.0)
        return np.where(t <= pp, left, right)

    elif arc.shape == ArcShape.RAMP_UP:
        return arc.start_level + (arc.peak_level - arc.start_level) * t

    elif arc.shape == ArcShape.RAMP_DOWN:
        return arc.start_level + (arc.end_level - arc.start_level) * t

    elif arc.shape == ArcShape.PLATEAU:
        rise_end = pp * 0.4
        fall_start = pp + (1 - pp) * 0.6
        curve = np.piecewise(t, [
            t < rise_end,
            (t >= rise_end) & (t <= fall_start),
            t > fall_start
        ], [
            lambda x: arc.start_level + (arc.peak_level - arc.start_level) * x / max(rise_end, 1e-10),
            arc.peak_level,
            lambda x: arc.peak_level + (arc.end_level - arc.peak_level) * (x - fall_start) / max(1 - fall_start, 1e-10),
        ])
        return curve

    elif arc.shape == ArcShape.WEDGE:
        left = np.where(t <= pp,
                        arc.start_level + (arc.end_level - arc.start_level) * (t / max(pp, 1e-10)) * 0.5,
                        0.0)
        right = np.where(t > pp,
                         arc.end_level * 0.5 + (arc.peak_level - arc.end_level * 0.5) * ((t - pp) / max(1 - pp, 1e-10)),
                         0.0)
        return np.where(t <= pp, left, right)

    elif arc.shape == ArcShape.TERRACED:
        n_steps = max(2, int(n_points / 32))
        levels = np.linspace(arc.start_level, arc.peak_level, n_steps)
        step_size = n_points // n_steps
        curve = np.zeros(n_points)
        for i, lev in enumerate(levels):
            start_idx = i * step_size
            end_idx = min((i + 1) * step_size, n_points)
            curve[start_idx:end_idx] = lev
        return curve

    return np.full(n_points, arc.start_level)


def velocity_to_dynamic(velocity: int) -> str:
    """Map MIDI velocity to dynamic marking."""
    for level in DynamicLevel:
        if velocity <= level.value:
            return level.name.lower()
    return "fff"


def dynamic_to_velocity(dynamic: str) -> int:
    """Map dynamic marking string to MIDI velocity."""
    mapping = {d.name.lower(): d.value for d in DynamicLevel}
    return mapping.get(dynamic.lower(), 80)


def crescendo_curve(start_vel: int, end_vel: int, n_beats: int,
                    shape: str = "linear") -> np.ndarray:
    """Generate a crescendo/decrescendo velocity curve."""
    t = np.linspace(0, 1, n_beats)
    if shape == "linear":
        frac = t
    elif shape == "exponential":
        frac = t ** 2
    elif shape == "logarithmic":
        frac = np.sqrt(t)
    elif shape == "sigmoid":
        frac = 1 / (1 + np.exp(-10 * (t - 0.5)))
    else:
        frac = t
    return np.clip(start_vel + (end_vel - start_vel) * frac, 0, 127).astype(int)


def phrase_dynamic_plan(phrase_lengths: List[int],
                        arcs: Optional[List[ArcShape]] = None,
                        global_shape: ArcShape = ArcShape.ARCH) -> np.ndarray:
    """Plan dynamics across multiple phrases with per-phrase arcs."""
    if arcs is None:
        arcs = [ArcShape.ARCH] * len(phrase_lengths)

    global_curve = generate_arc(DynamicArc(global_shape, 0.4, 1.0, 0.3, 0.6),
                                sum(phrase_lengths))

    result = []
    offset = 0
    for length, arc_shape in zip(phrase_lengths, arcs):
        local_arc = DynamicArc(arc_shape, 0.5, 0.9, 0.5, 0.5)
        local_curve = generate_arc(local_arc, length)
        global_segment = global_curve[offset:offset + length]
        blended = 0.6 * global_segment + 0.4 * local_curve
        result.append(np.clip(blended, 0, 1))
        offset += length
    return np.concatenate(result)


def dynamic_contrast(velocities: np.ndarray) -> float:
    """Ratio of dynamic range to mean level (higher = more contrast)."""
    if len(velocities) < 2:
        return 0.0
    dyn_range = float(np.max(velocities) - np.min(velocities))
    mean_level = float(np.mean(velocities))
    return dyn_range / max(mean_level, 1.0)


def dynamic_variance_by_phrase(velocities: np.ndarray,
                               phrase_boundaries: List[int]) -> List[float]:
    """Variance of dynamics within each phrase."""
    variances = []
    for i in range(len(phrase_boundaries) - 1):
        segment = velocities[phrase_boundaries[i]:phrase_boundaries[i + 1]]
        variances.append(float(np.var(segment)) if len(segment) > 1 else 0.0)
    return variances


def expressive_deviation(performed: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    """Compare performed dynamics against score markings."""
    diff = performed - score
    return {
        "mean_deviation": float(np.mean(diff)),
        "std_deviation": float(np.std(diff)),
        "max_overshoot": float(np.max(diff)),
        "max_undershoot": float(np.min(diff)),
        "correlation": float(np.corrcoef(performed, score)[0, 1]) if len(performed) > 1 else 1.0,
    }


def humanize_velocity(velocities: np.ndarray, variation: float = 5.0,
                      seed: int = 42) -> np.ndarray:
    """Add human-like random variation to MIDI velocities."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, variation, len(velocities))
    return np.clip(velocities + noise, 1, 127).astype(int)


def structural_dynamic_template(form: str = "sonata") -> List[Tuple[str, float, ArcShape]]:
    """Get dynamic arc template for a musical form."""
    templates = {
        "sonata": [
            ("exposition_1st", 0.7, ArcShape.RAMP_UP),
            ("exposition_2nd", 0.6, ArcShape.ARCH),
            ("development", 0.9, ArcShape.WEDGE),
            ("recapitulation_1st", 0.8, ArcShape.RAMP_UP),
            ("recapitulation_2nd", 0.7, ArcShape.ARCH),
            ("coda", 1.0, ArcShape.RAMP_UP),
        ],
        "rondo": [
            ("A", 0.7, ArcShape.ARCH),
            ("B", 0.6, ArcShape.RAMP_UP),
            ("A", 0.7, ArcShape.ARCH),
            ("C", 0.8, ArcShape.WEDGE),
            ("A", 0.9, ArcShape.RAMP_UP),
        ],
        "ternary": [
            ("A", 0.7, ArcShape.ARCH),
            ("B", 0.8, ArcShape.RAMP_UP),
            ("A'", 0.7, ArcShape.RAMP_DOWN),
        ],
        "binary": [
            ("A", 0.6, ArcShape.RAMP_UP),
            ("B", 0.8, ArcShape.ARCH),
        ],
    }
    return templates.get(form, templates["ternary"])
