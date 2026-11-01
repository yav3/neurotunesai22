"""4-D VADE Geometry on S³ (three-sphere).

Core projection:
    P_VADE = W_proxy · X + b, then normalize to S³
    d(p, q) = arccos(p · q)

Constraints (golden-ratio mood bounds):
    V ≥ 1.6·A (high valence), V ≥ 1.6·D (sleep), V ≥ 1.6²·A (deep regeneration)
    BPM ∈ [46, 186], max 4 BPM per 30 min

Additional methods from canonical VADE engine:
    dominance_proxy: D = sin(A²+V²) / √(A²+V²)
    herrnstein_matching: B = k·Rf / (Rf₀+Rf)
    lattice_partition: hyperspherical coordinates on S³
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass


FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


@dataclass
class VADEPoint:
    """A point in 4-D VADE space (pre-normalization)."""
    valence: float
    arousal: float
    dominance: float
    engagement: float

    def to_array(self) -> np.ndarray:
        return np.array([self.valence, self.arousal, self.dominance, self.engagement])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VADEPoint":
        return cls(*arr[:4])


class VADEGeometry:
    """Operations on the S³ VADE manifold."""

    @staticmethod
    def project_to_vade(
        features: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """P_VADE = W · X + b, then normalize to S³.

        Args:
            features: (N, d) input feature matrix
            W: (4, d) projection weight matrix
            b: (4,) bias vector
        Returns:
            (N, 4) points on S³
        """
        raw = features @ W.T + b
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return raw / norms

    @staticmethod
    def geodesic_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """d(p, q) = arccos(p · q) on S³.

        Args:
            p: (N, 4) or (4,) — points on S³
            q: (M, 4) or (4,) — target points on S³
        Returns:
            Pairwise geodesic distances.
        """
        dot = np.clip(np.sum(p * q, axis=-1), -1.0, 1.0)
        return np.arccos(dot)

    @staticmethod
    def geodesic_distance_matrix(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Full pairwise geodesic distance matrix.

        Args:
            points: (N, 4)
            targets: (M, 4)
        Returns:
            (N, M) distance matrix
        """
        dot = np.clip(points @ targets.T, -1.0, 1.0)
        return np.arccos(dot)

    @staticmethod
    def check_valence_constraint(vade: np.ndarray, ratio: float = 1.6) -> np.ndarray:
        """V ≥ ratio · A constraint. Returns boolean mask."""
        return vade[..., 0] >= ratio * vade[..., 1]

    @staticmethod
    def check_bpm_constraint(bpm: np.ndarray, lo: float = 46, hi: float = 186) -> np.ndarray:
        return (bpm >= lo) & (bpm <= hi)

    @staticmethod
    def fibonacci_schedule(
        base_minutes: float = 5.0,
        max_duration_minutes: float = 90.0,
    ) -> list[float]:
        """Generate Fibonacci-constrained session phase durations.
        Each phase duration = fib_i × base_minutes, until total ≤ max.
        """
        schedule = []
        total = 0.0
        for f in FIBONACCI:
            dur = f * base_minutes
            if total + dur > max_duration_minutes:
                break
            schedule.append(dur)
            total += dur
        return schedule

    @staticmethod
    def interpolate_geodesic(
        p: np.ndarray, q: np.ndarray, t: float,
    ) -> np.ndarray:
        """Spherical linear interpolation (slerp) on S³.

        slerp(p, q, t) = sin((1-t)·Ω)/sin(Ω) · p + sin(t·Ω)/sin(Ω) · q
        where Ω = arccos(p·q)
        """
        dot = np.clip(np.dot(p, q), -1.0, 1.0)
        omega = np.arccos(dot)
        if abs(omega) < 1e-8:
            return p
        sin_omega = np.sin(omega)
        return (np.sin((1 - t) * omega) / sin_omega) * p + (np.sin(t * omega) / sin_omega) * q

    @staticmethod
    def session_trajectory(
        start: np.ndarray,
        target: np.ndarray,
        n_phases: int,
    ) -> np.ndarray:
        """Generate a geodesic trajectory on S³ from start → target
        with n_phases intermediate waypoints (slerp)."""
        t_values = np.linspace(0, 1, n_phases + 2)
        trajectory = np.array([
            VADEGeometry.interpolate_geodesic(start, target, t)
            for t in t_values
        ])
        return trajectory


# ═══════════════════════════════════════════════════════════════════════
# Canonical VADE methods from the Merlin production engine
# ═══════════════════════════════════════════════════════════════════════

PHI = 1.618
BPM_BOUNDS = (46.0, 186.0)


def mood_bounds(v: float, a: float, d: float, ratio: float = 1.6) -> dict[str, bool]:
    return {
        "high_valence": v >= ratio * a,
        "sleep": v >= ratio * d and v >= ratio ** 2 * a,
        "deep_regeneration": v >= ratio ** 2 * a,
    }


def dominance_proxy(a: float, v: float) -> float:
    r2 = a * a + v * v
    return float(math.sin(r2) / math.sqrt(r2)) if r2 > 0 else 0.0


def bpm_schedule(start_bpm: float, end_bpm: float, minutes: float,
                 max_step_per_30min: float = 4.0,
                 bounds: tuple[float, float] = BPM_BOUNDS) -> list[float]:
    if not bounds[0] <= start_bpm <= bounds[1]:
        raise ValueError("start_bpm outside bounds")
    if not bounds[0] <= end_bpm <= bounds[1]:
        raise ValueError("end_bpm outside bounds")
    per_min = max_step_per_30min / 30.0
    path, cur = [start_bpm], start_bpm
    for _ in range(int(minutes)):
        step = np.clip(end_bpm - cur, -per_min, per_min)
        cur = float(cur + step)
        path.append(cur)
    return path


def fibonacci_dose(total_minutes: float = 30.0, n_segments: int = 5) -> list[float]:
    if n_segments < 1:
        raise ValueError("n_segments >= 1")
    f = [1, 1]
    while len(f) < n_segments:
        f.append(f[-1] + f[-2])
    f = f[:n_segments]
    s = sum(f)
    return [total_minutes * x / s for x in f]


def herrnstein_matching(k: float, rf: float, rf0: float) -> float:
    if rf0 + rf <= 0:
        raise ValueError("Rf0 + Rf must be positive")
    return k * rf / (rf0 + rf)


def lattice_partition(points: np.ndarray,
                      cells_per_axis: int = 4) -> dict[tuple[int, int, int], dict]:
    pts = np.asarray(points, dtype=float)
    norms = np.linalg.norm(pts, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    P = pts / norms
    cells: dict[tuple[int, int, int], dict] = {}
    for i, (x0, x1, x2, x3) in enumerate(P):
        psi = math.acos(np.clip(x0, -1, 1))
        r = math.sqrt(max(1 - x0 * x0, 0))
        theta = math.acos(np.clip(x1 / r, -1, 1)) if r > 0 else 0.0
        phi = math.atan2(x3, x2) % (2 * math.pi)
        idx = (min(int(psi / math.pi * cells_per_axis), cells_per_axis - 1),
               min(int(theta / math.pi * cells_per_axis), cells_per_axis - 1),
               min(int(phi / (2 * math.pi) * cells_per_axis), cells_per_axis - 1))
        cell = cells.setdefault(idx, {"count": 0, "members": []})
        cell["count"] += 1
        cell["members"].append(i)
    return cells
