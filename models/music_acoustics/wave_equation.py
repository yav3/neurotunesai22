"""Wave equation solvers for acoustic simulation.

1-D and 2-D FDTD (finite-difference time-domain) wave propagation,
boundary conditions, source injection, and measurement helpers.
Pure numpy — no external PDE libraries.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Medium:
    speed: float = 343.0
    density: float = 1.225
    damping: float = 0.0


@dataclass
class BoundaryCondition:
    """Reflection coefficient per edge (0 = absorbing, 1 = rigid, -1 = pressure release)."""
    left: float = 1.0
    right: float = 1.0


@dataclass
class Boundary2D:
    left: float = 1.0
    right: float = 1.0
    top: float = 1.0
    bottom: float = 1.0


def courant_number_1d(c: float, dt: float, dx: float) -> float:
    return c * dt / dx


def courant_number_2d(c: float, dt: float, dx: float, dy: float) -> float:
    return c * dt * np.sqrt(1.0 / dx**2 + 1.0 / dy**2)


def stable_dt_1d(c: float, dx: float, courant_max: float = 1.0) -> float:
    return courant_max * dx / c


def stable_dt_2d(c: float, dx: float, dy: float, courant_max: float = 1.0) -> float:
    return courant_max / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))


def fdtd_1d(
    n_points: int,
    n_steps: int,
    medium: Medium | None = None,
    boundary: BoundaryCondition | None = None,
    source_pos: int | None = None,
    source_signal: np.ndarray | None = None,
    dx: float = 0.01,
    dt: float | None = None,
    record_every: int = 1,
) -> np.ndarray:
    """1-D FDTD wave equation solver.

    Returns pressure field snapshots: shape (n_recorded, n_points).
    """
    med = medium or Medium()
    bc = boundary or BoundaryCondition()
    if dt is None:
        dt = stable_dt_1d(med.speed, dx, 0.95)
    C = courant_number_1d(med.speed, dt, dx)
    if C > 1.0:
        raise ValueError(f"Courant number {C:.3f} > 1; reduce dt or increase dx.")

    C2 = C * C
    damp = 1.0 - med.damping * dt

    p_prev = np.zeros(n_points)
    p_curr = np.zeros(n_points)

    if source_pos is None:
        source_pos = n_points // 2
    if source_signal is None:
        source_signal = np.zeros(n_steps)

    snapshots: list[np.ndarray] = []

    for t in range(n_steps):
        p_next = np.zeros(n_points)
        for i in range(1, n_points - 1):
            p_next[i] = (
                2.0 * p_curr[i]
                - p_prev[i]
                + C2 * (p_curr[i + 1] - 2.0 * p_curr[i] + p_curr[i - 1])
            )
        p_next *= damp

        # boundaries
        p_next[0] = bc.left * p_next[1]
        p_next[-1] = bc.right * p_next[-2]

        # source injection (soft source)
        if t < len(source_signal):
            p_next[source_pos] += source_signal[t]

        p_prev = p_curr
        p_curr = p_next

        if t % record_every == 0:
            snapshots.append(p_curr.copy())

    return np.array(snapshots)


def fdtd_2d(
    nx: int,
    ny: int,
    n_steps: int,
    medium: Medium | None = None,
    boundary: Boundary2D | None = None,
    source_pos: Tuple[int, int] | None = None,
    source_signal: np.ndarray | None = None,
    dx: float = 0.01,
    dy: float | None = None,
    dt: float | None = None,
    record_every: int = 10,
) -> np.ndarray:
    """2-D FDTD wave equation solver.

    Returns pressure snapshots: shape (n_recorded, nx, ny).
    """
    med = medium or Medium()
    bc = boundary or Boundary2D()
    if dy is None:
        dy = dx
    if dt is None:
        dt = stable_dt_2d(med.speed, dx, dy, 0.95)
    C = courant_number_2d(med.speed, dt, dx, dy)
    if C > 1.0:
        raise ValueError(f"Courant number {C:.3f} > 1; reduce dt or increase dx/dy.")

    cx2 = (med.speed * dt / dx) ** 2
    cy2 = (med.speed * dt / dy) ** 2
    damp = 1.0 - med.damping * dt

    p_prev = np.zeros((nx, ny))
    p_curr = np.zeros((nx, ny))

    if source_pos is None:
        source_pos = (nx // 2, ny // 2)
    if source_signal is None:
        source_signal = np.zeros(n_steps)

    snapshots: list[np.ndarray] = []

    for t in range(n_steps):
        lap_x = np.zeros_like(p_curr)
        lap_y = np.zeros_like(p_curr)
        lap_x[1:-1, :] = p_curr[2:, :] - 2.0 * p_curr[1:-1, :] + p_curr[:-2, :]
        lap_y[:, 1:-1] = p_curr[:, 2:] - 2.0 * p_curr[:, 1:-1] + p_curr[:, :-2]

        p_next = 2.0 * p_curr - p_prev + cx2 * lap_x + cy2 * lap_y
        p_next *= damp

        # boundaries
        p_next[0, :] = bc.left * p_next[1, :]
        p_next[-1, :] = bc.right * p_next[-2, :]
        p_next[:, 0] = bc.bottom * p_next[:, 1]
        p_next[:, -1] = bc.top * p_next[:, -2]

        if t < len(source_signal):
            p_next[source_pos[0], source_pos[1]] += source_signal[t]

        p_prev = p_curr
        p_curr = p_next

        if t % record_every == 0:
            snapshots.append(p_curr.copy())

    return np.array(snapshots)


def gaussian_pulse(
    n_samples: int,
    center: int | None = None,
    width: float = 10.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    if center is None:
        center = n_samples // 4
    t = np.arange(n_samples)
    return amplitude * np.exp(-((t - center) ** 2) / (2.0 * width**2))


def ricker_wavelet(
    n_samples: int,
    center: int | None = None,
    freq: float = 0.05,
    amplitude: float = 1.0,
) -> np.ndarray:
    if center is None:
        center = n_samples // 4
    t = np.arange(n_samples) - center
    u = (np.pi * freq * t) ** 2
    return amplitude * (1.0 - 2.0 * u) * np.exp(-u)


def sinusoidal_burst(
    n_samples: int,
    freq: float,
    n_cycles: int = 5,
    sr: float = 44100.0,
) -> np.ndarray:
    period = sr / freq
    burst_len = int(n_cycles * period)
    burst_len = min(burst_len, n_samples)
    t = np.arange(burst_len) / sr
    window = np.hanning(burst_len)
    sig = np.zeros(n_samples)
    sig[:burst_len] = window * np.sin(2.0 * np.pi * freq * t)
    return sig


def measure_at(snapshots: np.ndarray, position: int | Tuple[int, int]) -> np.ndarray:
    if snapshots.ndim == 2:
        return snapshots[:, position]
    elif snapshots.ndim == 3:
        return snapshots[:, position[0], position[1]]
    raise ValueError("snapshots must be 2-D (1-D sim) or 3-D (2-D sim)")


def energy_density(snapshots: np.ndarray) -> np.ndarray:
    return np.mean(snapshots**2, axis=tuple(range(1, snapshots.ndim)))


def decay_time_from_energy(energy: np.ndarray, threshold_db: float = -60.0) -> int:
    if energy[0] <= 0:
        return 0
    db = 10.0 * np.log10(np.maximum(energy / energy.max(), 1e-30))
    below = np.where(db <= threshold_db)[0]
    return int(below[0]) if len(below) > 0 else len(energy)


def diffraction_coefficient(
    freq: float,
    barrier_height: float,
    source_dist: float,
    receiver_dist: float,
    speed: float = 343.0,
) -> float:
    """Maekawa single-edge diffraction approximation.

    Returns attenuation in dB (negative = loss).
    """
    wavelength = speed / freq
    path_diff = (
        np.sqrt(source_dist**2 + barrier_height**2)
        + np.sqrt(receiver_dist**2 + barrier_height**2)
        - (source_dist + receiver_dist)
    )
    N = 2.0 * path_diff / wavelength
    if N <= -0.1916:
        return 0.0
    att = 10.0 * np.log10(3.0 + 20.0 * N)
    return -att


def diffraction_spectrum(
    freqs: np.ndarray,
    barrier_height: float,
    source_dist: float,
    receiver_dist: float,
    speed: float = 343.0,
) -> np.ndarray:
    return np.array([
        diffraction_coefficient(f, barrier_height, source_dist, receiver_dist, speed)
        for f in freqs
    ])
