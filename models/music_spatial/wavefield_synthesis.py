"""Wavefield synthesis: driving functions, spatial aliasing, focused sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "WFSArray", "WFSSource", "DrivingSignal",
    "planar_driving_function", "circular_driving_function",
    "spatial_aliasing_frequency", "focused_source_driving",
    "kirchhoff_helmholtz_weights", "compute_array_geometry",
    "render_wavefield_2d", "source_directivity_filter",
    "array_truncation_error", "secondary_source_selection",
]


@dataclass
class WFSArray:
    positions: np.ndarray
    normals: np.ndarray
    spacing: float
    n_sources: int

    @property
    def aperture(self) -> float:
        diffs = np.diff(self.positions, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


@dataclass
class WFSSource:
    position: np.ndarray
    source_type: str  # "plane", "point", "focused"
    amplitude: float = 1.0
    frequency: float = 1000.0


@dataclass
class DrivingSignal:
    weights: np.ndarray
    delays: np.ndarray
    gains: np.ndarray


def compute_array_geometry(
    array_type: str = "linear",
    n_sources: int = 64,
    spacing: float = 0.05,
    radius: float = 1.5,
) -> WFSArray:
    if array_type == "linear":
        x = np.arange(n_sources) * spacing - (n_sources - 1) * spacing / 2
        positions = np.column_stack([x, np.zeros(n_sources)])
        normals = np.column_stack([np.zeros(n_sources), np.ones(n_sources)])
    elif array_type == "circular":
        angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
        positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        normals = -positions / np.linalg.norm(positions, axis=1, keepdims=True)
    elif array_type == "rectangular":
        side = int(np.ceil(np.sqrt(n_sources)))
        xs = np.arange(side) * spacing - (side - 1) * spacing / 2
        ys = np.arange(side) * spacing - (side - 1) * spacing / 2
        xx, yy = np.meshgrid(xs, ys)
        positions = np.column_stack([xx.ravel()[:n_sources], yy.ravel()[:n_sources]])
        normals = np.zeros_like(positions)
        normals[:, 1] = 1.0
        n_sources = positions.shape[0]
    else:
        positions = np.zeros((n_sources, 2))
        normals = np.zeros((n_sources, 2))
        normals[:, 1] = 1.0
    return WFSArray(
        positions=positions, normals=normals,
        spacing=spacing, n_sources=n_sources,
    )


def _distance_2d(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    diff = p1 - p2
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def planar_driving_function(
    array: WFSArray,
    propagation_direction: np.ndarray,
    frequency: float,
    c: float = 343.0,
) -> DrivingSignal:
    d = np.asarray(propagation_direction, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-15)
    k = 2 * np.pi * frequency / c
    cos_alpha = np.sum(array.normals * d[np.newaxis, :], axis=1)
    selection = cos_alpha > 0
    delays = np.sum(array.positions * d[np.newaxis, :], axis=1) / c
    gains = np.where(selection, 2.0 * cos_alpha, 0.0)
    weights = gains * np.exp(-1j * k * delays * c).real
    return DrivingSignal(
        weights=weights, delays=delays, gains=gains,
    )


def circular_driving_function(
    array: WFSArray,
    source_pos: np.ndarray,
    frequency: float,
    c: float = 343.0,
) -> DrivingSignal:
    src = np.asarray(source_pos, dtype=float)
    r_vec = array.positions - src[np.newaxis, :]
    distances = np.sqrt(np.sum(r_vec ** 2, axis=1))
    distances = np.maximum(distances, 1e-6)
    cos_alpha = np.sum(r_vec * array.normals, axis=1) / distances
    selection = cos_alpha > 0
    k = 2 * np.pi * frequency / c
    amplitude_decay = 1.0 / np.sqrt(distances)
    gains = np.where(selection, 2.0 * cos_alpha * amplitude_decay, 0.0)
    delays = distances / c
    weights = gains * np.cos(k * distances)
    return DrivingSignal(weights=weights, delays=delays, gains=gains)


def focused_source_driving(
    array: WFSArray,
    focus_pos: np.ndarray,
    frequency: float,
    c: float = 343.0,
) -> DrivingSignal:
    fp = np.asarray(focus_pos, dtype=float)
    r_vec = fp[np.newaxis, :] - array.positions
    distances = np.sqrt(np.sum(r_vec ** 2, axis=1))
    distances = np.maximum(distances, 1e-6)
    cos_alpha = np.sum(r_vec * array.normals, axis=1) / distances
    selection = cos_alpha > 0
    k = 2 * np.pi * frequency / c
    gains = np.where(selection, 2.0 * np.abs(cos_alpha) / np.sqrt(distances), 0.0)
    delays = -distances / c
    weights = gains * np.cos(k * distances)
    return DrivingSignal(weights=weights, delays=delays, gains=gains)


def spatial_aliasing_frequency(
    spacing: float,
    c: float = 343.0,
    angle_max: float = np.pi / 2,
) -> float:
    return c / (2.0 * spacing * np.sin(angle_max))


def kirchhoff_helmholtz_weights(
    array: WFSArray,
    receiver_pos: np.ndarray,
    frequency: float,
    c: float = 343.0,
) -> np.ndarray:
    rp = np.asarray(receiver_pos, dtype=float)
    r_vec = rp[np.newaxis, :] - array.positions
    distances = np.sqrt(np.sum(r_vec ** 2, axis=1))
    distances = np.maximum(distances, 1e-6)
    k = 2 * np.pi * frequency / c
    green = np.exp(-1j * k * distances) / (4.0 * np.pi * distances)
    cos_n = np.sum(r_vec * array.normals, axis=1) / distances
    dg_dn = green * (-1j * k - 1.0 / distances) * cos_n
    return np.abs(dg_dn) * array.spacing


def render_wavefield_2d(
    array: WFSArray,
    driving: DrivingSignal,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    frequency: float,
    c: float = 343.0,
) -> np.ndarray:
    k = 2 * np.pi * frequency / c
    xx, yy = np.meshgrid(grid_x, grid_y)
    field = np.zeros_like(xx)
    for i in range(array.n_sources):
        if abs(driving.gains[i]) < 1e-12:
            continue
        dx = xx - array.positions[i, 0]
        dy = yy - array.positions[i, 1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        r = np.maximum(r, 1e-6)
        contribution = driving.weights[i] * np.cos(k * r) / np.sqrt(r)
        field += contribution
    return field


def source_directivity_filter(
    angles: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    a = np.asarray(angles, dtype=float)
    if order == 0:
        return np.ones_like(a)
    return np.maximum(np.cos(a), 0.0) ** order


def secondary_source_selection(
    array: WFSArray,
    source_pos: np.ndarray,
) -> np.ndarray:
    src = np.asarray(source_pos, dtype=float)
    r_vec = src[np.newaxis, :] - array.positions
    cos_alpha = np.sum(r_vec * array.normals, axis=1) / (
        np.sqrt(np.sum(r_vec ** 2, axis=1)) + 1e-15
    )
    return (cos_alpha > 0).astype(float)


def array_truncation_error(
    array: WFSArray,
    source_pos: np.ndarray,
    listen_pos: np.ndarray,
    frequency: float,
    c: float = 343.0,
) -> float:
    src = np.asarray(source_pos, dtype=float)
    lp = np.asarray(listen_pos, dtype=float)
    edges = np.array([array.positions[0], array.positions[-1]])
    edge_distances = _distance_2d(edges, lp[np.newaxis, :])
    min_edge_dist = float(edge_distances.min())
    wavelength = c / max(frequency, 1e-9)
    fresnel_number = min_edge_dist ** 2 / (wavelength * _distance_2d(src, lp))
    attenuation_db = float(5 + 20 * np.log10(max(np.sqrt(2 * np.pi * fresnel_number), 1e-9)))
    return max(0.0, attenuation_db)
