"""Room acoustics simulation: image source method, RT60, early reflections, diffusion.

Geometric room acoustics for reverb modeling, room mode analysis,
and critical distance estimation. Pure numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class RoomDimensions:
    length: float
    width: float
    height: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def surface_area(self) -> float:
        l, w, h = self.length, self.width, self.height
        return 2.0 * (l * w + l * h + w * h)


@dataclass(frozen=True)
class MaterialAbsorption:
    alpha_125: float = 0.1
    alpha_250: float = 0.1
    alpha_500: float = 0.1
    alpha_1000: float = 0.1
    alpha_2000: float = 0.1
    alpha_4000: float = 0.1

    def at_frequency(self, freq: float) -> float:
        freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
        alphas = np.array([
            self.alpha_125, self.alpha_250, self.alpha_500,
            self.alpha_1000, self.alpha_2000, self.alpha_4000,
        ])
        return float(np.interp(freq, freqs, alphas))


@dataclass(frozen=True)
class ReflectionPath:
    distance: float
    delay_seconds: float
    attenuation: float
    order: int


SPEED_OF_SOUND = 343.0
AIR_ABSORPTION_COEFFICIENT = 0.001


def sabine_rt60(room: RoomDimensions, avg_absorption: float) -> float:
    if avg_absorption <= 0 or room.surface_area <= 0:
        return float("inf")
    return 0.161 * room.volume / (avg_absorption * room.surface_area)


def eyring_rt60(room: RoomDimensions, avg_absorption: float) -> float:
    if avg_absorption <= 0 or avg_absorption >= 1 or room.surface_area <= 0:
        return float("inf")
    return -0.161 * room.volume / (room.surface_area * np.log(1 - avg_absorption))


def room_modes(room: RoomDimensions, max_freq: float = 300.0) -> list[float]:
    modes: list[float] = []
    max_n = int(max_freq * 2 * max(room.length, room.width, room.height) / SPEED_OF_SOUND) + 1
    for nx in range(max_n + 1):
        for ny in range(max_n + 1):
            for nz in range(max_n + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                f = 0.5 * SPEED_OF_SOUND * np.sqrt(
                    (nx / room.length) ** 2
                    + (ny / room.width) ** 2
                    + (nz / room.height) ** 2
                )
                if f <= max_freq:
                    modes.append(float(f))
    return sorted(modes)


def modal_density(room: RoomDimensions, freq: float) -> float:
    V = room.volume
    S = room.surface_area
    L = 4 * (room.length + room.width + room.height)
    c = SPEED_OF_SOUND
    return (4 * np.pi * freq ** 2 * V) / (c ** 3) + (np.pi * freq * S) / (2 * c ** 2) + L / (8 * c)


def schroeder_frequency(room: RoomDimensions, rt60: float) -> float:
    return 2000.0 * np.sqrt(rt60 / max(room.volume, 0.01))


def critical_distance(room: RoomDimensions, avg_absorption: float) -> float:
    A = avg_absorption * room.surface_area
    return 0.057 * np.sqrt(A)


def image_source_reflections(
    room: RoomDimensions,
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
    max_order: int = 3,
    wall_absorption: float = 0.1,
) -> list[ReflectionPath]:
    sx, sy, sz = source
    rx, ry, rz = receiver
    reflections: list[ReflectionPath] = []
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                order = abs(nx) + abs(ny) + abs(nz)
                if order == 0 or order > max_order:
                    continue
                ix = sx + 2 * nx * room.length if nx % 2 == 0 else (2 * (nx // 2 + 1) * room.length - sx)
                iy = sy + 2 * ny * room.width if ny % 2 == 0 else (2 * (ny // 2 + 1) * room.width - sy)
                iz = sz + 2 * nz * room.height if nz % 2 == 0 else (2 * (nz // 2 + 1) * room.height - sz)
                dist = np.sqrt((ix - rx) ** 2 + (iy - ry) ** 2 + (iz - rz) ** 2)
                atten = ((1 - wall_absorption) ** order) / max(dist, 0.01)
                delay = dist / SPEED_OF_SOUND
                reflections.append(ReflectionPath(
                    distance=float(dist),
                    delay_seconds=float(delay),
                    attenuation=float(atten),
                    order=order,
                ))
    reflections.sort(key=lambda r: r.delay_seconds)
    return reflections


def early_reflection_density(reflections: list[ReflectionPath], window_ms: float = 80.0) -> float:
    cutoff = window_ms / 1000.0
    early = [r for r in reflections if r.delay_seconds <= cutoff]
    return len(early) / max(window_ms / 1000.0, 0.001)


def clarity_c80(reflections: list[ReflectionPath]) -> float:
    early_energy = sum(r.attenuation ** 2 for r in reflections if r.delay_seconds <= 0.08)
    late_energy = sum(r.attenuation ** 2 for r in reflections if r.delay_seconds > 0.08)
    if late_energy < 1e-12:
        return 30.0
    return float(10 * np.log10(early_energy / late_energy))


def definition_d50(reflections: list[ReflectionPath]) -> float:
    early_energy = sum(r.attenuation ** 2 for r in reflections if r.delay_seconds <= 0.05)
    total_energy = sum(r.attenuation ** 2 for r in reflections)
    if total_energy < 1e-12:
        return 1.0
    return float(early_energy / total_energy)


def diffusion_index(mode_frequencies: list[float]) -> float:
    if len(mode_frequencies) < 3:
        return 0.0
    spacings = np.diff(sorted(mode_frequencies))
    if np.mean(spacings) < 1e-12:
        return 0.0
    cv = float(np.std(spacings) / np.mean(spacings))
    return float(np.exp(-cv))
