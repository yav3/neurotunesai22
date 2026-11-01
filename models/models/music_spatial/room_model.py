"""Simple room acoustic modeling: image source method, RT60, clarity."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class RoomGeometry:
    length: float
    width: float
    height: float
    absorption_coefficients: dict[str, float] = field(default_factory=lambda: {
        "floor": 0.1, "ceiling": 0.2, "walls": 0.15,
    })


@dataclass
class ImageSource:
    position: np.ndarray
    delay_s: float
    attenuation: float
    order: int


def _room_volume(room: RoomGeometry) -> float:
    return room.length * room.width * room.height


def _total_absorption(room: RoomGeometry) -> float:
    a = room.absorption_coefficients
    floor_ceil = room.length * room.width
    side_walls = 2 * room.height * room.width
    end_walls = 2 * room.height * room.length
    return (a.get("floor", 0.1) * floor_ceil +
            a.get("ceiling", 0.2) * floor_ceil +
            a.get("walls", 0.15) * (side_walls + end_walls))


def rt60_sabine(room: RoomGeometry) -> float:
    """Sabine reverberation time: T60 = 0.161 * V / A."""
    V = _room_volume(room)
    A = _total_absorption(room)
    if A <= 0:
        return float("inf")
    return 0.161 * V / A


def rt60_eyring(room: RoomGeometry) -> float:
    """Eyring reverberation time for higher absorption environments."""
    V = _room_volume(room)
    S = 2 * (room.length * room.width + room.length * room.height + room.width * room.height)
    A_total = _total_absorption(room)
    alpha_avg = A_total / (S + 1e-12)
    alpha_avg = min(alpha_avg, 0.999)
    return 0.161 * V / (-S * math.log(1.0 - alpha_avg))


def image_source_model(
    room: RoomGeometry,
    source_pos: np.ndarray,
    listener_pos: np.ndarray,
    max_order: int = 3,
) -> list[ImageSource]:
    """Compute image sources up to given order (2D simplification)."""
    c = 343.0
    sources: list[ImageSource] = []
    dims = np.array([room.length, room.width, room.height])
    alpha_avg = np.mean(list(room.absorption_coefficients.values()))

    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                order = abs(nx) + abs(ny) + abs(nz)
                if order == 0 or order > max_order:
                    continue
                image = source_pos.copy()
                if nx % 2 != 0:
                    image[0] = 2 * dims[0] * ((nx + 1) // 2) - source_pos[0]
                else:
                    image[0] = 2 * dims[0] * (nx // 2) + source_pos[0]
                if ny % 2 != 0:
                    image[1] = 2 * dims[1] * ((ny + 1) // 2) - source_pos[1]
                else:
                    image[1] = 2 * dims[1] * (ny // 2) + source_pos[1]
                if nz % 2 != 0:
                    image[2] = 2 * dims[2] * ((nz + 1) // 2) - source_pos[2]
                else:
                    image[2] = 2 * dims[2] * (nz // 2) + source_pos[2]

                dist = float(np.linalg.norm(image - listener_pos))
                if dist < 0.01:
                    continue
                delay = dist / c
                atten = (1.0 / dist) * ((1.0 - alpha_avg) ** order)
                sources.append(ImageSource(image, delay, atten, order))

    sources.sort(key=lambda s: s.delay_s)
    return sources


def early_decay_time(impulse_response: np.ndarray, sr: int) -> float:
    """EDT: time for first 10 dB of decay, extrapolated to 60 dB."""
    energy = impulse_response ** 2
    cumulative = np.cumsum(energy[::-1])[::-1]
    cumulative_db = 10 * np.log10(cumulative / (cumulative[0] + 1e-12) + 1e-12)
    idx_10 = np.searchsorted(-cumulative_db, 10)
    if idx_10 >= len(cumulative_db):
        idx_10 = len(cumulative_db) - 1
    return 6.0 * idx_10 / sr


def clarity_index(impulse_response: np.ndarray, sr: int, t_ms: float = 80) -> float:
    """C80 = 10*log10(E_early / E_late) in dB."""
    boundary = int(t_ms * sr / 1000)
    early = np.sum(impulse_response[:boundary] ** 2)
    late = np.sum(impulse_response[boundary:] ** 2)
    if late < 1e-12:
        return 60.0
    return 10 * np.log10(early / late)


def direct_to_reverberant_ratio(impulse_response: np.ndarray, sr: int, direct_window_ms: float = 2.5) -> float:
    """DRR in dB."""
    boundary = int(direct_window_ms * sr / 1000)
    direct = np.sum(impulse_response[:boundary] ** 2)
    reverberant = np.sum(impulse_response[boundary:] ** 2)
    if reverberant < 1e-12:
        return 60.0
    return 10 * np.log10(direct / reverberant)
