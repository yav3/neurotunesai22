"""Ambisonics encoding and decoding using real spherical harmonics."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


class AmbisonicsOrder(IntEnum):
    FIRST = 1
    SECOND = 2
    THIRD = 3


def n_channels(order: int) -> int:
    return (order + 1) ** 2


def _real_sh(l: int, m: int, azimuth: float, elevation: float) -> float:
    """ACN-ordered real spherical harmonic Y_l^m(az, el), SN3D normalised."""
    if l == 0:
        return 1.0
    if l == 1:
        if m == -1:
            return math.sin(azimuth) * math.cos(elevation)
        if m == 0:
            return math.sin(elevation)
        return math.cos(azimuth) * math.cos(elevation)
    if l == 2:
        cos_e = math.cos(elevation)
        sin_e = math.sin(elevation)
        if m == -2:
            return math.sin(2 * azimuth) * cos_e ** 2
        if m == -1:
            return math.sin(azimuth) * sin_e * cos_e
        if m == 0:
            return 1.5 * sin_e ** 2 - 0.5
        if m == 1:
            return math.cos(azimuth) * sin_e * cos_e
        return math.cos(2 * azimuth) * cos_e ** 2
    return 0.0


def encode_source(azimuth: float, elevation: float, order: int = 1) -> np.ndarray:
    """Encode a point source into B-format coefficients (ACN/SN3D)."""
    n = n_channels(order)
    coeffs = np.zeros(n)
    idx = 0
    for l in range(order + 1):
        for m in range(-l, l + 1):
            coeffs[idx] = _real_sh(l, m, azimuth, elevation)
            idx += 1
    return coeffs


@dataclass
class AmbisonicsDecoder:
    speaker_positions: np.ndarray  # (n_speakers, 2) — azimuth, elevation

    def decode_matrix(self, order: int = 1) -> np.ndarray:
        """Pseudo-inverse decoding matrix: speakers × ambi channels."""
        n_spk = len(self.speaker_positions)
        n_ch = n_channels(order)
        C = np.zeros((n_spk, n_ch))
        for i, (az, el) in enumerate(self.speaker_positions):
            C[i] = encode_source(az, el, order)
        return np.linalg.pinv(C).T

    def decode_to_speakers(self, b_format: np.ndarray, order: int = 1) -> np.ndarray:
        """Decode B-format signal (n_ch, n_samples) to speaker feeds."""
        D = self.decode_matrix(order)
        return D @ b_format


def rotate_soundfield(b_format: np.ndarray, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Rotate first-order B-format by yaw (Z), pitch (Y), roll (X) in radians."""
    if b_format.shape[0] < 4:
        return b_format
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])

    out = b_format.copy()
    xyz = b_format[1:4]
    out[1:4] = R @ xyz
    return out


def energy_vector(speaker_signals: np.ndarray, speaker_positions: np.ndarray) -> tuple[np.ndarray, float]:
    """Gerzon energy vector: direction and spread from speaker feeds."""
    energies = np.mean(speaker_signals ** 2, axis=-1)
    total = energies.sum() + 1e-12
    direction = np.zeros(3)
    for i, (az, el) in enumerate(speaker_positions):
        x = math.cos(az) * math.cos(el)
        y = math.sin(az) * math.cos(el)
        z = math.sin(el)
        direction += energies[i] * np.array([x, y, z])
    direction /= total
    spread = 1.0 - np.linalg.norm(direction)
    return direction, max(0.0, spread)
