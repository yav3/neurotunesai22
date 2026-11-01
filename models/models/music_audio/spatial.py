"""Spatial audio: panning, ambisonics, room simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpatialPosition:
    azimuth: float = 0.0
    elevation: float = 0.0
    distance: float = 1.0


class SpatialPanner:
    @staticmethod
    def stereo_pan(signal: np.ndarray, pan: float = 0.0) -> np.ndarray:
        pan = max(-1.0, min(1.0, pan))
        angle = (pan + 1.0) * np.pi / 4.0
        left = np.cos(angle)
        right = np.sin(angle)
        sig = np.asarray(signal, dtype=np.float32)
        return np.column_stack([sig * left, sig * right])

    @staticmethod
    def binaural_pan(
        signal: np.ndarray, azimuth: float, elevation: float = 0.0, sr: int = 22050,
    ) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32)
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        head_radius = 0.0875
        c = 343.0
        itd_s = (head_radius / c) * (az_rad + np.sin(az_rad))
        itd_samples = int(abs(itd_s) * sr)
        cos_el = np.cos(el_rad)
        ild_factor = 0.5 * np.sin(az_rad) * cos_el
        left_gain = max(0.0, 1.0 - ild_factor)
        right_gain = max(0.0, 1.0 + ild_factor)
        n = len(sig) + itd_samples
        left = np.zeros(n, dtype=np.float32)
        right = np.zeros(n, dtype=np.float32)
        if azimuth >= 0:
            left[:len(sig)] = sig * left_gain
            right[itd_samples:itd_samples + len(sig)] = sig * right_gain
        else:
            left[itd_samples:itd_samples + len(sig)] = sig * left_gain
            right[:len(sig)] = sig * right_gain
        return np.column_stack([left, right])

    @staticmethod
    def surround_encode(
        signal: np.ndarray, positions: list[SpatialPosition], n_channels: int = 6,
    ) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32)
        out = np.zeros((len(sig), n_channels), dtype=np.float32)
        if not positions:
            out[:, 0] = sig
            return out
        for pos in positions:
            az = np.radians(pos.azimuth % 360)
            gains = np.zeros(n_channels)
            gains[0] = max(0.0, np.cos(az)) * max(0.0, np.cos(np.radians(pos.elevation)))
            if n_channels > 1:
                gains[1] = max(0.0, np.cos(az - np.pi)) * max(0.0, np.cos(np.radians(pos.elevation)))
            for ch in range(2, n_channels):
                angle = 2 * np.pi * ch / n_channels
                gains[ch] = max(0.0, np.cos(az - angle))
            total = np.sum(gains)
            if total > 1e-12:
                gains /= total
            atten = 1.0 / max(pos.distance, 0.1)
            for ch in range(n_channels):
                out[:, ch] += sig * gains[ch] * atten
        return out


class AmbisonicsEncoder:
    @staticmethod
    def encode_bformat(
        signal: np.ndarray, azimuth: float, elevation: float = 0.0,
    ) -> dict[str, np.ndarray]:
        sig = np.asarray(signal, dtype=np.float32)
        az = np.radians(azimuth)
        el = np.radians(elevation)
        w = sig * (1.0 / np.sqrt(2.0))
        x = sig * np.cos(az) * np.cos(el)
        y = sig * np.sin(az) * np.cos(el)
        z = sig * np.sin(el)
        return {"W": w, "X": x, "Y": y, "Z": z}


class RoomSimulator:
    @staticmethod
    def early_reflections(
        signal: np.ndarray,
        room_dims: tuple[float, float, float],
        source_pos: tuple[float, float, float],
        listener_pos: tuple[float, float, float],
        sr: int = 22050,
    ) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32)
        c = 343.0
        reflections = []
        for axis in range(3):
            for sign in (-1, 1):
                image = list(source_pos)
                wall = room_dims[axis] if sign > 0 else 0.0
                image[axis] = 2 * wall - source_pos[axis]
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(image, listener_pos)))
                delay_s = dist / c
                atten = 0.7 / max(dist, 0.1)
                reflections.append((delay_s, atten))
        direct_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(source_pos, listener_pos)))
        max_delay = max(r[0] for r in reflections)
        out_len = len(sig) + int(max_delay * sr) + 1
        out = np.zeros(out_len, dtype=np.float32)
        direct_delay = int(direct_dist / c * sr)
        out[direct_delay:direct_delay + len(sig)] += sig
        for delay_s, atten in reflections:
            d = int(delay_s * sr)
            end = min(d + len(sig), out_len)
            seg = min(len(sig), end - d)
            out[d:d + seg] += sig[:seg] * atten
        return out

    @staticmethod
    def reverb_tail(signal: np.ndarray, rt60: float = 1.0, sr: int = 22050) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32)
        n_tail = int(rt60 * sr)
        total = len(sig) + n_tail
        out = np.zeros(total, dtype=np.float32)
        out[:len(sig)] = sig
        decay = np.exp(-np.log(1000.0) * np.arange(n_tail) / max(n_tail, 1))
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(n_tail).astype(np.float32)
        rms = np.sqrt(np.mean(sig ** 2)) if len(sig) > 0 else 0.01
        tail = noise * decay * rms * 0.3
        out[len(sig):] += tail
        return out
