"""Spatial audio coding: Ambisonics, binaural rendering, diffuseness estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "AmbisonicSignal", "AmbisonicConfig",
    "ambisonic_encode", "ambisonic_decode",
    "binaural_render_ambisonic", "directional_loudness_map",
    "diffuseness_estimate", "spatial_coherence",
    "spherical_harmonic_real", "ambisonic_rotation",
    "energy_vector", "velocity_vector",
    "max_re_decode_weights",
]


def _factorial(n: int) -> float:
    r = 1.0
    for i in range(2, n + 1):
        r *= i
    return r


def _associated_legendre(l: int, m: int, x: float) -> float:
    am = abs(m)
    if am > l:
        return 0.0
    pmm = 1.0
    if am > 0:
        somx2 = np.sqrt(max(1.0 - x * x, 0.0))
        fact = 1.0
        for i in range(1, am + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == am:
        return pmm
    pmmp1 = x * (2 * am + 1) * pmm
    if l == am + 1:
        return pmmp1
    pll = 0.0
    for ll in range(am + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + am - 1) * pmm) / (ll - am)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def spherical_harmonic_real(l: int, m: int, azimuth: float, elevation: float) -> float:
    norm = np.sqrt(
        (2 * l + 1) / (4 * np.pi)
        * _factorial(l - abs(m)) / max(_factorial(l + abs(m)), 1e-30)
    )
    plm = _associated_legendre(l, abs(m), np.sin(elevation))
    if m > 0:
        return float(norm * plm * np.sqrt(2) * np.cos(m * azimuth))
    elif m < 0:
        return float(norm * plm * np.sqrt(2) * np.sin(abs(m) * azimuth))
    return float(norm * plm)


@dataclass
class AmbisonicConfig:
    order: int = 1
    normalization: str = "SN3D"  # SN3D, N3D, FuMa
    channel_ordering: str = "ACN"  # ACN, FuMa

    @property
    def n_channels(self) -> int:
        return (self.order + 1) ** 2


@dataclass
class AmbisonicSignal:
    channels: np.ndarray
    config: AmbisonicConfig
    sr: int = 48000


def ambisonic_encode(
    mono_signal: np.ndarray,
    azimuth: float,
    elevation: float,
    config: AmbisonicConfig = AmbisonicConfig(),
) -> AmbisonicSignal:
    n_ch = config.n_channels
    n_samples = len(mono_signal)
    channels = np.zeros((n_ch, n_samples))
    ch_idx = 0
    for l in range(config.order + 1):
        for m in range(-l, l + 1):
            coeff = spherical_harmonic_real(l, m, azimuth, elevation)
            if config.normalization == "N3D":
                coeff *= np.sqrt(2 * l + 1)
            channels[ch_idx] = mono_signal * coeff
            ch_idx += 1
    return AmbisonicSignal(channels=channels, config=config)


def ambisonic_decode(
    ambi: AmbisonicSignal,
    speaker_azimuths: np.ndarray,
    speaker_elevations: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_speakers = len(speaker_azimuths)
    if speaker_elevations is None:
        speaker_elevations = np.zeros(n_speakers)
    decode_matrix = np.zeros((n_speakers, ambi.config.n_channels))
    for s in range(n_speakers):
        ch_idx = 0
        for l in range(ambi.config.order + 1):
            for m in range(-l, l + 1):
                decode_matrix[s, ch_idx] = spherical_harmonic_real(
                    l, m, speaker_azimuths[s], speaker_elevations[s]
                )
                ch_idx += 1
    return decode_matrix @ ambi.channels


def max_re_decode_weights(order: int) -> np.ndarray:
    weights = np.zeros(order + 1)
    for l in range(order + 1):
        weights[l] = np.cos(l * np.pi / (2 * (order + 1)))
    return weights


def binaural_render_ambisonic(
    ambi: AmbisonicSignal,
    hrtf_left: np.ndarray,
    hrtf_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_ch = min(ambi.config.n_channels, hrtf_left.shape[0], hrtf_right.shape[0])
    n_samples = ambi.channels.shape[1]
    left = np.zeros(n_samples)
    right = np.zeros(n_samples)
    for i in range(n_ch):
        left += ambi.channels[i] * hrtf_left[i, 0] if hrtf_left.ndim > 1 else ambi.channels[i] * hrtf_left[i]
        right += ambi.channels[i] * hrtf_right[i, 0] if hrtf_right.ndim > 1 else ambi.channels[i] * hrtf_right[i]
    return left, right


def directional_loudness_map(
    ambi: AmbisonicSignal,
    n_azimuth: int = 36,
    n_elevation: int = 18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuths = np.linspace(-np.pi, np.pi, n_azimuth, endpoint=False)
    elevations = np.linspace(-np.pi / 2, np.pi / 2, n_elevation)
    loudness = np.zeros((n_elevation, n_azimuth))
    for ei, el in enumerate(elevations):
        for ai, az in enumerate(azimuths):
            coeffs = []
            for l in range(ambi.config.order + 1):
                for m in range(-l, l + 1):
                    coeffs.append(spherical_harmonic_real(l, m, az, el))
            coeffs = np.array(coeffs)
            n_ch = min(len(coeffs), ambi.channels.shape[0])
            energy = np.mean(np.sum(ambi.channels[:n_ch] * coeffs[:n_ch, np.newaxis], axis=0) ** 2)
            loudness[ei, ai] = energy
    return azimuths, elevations, loudness


def diffuseness_estimate(
    ambi: AmbisonicSignal,
    frame_size: int = 1024,
) -> np.ndarray:
    if ambi.config.n_channels < 4:
        return np.zeros(1)
    n_samples = ambi.channels.shape[1]
    n_frames = max(1, n_samples // frame_size)
    diffuseness = np.zeros(n_frames)
    for f in range(n_frames):
        start = f * frame_size
        end = min(start + frame_size, n_samples)
        w = ambi.channels[0, start:end]
        xyz = ambi.channels[1:4, start:end]
        w_energy = np.mean(w ** 2)
        xyz_energy = np.mean(np.sum(xyz ** 2, axis=0))
        if w_energy < 1e-15:
            diffuseness[f] = 1.0
            continue
        intensity_ratio = np.sqrt(xyz_energy) / (np.sqrt(w_energy) * np.sqrt(3))
        diffuseness[f] = 1.0 - np.clip(intensity_ratio, 0.0, 1.0)
    return diffuseness


def spatial_coherence(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    frame_size: int = 1024,
) -> np.ndarray:
    n = min(len(signal_a), len(signal_b))
    n_frames = max(1, n // frame_size)
    coh = np.zeros(n_frames)
    for f in range(n_frames):
        s = f * frame_size
        e = min(s + frame_size, n)
        a = signal_a[s:e]
        b = signal_b[s:e]
        cross = np.mean(a * b)
        auto_a = np.mean(a ** 2)
        auto_b = np.mean(b ** 2)
        denom = np.sqrt(auto_a * auto_b)
        coh[f] = abs(cross) / max(denom, 1e-15)
    return coh


def energy_vector(ambi: AmbisonicSignal) -> np.ndarray:
    if ambi.config.n_channels < 4:
        return np.zeros(3)
    w = ambi.channels[0]
    xyz = ambi.channels[1:4]
    w_sq = np.mean(w ** 2)
    if w_sq < 1e-15:
        return np.zeros(3)
    ev = np.array([np.mean(xyz[i] * w) for i in range(3)])
    return ev / max(np.linalg.norm(ev), 1e-15)


def velocity_vector(ambi: AmbisonicSignal) -> np.ndarray:
    if ambi.config.n_channels < 4:
        return np.zeros(3)
    w = ambi.channels[0]
    xyz = ambi.channels[1:4]
    w_mean = np.mean(w ** 2)
    if w_mean < 1e-15:
        return np.zeros(3)
    vv = np.array([np.mean(xyz[i] * w) for i in range(3)]) / w_mean
    norm = np.linalg.norm(vv)
    return vv / max(norm, 1e-15)


def ambisonic_rotation(
    ambi: AmbisonicSignal,
    yaw: float,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> AmbisonicSignal:
    if ambi.config.order == 0:
        return AmbisonicSignal(ambi.channels.copy(), ambi.config, ambi.sr)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr_val = np.cos(roll), np.sin(roll)
    R = np.array([
        [cy * cp, cy * sp * sr_val - sy * cr, cy * sp * cr + sy * sr_val],
        [sy * cp, sy * sp * sr_val + cy * cr, sy * sp * cr - cy * sr_val],
        [-sp, cp * sr_val, cp * cr],
    ])
    new_channels = ambi.channels.copy()
    if ambi.config.n_channels >= 4:
        xyz = ambi.channels[1:4]
        new_channels[1:4] = R @ xyz
    return AmbisonicSignal(new_channels, ambi.config, ambi.sr)
