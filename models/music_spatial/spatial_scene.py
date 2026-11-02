"""Spatial audio scene description, rendering, and analysis.

A scene graph with positioned sources, listener, and room. Supports
rendering to binaural, stereo, or ambisonics; trajectory animation;
spatial metrics (envelopment, width, localisation accuracy).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> NDArray:
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: "Vec3") -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def azimuth_elevation(self, origin: "Vec3") -> Tuple[float, float]:
        d = self.to_array() - origin.to_array()
        r = np.linalg.norm(d)
        if r < 1e-10:
            return 0.0, 0.0
        az = np.arctan2(d[1], d[0])
        el = np.arcsin(np.clip(d[2] / r, -1, 1))
        return float(az), float(el)


@dataclass
class SpatialSource:
    signal: NDArray
    position: Vec3
    gain_db: float = 0.0
    directivity: float = 1.0  # 0=omni, 1=cardioid, 2=figure-8


@dataclass
class Listener:
    position: Vec3 = field(default_factory=Vec3)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class SpatialScene:
    """Container for a spatial audio scene."""
    sources: List[SpatialSource] = field(default_factory=list)
    listener: Listener = field(default_factory=Listener)
    sr: int = 44100

    def add_source(
        self,
        signal: NDArray,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        gain_db: float = 0.0,
    ) -> int:
        src = SpatialSource(
            signal=np.asarray(signal, dtype=np.float64),
            position=Vec3(x, y, z),
            gain_db=gain_db,
        )
        self.sources.append(src)
        return len(self.sources) - 1

    @property
    def duration_samples(self) -> int:
        if not self.sources:
            return 0
        return max(len(s.signal) for s in self.sources)


# --- Rendering ---

def _distance_gain(distance: float, ref: float = 1.0) -> float:
    return ref / max(distance, 0.01)


def _directivity_gain(
    source: SpatialSource,
    listener: Listener,
) -> float:
    """Cardioid-family directivity: 1 on-axis, reduced off-axis."""
    az, el = source.position.azimuth_elevation(listener.position)
    # Relative angle w.r.t. source facing direction (assume facing listener)
    # For simplicity, directivity pattern = (1 + d·cos(θ)) / (1 + d)
    # where d is directivity factor and θ is off-axis angle (0 here = facing listener)
    d = source.directivity
    return 1.0 / (1.0 + d)  # on-axis gain (normalised peak)


def render_stereo(scene: SpatialScene) -> NDArray:
    """Render scene to stereo (2, N) using constant-power panning."""
    N = scene.duration_samples
    out = np.zeros((2, N))

    for src in scene.sources:
        az, el = src.position.azimuth_elevation(scene.listener.position)
        # Adjust for listener yaw
        az_rel = az - scene.listener.yaw
        dist = src.position.distance_to(scene.listener.position)
        gain_linear = 10.0 ** (src.gain_db / 20.0) * _distance_gain(dist)

        # Constant-power pan: L = cos(θ), R = sin(θ) where θ ∈ [0, π/2]
        pan_angle = np.clip((az_rel + np.pi / 2) / np.pi, 0.0, 1.0) * (np.pi / 2)
        gain_l = gain_linear * np.cos(pan_angle)
        gain_r = gain_linear * np.sin(pan_angle)

        n = len(src.signal)
        out[0, :n] += gain_l * src.signal
        out[1, :n] += gain_r * src.signal

    return out


def render_binaural_simple(
    scene: SpatialScene,
    head_radius: float = 0.0875,
) -> NDArray:
    """Simple binaural rendering with ITD + ILD (no HRTF convolution).

    Returns (2, N) — left ear, right ear.
    """
    c = 343.0
    N = scene.duration_samples
    out = np.zeros((2, N))

    for src in scene.sources:
        az, el = src.position.azimuth_elevation(scene.listener.position)
        az_rel = az - scene.listener.yaw
        dist = src.position.distance_to(scene.listener.position)
        gain = 10.0 ** (src.gain_db / 20.0) * _distance_gain(dist)

        # Woodworth ITD model
        if abs(az_rel) <= np.pi / 2:
            itd = head_radius / c * (az_rel + np.sin(az_rel))
        else:
            itd = head_radius / c * (np.pi - az_rel + np.sin(az_rel))

        itd_samples = itd * scene.sr
        n = len(src.signal)

        # ILD: simplified head-shadow
        ild_db = 1.5 * np.sin(az_rel)  # dB, positive = louder on near ear
        gain_l = gain * 10.0 ** (ild_db / 20.0)
        gain_r = gain * 10.0 ** (-ild_db / 20.0)

        # Apply fractional delay via linear interpolation
        delay_l = max(-itd_samples, 0)
        delay_r = max(itd_samples, 0)

        for ch_idx, (delay, g) in enumerate([(delay_l, gain_l), (delay_r, gain_r)]):
            d_int = int(np.floor(delay))
            d_frac = delay - d_int
            sig = src.signal * g
            if d_int + n <= N:
                out[ch_idx, d_int:d_int + n] += sig * (1 - d_frac)
                if d_int + 1 + n <= N:
                    out[ch_idx, d_int + 1:d_int + 1 + n] += sig * d_frac

    return out


# --- Trajectory ---

@dataclass
class Keyframe:
    time_s: float
    position: Vec3


def interpolate_trajectory(
    keyframes: List[Keyframe],
    sr: int,
    duration_s: float,
) -> NDArray:
    """Linearly interpolate a source trajectory from keyframes.

    Returns (N, 3) position array at sample rate.
    """
    N = int(duration_s * sr)
    times = np.array([kf.time_s for kf in keyframes])
    positions = np.array([[kf.position.x, kf.position.y, kf.position.z]
                          for kf in keyframes])
    t = np.linspace(0, duration_s, N)
    trajectory = np.zeros((N, 3))
    for dim in range(3):
        trajectory[:, dim] = np.interp(t, times, positions[:, dim])
    return trajectory


def source_velocity(trajectory: NDArray, sr: int) -> NDArray:
    """Instantaneous velocity (N, 3) from position trajectory via finite differences."""
    vel = np.zeros_like(trajectory)
    vel[1:-1] = (trajectory[2:] - trajectory[:-2]) * sr / 2.0
    vel[0] = (trajectory[1] - trajectory[0]) * sr
    vel[-1] = (trajectory[-1] - trajectory[-2]) * sr
    return vel


# --- Spatial metrics ---

def apparent_source_width(
    left: NDArray,
    right: NDArray,
    frame_size: int = 2048,
) -> NDArray:
    """Estimate apparent source width (ASW) from IACC.

    ASW ∝ 1 − IACC_early. Returns per-frame ASW ∈ [0, 1].
    """
    n_frames = min(len(left), len(right)) // frame_size
    asw = np.zeros(n_frames)
    for i in range(n_frames):
        s, e = i * frame_size, (i + 1) * frame_size
        l_frame = left[s:e]
        r_frame = right[s:e]
        # Normalised cross-correlation at lag 0
        denom = np.sqrt(np.sum(l_frame ** 2) * np.sum(r_frame ** 2))
        if denom < 1e-12:
            asw[i] = 0.0
            continue
        iacc = np.abs(np.sum(l_frame * r_frame)) / denom
        asw[i] = 1.0 - iacc
    return asw


def listener_envelopment(
    left: NDArray,
    right: NDArray,
    frame_size: int = 4096,
) -> NDArray:
    """Estimate listener envelopment (LEV) from late IACC.

    LEV ∝ 1 − IACC_late. Approximated by using larger analysis frames.
    """
    return apparent_source_width(left, right, frame_size)


def interaural_cross_correlation(
    left: NDArray,
    right: NDArray,
    max_lag_samples: int = 44,
) -> Tuple[float, int]:
    """Maximum IACC and the lag at which it occurs.

    max_lag_samples: typically ±1 ms at 44100 Hz.
    """
    left = left - np.mean(left)
    right = right - np.mean(right)
    denom = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    if denom < 1e-12:
        return 0.0, 0

    best_cc = -1.0
    best_lag = 0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            cc = np.sum(left[lag:] * right[:len(right) - lag]) / denom
        else:
            cc = np.sum(left[:len(left) + lag] * right[-lag:]) / denom
        if abs(cc) > best_cc:
            best_cc = abs(cc)
            best_lag = lag

    return float(best_cc), best_lag


def spatial_complexity(scene: SpatialScene) -> Dict[str, float]:
    """Compute spatial complexity metrics for a scene.

    Returns dict with: n_sources, mean_distance, spatial_spread (std of
    inter-source angles), angular_coverage (fraction of sphere covered).
    """
    if not scene.sources:
        return {"n_sources": 0, "mean_distance": 0.0,
                "spatial_spread": 0.0, "angular_coverage": 0.0}

    positions = np.array([s.position.to_array() for s in scene.sources])
    listener = scene.listener.position.to_array()
    relative = positions - listener
    distances = np.linalg.norm(relative, axis=1)

    # Azimuth/elevation pairs
    azimuths = np.arctan2(relative[:, 1], relative[:, 0])
    elevations = np.arcsin(np.clip(
        relative[:, 2] / np.maximum(distances, 1e-10), -1, 1
    ))

    # Spatial spread: circular std of azimuths
    mean_cos = np.mean(np.cos(azimuths))
    mean_sin = np.mean(np.sin(azimuths))
    R = np.sqrt(mean_cos ** 2 + mean_sin ** 2)
    circular_std = np.sqrt(-2.0 * np.log(max(R, 1e-10)))

    # Angular coverage: fraction of 2π covered by source positions
    az_sorted = np.sort(azimuths)
    if len(az_sorted) > 1:
        gaps = np.diff(az_sorted)
        gaps = np.append(gaps, 2 * np.pi - (az_sorted[-1] - az_sorted[0]))
        max_gap = np.max(gaps)
        coverage = 1.0 - max_gap / (2 * np.pi)
    else:
        coverage = 0.0

    return {
        "n_sources": len(scene.sources),
        "mean_distance": float(np.mean(distances)),
        "spatial_spread": float(circular_std),
        "angular_coverage": float(np.clip(coverage, 0, 1)),
    }
