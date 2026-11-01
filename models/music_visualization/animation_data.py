"""Animation data generation: keyframes from audio, beat-sync curves, waveform geometry, color mapping.

Provides keyframe generation driven by spectral features, beat-synchronised
easing functions, progressive spectrogram tiles, and feature-to-color mappings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal as sig


# ---------------------------------------------------------------------------
# Easing functions
# ---------------------------------------------------------------------------

class EasingType(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    CUBIC_BEZIER = "cubic_bezier"


def ease(t: float, kind: EasingType = EasingType.LINEAR) -> float:
    """Apply easing function to normalised parameter t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    if kind == EasingType.LINEAR:
        return t
    elif kind == EasingType.EASE_IN:
        return t * t
    elif kind == EasingType.EASE_OUT:
        return 1.0 - (1.0 - t) ** 2
    elif kind == EasingType.EASE_IN_OUT:
        return 3.0 * t * t - 2.0 * t * t * t
    elif kind == EasingType.CUBIC_BEZIER:
        return 3.0 * t * t - 2.0 * t * t * t
    return t


def interpolate_keyframes(times: np.ndarray, values: np.ndarray,
                          target_times: np.ndarray,
                          easing: EasingType = EasingType.LINEAR) -> np.ndarray:
    """Interpolate between keyframes with easing."""
    result = np.zeros(len(target_times))
    for i, tt in enumerate(target_times):
        idx = np.searchsorted(times, tt, side="right") - 1
        idx = max(0, min(idx, len(times) - 2))
        t0, t1 = times[idx], times[min(idx + 1, len(times) - 1)]
        v0, v1 = values[idx], values[min(idx + 1, len(values) - 1)]
        if t1 <= t0:
            result[i] = v0
        else:
            frac = (tt - t0) / (t1 - t0)
            result[i] = v0 + (v1 - v0) * ease(frac, easing)
    return result


# ---------------------------------------------------------------------------
# Keyframe generation from audio features
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    time: float
    position: np.ndarray   # (x, y, z)
    scale: float
    color_hsl: np.ndarray  # (h, s, l)


def keyframes_from_features(times: np.ndarray,
                            rms: np.ndarray,
                            centroid: np.ndarray,
                            onset_strength: np.ndarray) -> list[Keyframe]:
    """Generate animation keyframes from audio analysis features.

    rms → scale, centroid → hue, onset_strength → position displacement.
    """
    n = len(times)
    kfs: list[Keyframe] = []

    rms_norm = rms / (np.max(rms) + 1e-10)
    cent_norm = centroid / (np.max(centroid) + 1e-10)
    onset_norm = onset_strength / (np.max(onset_strength) + 1e-10)

    for i in range(n):
        hue = float(cent_norm[i]) * 360.0
        saturation = 0.5 + 0.5 * float(rms_norm[i])
        lightness = 0.3 + 0.4 * float(rms_norm[i])

        dx = float(onset_norm[i]) * 2.0 - 1.0
        dy = float(cent_norm[i]) * 2.0 - 1.0
        dz = float(rms_norm[i]) * 0.5

        kfs.append(Keyframe(
            time=float(times[i]),
            position=np.array([dx, dy, dz]),
            scale=0.5 + float(rms_norm[i]),
            color_hsl=np.array([hue, saturation, lightness]),
        ))

    return kfs


# ---------------------------------------------------------------------------
# Beat-synchronised animation
# ---------------------------------------------------------------------------

def snap_to_beats(times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Snap each time to the nearest beat."""
    snapped = np.zeros_like(times)
    for i, t in enumerate(times):
        idx = int(np.argmin(np.abs(beat_times - t)))
        snapped[i] = beat_times[idx]
    return snapped


def beat_sync_curve(beat_times: np.ndarray, values: np.ndarray,
                    sr: float, duration: float,
                    easing: EasingType = EasingType.EASE_IN_OUT) -> np.ndarray:
    """Generate a sample-rate curve that transitions on beats with easing."""
    n_samples = int(duration * sr)
    sample_times = np.arange(n_samples) / sr
    return interpolate_keyframes(beat_times, values, sample_times, easing)


# ---------------------------------------------------------------------------
# Waveform geometry
# ---------------------------------------------------------------------------

@dataclass
class WaveformGeometry:
    times: np.ndarray
    min_envelope: np.ndarray
    max_envelope: np.ndarray
    rms_envelope: np.ndarray


def waveform_display_data(audio: np.ndarray, sr: float,
                          n_points: int = 2000) -> WaveformGeometry:
    """Downsample waveform to N points preserving peaks for display."""
    n = len(audio)
    if n <= n_points:
        t = np.arange(n) / sr
        return WaveformGeometry(t, audio, audio, np.abs(audio))

    chunk_size = n // n_points
    n_use = chunk_size * n_points
    reshaped = audio[:n_use].reshape(n_points, chunk_size)

    times = np.arange(n_points) * chunk_size / sr
    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)
    rms = np.sqrt(np.mean(reshaped ** 2, axis=1))

    return WaveformGeometry(times, mins, maxs, rms)


# ---------------------------------------------------------------------------
# Frequency band energy series
# ---------------------------------------------------------------------------

def band_energy_series(spectrogram: np.ndarray, freqs: np.ndarray,
                       band_edges: list[float] | None = None) -> np.ndarray:
    """Split spectrogram into frequency bands, return per-band energy over time.

    Default bands: sub (<60), bass (60-250), low_mid (250-500),
    mid (500-2k), high_mid (2k-6k), air (6k+).

    Returns (n_bands, n_frames).
    """
    if band_edges is None:
        band_edges = [0, 60, 250, 500, 2000, 6000, float(freqs[-1]) + 1]

    n_bands = len(band_edges) - 1
    n_frames = spectrogram.shape[1]
    energies = np.zeros((n_bands, n_frames))

    for b in range(n_bands):
        lo, hi = band_edges[b], band_edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            energies[b] = np.sum(spectrogram[mask, :] ** 2, axis=0)

    return energies


# ---------------------------------------------------------------------------
# Spectrogram tiles for progressive rendering
# ---------------------------------------------------------------------------

@dataclass
class SpectrogramTile:
    zoom_level: int
    time_start: float
    time_end: float
    freq_start: float
    freq_end: float
    data: np.ndarray


def spectrogram_tiles(spectrogram: np.ndarray, times: np.ndarray,
                      freqs: np.ndarray,
                      n_zoom_levels: int = 4) -> list[SpectrogramTile]:
    """Generate multi-resolution spectrogram tiles for progressive rendering."""
    tiles: list[SpectrogramTile] = []
    F, T = spectrogram.shape

    for level in range(n_zoom_levels):
        factor = 2 ** level
        # Downsample by averaging
        f_step = max(1, F // (F // factor) if F // factor > 0 else 1)
        t_step = max(1, T // (T // factor) if T // factor > 0 else 1)
        f_step = min(f_step, factor)
        t_step = min(t_step, factor)

        f_new = F // f_step
        t_new = T // t_step
        if f_new < 1 or t_new < 1:
            break

        data = spectrogram[:f_new * f_step, :t_new * t_step]
        data = data.reshape(f_new, f_step, t_new, t_step).mean(axis=(1, 3))

        tiles.append(SpectrogramTile(
            zoom_level=level,
            time_start=float(times[0]),
            time_end=float(times[-1]),
            freq_start=float(freqs[0]),
            freq_end=float(freqs[-1]),
            data=data,
        ))

    return tiles


# ---------------------------------------------------------------------------
# Feature-to-color mapping
# ---------------------------------------------------------------------------

@dataclass
class ColorMapping:
    hue: np.ndarray         # 0..360
    saturation: np.ndarray  # 0..1
    lightness: np.ndarray   # 0..1


def feature_to_color(pitch: np.ndarray | None = None,
                     loudness: np.ndarray | None = None,
                     brightness: np.ndarray | None = None,
                     n_frames: int = 0) -> ColorMapping:
    """Map spectral features to HSL color space.

    pitch → hue (C=0°, wraps at 360°)
    loudness → lightness
    brightness (spectral centroid normalised) → saturation
    """
    if n_frames == 0:
        for arr in [pitch, loudness, brightness]:
            if arr is not None:
                n_frames = len(arr)
                break
    if n_frames == 0:
        return ColorMapping(np.array([]), np.array([]), np.array([]))

    if pitch is not None:
        hue = (pitch % 12) / 12.0 * 360.0
    else:
        hue = np.full(n_frames, 200.0)

    if loudness is not None:
        l_norm = loudness / (np.max(np.abs(loudness)) + 1e-10)
        lightness = 0.2 + 0.6 * np.clip(l_norm, 0.0, 1.0)
    else:
        lightness = np.full(n_frames, 0.5)

    if brightness is not None:
        b_norm = brightness / (np.max(brightness) + 1e-10)
        saturation = 0.3 + 0.7 * np.clip(b_norm, 0.0, 1.0)
    else:
        saturation = np.full(n_frames, 0.7)

    return ColorMapping(hue, saturation, lightness)


# ---------------------------------------------------------------------------
# Camera path generation
# ---------------------------------------------------------------------------

@dataclass
class CameraKeyframe:
    time: float
    position: np.ndarray   # (x, y, z)
    look_at: np.ndarray    # (x, y, z)
    fov: float


def camera_path(section_times: np.ndarray,
                rms_curve: np.ndarray,
                duration: float,
                n_keyframes: int = 60) -> list[CameraKeyframe]:
    """Generate a smooth 3D camera path responding to musical structure.

    Section boundaries → camera angle changes.
    Crescendo → zoom in (narrower FOV).
    """
    kf_times = np.linspace(0, duration, n_keyframes)
    kfs: list[CameraKeyframe] = []

    # Base circular orbit
    angles = np.linspace(0, 2 * np.pi, n_keyframes)

    # Interpolate rms to keyframe times
    rms_times = np.linspace(0, duration, len(rms_curve))
    rms_interp = np.interp(kf_times, rms_times, rms_curve)
    rms_norm = rms_interp / (np.max(rms_interp) + 1e-10)

    # Section change detection: nearest section boundary
    for i, t in enumerate(kf_times):
        radius = 5.0 - 2.0 * rms_norm[i]  # closer during loud sections
        x = radius * np.cos(angles[i])
        z = radius * np.sin(angles[i])
        y = 1.0 + rms_norm[i] * 2.0

        # Jitter on section boundaries
        if len(section_times) > 0:
            dist_to_section = np.min(np.abs(section_times - t))
            if dist_to_section < 0.5:
                y += 1.5  # quick upward pan at section changes

        fov = 60.0 - 20.0 * rms_norm[i]  # zoom during loud passages

        kfs.append(CameraKeyframe(
            time=float(t),
            position=np.array([x, y, z]),
            look_at=np.array([0.0, 0.0, 0.0]),
            fov=fov,
        ))

    return kfs
