"""Binaural rendering: virtual source placement over headphones."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class BinauralSource:
    azimuth_deg: float
    elevation_deg: float
    distance_m: float = 1.5
    gain_db: float = 0.0


@dataclass
class BinauralScene:
    sources: list[BinauralSource] = field(default_factory=list)
    listener_yaw_deg: float = 0.0
    room_size_m: float = 0.0

    def add_source(self, azimuth: float, elevation: float = 0.0,
                   distance: float = 1.5, gain_db: float = 0.0) -> None:
        self.sources.append(BinauralSource(azimuth, elevation, distance, gain_db))


HEAD_RADIUS_M = 0.0875


def itd_from_azimuth(azimuth_deg: float, head_radius: float = HEAD_RADIUS_M) -> float:
    """Interaural time difference (Woodworth model) in seconds."""
    theta = math.radians(azimuth_deg)
    c = 343.0
    return (head_radius / c) * (math.sin(abs(theta)) + abs(theta))


def ild_from_azimuth(azimuth_deg: float, frequency: float) -> float:
    """Approximate interaural level difference in dB."""
    theta = math.radians(azimuth_deg)
    if frequency < 1500.0:
        return 0.0
    shadow = 1.0 + (frequency / 3000.0) * abs(math.sin(theta))
    return 20.0 * math.log10(shadow) * (1.0 if theta > 0 else -1.0)


def crossfeed(left: np.ndarray, right: np.ndarray,
              amount: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Simple crossfeed to reduce extreme stereo separation in headphones."""
    amount = max(0.0, min(amount, 1.0))
    new_left = left * (1.0 - amount) + right * amount
    new_right = right * (1.0 - amount) + left * amount
    return new_left, new_right


def render_binaural_mono(
    mono: np.ndarray,
    source: BinauralSource,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a mono signal to binaural stereo for a single source."""
    itd = itd_from_azimuth(source.azimuth_deg)
    delay_samples = int(abs(itd) * sample_rate)
    gain_linear = 10.0 ** (source.gain_db / 20.0)
    distance_atten = 1.0 / max(source.distance_m, 0.1)
    amplitude = gain_linear * distance_atten

    n = len(mono) + delay_samples
    left = np.zeros(n)
    right = np.zeros(n)

    if itd >= 0:
        left[:len(mono)] = mono * amplitude
        right[delay_samples:delay_samples + len(mono)] = mono * amplitude * 0.9
    else:
        right[:len(mono)] = mono * amplitude
        left[delay_samples:delay_samples + len(mono)] = mono * amplitude * 0.9
    return left, right


def render_scene(
    mono_signals: list[np.ndarray],
    scene: BinauralScene,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a full binaural scene from mono sources."""
    if not mono_signals or not scene.sources:
        return np.zeros(1), np.zeros(1)
    max_len = max(len(s) for s in mono_signals) + int(0.002 * sample_rate)
    left_mix = np.zeros(max_len)
    right_mix = np.zeros(max_len)
    for sig, src in zip(mono_signals, scene.sources):
        rotated = BinauralSource(
            azimuth_deg=src.azimuth_deg - scene.listener_yaw_deg,
            elevation_deg=src.elevation_deg,
            distance_m=src.distance_m,
            gain_db=src.gain_db,
        )
        l, r = render_binaural_mono(sig, rotated, sample_rate)
        left_mix[:len(l)] += l
        right_mix[:len(r)] += r
    return left_mix, right_mix


def head_shadow_filter(signal: np.ndarray, azimuth_deg: float,
                       cutoff_hz: float = 1500.0,
                       sample_rate: int = 44100) -> np.ndarray:
    """Apply a simple low-pass head shadow to the contralateral ear."""
    theta = abs(math.radians(azimuth_deg))
    if theta < 0.1:
        return signal.copy()
    shadow_factor = max(0.3, 1.0 - 0.7 * math.sin(theta))
    n = len(signal)
    freq = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spectrum = np.fft.rfft(signal)
    mask = np.ones_like(freq)
    above = freq > cutoff_hz
    mask[above] *= shadow_factor
    return np.fft.irfft(spectrum * mask, n=n)
