"""Granular synthesis: grain extraction, scattering, time-stretching, freezing.

Microsound-level synthesis following Roads (2001) and Truax (1988).
All operations are pure numpy with sample-accurate grain placement.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EnvelopeShape(str, Enum):
    HANN = "hann"
    GAUSSIAN = "gaussian"
    TRAPEZOIDAL = "trapezoidal"
    TRIANGULAR = "triangular"
    TUKEY = "tukey"


@dataclass
class GrainEnvelope:
    shape: EnvelopeShape = EnvelopeShape.HANN
    attack_ratio: float = 0.1
    release_ratio: float = 0.1

    def generate(self, n_samples: int) -> np.ndarray:
        if self.shape == EnvelopeShape.HANN:
            return np.hanning(n_samples)
        elif self.shape == EnvelopeShape.GAUSSIAN:
            t = np.linspace(-3, 3, n_samples)
            return np.exp(-0.5 * t ** 2)
        elif self.shape == EnvelopeShape.TRIANGULAR:
            return np.bartlett(n_samples)
        elif self.shape == EnvelopeShape.TUKEY:
            alpha = self.attack_ratio + self.release_ratio
            alpha = min(alpha, 1.0)
            n = n_samples
            env = np.ones(n)
            rise = int(n * alpha / 2)
            if rise > 0:
                env[:rise] = 0.5 * (1 - np.cos(np.pi * np.arange(rise) / rise))
                env[-rise:] = 0.5 * (1 - np.cos(np.pi * np.arange(rise, 0, -1) / rise))
            return env
        else:  # trapezoidal
            env = np.ones(n_samples)
            attack_len = max(1, int(n_samples * self.attack_ratio))
            release_len = max(1, int(n_samples * self.release_ratio))
            env[:attack_len] = np.linspace(0, 1, attack_len)
            env[-release_len:] = np.linspace(1, 0, release_len)
            return env


@dataclass
class Grain:
    samples: np.ndarray
    position: int
    amplitude: float = 1.0


class GranularSynthesizer:
    """Granular synthesis engine for time-stretching, freezing, and texture."""

    @staticmethod
    def extract_grain(
        source: np.ndarray,
        start: int,
        duration: int,
        envelope: GrainEnvelope | None = None,
        pitch_shift: float = 1.0,
    ) -> np.ndarray:
        """Extract a windowed grain from source, with optional pitch shift.

        pitch_shift > 1.0 raises pitch (shorter grain resampled),
        pitch_shift < 1.0 lowers pitch.
        """
        env = envelope or GrainEnvelope()
        end = min(start + duration, len(source))
        grain = source[start:end].copy()
        if len(grain) < duration:
            grain = np.pad(grain, (0, duration - len(grain)))

        if abs(pitch_shift - 1.0) > 1e-6:
            n_new = max(1, int(len(grain) / pitch_shift))
            indices = np.linspace(0, len(grain) - 1, n_new)
            grain = np.interp(indices, np.arange(len(grain)), grain)
            if len(grain) < duration:
                grain = np.pad(grain, (0, duration - len(grain)))
            else:
                grain = grain[:duration]

        window = env.generate(len(grain))
        return grain * window

    @staticmethod
    def scatter_grains(
        source: np.ndarray,
        grain_density: float = 50.0,
        grain_duration: int = 1024,
        jitter: float = 0.5,
        pitch_spread: float = 0.0,
        duration: int | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Asynchronous granular synthesis: randomly scattered grains.

        Parameters
        ----------
        source : source waveform to extract grains from
        grain_density : average grains per second (at sr=22050)
        grain_duration : grain length in samples
        jitter : randomness in grain spacing [0, 1]
        pitch_spread : semitone range for random pitch variation
        duration : output length in samples (default: len(source))
        """
        rng = np.random.default_rng(seed)
        out_len = duration or len(source)
        output = np.zeros(out_len)

        avg_spacing = max(1, int(22050 / grain_density))
        pos = 0
        while pos < out_len:
            src_start = rng.integers(0, max(1, len(source) - grain_duration))
            shift = 1.0
            if pitch_spread > 0:
                semitones = rng.uniform(-pitch_spread, pitch_spread)
                shift = 2.0 ** (semitones / 12.0)

            grain = GranularSynthesizer.extract_grain(
                source, src_start, grain_duration, pitch_shift=shift
            )
            end = min(pos + len(grain), out_len)
            output[pos:end] += grain[: end - pos]

            spacing = avg_spacing
            if jitter > 0:
                spacing = int(spacing * (1.0 + rng.uniform(-jitter, jitter)))
            pos += max(1, spacing)

        peak = np.max(np.abs(output))
        if peak > 1e-6:
            output /= peak
        return output

    @staticmethod
    def time_stretch_granular(
        source: np.ndarray,
        rate: float = 1.0,
        grain_size: int = 2048,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """Granular time-stretching: overlap-add with analysis/synthesis hop.

        rate > 1.0 = faster (shorter output), rate < 1.0 = slower (longer).
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        analysis_hop = max(1, int(grain_size * (1 - overlap)))
        synthesis_hop = max(1, int(analysis_hop / rate))
        n_grains = max(1, (len(source) - grain_size) // analysis_hop + 1)
        out_len = (n_grains - 1) * synthesis_hop + grain_size
        output = np.zeros(out_len)
        norm = np.zeros(out_len)
        window = np.hanning(grain_size)

        for i in range(n_grains):
            a_start = i * analysis_hop
            if a_start + grain_size > len(source):
                break
            grain = source[a_start : a_start + grain_size] * window
            s_start = i * synthesis_hop
            end = min(s_start + grain_size, out_len)
            g_len = end - s_start
            output[s_start:end] += grain[:g_len]
            norm[s_start:end] += window[:g_len] ** 2

        norm[norm < 1e-10] = 1.0
        return output / norm

    @staticmethod
    def freeze(
        source: np.ndarray,
        position: int,
        grain_size: int = 2048,
        n_grains: int = 50,
        spread: int = 0,
        seed: int = 42,
    ) -> np.ndarray:
        """Spectral freeze at a position: repeated grains from one location.

        spread controls how many samples around the position grains can start from.
        """
        rng = np.random.default_rng(seed)
        env = GrainEnvelope(shape=EnvelopeShape.HANN)
        hop = grain_size // 2
        out_len = (n_grains - 1) * hop + grain_size
        output = np.zeros(out_len)
        norm = np.zeros(out_len)
        window = env.generate(grain_size)

        for i in range(n_grains):
            offset = 0
            if spread > 0:
                offset = rng.integers(-spread, spread + 1)
            start = max(0, min(position + offset, len(source) - grain_size))
            grain = source[start : start + grain_size]
            if len(grain) < grain_size:
                grain = np.pad(grain, (0, grain_size - len(grain)))
            grain = grain * window
            s = i * hop
            end = min(s + grain_size, out_len)
            g_len = end - s
            output[s:end] += grain[:g_len]
            norm[s:end] += window[:g_len] ** 2

        norm[norm < 1e-10] = 1.0
        return output / norm

    @staticmethod
    def granular_delay(
        source: np.ndarray,
        delay_samples: int = 11025,
        feedback: float = 0.5,
        grain_size: int = 1024,
        n_repeats: int = 5,
    ) -> np.ndarray:
        """Granular delay: each repeat is re-granulated with scatter."""
        feedback = np.clip(feedback, 0.0, 0.99)
        out_len = len(source) + delay_samples * n_repeats
        output = np.zeros(out_len)
        output[: len(source)] = source

        current = source.copy()
        for i in range(1, n_repeats + 1):
            amplitude = feedback ** i
            if amplitude < 1e-6:
                break
            scattered = GranularSynthesizer.scatter_grains(
                current, grain_duration=grain_size, duration=len(current), seed=42 + i
            )
            offset = i * delay_samples
            end = min(offset + len(scattered), out_len)
            output[offset:end] += scattered[: end - offset] * amplitude

        peak = np.max(np.abs(output))
        if peak > 1e-6:
            output /= peak
        return output
