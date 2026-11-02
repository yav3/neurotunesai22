"""ADSR and amplitude envelope modelling for timbre characterisation."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ADSRParams:
    """Attack-Decay-Sustain-Release envelope parameters."""
    attack_ms: float
    decay_ms: float
    sustain_level: float
    release_ms: float


@dataclass
class EnvelopeResult:
    """Extracted envelope and its ADSR estimate."""
    amplitude_envelope: np.ndarray
    adsr: ADSRParams
    attack_slope: float
    decay_slope: float
    release_slope: float


def amplitude_envelope(signal: np.ndarray, hop_size: int = 256) -> np.ndarray:
    """Extract amplitude envelope via peak-picking per frame."""
    n_frames = max(1, len(signal) // hop_size)
    env = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + hop_size, len(signal))
        env[i] = np.max(np.abs(signal[start:end]))
    return env


def rms_envelope(signal: np.ndarray, hop_size: int = 256) -> np.ndarray:
    """RMS-based amplitude envelope."""
    n_frames = max(1, len(signal) // hop_size)
    env = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + hop_size, len(signal))
        env[i] = np.sqrt(np.mean(signal[start:end] ** 2))
    return env


def estimate_adsr(
    env: np.ndarray,
    sr: int,
    hop_size: int = 256,
    sustain_threshold: float = 0.1,
) -> ADSRParams:
    """Estimate ADSR parameters from an amplitude envelope.

    Simple heuristic: attack = onset to peak; decay = peak to sustain;
    sustain = stable region; release = final drop.
    """
    if len(env) < 3:
        frame_ms = 1000.0 * hop_size / sr
        return ADSRParams(frame_ms, frame_ms, 0.5, frame_ms)

    peak_idx = int(np.argmax(env))
    peak_val = env[peak_idx]
    if peak_val < 1e-10:
        frame_ms = 1000.0 * hop_size / sr
        return ADSRParams(frame_ms, frame_ms, 0.0, frame_ms)

    frame_ms = 1000.0 * hop_size / sr

    attack_ms = max(frame_ms, peak_idx * frame_ms)

    sustain_level = 0.0
    decay_end = peak_idx
    for i in range(peak_idx + 1, len(env)):
        if i + 1 < len(env) and abs(env[i] - env[i + 1]) < sustain_threshold * peak_val:
            sustain_level = env[i] / peak_val
            decay_end = i
            break
    else:
        decay_end = len(env) - 1
        sustain_level = env[-1] / peak_val if peak_val > 0 else 0.0

    decay_ms = max(frame_ms, (decay_end - peak_idx) * frame_ms)

    release_start = decay_end
    for i in range(len(env) - 1, decay_end, -1):
        if env[i] > sustain_threshold * peak_val:
            release_start = i
            break

    release_ms = max(frame_ms, (len(env) - 1 - release_start) * frame_ms)

    return ADSRParams(
        attack_ms=attack_ms,
        decay_ms=decay_ms,
        sustain_level=float(np.clip(sustain_level, 0.0, 1.0)),
        release_ms=release_ms,
    )


def synthesise_adsr(
    adsr: ADSRParams,
    total_ms: float,
    sr: int,
) -> np.ndarray:
    """Generate an ADSR envelope waveform from parameters."""
    total_samples = int(sr * total_ms / 1000.0)
    attack_samples = int(sr * adsr.attack_ms / 1000.0)
    decay_samples = int(sr * adsr.decay_ms / 1000.0)
    release_samples = int(sr * adsr.release_ms / 1000.0)
    sustain_samples = max(0, total_samples - attack_samples - decay_samples - release_samples)

    parts = []
    if attack_samples > 0:
        parts.append(np.linspace(0.0, 1.0, attack_samples))
    if decay_samples > 0:
        parts.append(np.linspace(1.0, adsr.sustain_level, decay_samples))
    if sustain_samples > 0:
        parts.append(np.full(sustain_samples, adsr.sustain_level))
    if release_samples > 0:
        parts.append(np.linspace(adsr.sustain_level, 0.0, release_samples))

    if not parts:
        return np.zeros(1)
    env = np.concatenate(parts)
    return env[:total_samples]


def extract_envelope(
    signal: np.ndarray,
    sr: int,
    hop_size: int = 256,
) -> EnvelopeResult:
    """Full envelope extraction and ADSR estimation."""
    env = amplitude_envelope(signal, hop_size)
    adsr = estimate_adsr(env, sr, hop_size)

    frame_ms = 1000.0 * hop_size / sr
    peak_idx = int(np.argmax(env))
    peak_val = max(env[peak_idx], 1e-10)

    attack_slope = peak_val / max(adsr.attack_ms, frame_ms)
    decay_slope = (peak_val - adsr.sustain_level * peak_val) / max(adsr.decay_ms, frame_ms)
    release_slope = (adsr.sustain_level * peak_val) / max(adsr.release_ms, frame_ms)

    return EnvelopeResult(
        amplitude_envelope=env,
        adsr=adsr,
        attack_slope=float(attack_slope),
        decay_slope=float(decay_slope),
        release_slope=float(release_slope),
    )
