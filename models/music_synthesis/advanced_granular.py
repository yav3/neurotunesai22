"""Advanced granular synthesis: stereo clouds, spectral grains, Poisson scheduling.

Extends the basic granular engine with:
- Multiple envelope shapes (Gaussian, Tukey, trapezoidal, exponential decay)
- Stereo grain rendering with per-grain panning
- Synchronous and asynchronous (Poisson) grain scheduling
- FFT-domain spectral grain manipulation (phase randomization)
- Grain density analysis and cloud statistics
Pure numpy/scipy — no external dependencies.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence
from enum import Enum


class GrainEnvelopeType(Enum):
    GAUSSIAN = "gaussian"
    HANNING = "hanning"
    TUKEY = "tukey"
    TRAPEZOIDAL = "trapezoidal"
    EXPODEC = "expodec"


@dataclass
class GrainSpec:
    start_sample: int
    duration_samples: int
    pitch_shift: float = 0.0       # semitones
    amplitude: float = 1.0
    envelope: GrainEnvelopeType = GrainEnvelopeType.HANNING
    pan: float = 0.0               # -1 left, 0 centre, +1 right


@dataclass
class StereoCloud:
    grains: List[GrainSpec]
    base_signal: np.ndarray
    sample_rate: float


def envelope_function(n: int, env_type: GrainEnvelopeType,
                      tukey_alpha: float = 0.5,
                      decay_tau: float = 0.3) -> np.ndarray:
    if n < 2:
        return np.ones(max(n, 1))
    if env_type == GrainEnvelopeType.GAUSSIAN:
        t = np.linspace(-3, 3, n)
        return np.exp(-0.5 * t * t)
    elif env_type == GrainEnvelopeType.HANNING:
        return np.hanning(n)
    elif env_type == GrainEnvelopeType.TUKEY:
        w = np.ones(n)
        ramp = max(1, int(tukey_alpha * n / 2))
        w[:ramp] = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))
        w[-ramp:] = w[:ramp][::-1]
        return w
    elif env_type == GrainEnvelopeType.TRAPEZOIDAL:
        w = np.ones(n)
        ramp = max(1, n // 4)
        w[:ramp] = np.linspace(0, 1, ramp)
        w[-ramp:] = np.linspace(1, 0, ramp)
        return w
    elif env_type == GrainEnvelopeType.EXPODEC:
        t = np.linspace(0, 1, n)
        env = np.exp(-t / max(decay_tau, 1e-6))
        attack = max(1, n // 10)
        env[:attack] *= np.linspace(0, 1, attack)
        return env
    return np.hanning(n)


def extract_grain(signal: np.ndarray, start: int, duration: int) -> np.ndarray:
    n = len(signal)
    s = max(0, min(start, n - 1))
    e = min(s + duration, n)
    grain = signal[s:e].copy().astype(np.float64)
    if len(grain) < duration:
        grain = np.pad(grain, (0, duration - len(grain)))
    return grain


def pitch_shift_grain(grain: np.ndarray, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-6:
        return grain.copy()
    ratio = 2.0 ** (semitones / 12.0)
    new_len = max(2, int(len(grain) / ratio))
    shifted = resample(grain, new_len)
    if len(shifted) < len(grain):
        shifted = np.pad(shifted, (0, len(grain) - len(shifted)))
    return shifted[: len(grain)]


def synthesize_grain(grain_spec: GrainSpec, base_signal: np.ndarray,
                     output_left: np.ndarray, output_right: np.ndarray) -> None:
    raw = extract_grain(base_signal, grain_spec.start_sample,
                        grain_spec.duration_samples)
    env = envelope_function(grain_spec.duration_samples, grain_spec.envelope)
    raw *= env
    if abs(grain_spec.pitch_shift) > 1e-6:
        raw = pitch_shift_grain(raw, grain_spec.pitch_shift)
    raw *= grain_spec.amplitude
    pan_r = (grain_spec.pan + 1.0) / 2.0
    pan_l = 1.0 - pan_r
    gain_l = np.sqrt(pan_l)
    gain_r = np.sqrt(pan_r)
    n = min(len(raw), len(output_left) - grain_spec.start_sample)
    if n <= 0:
        return
    s = grain_spec.start_sample
    output_left[s: s + n] += raw[:n] * gain_l
    output_right[s: s + n] += raw[:n] * gain_r


def render_cloud(cloud: StereoCloud) -> np.ndarray:
    total = len(cloud.base_signal)
    for g in cloud.grains:
        total = max(total, g.start_sample + g.duration_samples)
    left = np.zeros(total)
    right = np.zeros(total)
    for g in cloud.grains:
        synthesize_grain(g, cloud.base_signal, left, right)
    return np.stack([left, right])


def synchronous_granular(signal: np.ndarray, sr: float,
                         grain_dur_ms: float = 40.0,
                         overlap: float = 0.5,
                         pitch_shift: float = 0.0,
                         envelope: GrainEnvelopeType = GrainEnvelopeType.HANNING
                         ) -> StereoCloud:
    dur_samp = max(2, int(grain_dur_ms * sr / 1000.0))
    hop = max(1, int(dur_samp * (1.0 - overlap)))
    grains: List[GrainSpec] = []
    pos = 0
    while pos + dur_samp <= len(signal):
        grains.append(GrainSpec(
            start_sample=pos, duration_samples=dur_samp,
            pitch_shift=pitch_shift, amplitude=1.0,
            envelope=envelope, pan=0.0,
        ))
        pos += hop
    return StereoCloud(grains=grains, base_signal=signal, sample_rate=sr)


def asynchronous_granular(signal: np.ndarray, sr: float,
                          density_hz: float = 50.0,
                          grain_dur_ms: float = 40.0,
                          duration_s: Optional[float] = None,
                          pitch_spread: float = 0.0,
                          pan_spread: float = 0.0,
                          envelope: GrainEnvelopeType = GrainEnvelopeType.GAUSSIAN,
                          seed: int = 42) -> StereoCloud:
    rng = np.random.default_rng(seed)
    dur_samp = max(2, int(grain_dur_ms * sr / 1000.0))
    total_s = duration_s if duration_s else len(signal) / sr
    total_samp = int(total_s * sr)
    n_grains = max(1, int(density_hz * total_s))
    # Poisson inter-arrival: exponential intervals
    intervals = rng.exponential(sr / max(density_hz, 1e-6), size=n_grains)
    positions = np.cumsum(intervals).astype(int)
    positions = positions[positions < total_samp]
    grains: List[GrainSpec] = []
    for pos in positions:
        ps = rng.normal(0, pitch_spread) if pitch_spread > 0 else 0.0
        pn = np.clip(rng.normal(0, pan_spread), -1, 1) if pan_spread > 0 else 0.0
        src_start = rng.integers(0, max(1, len(signal) - dur_samp))
        grains.append(GrainSpec(
            start_sample=int(pos), duration_samples=dur_samp,
            pitch_shift=ps, amplitude=1.0, envelope=envelope, pan=float(pn),
        ))
    return StereoCloud(grains=grains, base_signal=signal, sample_rate=sr)


def granular_time_stretch(signal: np.ndarray, sr: float,
                          stretch_factor: float,
                          grain_dur_ms: float = 50.0,
                          overlap: float = 0.5) -> np.ndarray:
    dur_samp = max(2, int(grain_dur_ms * sr / 1000.0))
    hop_out = max(1, int(dur_samp * (1.0 - overlap)))
    hop_in = max(1, int(hop_out / stretch_factor))
    out_len = int(len(signal) * stretch_factor)
    output = np.zeros(out_len)
    src_pos = 0
    out_pos = 0
    env = envelope_function(dur_samp, GrainEnvelopeType.HANNING)
    while out_pos + dur_samp <= out_len and src_pos + dur_samp <= len(signal):
        grain = signal[src_pos: src_pos + dur_samp].astype(np.float64) * env
        output[out_pos: out_pos + dur_samp] += grain
        out_pos += hop_out
        src_pos += hop_in
    return output


def granular_freeze(signal: np.ndarray, sr: float,
                    position_s: float, duration_s: float,
                    grain_dur_ms: float = 40.0,
                    spread_ms: float = 10.0,
                    density_hz: float = 60.0,
                    seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dur_samp = max(2, int(grain_dur_ms * sr / 1000.0))
    spread_samp = int(spread_ms * sr / 1000.0)
    centre = int(position_s * sr)
    out_len = int(duration_s * sr)
    output = np.zeros(out_len)
    env = envelope_function(dur_samp, GrainEnvelopeType.GAUSSIAN)
    n_grains = max(1, int(density_hz * duration_s))
    for _ in range(n_grains):
        out_pos = rng.integers(0, max(1, out_len - dur_samp))
        src_pos = centre + rng.integers(-spread_samp, spread_samp + 1)
        src_pos = max(0, min(src_pos, len(signal) - dur_samp))
        grain = signal[src_pos: src_pos + dur_samp].astype(np.float64) * env
        output[out_pos: out_pos + len(grain)] += grain[: out_len - out_pos]
    peak = np.max(np.abs(output))
    if peak > 1e-10:
        output /= peak
    return output


def spectral_grain(grain: np.ndarray, phase_randomize: bool = True,
                   spectral_tilt_db: float = 0.0,
                   seed: int = 0) -> np.ndarray:
    spec = np.fft.rfft(grain)
    mag = np.abs(spec)
    if phase_randomize:
        rng = np.random.default_rng(seed)
        new_phase = rng.uniform(-np.pi, np.pi, len(spec))
        spec = mag * np.exp(1j * new_phase)
    if abs(spectral_tilt_db) > 1e-6:
        n_bins = len(mag)
        tilt = np.linspace(0, spectral_tilt_db / 20.0, n_bins)
        spec *= 10.0 ** tilt
    return np.fft.irfft(spec, n=len(grain))


def grain_density_curve(cloud: StereoCloud, hop_samples: int = 512) -> np.ndarray:
    total = len(cloud.base_signal)
    for g in cloud.grains:
        total = max(total, g.start_sample + g.duration_samples)
    n_frames = max(1, total // hop_samples)
    density = np.zeros(n_frames)
    for g in cloud.grains:
        frame_start = g.start_sample // hop_samples
        frame_end = min((g.start_sample + g.duration_samples) // hop_samples, n_frames - 1)
        for f in range(frame_start, frame_end + 1):
            density[f] += 1.0
    density /= (hop_samples / cloud.sample_rate)
    return density


@dataclass
class CloudStatistics:
    n_grains: int
    mean_duration_ms: float
    std_duration_ms: float
    mean_density_hz: float
    pitch_spread_semitones: float
    pan_spread: float
    total_duration_s: float


def cloud_statistics(cloud: StereoCloud) -> CloudStatistics:
    if not cloud.grains:
        return CloudStatistics(0, 0, 0, 0, 0, 0, 0)
    durs = np.array([g.duration_samples for g in cloud.grains])
    pitches = np.array([g.pitch_shift for g in cloud.grains])
    pans = np.array([g.pan for g in cloud.grains])
    total = max(g.start_sample + g.duration_samples for g in cloud.grains)
    total_s = total / cloud.sample_rate
    dur_ms = durs / cloud.sample_rate * 1000.0
    return CloudStatistics(
        n_grains=len(cloud.grains),
        mean_duration_ms=float(np.mean(dur_ms)),
        std_duration_ms=float(np.std(dur_ms)),
        mean_density_hz=len(cloud.grains) / max(total_s, 1e-10),
        pitch_spread_semitones=float(np.std(pitches)),
        pan_spread=float(np.std(pans)),
        total_duration_s=total_s,
    )
