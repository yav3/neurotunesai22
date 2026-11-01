"""Real-time streaming audio processing pipeline.

Ring-buffer-backed overlap-add/overlap-save STFT processing with
frame-level feature extraction, onset detection, and adaptive gain
control.  All operations are sample-accurate and run in constant
memory per frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import windows


@dataclass
class RingBuffer:
    capacity: int
    data: np.ndarray = field(init=False)
    write_pos: int = 0
    read_pos: int = 0
    level: int = 0

    def __post_init__(self) -> None:
        self.data = np.zeros(self.capacity, dtype=np.float64)


@dataclass
class StreamConfig:
    sample_rate: float = 44100.0
    frame_size: int = 1024
    hop_size: int = 512
    window_type: str = "hann"
    n_fft: int = 2048


@dataclass
class StreamState:
    frames_processed: int = 0
    buffer_level: int = 0
    latency_samples: int = 0
    features_buffer: list = field(default_factory=list)


@dataclass
class FeatureFrame:
    timestamp: float
    features: dict[str, float]
    frame_index: int


def create_ring_buffer(capacity: int) -> RingBuffer:
    return RingBuffer(capacity=capacity)


def push_samples(buf: RingBuffer, samples: np.ndarray) -> int:
    s = np.asarray(samples, dtype=np.float64)
    n = len(s)
    available = buf.capacity - buf.level
    to_write = min(n, available)
    for i in range(to_write):
        buf.data[buf.write_pos] = s[i]
        buf.write_pos = (buf.write_pos + 1) % buf.capacity
    buf.level += to_write
    return to_write


def pull_frame(buf: RingBuffer, frame_size: int) -> np.ndarray | None:
    if buf.level < frame_size:
        return None
    frame = np.zeros(frame_size, dtype=np.float64)
    for i in range(frame_size):
        frame[i] = buf.data[(buf.read_pos + i) % buf.capacity]
    return frame


def advance_read(buf: RingBuffer, hop: int) -> None:
    actual = min(hop, buf.level)
    buf.read_pos = (buf.read_pos + actual) % buf.capacity
    buf.level -= actual


def _get_window(name: str, size: int) -> np.ndarray:
    if name == "hamming":
        return windows.hamming(size, sym=False)
    if name == "blackman":
        return windows.blackman(size, sym=False)
    return windows.hann(size, sym=False)


def overlap_add_process(
    buf: RingBuffer,
    modify_fn,
    config: StreamConfig,
    output: np.ndarray,
    out_pos: int,
) -> int:
    win = _get_window(config.window_type, config.frame_size)
    frame = pull_frame(buf, config.frame_size)
    if frame is None:
        return out_pos
    windowed = frame * win
    spectrum = np.fft.rfft(windowed, n=config.n_fft)
    spectrum = modify_fn(spectrum)
    reconstructed = np.fft.irfft(spectrum, n=config.n_fft)[: config.frame_size] * win
    end = min(out_pos + config.frame_size, len(output))
    output[out_pos:end] += reconstructed[: end - out_pos]
    advance_read(buf, config.hop_size)
    return out_pos + config.hop_size


def streaming_stft(
    buf: RingBuffer, config: StreamConfig
) -> np.ndarray | None:
    frame = pull_frame(buf, config.frame_size)
    if frame is None:
        return None
    win = _get_window(config.window_type, config.frame_size)
    return np.fft.rfft(frame * win, n=config.n_fft)


def streaming_features(
    frame: np.ndarray, sr: float, n_fft: int = 2048
) -> dict[str, float]:
    f = np.asarray(frame, dtype=np.float64)
    rms = float(np.sqrt(np.mean(f ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(f))) > 0))
    spectrum = np.abs(np.fft.rfft(f, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    power = spectrum ** 2
    total_power = np.sum(power) + 1e-12
    centroid = float(np.sum(freqs * power) / total_power)
    cumsum = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total_power)
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    log_power = np.log(power + 1e-12)
    n_mel = 13
    mel_energies = np.zeros(n_mel)
    bin_edges = np.linspace(0, len(log_power) - 1, n_mel + 2, dtype=int)
    for i in range(n_mel):
        mel_energies[i] = np.mean(log_power[bin_edges[i] : bin_edges[i + 2] + 1])
    from scipy.fft import dct
    mfcc = dct(mel_energies, type=2, norm="ortho")
    return {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "mfcc_0": float(mfcc[0]) if len(mfcc) > 0 else 0.0,
        "mfcc_1": float(mfcc[1]) if len(mfcc) > 1 else 0.0,
        "mfcc_2": float(mfcc[2]) if len(mfcc) > 2 else 0.0,
    }


def detect_onset_streaming(
    prev_spectrum: np.ndarray | None,
    curr_spectrum: np.ndarray,
    threshold: float = 0.5,
) -> bool:
    if prev_spectrum is None:
        return False
    flux = np.sum(np.maximum(np.abs(curr_spectrum) - np.abs(prev_spectrum), 0.0))
    norm = np.sum(np.abs(curr_spectrum)) + 1e-12
    return float(flux / norm) > threshold


def adaptive_gain(
    signal: np.ndarray,
    target_rms: float = 0.1,
    attack_coeff: float = 0.01,
    release_coeff: float = 0.001,
) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float64)
    out = np.zeros_like(sig)
    gain = 1.0
    for i in range(len(sig)):
        env = abs(sig[i])
        if env > 1e-9:
            desired_gain = target_rms / env
            if desired_gain < gain:
                gain += attack_coeff * (desired_gain - gain)
            else:
                gain += release_coeff * (desired_gain - gain)
        out[i] = sig[i] * gain
    return out


def latency_report(config: StreamConfig) -> dict[str, float]:
    algo_latency_samples = config.frame_size + config.hop_size
    latency_ms = algo_latency_samples / config.sample_rate * 1000.0
    return {
        "algorithmic_latency_samples": float(algo_latency_samples),
        "algorithmic_latency_ms": latency_ms,
        "frame_duration_ms": config.frame_size / config.sample_rate * 1000.0,
        "hop_duration_ms": config.hop_size / config.sample_rate * 1000.0,
    }
