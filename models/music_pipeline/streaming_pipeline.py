"""Streaming audio pipeline: chunked processing, ring buffers, real-time feature extraction."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any


@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    timestamp: float
    channel_count: int = 1

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate if self.sample_rate > 0 else 0.0


class RingBuffer:
    def __init__(self, capacity: int, dtype: type = np.float64):
        self._buf = np.zeros(capacity, dtype=dtype)
        self._capacity = capacity
        self._write_pos = 0
        self._count = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return min(self._count, self._capacity)

    def write(self, data: np.ndarray) -> int:
        n = len(data)
        if n >= self._capacity:
            self._buf[:] = data[-self._capacity:]
            self._write_pos = 0
            self._count = self._capacity
            return self._capacity
        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos:end] = data
        else:
            first = self._capacity - self._write_pos
            self._buf[self._write_pos:] = data[:first]
            self._buf[:n - first] = data[first:]
        self._write_pos = end % self._capacity
        self._count += n
        return n

    def read(self, n: Optional[int] = None) -> np.ndarray:
        available = self.count
        if n is None:
            n = available
        n = min(n, available)
        if n == 0:
            return np.array([], dtype=self._buf.dtype)
        start = (self._write_pos - available) % self._capacity
        end = (start + n) % self._capacity
        if end > start or end == 0:
            return self._buf[start:start + n].copy()
        return np.concatenate([self._buf[start:], self._buf[:end]])

    def clear(self) -> None:
        self._write_pos = 0
        self._count = 0


@dataclass
class FeatureFrame:
    timestamp: float
    features: Dict[str, float]


def rms_energy(chunk: np.ndarray) -> float:
    if len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk ** 2)))


def zero_crossing_rate(chunk: np.ndarray) -> float:
    if len(chunk) < 2:
        return 0.0
    signs = np.sign(chunk)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (len(chunk) - 1))


def spectral_centroid(chunk: np.ndarray, sample_rate: int) -> float:
    if len(chunk) < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
    total = spectrum.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


def spectral_rolloff(chunk: np.ndarray, sample_rate: int,
                     threshold: float = 0.85) -> float:
    if len(chunk) < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
    total = spectrum.sum()
    if total < 1e-12:
        return 0.0
    cumsum = np.cumsum(spectrum)
    idx = np.searchsorted(cumsum, threshold * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def spectral_flux(prev_spectrum: np.ndarray, curr_spectrum: np.ndarray) -> float:
    if len(prev_spectrum) != len(curr_spectrum):
        return 0.0
    diff = curr_spectrum - prev_spectrum
    return float(np.sum(np.maximum(diff, 0.0) ** 2))


@dataclass
class OnsetDetector:
    threshold: float = 0.3
    _prev_spectrum: Optional[np.ndarray] = field(default=None, repr=False)
    _flux_history: List[float] = field(default_factory=list, repr=False)

    def process(self, chunk: np.ndarray) -> bool:
        spectrum = np.abs(np.fft.rfft(chunk))
        if self._prev_spectrum is None:
            self._prev_spectrum = spectrum
            return False
        flux = spectral_flux(self._prev_spectrum, spectrum)
        self._flux_history.append(flux)
        self._prev_spectrum = spectrum
        if len(self._flux_history) < 3:
            return False
        mean_flux = np.mean(self._flux_history[-10:])
        return flux > mean_flux * (1.0 + self.threshold)


@dataclass
class StreamProcessor:
    sample_rate: int
    hop_size: int
    extractors: Dict[str, Callable[[np.ndarray, int], float]] = field(default_factory=dict)
    _buffer: Optional[RingBuffer] = field(default=None, repr=False)
    _frame_count: int = 0
    _features: List[FeatureFrame] = field(default_factory=list, repr=False)

    def __post_init__(self):
        buf_size = self.hop_size * 4
        self._buffer = RingBuffer(buf_size)
        if not self.extractors:
            self.extractors = {
                "rms": lambda c, sr: rms_energy(c),
                "zcr": lambda c, sr: zero_crossing_rate(c),
                "centroid": spectral_centroid,
                "rolloff": spectral_rolloff,
            }

    def push(self, chunk: AudioChunk) -> List[FeatureFrame]:
        self._buffer.write(chunk.data)
        frames: List[FeatureFrame] = []
        while self._buffer.count >= self.hop_size:
            window = self._buffer.read(self.hop_size)
            self._buffer._count -= self.hop_size
            start = (self._buffer._write_pos - self._buffer._count) % self._buffer._capacity
            self._buffer._write_pos = (start + self._buffer._count) % self._buffer._capacity
            # actually just advance the read pointer
            ts = self._frame_count * self.hop_size / self.sample_rate
            feats = {}
            for name, fn in self.extractors.items():
                feats[name] = fn(window, self.sample_rate)
            frame = FeatureFrame(timestamp=ts, features=feats)
            frames.append(frame)
            self._features.append(frame)
            self._frame_count += 1
        return frames

    def get_all_features(self) -> List[FeatureFrame]:
        return list(self._features)

    def reset(self) -> None:
        self._buffer.clear()
        self._frame_count = 0
        self._features.clear()


def chunk_audio(signal: np.ndarray, sample_rate: int,
                chunk_size: int) -> List[AudioChunk]:
    chunks: List[AudioChunk] = []
    for start in range(0, len(signal), chunk_size):
        end = min(start + chunk_size, len(signal))
        ts = start / sample_rate
        chunks.append(AudioChunk(data=signal[start:end],
                                 sample_rate=sample_rate, timestamp=ts))
    return chunks


def process_full_signal(signal: np.ndarray, sample_rate: int,
                        hop_size: int = 1024) -> List[FeatureFrame]:
    proc = StreamProcessor(sample_rate=sample_rate, hop_size=hop_size)
    chunks = chunk_audio(signal, sample_rate, hop_size * 2)
    all_frames: List[FeatureFrame] = []
    for chunk in chunks:
        frames = proc.push(chunk)
        all_frames.extend(frames)
    return all_frames


@dataclass
class LatencyTracker:
    _timestamps: List[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self._timestamps.append(latency_ms)

    def mean_latency(self) -> float:
        if not self._timestamps:
            return 0.0
        return float(np.mean(self._timestamps))

    def p99_latency(self) -> float:
        if not self._timestamps:
            return 0.0
        return float(np.percentile(self._timestamps, 99))

    def jitter(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        return float(np.std(self._timestamps))


def overlap_add(frames: List[np.ndarray], hop_size: int,
                window: Optional[np.ndarray] = None) -> np.ndarray:
    if not frames:
        return np.array([])
    frame_len = len(frames[0])
    out_len = (len(frames) - 1) * hop_size + frame_len
    output = np.zeros(out_len)
    norm = np.zeros(out_len)
    if window is None:
        window = np.hanning(frame_len)
    for i, frame in enumerate(frames):
        start = i * hop_size
        output[start:start + frame_len] += frame * window
        norm[start:start + frame_len] += window ** 2
    norm = np.maximum(norm, 1e-12)
    return output / norm
