"""Network latency compensation for remote music collaboration.

Adaptive jitter buffering, clock drift estimation, packet-loss
concealment, and sample-accurate synchronisation for networked
musical performance (NMP) scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NetworkStats:
    rtt_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    clock_offset_ms: float = 0.0


@dataclass
class JitterBuffer:
    buffer_ms: float = 40.0
    target_latency: float = 40.0
    adaptive: bool = True
    _samples: list[np.ndarray] = field(default_factory=list, repr=False)
    _timestamps: list[float] = field(default_factory=list, repr=False)


@dataclass
class SyncState:
    local_clock: float = 0.0
    remote_clock: float = 0.0
    drift_rate: float = 0.0
    confidence: float = 0.0


def estimate_rtt(
    rtt_samples: np.ndarray, alpha: float = 0.125
) -> NetworkStats:
    samples = np.asarray(rtt_samples, dtype=np.float64)
    if len(samples) == 0:
        return NetworkStats()
    srtt = samples[0]
    rttvar = samples[0] / 2.0
    for s in samples[1:]:
        rttvar = (1 - 0.25) * rttvar + 0.25 * abs(s - srtt)
        srtt = (1 - alpha) * srtt + alpha * s
    return NetworkStats(
        rtt_ms=float(srtt),
        jitter_ms=float(rttvar),
        packet_loss_rate=0.0,
        clock_offset_ms=0.0,
    )


def adaptive_buffer_size(
    stats: NetworkStats, k: float = 2.0, min_ms: float = 10.0, max_ms: float = 200.0
) -> float:
    target = stats.rtt_ms / 2.0 + k * stats.jitter_ms
    return float(np.clip(target, min_ms, max_ms))


def clock_drift_estimate(
    local_times: np.ndarray, remote_times: np.ndarray
) -> SyncState:
    lt = np.asarray(local_times, dtype=np.float64)
    rt = np.asarray(remote_times, dtype=np.float64)
    n = min(len(lt), len(rt))
    if n < 2:
        return SyncState(
            local_clock=float(lt[-1]) if len(lt) else 0.0,
            remote_clock=float(rt[-1]) if len(rt) else 0.0,
        )
    lt, rt = lt[:n], rt[:n]
    offsets = rt - lt
    x = lt - lt[0]
    mean_x = np.mean(x)
    mean_y = np.mean(offsets)
    slope = np.sum((x - mean_x) * (offsets - mean_y)) / max(np.sum((x - mean_x) ** 2), 1e-12)
    residuals = offsets - (mean_y + slope * (x - mean_x))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    conf = float(np.clip(1.0 - rmse / max(abs(mean_y), 1.0), 0.0, 1.0))
    return SyncState(
        local_clock=float(lt[-1]),
        remote_clock=float(rt[-1]),
        drift_rate=float(slope),
        confidence=conf,
    )


def compensate_drift(
    audio: np.ndarray, drift_rate: float, sr: float
) -> np.ndarray:
    sig = np.asarray(audio, dtype=np.float64)
    n = len(sig)
    if abs(drift_rate) < 1e-9 or n < 2:
        return sig.copy()
    new_rate = 1.0 + drift_rate
    new_n = int(n / new_rate)
    if new_n < 2:
        return sig.copy()
    old_indices = np.linspace(0, n - 1, new_n)
    return np.interp(old_indices, np.arange(n), sig)


def packet_loss_concealment(
    frames: list[np.ndarray | None], order: int = 10
) -> np.ndarray:
    result_parts: list[np.ndarray] = []
    frame_size = 0
    for f in frames:
        if f is not None:
            frame_size = len(f)
            break
    if frame_size == 0:
        return np.array([], dtype=np.float64)
    for i, frame in enumerate(frames):
        if frame is not None:
            result_parts.append(np.asarray(frame, dtype=np.float64))
        else:
            prev_frames = [f for f in frames[max(0, i - 3) : i] if f is not None]
            if prev_frames:
                ref = np.concatenate(prev_frames)
                n = min(len(ref), order)
                autocorr = np.correlate(ref[-frame_size * 2 :], ref[-frame_size * 2 :], "full")
                mid = len(autocorr) // 2
                if mid + frame_size < len(autocorr):
                    pattern = ref[-frame_size:]
                    fade = np.linspace(1.0, 0.8, frame_size)
                    result_parts.append(pattern * fade)
                else:
                    result_parts.append(ref[-frame_size:] * 0.9)
            else:
                result_parts.append(np.zeros(frame_size))
    return np.concatenate(result_parts) if result_parts else np.array([], dtype=np.float64)


def sync_score(stats: NetworkStats, target_ms: float = 20.0) -> float:
    one_way = stats.rtt_ms / 2.0
    total_latency = one_way + stats.jitter_ms
    if total_latency <= target_ms:
        return 1.0
    ratio = target_ms / max(total_latency, 1e-9)
    loss_penalty = 1.0 - stats.packet_loss_rate
    return float(np.clip(ratio * loss_penalty, 0.0, 1.0))


def optimal_monitoring_latency(
    stats: NetworkStats, safety_factor: float = 1.5
) -> float:
    one_way = stats.rtt_ms / 2.0
    buffer = safety_factor * stats.jitter_ms
    return float(one_way + buffer)


def jitter_buffer_push(
    buf: JitterBuffer, frame: np.ndarray, timestamp: float
) -> None:
    buf._samples.append(np.asarray(frame, dtype=np.float64))
    buf._timestamps.append(timestamp)
    if buf.adaptive and len(buf._timestamps) > 10:
        jitter = float(np.std(np.diff(buf._timestamps[-10:])))
        buf.target_latency = max(buf.buffer_ms, 2.0 * jitter * 1000.0)


def jitter_buffer_pull(buf: JitterBuffer) -> np.ndarray | None:
    if not buf._samples:
        return None
    return buf._samples.pop(0)


def jitter_buffer_level(buf: JitterBuffer) -> int:
    return len(buf._samples)
