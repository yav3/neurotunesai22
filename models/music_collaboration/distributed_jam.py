"""Distributed jam session: clock sync, latency-aware quantization, jitter buffer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict

import numpy as np


class SyncStrategy(str, Enum):
    NTP = "ntp"
    PTP = "ptp"
    AUDIO_BEACON = "audio_beacon"
    LEADER_FOLLOWER = "leader_follower"


@dataclass
class ClockState:
    local_ms: float
    remote_ms: float
    rtt_ms: float
    offset_ms: float = 0.0

    def __post_init__(self) -> None:
        self.offset_ms = self.remote_ms - self.local_ms - self.rtt_ms / 2.0


@dataclass
class JitterStats:
    mean_ms: float
    std_ms: float
    max_ms: float
    percentile_95_ms: float
    loss_ratio: float


@dataclass
class QuantizeConfig:
    grid_ms: float = 125.0
    strength: float = 1.0
    lookahead_ms: float = 0.0
    latency_compensation_ms: float = 0.0


@dataclass
class JamParticipant:
    id: str
    rtt_ms: float
    jitter_ms: float
    clock_offset_ms: float = 0.0
    buffer_depth_ms: float = 0.0


def ntp_offset(samples: List[ClockState]) -> float:
    if not samples:
        return 0.0
    offsets = np.array([s.offset_ms for s in samples])
    rtts = np.array([s.rtt_ms for s in samples])
    weights = 1.0 / (rtts + 1e-6)
    return float(np.average(offsets, weights=weights))


def ptp_offset(master_timestamps: np.ndarray,
               slave_timestamps: np.ndarray) -> float:
    if len(master_timestamps) != len(slave_timestamps):
        raise ValueError("timestamp arrays must have equal length")
    delays = slave_timestamps - master_timestamps
    return float(np.median(delays))


def audio_beacon_offset(beacon_send_ms: float, beacon_recv_ms: float,
                        propagation_ms: float = 0.0) -> float:
    return beacon_recv_ms - beacon_send_ms - propagation_ms


def optimal_buffer_depth(rtt_ms: float, jitter_std_ms: float,
                         target_loss: float = 0.01) -> float:
    z = 2.326 if target_loss <= 0.01 else 1.645
    return rtt_ms / 2.0 + z * jitter_std_ms


def jitter_buffer_delay(timestamps_ms: np.ndarray,
                        buffer_depth_ms: float) -> np.ndarray:
    if len(timestamps_ms) == 0:
        return np.array([])
    expected = np.arange(len(timestamps_ms)) * np.median(np.diff(timestamps_ms)) if len(timestamps_ms) > 1 else timestamps_ms.copy()
    if len(timestamps_ms) > 1:
        expected = expected - expected[0] + timestamps_ms[0]
    delays = np.maximum(0.0, expected + buffer_depth_ms - timestamps_ms)
    return delays


def jitter_statistics(arrival_ms: np.ndarray,
                      expected_interval_ms: float) -> JitterStats:
    if len(arrival_ms) < 2:
        return JitterStats(0.0, 0.0, 0.0, 0.0, 0.0)
    ioi = np.diff(arrival_ms)
    jitter = ioi - expected_interval_ms
    abs_jitter = np.abs(jitter)
    return JitterStats(
        mean_ms=float(np.mean(abs_jitter)),
        std_ms=float(np.std(jitter)),
        max_ms=float(np.max(abs_jitter)),
        percentile_95_ms=float(np.percentile(abs_jitter, 95)),
        loss_ratio=0.0,
    )


def packet_loss_interpolation(values: np.ndarray,
                              received: np.ndarray) -> np.ndarray:
    mask = received.astype(bool)
    if mask.all() or not mask.any():
        return values.copy()
    indices = np.arange(len(values))
    result = values.copy()
    result[~mask] = np.interp(indices[~mask], indices[mask], values[mask])
    return result


def latency_aware_quantize(onset_ms: float, config: QuantizeConfig) -> float:
    adjusted = onset_ms - config.latency_compensation_ms + config.lookahead_ms
    grid_pos = round(adjusted / config.grid_ms) * config.grid_ms
    quantized = onset_ms + config.strength * (grid_pos - adjusted)
    return quantized


def beat_align_to_master(local_onsets_ms: np.ndarray,
                         master_beat_ms: np.ndarray,
                         offset_ms: float) -> np.ndarray:
    aligned = local_onsets_ms - offset_ms
    result = np.empty_like(aligned)
    for i, t in enumerate(aligned):
        idx = np.argmin(np.abs(master_beat_ms - t))
        result[i] = master_beat_ms[idx]
    return result


def round_trip_estimate(ping_samples_ms: np.ndarray) -> Tuple[float, float]:
    if len(ping_samples_ms) == 0:
        return (0.0, 0.0)
    return (float(np.median(ping_samples_ms)),
            float(np.std(ping_samples_ms)))


def session_latency_budget(participants: List[JamParticipant],
                           tempo_bpm: float) -> Dict[str, float]:
    beat_ms = 60000.0 / tempo_bpm
    max_rtt = max(p.rtt_ms for p in participants) if participants else 0.0
    max_jitter = max(p.jitter_ms for p in participants) if participants else 0.0
    buffer = optimal_buffer_depth(max_rtt, max_jitter)
    perceptual_limit = 30.0
    return {
        "beat_duration_ms": beat_ms,
        "max_rtt_ms": max_rtt,
        "recommended_buffer_ms": buffer,
        "total_latency_ms": max_rtt / 2.0 + buffer,
        "within_perceptual_limit": (max_rtt / 2.0 + buffer) < perceptual_limit,
        "max_safe_tempo_bpm": 60000.0 / (max_rtt / 2.0 + buffer + 10.0) if (max_rtt / 2.0 + buffer + 10.0) > 0 else 999.0,
    }


def mix_remote_streams(streams: Dict[str, np.ndarray],
                       gains: Optional[Dict[str, float]] = None) -> np.ndarray:
    if not streams:
        return np.array([])
    max_len = max(len(s) for s in streams.values())
    mix = np.zeros(max_len)
    for uid, s in streams.items():
        g = gains.get(uid, 1.0) if gains else 1.0
        mix[:len(s)] += g * s
    n = len(streams)
    if n > 0:
        mix /= math.sqrt(n)
    return mix


def consensus_tempo(local_tempos: Dict[str, float],
                    weights: Optional[Dict[str, float]] = None) -> float:
    if not local_tempos:
        return 120.0
    ids = list(local_tempos.keys())
    vals = np.array([local_tempos[k] for k in ids])
    if weights:
        w = np.array([weights.get(k, 1.0) for k in ids])
    else:
        w = np.ones(len(vals))
    return float(np.average(vals, weights=w))


def network_score(rtt_ms: float, jitter_ms: float,
                  loss_ratio: float) -> float:
    rtt_score = max(0.0, 1.0 - rtt_ms / 200.0)
    jitter_score = max(0.0, 1.0 - jitter_ms / 50.0)
    loss_score = max(0.0, 1.0 - loss_ratio * 20.0)
    return 0.4 * rtt_score + 0.35 * jitter_score + 0.25 * loss_score
