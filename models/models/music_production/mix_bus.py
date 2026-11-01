"""Mix bus processing: summing, stereo bus compression, master EQ, loudness targeting.

Implements a signal-processing chain for mixing and mastering without
external audio frameworks — pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import signal as sp_signal


@dataclass
class BusInsert:
    name: str
    process: object  # callable(signal, sr) -> signal
    active: bool = True


@dataclass
class MixBusConfig:
    target_lufs: float = -14.0
    headroom_db: float = -1.0
    bus_comp_threshold_db: float = -12.0
    bus_comp_ratio: float = 2.0
    bus_comp_attack_ms: float = 30.0
    bus_comp_release_ms: float = 300.0
    hp_freq: float = 30.0
    lp_freq: float = 18000.0


def sum_stems(
    stems: List[np.ndarray],
    gains_db: Optional[List[float]] = None,
    pans: Optional[List[float]] = None,
) -> np.ndarray:
    """Sum mono/stereo stems with gain and pan.

    Parameters
    ----------
    stems : list of arrays
        Each shape (N,) mono or (N, 2) stereo.
    gains_db : optional list of float
        Per-stem gain in dB. Default 0 dB each.
    pans : optional list of float
        Pan position -1 (left) to +1 (right). Ignored for stereo stems.

    Returns
    -------
    np.ndarray shape (max_len, 2) — stereo mix.
    """
    if not stems:
        return np.zeros((0, 2))
    n_stems = len(stems)
    if gains_db is None:
        gains_db = [0.0] * n_stems
    if pans is None:
        pans = [0.0] * n_stems
    max_len = max(s.shape[0] if s.ndim == 1 else s.shape[0] for s in stems)
    mix = np.zeros((max_len, 2))
    for stem, gain_db, pan in zip(stems, gains_db, pans):
        gain_lin = 10 ** (gain_db / 20.0)
        s = np.asarray(stem, dtype=float)
        if s.ndim == 1:
            left_gain = np.cos((pan + 1) * np.pi / 4)
            right_gain = np.sin((pan + 1) * np.pi / 4)
            mix[:len(s), 0] += s * gain_lin * left_gain
            mix[:len(s), 1] += s * gain_lin * right_gain
        else:
            mix[:s.shape[0], :2] += s[:, :2] * gain_lin
    return mix


def rms_db(audio: np.ndarray) -> float:
    """RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    return float(20 * np.log10(rms + 1e-12))


def peak_db(audio: np.ndarray) -> float:
    """Peak level in dBFS."""
    pk = np.max(np.abs(audio))
    return float(20 * np.log10(pk + 1e-12))


def lufs_integrated(audio: np.ndarray, sr: int) -> float:
    """Approximate integrated LUFS (ITU-R BS.1770-4 simplified).

    Uses a K-weighting approximation (high-shelf + high-pass) and
    gated mean-square loudness.
    """
    x = np.asarray(audio, dtype=float)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    nyq = sr / 2.0
    if nyq > 100:
        b_hp, a_hp = sp_signal.butter(2, 100.0 / nyq, btype="high")
        x = sp_signal.lfilter(b_hp, a_hp, x)
    if nyq > 2000:
        shelf_freq = min(1500.0 / nyq, 0.99)
        b_hs, a_hs = sp_signal.butter(1, shelf_freq, btype="high")
        boost = sp_signal.lfilter(b_hs, a_hs, x)
        x = x + 0.5 * boost
    block_dur = 0.4
    block_samples = int(sr * block_dur)
    hop = block_samples // 4
    if block_samples < 1:
        return -70.0
    blocks = []
    for i in range(0, len(x) - block_samples + 1, hop):
        block = x[i:i + block_samples]
        blocks.append(np.mean(block ** 2))
    if not blocks:
        return -70.0
    blocks = np.array(blocks)
    abs_threshold = -70.0
    block_lufs = -0.691 + 10 * np.log10(blocks + 1e-20)
    above = blocks[block_lufs > abs_threshold]
    if len(above) == 0:
        return -70.0
    rel_threshold = -0.691 + 10 * np.log10(np.mean(above) + 1e-20) - 10.0
    gated = blocks[(block_lufs > abs_threshold) & (block_lufs > rel_threshold)]
    if len(gated) == 0:
        return -70.0
    return float(-0.691 + 10 * np.log10(np.mean(gated) + 1e-20))


def bus_compressor(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -12.0,
    ratio: float = 2.0,
    attack_ms: float = 30.0,
    release_ms: float = 300.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """Stereo bus compressor with RMS detection and soft-knee."""
    x = np.array(audio, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    attack = np.exp(-1.0 / (sr * attack_ms / 1000.0)) if attack_ms > 0 else 0.0
    release = np.exp(-1.0 / (sr * release_ms / 1000.0)) if release_ms > 0 else 0.0
    threshold_lin = 10 ** (threshold_db / 20.0)
    env = np.zeros(len(x))
    rms_val = 0.0
    for i in range(len(x)):
        sample_sq = np.mean(x[i] ** 2)
        coeff = attack if sample_sq > rms_val else release
        rms_val = coeff * rms_val + (1 - coeff) * sample_sq
        env[i] = np.sqrt(rms_val + 1e-12)
    gain = np.ones(len(x))
    for i in range(len(x)):
        if env[i] > threshold_lin:
            over_db = 20 * np.log10(env[i] / threshold_lin)
            reduction_db = over_db * (1 - 1.0 / ratio)
            gain[i] = 10 ** (-reduction_db / 20.0)
    makeup_lin = 10 ** (makeup_db / 20.0)
    if x.ndim == 2:
        return x * gain[:, np.newaxis] * makeup_lin
    return (x.flatten() * gain * makeup_lin)


def master_eq(
    audio: np.ndarray,
    sr: int,
    low_shelf_db: float = 0.0,
    low_shelf_freq: float = 100.0,
    high_shelf_db: float = 0.0,
    high_shelf_freq: float = 10000.0,
    mid_gain_db: float = 0.0,
    mid_freq: float = 2500.0,
    mid_q: float = 1.0,
) -> np.ndarray:
    """Three-band master EQ: low shelf + parametric mid + high shelf."""
    x = np.array(audio, dtype=float)
    nyq = sr / 2.0
    if abs(low_shelf_db) > 0.01 and low_shelf_freq < nyq:
        sos = sp_signal.iirfilter(
            1, low_shelf_freq / nyq, btype="low", ftype="butter", output="sos"
        )
        low = sp_signal.sosfilt(sos, x, axis=0)
        gain = 10 ** (low_shelf_db / 20.0) - 1.0
        x = x + gain * low
    if abs(high_shelf_db) > 0.01 and high_shelf_freq < nyq:
        sos = sp_signal.iirfilter(
            1, high_shelf_freq / nyq, btype="high", ftype="butter", output="sos"
        )
        high = sp_signal.sosfilt(sos, x, axis=0)
        gain = 10 ** (high_shelf_db / 20.0) - 1.0
        x = x + gain * high
    if abs(mid_gain_db) > 0.01 and mid_freq < nyq:
        bw = mid_freq / max(mid_q, 0.1)
        low_cut = max((mid_freq - bw / 2) / nyq, 0.001)
        high_cut = min((mid_freq + bw / 2) / nyq, 0.999)
        if low_cut < high_cut:
            sos = sp_signal.iirfilter(
                1, [low_cut, high_cut], btype="band", ftype="butter", output="sos"
            )
            mid = sp_signal.sosfilt(sos, x, axis=0)
            gain = 10 ** (mid_gain_db / 20.0) - 1.0
            x = x + gain * mid
    return x


def loudness_normalize(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
) -> np.ndarray:
    """Normalize audio to a target LUFS level."""
    current = lufs_integrated(audio, sr)
    if current < -60:
        return audio.copy()
    adjustment_db = target_lufs - current
    gain = 10 ** (adjustment_db / 20.0)
    return audio * gain


def true_peak(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Estimate true peak level via oversampling."""
    x = np.asarray(audio, dtype=float)
    if x.ndim == 2:
        x = np.max(np.abs(x), axis=1)
    else:
        x = np.abs(x)
    if oversample > 1 and len(x) > 1:
        n_out = len(x) * oversample
        t_in = np.arange(len(x))
        t_out = np.linspace(0, len(x) - 1, n_out)
        x_up = np.interp(t_out, t_in, x)
        return float(20 * np.log10(np.max(x_up) + 1e-12))
    return float(20 * np.log10(np.max(x) + 1e-12))


def process_mix_bus(
    stems: List[np.ndarray],
    sr: int,
    config: Optional[MixBusConfig] = None,
    gains_db: Optional[List[float]] = None,
    pans: Optional[List[float]] = None,
) -> np.ndarray:
    """Full mix bus chain: sum → HP/LP → bus comp → master EQ → loudness normalize."""
    if config is None:
        config = MixBusConfig()
    mix = sum_stems(stems, gains_db, pans)
    if mix.size == 0:
        return mix
    nyq = sr / 2.0
    if config.hp_freq > 0 and config.hp_freq < nyq:
        sos = sp_signal.iirfilter(
            2, config.hp_freq / nyq, btype="high", ftype="butter", output="sos"
        )
        mix = sp_signal.sosfilt(sos, mix, axis=0)
    if config.lp_freq > 0 and config.lp_freq < nyq:
        sos = sp_signal.iirfilter(
            2, config.lp_freq / nyq, btype="low", ftype="butter", output="sos"
        )
        mix = sp_signal.sosfilt(sos, mix, axis=0)
    mix = bus_compressor(
        mix, sr,
        threshold_db=config.bus_comp_threshold_db,
        ratio=config.bus_comp_ratio,
        attack_ms=config.bus_comp_attack_ms,
        release_ms=config.bus_comp_release_ms,
    )
    mix = loudness_normalize(mix, sr, target_lufs=config.target_lufs)
    tp = true_peak(mix, sr)
    if tp > config.headroom_db:
        reduction = config.headroom_db - tp
        mix *= 10 ** (reduction / 20.0)
    return mix
