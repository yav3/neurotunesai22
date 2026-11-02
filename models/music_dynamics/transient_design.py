"""Transient design: attack/sustain separation, transient shaping,
percussive enhancement, envelope-based dynamics control.

Based on Verfaille & Arfib (2001) spectral envelope separation
and Giannoulis et al. (2012) dynamic range control. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class TransientMode(Enum):
    ENHANCE = "enhance"
    SUPPRESS = "suppress"
    CLIP = "clip"


@dataclass
class TransientProfile:
    attack_energy: float
    sustain_energy: float
    ratio: float
    peak_index: int
    attack_time_ms: float


@dataclass
class TransientParams:
    attack_gain_db: float = 0.0
    sustain_gain_db: float = 0.0
    attack_time_ms: float = 5.0
    release_time_ms: float = 50.0
    sensitivity: float = 0.5


def envelope_follower(signal: np.ndarray, sr: float,
                      attack_ms: float = 1.0,
                      release_ms: float = 50.0) -> np.ndarray:
    n = len(signal)
    env = np.zeros(n)
    att = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    rel = np.exp(-1.0 / (sr * release_ms / 1000.0))
    rect = np.abs(signal)
    for i in range(1, n):
        if rect[i] > env[i - 1]:
            env[i] = att * env[i - 1] + (1.0 - att) * rect[i]
        else:
            env[i] = rel * env[i - 1]
    return env


def fast_envelope(signal: np.ndarray, sr: float,
                  attack_ms: float) -> np.ndarray:
    return envelope_follower(signal, sr, attack_ms, attack_ms * 10.0)


def slow_envelope(signal: np.ndarray, sr: float,
                  release_ms: float) -> np.ndarray:
    return envelope_follower(signal, sr, release_ms * 0.1, release_ms)


def separate_transient_sustain(signal: np.ndarray, sr: float,
                               attack_ms: float = 5.0,
                               release_ms: float = 100.0
                               ) -> Tuple[np.ndarray, np.ndarray]:
    fast = fast_envelope(signal, sr, attack_ms)
    slow = slow_envelope(signal, sr, release_ms)
    ratio = np.where(slow > 1e-10, fast / slow, 1.0)
    transient_mask = np.clip(ratio - 1.0, 0.0, None)
    transient_mask = np.minimum(transient_mask / (np.max(transient_mask) + 1e-10), 1.0)
    transient = signal * transient_mask
    sustain = signal * (1.0 - transient_mask)
    return transient, sustain


def transient_shape(signal: np.ndarray, sr: float,
                    params: TransientParams) -> np.ndarray:
    transient, sustain = separate_transient_sustain(
        signal, sr, params.attack_time_ms, params.release_time_ms
    )
    att_gain = 10.0 ** (params.attack_gain_db / 20.0)
    sus_gain = 10.0 ** (params.sustain_gain_db / 20.0)
    return transient * att_gain + sustain * sus_gain


def detect_transients(signal: np.ndarray, sr: float,
                      threshold: float = 0.3,
                      attack_ms: float = 2.0) -> np.ndarray:
    fast = fast_envelope(signal, sr, attack_ms)
    slow = slow_envelope(signal, sr, attack_ms * 20.0)
    ratio = np.where(slow > 1e-10, fast / slow, 0.0)
    diff = np.diff(ratio, prepend=0.0)
    peaks = np.where(diff > threshold)[0]
    return peaks


def transient_profile(signal: np.ndarray, sr: float,
                      attack_ms: float = 5.0) -> TransientProfile:
    transient, sustain = separate_transient_sustain(signal, sr, attack_ms)
    att_energy = float(np.sum(transient ** 2))
    sus_energy = float(np.sum(sustain ** 2))
    total = att_energy + sus_energy + 1e-20
    ratio = att_energy / total
    peak_idx = int(np.argmax(np.abs(transient)))
    attack_time = peak_idx / sr * 1000.0
    return TransientProfile(att_energy, sus_energy, ratio, peak_idx, attack_time)


def percussive_enhance(signal: np.ndarray, sr: float,
                       amount_db: float = 6.0,
                       attack_ms: float = 3.0) -> np.ndarray:
    params = TransientParams(attack_gain_db=amount_db, sustain_gain_db=0.0,
                             attack_time_ms=attack_ms)
    return transient_shape(signal, sr, params)


def sustain_enhance(signal: np.ndarray, sr: float,
                    amount_db: float = 6.0,
                    release_ms: float = 80.0) -> np.ndarray:
    params = TransientParams(attack_gain_db=0.0, sustain_gain_db=amount_db,
                             release_time_ms=release_ms)
    return transient_shape(signal, sr, params)


def de_esser(signal: np.ndarray, sr: float,
             freq_low: float = 4000.0,
             freq_high: float = 9000.0,
             threshold_db: float = -20.0,
             ratio: float = 4.0) -> np.ndarray:
    n = len(signal)
    nfft = 2048
    hop = nfft // 4
    window = np.hanning(nfft)
    output = np.zeros(n)
    norm = np.zeros(n)
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= freq_low) & (freqs <= freq_high)
    for start in range(0, n - nfft, hop):
        frame = signal[start:start + nfft] * window
        spec = np.fft.rfft(frame)
        band_energy = np.sqrt(np.mean(np.abs(spec[band]) ** 2))
        if band_energy > threshold_lin:
            gain = threshold_lin + (band_energy - threshold_lin) / ratio
            reduction = gain / (band_energy + 1e-20)
            spec[band] *= reduction
        result = np.fft.irfft(spec, nfft) * window
        output[start:start + nfft] += result
        norm[start:start + nfft] += window ** 2
    norm = np.maximum(norm, 1e-10)
    return output / norm


def noise_gate(signal: np.ndarray, sr: float,
               threshold_db: float = -40.0,
               attack_ms: float = 0.5,
               hold_ms: float = 10.0,
               release_ms: float = 50.0,
               range_db: float = -80.0) -> np.ndarray:
    n = len(signal)
    env = envelope_follower(signal, sr, attack_ms, release_ms)
    env_db = 20.0 * np.log10(env + 1e-20)
    gate_open = env_db >= threshold_db
    hold_samples = int(sr * hold_ms / 1000.0)
    held = np.copy(gate_open)
    last_open = -hold_samples - 1
    for i in range(n):
        if gate_open[i]:
            last_open = i
        elif i - last_open <= hold_samples:
            held[i] = True
    range_lin = 10.0 ** (range_db / 20.0)
    gain = np.where(held, 1.0, range_lin)
    att_coeff = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    rel_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))
    smooth_gain = np.zeros(n)
    smooth_gain[0] = gain[0]
    for i in range(1, n):
        if gain[i] > smooth_gain[i - 1]:
            smooth_gain[i] = att_coeff * smooth_gain[i - 1] + (1.0 - att_coeff) * gain[i]
        else:
            smooth_gain[i] = rel_coeff * smooth_gain[i - 1] + (1.0 - rel_coeff) * gain[i]
    return signal * smooth_gain


def attack_time_ms(signal: np.ndarray, sr: float,
                   threshold: float = 0.1) -> float:
    env = np.abs(signal)
    peak = np.max(env)
    if peak < 1e-10:
        return 0.0
    onset = np.argmax(env > threshold * peak)
    peak_idx = np.argmax(env)
    if peak_idx <= onset:
        return 0.0
    return float((peak_idx - onset) / sr * 1000.0)


def crest_factor(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-10:
        return 0.0
    return float(np.max(np.abs(signal)) / rms)


def dynamic_eq_gain(signal: np.ndarray, sr: float,
                    center_freq: float, bandwidth: float,
                    threshold_db: float, max_gain_db: float,
                    nfft: int = 2048) -> np.ndarray:
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= center_freq - bandwidth / 2) & (freqs <= center_freq + bandwidth / 2)
    hop = nfft // 4
    n = len(signal)
    n_frames = max(1, (n - nfft) // hop + 1)
    gains = np.zeros(n_frames)
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    max_gain_lin = 10.0 ** (max_gain_db / 20.0)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + nfft]
        if len(frame) < nfft:
            frame = np.pad(frame, (0, nfft - len(frame)))
        spec = np.abs(np.fft.rfft(frame))
        band_level = np.sqrt(np.mean(spec[band] ** 2)) if np.any(band) else 0.0
        if band_level > threshold_lin:
            excess_db = 20.0 * np.log10(band_level / threshold_lin + 1e-20)
            gain_db = min(excess_db * 0.5, max_gain_db)
            gains[i] = 10.0 ** (gain_db / 20.0)
        else:
            gains[i] = 1.0
    gains = np.clip(gains, 1.0 / max_gain_lin, max_gain_lin)
    return gains
