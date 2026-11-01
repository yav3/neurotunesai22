"""Audio effects: reverb, delay, compressor, EQ, effects chain."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class AudioEffect:
    def process(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        return np.asarray(signal, dtype=np.float32)


class Reverb(AudioEffect):
    def __init__(self, room_size: float = 0.5, damping: float = 0.5, mix: float = 0.3):
        self.room_size = max(0.0, min(1.0, room_size))
        self.damping = max(0.0, min(1.0, damping))
        self.mix = max(0.0, min(1.0, mix))

    def process(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32)
        delays_ms = [29.7, 37.1, 41.1, 43.7]
        decay_base = 0.5 + 0.45 * self.room_size
        wet = np.zeros_like(sig)
        for d_ms in delays_ms:
            d_samples = int(d_ms * sr / 1000)
            if d_samples >= len(sig):
                continue
            decay = decay_base * (1.0 - 0.3 * self.damping)
            buf = np.zeros(len(sig), dtype=np.float32)
            for i in range(d_samples, len(sig)):
                buf[i] = sig[i - d_samples] + decay * buf[i - d_samples] if i >= d_samples else 0
            wet += buf * 0.25
        return sig * (1.0 - self.mix) + wet * self.mix


class Delay(AudioEffect):
    def __init__(self, delay_time: float = 0.3, feedback: float = 0.4, mix: float = 0.3):
        self.delay_time = max(0.001, delay_time)
        self.feedback = max(0.0, min(0.95, feedback))
        self.mix = max(0.0, min(1.0, mix))

    def process(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32)
        d = int(self.delay_time * sr)
        n_echoes = min(20, int(-3 / np.log10(max(self.feedback, 0.01))))
        out_len = len(sig) + d * n_echoes
        out = np.zeros(out_len, dtype=np.float32)
        out[:len(sig)] = sig
        wet = np.zeros(out_len, dtype=np.float32)
        for i in range(1, n_echoes + 1):
            offset = d * i
            gain = self.feedback ** i
            if gain < 0.001:
                break
            end = min(offset + len(sig), out_len)
            seg = end - offset
            wet[offset:end] += sig[:seg] * gain
        result = np.zeros(out_len, dtype=np.float32)
        result[:] = out * (1.0 - self.mix) + wet * self.mix
        return result[:len(sig)]


class Compressor(AudioEffect):
    def __init__(
        self, threshold_db: float = -20.0, ratio: float = 4.0,
        attack: float = 0.005, release: float = 0.05,
    ):
        self.threshold_db = threshold_db
        self.ratio = max(1.0, ratio)
        self.attack = max(0.0001, attack)
        self.release = max(0.001, release)

    def process(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float64)
        threshold_lin = 10 ** (self.threshold_db / 20.0)
        attack_coeff = np.exp(-1.0 / (self.attack * sr))
        release_coeff = np.exp(-1.0 / (self.release * sr))
        envelope = np.zeros(len(sig), dtype=np.float64)
        env = 0.0
        for i in range(len(sig)):
            level = abs(sig[i])
            if level > env:
                env = attack_coeff * env + (1 - attack_coeff) * level
            else:
                env = release_coeff * env + (1 - release_coeff) * level
            envelope[i] = env
        gain = np.ones(len(sig), dtype=np.float64)
        above = envelope > threshold_lin
        if np.any(above):
            over_db = 20 * np.log10(np.maximum(envelope[above], 1e-10) / threshold_lin)
            reduced = over_db / self.ratio
            gain[above] = 10 ** ((reduced - over_db) / 20.0)
        return (sig * gain).astype(np.float32)


@dataclass
class EQBand:
    center_freq: float = 1000.0
    gain_db: float = 0.0
    q: float = 1.0


class EQ(AudioEffect):
    def __init__(self, bands: list[EQBand] | None = None):
        self.bands = bands or []

    def process(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float64)
        if not self.bands:
            return sig.astype(np.float32)
        n = len(sig)
        spectrum = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        gain_curve = np.ones(len(freqs), dtype=np.float64)
        for band in self.bands:
            if band.center_freq <= 0 or band.q <= 0:
                continue
            bw = band.center_freq / band.q
            lo = band.center_freq - bw / 2
            hi = band.center_freq + bw / 2
            mask = (freqs >= lo) & (freqs <= hi)
            gain_lin = 10 ** (band.gain_db / 20.0)
            bell = np.ones(len(freqs))
            in_band = mask.astype(float)
            bell = 1.0 + (gain_lin - 1.0) * in_band
            gain_curve *= bell
        spectrum *= gain_curve
        return np.fft.irfft(spectrum, n=n).astype(np.float32)


class EffectsChain:
    def __init__(self):
        self._effects: list[AudioEffect] = []

    def add(self, effect: AudioEffect) -> "EffectsChain":
        self._effects.append(effect)
        return self

    def process(self, signal: np.ndarray, sr: int = 22050) -> np.ndarray:
        result = np.asarray(signal, dtype=np.float32)
        for fx in self._effects:
            result = fx.process(result, sr)
        return result

    def __len__(self) -> int:
        return len(self._effects)

    def clear(self) -> None:
        self._effects.clear()
