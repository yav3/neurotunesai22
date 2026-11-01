"""Mix engineering: buses, channels, stereo mixer, gain staging."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MixBus:
    name: str = "Main"
    gain_db: float = 0.0
    pan: float = 0.0  # -1 (left) to +1 (right)
    mute: bool = False
    solo: bool = False
    sends: list[str] = field(default_factory=list)

    def linear_gain(self) -> float:
        return 10.0 ** (self.gain_db / 20.0)


@dataclass
class MixerChannel:
    name: str = ""
    signal: np.ndarray | None = None
    bus: MixBus = field(default_factory=MixBus)
    inserts: list[Any] = field(default_factory=list)

    def rendered(self) -> np.ndarray:
        if self.signal is None:
            return np.zeros(0, dtype=np.float32)
        sig = self.signal.astype(np.float32)
        for fx in self.inserts:
            if hasattr(fx, "process"):
                sig = fx.process(sig)
        return sig


class StereoMixer:
    def __init__(self) -> None:
        self._channels: list[MixerChannel] = []

    def add_channel(self, channel: MixerChannel) -> None:
        self._channels.append(channel)

    @property
    def channels(self) -> list[MixerChannel]:
        return list(self._channels)

    def set_levels(self, levels: dict[str, float]) -> None:
        for ch in self._channels:
            if ch.name in levels:
                ch.bus.gain_db = levels[ch.name]

    def get_peak_levels(self) -> dict[str, float]:
        peaks: dict[str, float] = {}
        for ch in self._channels:
            sig = ch.rendered()
            if len(sig) == 0:
                peaks[ch.name] = -np.inf
            else:
                peak = float(np.max(np.abs(sig)))
                peaks[ch.name] = 20.0 * np.log10(peak) if peak > 1e-12 else -np.inf
        return peaks

    def mix_down(self, sr: int = 22050) -> np.ndarray:
        has_solo = any(ch.bus.solo for ch in self._channels)
        max_len = 0
        for ch in self._channels:
            sig = ch.rendered()
            if len(sig) > max_len:
                max_len = len(sig)
        if max_len == 0:
            return np.zeros((0, 2), dtype=np.float32)

        stereo = np.zeros((max_len, 2), dtype=np.float64)
        for ch in self._channels:
            if ch.bus.mute:
                continue
            if has_solo and not ch.bus.solo:
                continue
            sig = ch.rendered()
            if len(sig) == 0:
                continue
            gain = ch.bus.linear_gain()
            pan = np.clip(ch.bus.pan, -1.0, 1.0)
            left_gain = gain * np.sqrt(0.5 * (1.0 - pan))
            right_gain = gain * np.sqrt(0.5 * (1.0 + pan))
            n = len(sig)
            stereo[:n, 0] += sig * left_gain
            stereo[:n, 1] += sig * right_gain

        return stereo.astype(np.float32)


class GainStaging:
    @staticmethod
    def _rms(signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))

    @staticmethod
    def auto_gain(
        signal: np.ndarray,
        target_lufs: float = -14.0,
    ) -> tuple[np.ndarray, float]:
        rms = GainStaging._rms(signal)
        if rms < 1e-12:
            return signal.copy(), 0.0
        current_db = 20.0 * np.log10(rms)
        target_rms_db = target_lufs + 3.0
        gain_db = target_rms_db - current_db
        gain_lin = 10.0 ** (gain_db / 20.0)
        return (signal * gain_lin).astype(signal.dtype), float(gain_db)

    @staticmethod
    def headroom_check(signal: np.ndarray, ceiling_db: float = -1.0) -> dict[str, Any]:
        peak = float(np.max(np.abs(signal)))
        peak_db = 20.0 * np.log10(peak) if peak > 1e-12 else -np.inf
        headroom = ceiling_db - peak_db
        return {
            "peak_linear": peak,
            "peak_db": peak_db,
            "ceiling_db": ceiling_db,
            "headroom_db": headroom,
            "clipping": peak_db > ceiling_db,
        }
