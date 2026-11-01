"""Granular synthesis: grains, grain clouds, time-stretch, pitch-shift, freeze."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Grain:
    position: int = 0
    length: int = 1024
    pitch_shift: float = 1.0
    amplitude: float = 1.0

    def extract(self, source: np.ndarray, window: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(source)
        start = max(0, self.position)
        end = min(n, self.position + self.length)
        if start >= end:
            return np.zeros(self.length, dtype=np.float32)
        segment = source[start:end].astype(np.float64)
        if len(segment) < self.length:
            segment = np.pad(segment, (0, self.length - len(segment)))

        if window is None:
            window = np.hanning(self.length)
        if len(window) != self.length:
            window = np.hanning(self.length)
        segment = segment * window

        if abs(self.pitch_shift - 1.0) > 1e-6:
            original_len = len(segment)
            indices = np.linspace(0, original_len - 1, int(original_len / self.pitch_shift))
            indices = np.clip(indices, 0, original_len - 1)
            idx_floor = np.floor(indices).astype(int)
            idx_ceil = np.minimum(idx_floor + 1, original_len - 1)
            frac = indices - idx_floor
            segment = segment[idx_floor] * (1.0 - frac) + segment[idx_ceil] * frac
            if len(segment) < self.length:
                segment = np.pad(segment, (0, self.length - len(segment)))
            else:
                segment = segment[: self.length]

        return (self.amplitude * segment).astype(np.float32)


@dataclass
class GrainCloud:
    density: float = 50.0
    grain_length_ms: float = 50.0
    spray_ms: float = 5.0
    pitch_variation: float = 0.0
    amplitude_variation: float = 0.0

    def schedule_grains(
        self, duration_s: float, sr: int = 44100, seed: Optional[int] = None
    ) -> list[Grain]:
        rng = np.random.default_rng(seed)
        grain_len = int(sr * self.grain_length_ms / 1000.0)
        n_grains = max(1, int(self.density * duration_s))
        spray_samp = int(sr * self.spray_ms / 1000.0)
        interval = duration_s / n_grains
        grains = []
        for i in range(n_grains):
            t = i * interval
            pos = int(t * sr)
            if spray_samp > 0:
                pos += int(rng.uniform(-spray_samp, spray_samp))
            ps = 1.0
            if self.pitch_variation > 0:
                ps = 2.0 ** (rng.uniform(-self.pitch_variation, self.pitch_variation))
            amp = 1.0
            if self.amplitude_variation > 0:
                amp = max(0.1, 1.0 + rng.uniform(-self.amplitude_variation, self.amplitude_variation))
            grains.append(Grain(position=pos, length=grain_len, pitch_shift=ps, amplitude=amp))
        return grains


class GranularSynthesizer:
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def render(
        self,
        source: np.ndarray,
        cloud: GrainCloud,
        duration_s: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        n_out = int(self.sr * duration_s)
        out = np.zeros(n_out, dtype=np.float64)
        grains = cloud.schedule_grains(duration_s, self.sr, seed)

        for g in grains:
            grain_audio = g.extract(source)
            start = max(0, g.position)
            end = min(n_out, start + len(grain_audio))
            seg_len = end - start
            if seg_len > 0:
                out[start:end] += grain_audio[:seg_len].astype(np.float64)

        mx = np.max(np.abs(out))
        if mx > 1.0:
            out /= mx
        return out.astype(np.float32)

    def time_stretch(
        self,
        source: np.ndarray,
        stretch_factor: float,
        grain_length_ms: float = 50.0,
        overlap: float = 0.5,
    ) -> np.ndarray:
        src_len = len(source)
        grain_len = int(self.sr * grain_length_ms / 1000.0)
        hop_in = int(grain_len * (1.0 - overlap))
        hop_out = int(hop_in * stretch_factor)
        out_len = int(src_len * stretch_factor)
        out = np.zeros(out_len, dtype=np.float64)
        window = np.hanning(grain_len)
        norm = np.zeros(out_len, dtype=np.float64)

        pos_in = 0
        pos_out = 0
        while pos_in + grain_len <= src_len and pos_out + grain_len <= out_len:
            segment = source[pos_in: pos_in + grain_len].astype(np.float64) * window
            out[pos_out: pos_out + grain_len] += segment
            norm[pos_out: pos_out + grain_len] += window
            pos_in += hop_in
            pos_out += hop_out

        mask = norm > 1e-8
        out[mask] /= norm[mask]
        return out.astype(np.float32)

    def pitch_shift(
        self,
        source: np.ndarray,
        semitones: float,
        grain_length_ms: float = 50.0,
    ) -> np.ndarray:
        ratio = 2.0 ** (semitones / 12.0)
        stretched = self.time_stretch(source, ratio, grain_length_ms)
        original_len = len(source)
        indices = np.linspace(0, len(stretched) - 1, original_len)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(stretched) - 1)
        frac = indices - idx_floor
        resampled = stretched[idx_floor] * (1.0 - frac) + stretched[idx_ceil] * frac
        return resampled.astype(np.float32)


class GranularFreeze:
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def freeze(
        self,
        source: np.ndarray,
        position_s: float,
        duration_s: float,
        grain_length_ms: float = 80.0,
        density: float = 40.0,
        scatter_ms: float = 20.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        pos_samp = int(position_s * self.sr)
        grain_len = int(self.sr * grain_length_ms / 1000.0)
        scatter_samp = int(self.sr * scatter_ms / 1000.0)
        n_out = int(self.sr * duration_s)
        out = np.zeros(n_out, dtype=np.float64)
        window = np.hanning(grain_len)

        n_grains = max(1, int(density * duration_s))
        for _ in range(n_grains):
            offset = int(rng.uniform(-scatter_samp, scatter_samp))
            src_start = max(0, pos_samp + offset)
            src_end = min(len(source), src_start + grain_len)
            if src_end <= src_start:
                continue
            seg = source[src_start:src_end].astype(np.float64)
            if len(seg) < grain_len:
                seg = np.pad(seg, (0, grain_len - len(seg)))
            seg = seg * window

            out_pos = int(rng.uniform(0, max(1, n_out - grain_len)))
            end_pos = min(n_out, out_pos + grain_len)
            seg_len = end_pos - out_pos
            out[out_pos:end_pos] += seg[:seg_len]

        mx = np.max(np.abs(out))
        if mx > 1.0:
            out /= mx
        return out.astype(np.float32)
