"""Physical modeling synthesis: Karplus-Strong, waveguides, modal, tube models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class KarplusStrong:
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def pluck(
        self,
        frequency: float,
        duration_s: float,
        decay: float = 0.996,
        brightness: float = 0.5,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        delay_len = max(2, int(self.sr / frequency))
        n = int(self.sr * duration_s)
        buf = rng.uniform(-1.0, 1.0, delay_len).astype(np.float64)
        out = np.zeros(n, dtype=np.float64)
        idx = 0
        for i in range(n):
            out[i] = buf[idx]
            next_idx = (idx + 1) % delay_len
            filtered = brightness * buf[idx] + (1.0 - brightness) * buf[next_idx]
            buf[idx] = filtered * decay
            idx = next_idx
        return out.astype(np.float32)

    def drum(
        self,
        frequency: float,
        duration_s: float,
        blend: float = 0.5,
        decay: float = 0.990,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        delay_len = max(2, int(self.sr / frequency))
        n = int(self.sr * duration_s)
        buf = rng.uniform(-1.0, 1.0, delay_len).astype(np.float64)
        out = np.zeros(n, dtype=np.float64)
        idx = 0
        for i in range(n):
            out[i] = buf[idx]
            next_idx = (idx + 1) % delay_len
            avg = 0.5 * (buf[idx] + buf[next_idx])
            sign = 1.0 if rng.random() < blend else -1.0
            buf[idx] = sign * avg * decay
            idx = next_idx
        return out.astype(np.float32)


class WaveguideString:
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def synthesize(
        self,
        frequency: float,
        duration_s: float,
        pluck_position: float = 0.25,
        damping: float = 0.995,
        stiffness: float = 0.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        total_delay = max(4, int(self.sr / frequency))
        half = total_delay // 2
        n = int(self.sr * duration_s)

        right = np.zeros(half, dtype=np.float64)
        left = np.zeros(half, dtype=np.float64)

        pluck_idx = max(1, int(pluck_position * half))
        excitation = rng.uniform(-1.0, 1.0, half).astype(np.float64)
        for k in range(half):
            if k < pluck_idx:
                excitation[k] *= k / pluck_idx
            else:
                excitation[k] *= (half - k) / (half - pluck_idx)
        right[:] = excitation * 0.5
        left[:] = excitation * 0.5

        r_idx = 0
        l_idx = 0
        out = np.zeros(n, dtype=np.float64)
        prev_r = 0.0
        prev_l = 0.0

        for i in range(n):
            r_out = right[r_idx]
            l_out = left[l_idx]
            out[i] = r_out + l_out

            r_reflected = -r_out * damping
            l_reflected = -l_out * damping

            if stiffness > 0:
                r_reflected = (1.0 - stiffness) * r_reflected + stiffness * prev_r
                l_reflected = (1.0 - stiffness) * l_reflected + stiffness * prev_l

            prev_r = r_reflected
            prev_l = l_reflected

            left[(l_idx - 1) % half] = r_reflected
            right[(r_idx - 1) % half] = l_reflected
            r_idx = (r_idx + 1) % half
            l_idx = (l_idx + 1) % half

        mx = np.max(np.abs(out))
        if mx > 1e-12:
            out /= mx
        return out.astype(np.float32)


class ModalSynthesizer:
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def synthesize(
        self,
        modes: list[tuple[float, float, float]],
        duration_s: float,
        excitation_amplitude: float = 1.0,
    ) -> np.ndarray:
        n = int(self.sr * duration_s)
        t = np.arange(n, dtype=np.float64) / self.sr
        out = np.zeros(n, dtype=np.float64)
        for freq, amp, decay_rate in modes:
            if freq >= self.sr / 2.0:
                continue
            mode_sig = amp * np.exp(-decay_rate * t) * np.sin(2.0 * np.pi * freq * t)
            out += mode_sig
        out *= excitation_amplitude
        mx = np.max(np.abs(out))
        if mx > 1e-12:
            out /= mx
        return out.astype(np.float32)

    @staticmethod
    def bar_modes(fundamental: float, n_modes: int = 8) -> list[tuple[float, float, float]]:
        modes = []
        for k in range(1, n_modes + 1):
            freq = fundamental * k * k
            amp = 1.0 / k
            decay = 2.0 * k
            modes.append((freq, amp, decay))
        return modes

    @staticmethod
    def plate_modes(fundamental: float, n_modes: int = 10) -> list[tuple[float, float, float]]:
        modes = []
        idx = 0
        for m in range(1, n_modes + 1):
            for n_m in range(1, n_modes + 1):
                freq = fundamental * np.sqrt(m * m + n_m * n_m)
                amp = 1.0 / (m * n_m)
                decay = 1.5 * (m + n_m)
                modes.append((freq, amp, decay))
                idx += 1
                if idx >= n_modes:
                    return modes
        return modes

    @staticmethod
    def string_modes(fundamental: float, n_modes: int = 12) -> list[tuple[float, float, float]]:
        modes = []
        for k in range(1, n_modes + 1):
            freq = fundamental * k
            amp = 1.0 / k
            decay = 1.0 + 0.5 * k
            modes.append((freq, amp, decay))
        return modes


class TubeSynthesizer:
    def __init__(self, sr: int = 44100):
        self.sr = sr

    def synthesize(
        self,
        frequency: float,
        duration_s: float,
        n_sections: int = 8,
        radii: Optional[list[float]] = None,
        breath_noise: float = 0.05,
        damping: float = 0.995,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if radii is None:
            radii = [1.0 + 0.3 * np.sin(np.pi * i / n_sections) for i in range(n_sections)]
        else:
            n_sections = len(radii)

        reflection = np.zeros(n_sections - 1, dtype=np.float64)
        for i in range(n_sections - 1):
            a1 = radii[i] ** 2
            a2 = radii[i + 1] ** 2
            denom = a1 + a2
            if denom > 1e-12:
                reflection[i] = (a2 - a1) / denom

        section_len = max(1, int(self.sr / (frequency * 2 * n_sections)))
        total_n = int(self.sr * duration_s)

        forward = [np.zeros(section_len, dtype=np.float64) for _ in range(n_sections)]
        backward = [np.zeros(section_len, dtype=np.float64) for _ in range(n_sections)]
        indices = [0] * n_sections

        out = np.zeros(total_n, dtype=np.float64)

        for i in range(total_n):
            excitation = rng.standard_normal() * breath_noise
            pressure = 0.3 * np.sin(2.0 * np.pi * frequency * i / self.sr)
            forward[0][indices[0]] += (pressure + excitation)

            for s in range(n_sections - 1):
                f_out = forward[s][indices[s]]
                b_out = backward[s + 1][indices[s + 1]]
                k = reflection[s]
                forward[s + 1][(indices[s + 1] - 1) % section_len] = (1.0 + k) * f_out * 0.5 + k * b_out * 0.5
                backward[s][(indices[s] - 1) % section_len] = -k * f_out * 0.5 + (1.0 - k) * b_out * 0.5

            last = n_sections - 1
            lip_reflection = -0.9
            out_sample = forward[last][indices[last]] * (1.0 - abs(lip_reflection))
            backward[last][(indices[last] - 1) % section_len] += forward[last][indices[last]] * lip_reflection

            for s in range(n_sections):
                forward[s][indices[s]] *= damping
                backward[s][indices[s]] *= damping
                indices[s] = (indices[s] + 1) % section_len

            out[i] = out_sample

        mx = np.max(np.abs(out))
        if mx > 1e-12:
            out /= mx
        return out.astype(np.float32)
