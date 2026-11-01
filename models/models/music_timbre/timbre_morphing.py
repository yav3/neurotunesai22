"""Timbre morphing: spectral interpolation, cross-synthesis, timbre transfer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class MorphMethod(str, Enum):
    SPECTRAL_LERP = "spectral_lerp"
    ENVELOPE_SWAP = "envelope_swap"
    HARMONIC_BLEND = "harmonic_blend"


@dataclass
class SpectralFrame:
    magnitudes: List[float]
    phases: List[float]

    @property
    def n_bins(self) -> int:
        return len(self.magnitudes)


@dataclass
class TimbreMorphResult:
    frames: List[SpectralFrame]
    method: MorphMethod
    blend_factor: float


def spectral_lerp(frame_a: SpectralFrame, frame_b: SpectralFrame,
                  alpha: float) -> SpectralFrame:
    alpha = max(0.0, min(1.0, alpha))
    n = min(frame_a.n_bins, frame_b.n_bins)
    mags = [(1 - alpha) * frame_a.magnitudes[i] + alpha * frame_b.magnitudes[i]
            for i in range(n)]
    phases = [(1 - alpha) * frame_a.phases[i] + alpha * frame_b.phases[i]
              for i in range(n)]
    return SpectralFrame(magnitudes=mags, phases=phases)


def spectral_envelope(magnitudes: List[float], order: int = 20) -> List[float]:
    n = len(magnitudes)
    if n == 0:
        return []
    log_mags = [math.log(max(m, 1e-10)) for m in magnitudes]
    if order >= n:
        return list(magnitudes)
    env: List[float] = []
    for i in range(n):
        lo = max(0, i - order)
        hi = min(n, i + order + 1)
        env.append(math.exp(sum(log_mags[lo:hi]) / (hi - lo)))
    return env


def envelope_swap(frame_source: SpectralFrame, frame_target: SpectralFrame,
                  order: int = 20) -> SpectralFrame:
    env_src = spectral_envelope(frame_source.magnitudes, order)
    env_tgt = spectral_envelope(frame_target.magnitudes, order)
    n = min(frame_source.n_bins, frame_target.n_bins)
    residual = [frame_source.magnitudes[i] / max(env_src[i], 1e-10) for i in range(n)]
    mags = [residual[i] * env_tgt[i] for i in range(n)]
    return SpectralFrame(magnitudes=mags, phases=list(frame_source.phases[:n]))


def harmonic_blend(partials_a: List[Tuple[float, float]],
                   partials_b: List[Tuple[float, float]],
                   alpha: float) -> List[Tuple[float, float]]:
    alpha = max(0.0, min(1.0, alpha))
    n = min(len(partials_a), len(partials_b))
    result: List[Tuple[float, float]] = []
    for i in range(n):
        freq = (1 - alpha) * partials_a[i][0] + alpha * partials_b[i][0]
        amp = (1 - alpha) * partials_a[i][1] + alpha * partials_b[i][1]
        result.append((freq, amp))
    return result


def morph_trajectory(frames_a: List[SpectralFrame], frames_b: List[SpectralFrame],
                     alpha_start: float = 0.0, alpha_end: float = 1.0,
                     method: MorphMethod = MorphMethod.SPECTRAL_LERP) -> TimbreMorphResult:
    n = min(len(frames_a), len(frames_b))
    out: List[SpectralFrame] = []
    for i in range(n):
        t = i / max(n - 1, 1)
        alpha = alpha_start + t * (alpha_end - alpha_start)
        if method == MorphMethod.ENVELOPE_SWAP:
            frame = envelope_swap(frames_a[i], frames_b[i])
        else:
            frame = spectral_lerp(frames_a[i], frames_b[i], alpha)
        out.append(frame)
    return TimbreMorphResult(frames=out, method=method, blend_factor=(alpha_start + alpha_end) / 2.0)


def spectral_distance(frame_a: SpectralFrame, frame_b: SpectralFrame) -> float:
    n = min(frame_a.n_bins, frame_b.n_bins)
    if n == 0:
        return 0.0
    log_a = [math.log(max(m, 1e-10)) for m in frame_a.magnitudes[:n]]
    log_b = [math.log(max(m, 1e-10)) for m in frame_b.magnitudes[:n]]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(log_a, log_b)) / n)


def brightness_transfer(frame: SpectralFrame, brightness_shift: float) -> SpectralFrame:
    n = frame.n_bins
    if n == 0:
        return frame
    mags = list(frame.magnitudes)
    for i in range(n):
        freq_ratio = (i + 1) / n
        scale = 1.0 + brightness_shift * freq_ratio
        mags[i] *= max(0.0, scale)
    return SpectralFrame(magnitudes=mags, phases=list(frame.phases))
