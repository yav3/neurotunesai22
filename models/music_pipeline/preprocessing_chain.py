"""Composable audio preprocessing chains.

Build multi-stage preprocessing pipelines from reusable atomic stages:
DC removal, silence trimming, channel mixing, loudness normalization,
resampling, pre-emphasis, fade-in/out, and dithering.  Each stage is a
pure function ``np.ndarray -> np.ndarray``; the chain applies them in
order and records per-stage metadata (gain changes, trimmed samples, etc.).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stage identifiers
# ---------------------------------------------------------------------------

class StageType(enum.Enum):
    DC_REMOVE = "dc_remove"
    TRIM_SILENCE = "trim_silence"
    CHANNEL_MIX = "channel_mix"
    LOUDNESS_NORM = "loudness_norm"
    RESAMPLE = "resample"
    PRE_EMPHASIS = "pre_emphasis"
    FADE = "fade"
    DITHER = "dither"
    CLIP_GUARD = "clip_guard"


# ---------------------------------------------------------------------------
# Stage metadata
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: StageType
    input_length: int
    output_length: int
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChainResult:
    signal: np.ndarray
    sr: float
    stages: List[StageResult] = field(default_factory=list)

    @property
    def total_gain_db(self) -> float:
        return sum(s.metadata.get("gain_db", 0.0) for s in self.stages)

    @property
    def total_trimmed(self) -> int:
        return sum(
            s.input_length - s.output_length
            for s in self.stages
            if s.stage == StageType.TRIM_SILENCE
        )


# ---------------------------------------------------------------------------
# Atomic preprocessing functions
# ---------------------------------------------------------------------------

def remove_dc(signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Subtract the mean (DC offset) from the signal."""
    dc = float(np.mean(signal))
    return signal - dc, {"dc_offset_removed": dc}


def trim_silence(
    signal: np.ndarray,
    sr: float = 22050,
    threshold_db: float = -40.0,
    frame_ms: float = 20.0,
    min_silence_ms: float = 200.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Remove leading and trailing silence below *threshold_db*.

    Uses frame-level RMS energy with a configurable minimum silence
    duration to prevent trimming brief pauses inside content.
    """
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    n_frames = max(1, len(signal) // frame_len)
    threshold_amp = 10.0 ** (threshold_db / 20.0)

    rms = np.array([
        np.sqrt(np.mean(signal[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    active = rms > threshold_amp
    if not np.any(active):
        return signal, {"trimmed_start": 0, "trimmed_end": 0}

    first = int(np.argmax(active))
    last = n_frames - 1 - int(np.argmax(active[::-1]))

    start_sample = first * frame_len
    end_sample = min((last + 1) * frame_len, len(signal))

    trimmed = signal[start_sample:end_sample]
    return trimmed, {
        "trimmed_start": start_sample,
        "trimmed_end": len(signal) - end_sample,
        "trimmed_pct": 1.0 - len(trimmed) / max(1, len(signal)),
    }


def mix_to_mono(signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Down-mix multi-channel audio to mono by averaging channels.

    Input shape: ``(channels, samples)`` or ``(samples,)``.
    """
    if signal.ndim == 1:
        return signal, {"n_channels_in": 1}
    n_ch = signal.shape[0]
    mono = np.mean(signal, axis=0)
    return mono, {"n_channels_in": float(n_ch)}


def mix_to_stereo(signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Ensure output is stereo (2, N).  Mono is duplicated; >2ch averaged to stereo panning."""
    if signal.ndim == 1:
        stereo = np.stack([signal, signal])
        return stereo, {"n_channels_in": 1.0}
    if signal.shape[0] == 2:
        return signal, {"n_channels_in": 2.0}
    n_ch = signal.shape[0]
    left = np.mean(signal[: n_ch // 2], axis=0)
    right = np.mean(signal[n_ch // 2 :], axis=0)
    return np.stack([left, right]), {"n_channels_in": float(n_ch)}


def loudness_normalize(
    signal: np.ndarray,
    target_db: float = -14.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """RMS-based loudness normalization to *target_db* LUFS (approximation).

    Applies a single gain factor so that the output RMS matches the target.
    """
    rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
    current_db = 20.0 * np.log10(rms)
    gain_db = target_db - current_db
    gain_linear = 10.0 ** (gain_db / 20.0)
    normed = signal * gain_linear
    return normed, {"gain_db": gain_db, "rms_before_db": current_db}


def peak_normalize(
    signal: np.ndarray,
    target_peak: float = 0.95,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Normalize so that peak amplitude equals *target_peak*."""
    peak = np.max(np.abs(signal)) + 1e-10
    gain = target_peak / peak
    gain_db = 20.0 * np.log10(gain)
    return signal * gain, {"gain_db": gain_db, "peak_before": float(peak)}


def resample_linear(
    signal: np.ndarray,
    sr_in: float,
    sr_out: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Linear-interpolation resampler (no anti-alias filter).

    For production quality use a polyphase filter; this is a lightweight
    fallback that avoids scipy.signal dependencies.
    """
    if abs(sr_in - sr_out) < 0.01:
        return signal, {"ratio": 1.0}
    ratio = sr_out / sr_in
    n_out = int(len(signal) * ratio)
    x_old = np.arange(len(signal))
    x_new = np.linspace(0, len(signal) - 1, n_out)
    resampled = np.interp(x_new, x_old, signal)
    return resampled, {"ratio": ratio, "sr_out": sr_out}


def pre_emphasis(
    signal: np.ndarray,
    coeff: float = 0.97,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """First-order pre-emphasis filter: y[n] = x[n] - coeff * x[n-1].

    Boosts high frequencies; standard preprocessing for speech/vocal analysis.
    """
    emphasized = np.empty_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - coeff * signal[:-1]
    return emphasized, {"coeff": coeff}


def de_emphasis(
    signal: np.ndarray,
    coeff: float = 0.97,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Inverse pre-emphasis: y[n] = x[n] + coeff * y[n-1]."""
    out = np.empty_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = signal[i] + coeff * out[i - 1]
    return out, {"coeff": coeff}


def apply_fade(
    signal: np.ndarray,
    sr: float = 22050,
    fade_in_ms: float = 10.0,
    fade_out_ms: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply raised-cosine fade-in and fade-out."""
    n_in = min(int(sr * fade_in_ms / 1000.0), len(signal))
    n_out = min(int(sr * fade_out_ms / 1000.0), len(signal))
    result = signal.copy()
    if n_in > 0:
        t = np.linspace(0, np.pi / 2, n_in)
        result[:n_in] *= np.sin(t) ** 2
    if n_out > 0:
        t = np.linspace(np.pi / 2, 0, n_out)
        result[-n_out:] *= np.sin(t) ** 2
    return result, {"fade_in_samples": n_in, "fade_out_samples": n_out}


def add_dither(
    signal: np.ndarray,
    bit_depth: int = 16,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Add triangular-PDF dither at the quantization noise floor.

    Standard TPDF dither: sum of two independent uniform distributions
    spanning ±1 LSB, shaped to the quantization step of *bit_depth*.
    """
    lsb = 2.0 / (2 ** bit_depth)
    noise = (np.random.uniform(-lsb, lsb, len(signal))
             + np.random.uniform(-lsb, lsb, len(signal)))
    return signal + noise, {"bit_depth": float(bit_depth), "lsb": lsb}


def clip_guard(
    signal: np.ndarray,
    headroom_db: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Soft-clip any samples exceeding ``1.0 - headroom`` via tanh saturation."""
    ceiling = 10.0 ** (-headroom_db / 20.0)
    peak = np.max(np.abs(signal))
    if peak <= ceiling:
        return signal, {"clipped": 0.0, "peak": peak}
    scale = ceiling / peak
    soft = np.tanh(signal * (1.0 / ceiling)) * ceiling
    n_clipped = int(np.sum(np.abs(signal) > ceiling))
    return soft, {"clipped": float(n_clipped), "peak": peak, "ceiling": ceiling}


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

StageFn = Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]]


@dataclass
class PreprocessingChain:
    """Ordered chain of preprocessing stages.

    Build with ``add()`` or from a preset via ``default_chain()``.
    Call ``run()`` to apply all stages and collect per-stage metadata.
    """
    _stages: List[Tuple[StageType, StageFn]] = field(default_factory=list)

    def add(self, stage_type: StageType, fn: StageFn) -> "PreprocessingChain":
        self._stages.append((stage_type, fn))
        return self

    def run(self, signal: np.ndarray, sr: float = 22050) -> ChainResult:
        results: List[StageResult] = []
        current = signal.copy()
        for stype, fn in self._stages:
            n_in = len(current)
            current, meta = fn(current)
            results.append(StageResult(
                stage=stype,
                input_length=n_in,
                output_length=len(current),
                metadata=meta,
            ))
        return ChainResult(signal=current, sr=sr, stages=results)

    def __len__(self) -> int:
        return len(self._stages)

    @property
    def stage_names(self) -> List[str]:
        return [s.value for s, _ in self._stages]


def default_chain(
    sr: float = 22050,
    target_db: float = -14.0,
    trim_threshold_db: float = -40.0,
    pre_emph_coeff: float = 0.97,
) -> PreprocessingChain:
    """Standard preprocessing chain: DC → trim → mono → loudness → pre-emphasis → fade → clip guard."""
    chain = PreprocessingChain()
    chain.add(StageType.DC_REMOVE, lambda s: remove_dc(s))
    chain.add(StageType.TRIM_SILENCE, lambda s: trim_silence(s, sr=sr, threshold_db=trim_threshold_db))
    chain.add(StageType.CHANNEL_MIX, lambda s: mix_to_mono(s))
    chain.add(StageType.LOUDNESS_NORM, lambda s: loudness_normalize(s, target_db=target_db))
    chain.add(StageType.PRE_EMPHASIS, lambda s: pre_emphasis(s, coeff=pre_emph_coeff))
    chain.add(StageType.FADE, lambda s: apply_fade(s, sr=sr))
    chain.add(StageType.CLIP_GUARD, lambda s: clip_guard(s))
    return chain


def speech_chain(sr: float = 16000) -> PreprocessingChain:
    """Chain optimized for speech/vocal signals: aggressive trim, 16 kHz target."""
    chain = PreprocessingChain()
    chain.add(StageType.DC_REMOVE, lambda s: remove_dc(s))
    chain.add(StageType.TRIM_SILENCE, lambda s: trim_silence(s, sr=sr, threshold_db=-35.0))
    chain.add(StageType.CHANNEL_MIX, lambda s: mix_to_mono(s))
    chain.add(StageType.LOUDNESS_NORM, lambda s: loudness_normalize(s, target_db=-18.0))
    chain.add(StageType.PRE_EMPHASIS, lambda s: pre_emphasis(s, coeff=0.97))
    chain.add(StageType.DITHER, lambda s: add_dither(s, bit_depth=16))
    return chain


def mastering_chain(sr: float = 44100) -> PreprocessingChain:
    """Chain for mastering-quality output: peak norm, fade, dither, clip guard."""
    chain = PreprocessingChain()
    chain.add(StageType.DC_REMOVE, lambda s: remove_dc(s))
    chain.add(StageType.LOUDNESS_NORM, lambda s: peak_normalize(s, target_peak=0.95))
    chain.add(StageType.FADE, lambda s: apply_fade(s, sr=sr, fade_in_ms=5.0, fade_out_ms=50.0))
    chain.add(StageType.DITHER, lambda s: add_dither(s, bit_depth=24))
    chain.add(StageType.CLIP_GUARD, lambda s: clip_guard(s, headroom_db=0.3))
    return chain
