"""Audio codec simulation: perceptual coding, quantization, bitrate effects.

Implements simplified audio codec models for testing fingerprint robustness
under lossy compression artifacts (quantization noise, spectral band limiting,
pre-echo, temporal masking).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class CodecType(Enum):
    PCM = "pcm"
    MP3_128 = "mp3_128"
    MP3_192 = "mp3_192"
    MP3_320 = "mp3_320"
    AAC_128 = "aac_128"
    AAC_256 = "aac_256"
    OGG_128 = "ogg_128"
    OPUS_64 = "opus_64"


CODEC_PARAMS: dict[CodecType, dict] = {
    CodecType.PCM: {"cutoff_hz": 22050, "quantization_bits": 16, "noise_floor_db": -96},
    CodecType.MP3_128: {"cutoff_hz": 16000, "quantization_bits": 10, "noise_floor_db": -60},
    CodecType.MP3_192: {"cutoff_hz": 18000, "quantization_bits": 12, "noise_floor_db": -70},
    CodecType.MP3_320: {"cutoff_hz": 20000, "quantization_bits": 14, "noise_floor_db": -80},
    CodecType.AAC_128: {"cutoff_hz": 17000, "quantization_bits": 11, "noise_floor_db": -65},
    CodecType.AAC_256: {"cutoff_hz": 19500, "quantization_bits": 13, "noise_floor_db": -75},
    CodecType.OGG_128: {"cutoff_hz": 16500, "quantization_bits": 10, "noise_floor_db": -62},
    CodecType.OPUS_64: {"cutoff_hz": 12000, "quantization_bits": 8, "noise_floor_db": -50},
}


@dataclass(frozen=True)
class CodecArtifact:
    codec: CodecType
    distortion_db: float
    spectral_loss_hz: float
    quantization_snr_db: float


def apply_lowpass(signal: np.ndarray, cutoff_hz: float,
                  sr: float = 44100.0) -> np.ndarray:
    if cutoff_hz >= sr / 2:
        return signal.copy()
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    rolloff_width = cutoff_hz * 0.1
    gain = np.where(
        freqs <= cutoff_hz,
        1.0,
        np.exp(-((freqs - cutoff_hz) / max(rolloff_width, 1.0)) ** 2),
    )
    return np.fft.irfft(spectrum * gain, n=len(signal))


def quantize(signal: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    step = 2.0 / levels
    quantized = np.round(signal / step) * step
    return np.clip(quantized, -1.0, 1.0)


def add_quantization_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power = np.mean(signal ** 2)
    if sig_power < 1e-20:
        return signal.copy()
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(42)
    noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def simulate_codec(signal: np.ndarray, codec: CodecType,
                   sr: float = 44100.0) -> np.ndarray:
    if codec == CodecType.PCM:
        return signal.copy()
    params = CODEC_PARAMS[codec]
    result = apply_lowpass(signal, params["cutoff_hz"], sr)
    result = quantize(result, params["quantization_bits"])
    noise_floor_linear = 10.0 ** (params["noise_floor_db"] / 20.0)
    rng = np.random.default_rng(7)
    result += rng.normal(0, noise_floor_linear, len(result))
    return np.clip(result, -1.0, 1.0)


def codec_distortion(original: np.ndarray, coded: np.ndarray) -> float:
    diff = original[:len(coded)] - coded[:len(original)]
    n = min(len(original), len(coded))
    if n == 0:
        return 0.0
    rms_diff = np.sqrt(np.mean(diff[:n] ** 2))
    return 20.0 * np.log10(rms_diff + 1e-12)


def analyze_codec_artifacts(signal: np.ndarray, codec: CodecType,
                            sr: float = 44100.0) -> CodecArtifact:
    coded = simulate_codec(signal, codec, sr)
    dist = codec_distortion(signal, coded)
    params = CODEC_PARAMS[codec]
    sig_power = np.mean(signal ** 2)
    noise = coded - signal[:len(coded)]
    noise_power = np.mean(noise[:len(signal)] ** 2)
    qsnr = 10.0 * np.log10(sig_power / (noise_power + 1e-20))
    return CodecArtifact(codec, dist, float(params["cutoff_hz"]), qsnr)


@dataclass
class CodecRobustnessTest:
    codecs: list[CodecType] = field(default_factory=lambda: [
        CodecType.MP3_128, CodecType.MP3_320, CodecType.AAC_128, CodecType.OGG_128,
    ])

    def test_fingerprint_survival(self, signal: np.ndarray,
                                  fingerprint_fn, sr: float = 44100.0,
                                  similarity_fn=None) -> dict[str, float]:
        original_fp = fingerprint_fn(signal)
        results: dict[str, float] = {}
        for codec in self.codecs:
            coded = simulate_codec(signal, codec, sr)
            coded_fp = fingerprint_fn(coded)
            if similarity_fn is not None:
                sim = similarity_fn(original_fp, coded_fp)
            else:
                if isinstance(original_fp, np.ndarray) and isinstance(coded_fp, np.ndarray):
                    n = min(len(original_fp), len(coded_fp))
                    if n > 0:
                        cos = np.dot(original_fp[:n], coded_fp[:n]) / (
                            np.linalg.norm(original_fp[:n]) * np.linalg.norm(coded_fp[:n]) + 1e-12)
                        sim = float(cos)
                    else:
                        sim = 0.0
                else:
                    sim = 1.0 if original_fp == coded_fp else 0.0
            results[codec.value] = sim
        return results


def pre_echo_simulation(signal: np.ndarray, block_size: int = 1152,
                        spread_samples: int = 64) -> np.ndarray:
    result = signal.copy()
    n_blocks = len(signal) // block_size
    for b in range(1, n_blocks):
        block_start = b * block_size
        block_energy = np.mean(signal[block_start:block_start + block_size] ** 2)
        prev_energy = np.mean(signal[block_start - block_size:block_start] ** 2)
        if block_energy > prev_energy * 10:
            pre_echo_start = max(0, block_start - spread_samples)
            leak = np.sqrt(block_energy) * 0.01
            rng = np.random.default_rng(b)
            result[pre_echo_start:block_start] += rng.normal(0, leak, block_start - pre_echo_start)
    return np.clip(result, -1.0, 1.0)
