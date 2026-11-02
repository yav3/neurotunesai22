"""Audio codec utilities: PCM encoding, mu-law/A-law companding, dithering, sample rate conversion."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AudioFormat:
    sample_rate: int
    bit_depth: int
    channels: int
    encoding: str = "pcm_s16le"


def mu_law_encode(signal: np.ndarray, mu: int = 255) -> np.ndarray:
    """Mu-law companding encode: [-1,1] -> [0, mu]."""
    x = np.clip(signal, -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def mu_law_decode(encoded: np.ndarray, mu: int = 255) -> np.ndarray:
    """Mu-law companding decode."""
    return np.sign(encoded) * (np.exp(np.abs(encoded) * np.log1p(mu)) - 1.0) / mu


def a_law_encode(signal: np.ndarray, a: float = 87.6) -> np.ndarray:
    """A-law companding encode."""
    x = np.clip(signal, -1.0, 1.0)
    ax = np.abs(x)
    mask = ax < (1.0 / a)
    result = np.zeros_like(x)
    result[mask] = a * ax[mask] / (1.0 + np.log(a))
    result[~mask] = (1.0 + np.log(a * ax[~mask])) / (1.0 + np.log(a))
    return np.sign(x) * result


def a_law_decode(encoded: np.ndarray, a: float = 87.6) -> np.ndarray:
    """A-law companding decode."""
    y = np.abs(encoded)
    log_a = np.log(a)
    threshold = 1.0 / (1.0 + log_a)
    mask = y < threshold
    result = np.zeros_like(encoded)
    result[mask] = y[mask] * (1.0 + log_a) / a
    result[~mask] = np.exp(y[~mask] * (1.0 + log_a) - 1.0) / a
    return np.sign(encoded) * result


def float_to_pcm(signal: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """Convert float [-1,1] to signed integer PCM."""
    max_val = (1 << (bit_depth - 1)) - 1
    clipped = np.clip(signal, -1.0, 1.0)
    return np.round(clipped * max_val).astype(np.int32)


def pcm_to_float(pcm: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """Convert signed integer PCM to float [-1,1]."""
    max_val = (1 << (bit_depth - 1)) - 1
    return pcm.astype(np.float64) / max_val


def triangular_dither(signal: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """Add triangular probability density dither before quantisation."""
    lsb = 2.0 / (1 << bit_depth)
    noise = np.random.triangular(-lsb, 0, lsb, size=signal.shape)
    return signal + noise


def noise_shape_first_order(signal: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """First-order noise shaping for dithered quantisation."""
    max_val = (1 << (bit_depth - 1)) - 1
    lsb = 1.0 / max_val
    output = np.zeros_like(signal)
    error = 0.0
    for i in range(len(signal)):
        shaped = signal[i] + error
        quantised = np.round(shaped * max_val) / max_val
        output[i] = quantised
        error = shaped - quantised
    return output


def resample_sinc(signal: np.ndarray, orig_sr: int, target_sr: int,
                  num_zeros: int = 64) -> np.ndarray:
    """Windowed sinc interpolation resampling."""
    if orig_sr == target_sr:
        return signal.copy()
    ratio = target_sr / orig_sr
    n_out = int(np.ceil(len(signal) * ratio))
    output = np.zeros(n_out)
    for i in range(n_out):
        t = i / ratio
        n_center = int(np.round(t))
        for n in range(max(0, n_center - num_zeros), min(len(signal), n_center + num_zeros)):
            x = t - n
            if abs(x) < 1e-12:
                sinc_val = 1.0
            else:
                sinc_val = np.sin(np.pi * x) / (np.pi * x)
            window = 0.5 * (1.0 + np.cos(np.pi * x / num_zeros)) if abs(x) < num_zeros else 0.0
            output[i] += signal[n] * sinc_val * window
    return output


def resample_linear(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear interpolation resampling (fast, lower quality)."""
    if orig_sr == target_sr:
        return signal.copy()
    n_out = int(np.ceil(len(signal) * target_sr / orig_sr))
    indices = np.linspace(0, len(signal) - 1, n_out)
    floor_idx = np.floor(indices).astype(int)
    ceil_idx = np.minimum(floor_idx + 1, len(signal) - 1)
    frac = indices - floor_idx
    return signal[floor_idx] * (1.0 - frac) + signal[ceil_idx] * frac


def split_channels(interleaved: np.ndarray, n_channels: int) -> np.ndarray:
    """De-interleave multi-channel audio to (channels, samples)."""
    n_samples = len(interleaved) // n_channels
    return interleaved[:n_samples * n_channels].reshape(n_samples, n_channels).T


def interleave_channels(channels: np.ndarray) -> np.ndarray:
    """Interleave (channels, samples) to flat array."""
    return channels.T.flatten()


def compute_rms_db(signal: np.ndarray) -> float:
    """RMS level in dBFS."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-12:
        return -120.0
    return float(20.0 * np.log10(rms))


def compute_peak_db(signal: np.ndarray) -> float:
    """Peak level in dBFS."""
    peak = np.max(np.abs(signal))
    if peak < 1e-12:
        return -120.0
    return float(20.0 * np.log10(peak))


def compute_crest_factor(signal: np.ndarray) -> float:
    """Crest factor (peak/RMS) in dB."""
    return compute_peak_db(signal) - compute_rms_db(signal)


def detect_clipping(signal: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of samples at or above clipping threshold."""
    return float(np.mean(np.abs(signal) >= threshold))


def dc_offset(signal: np.ndarray) -> float:
    """DC offset (mean of signal)."""
    return float(np.mean(signal))


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Remove DC offset."""
    return signal - np.mean(signal)


def fade_in(signal: np.ndarray, n_samples: int) -> np.ndarray:
    """Apply raised-cosine fade-in."""
    out = signal.copy()
    n = min(n_samples, len(signal))
    out[:n] *= 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))
    return out


def fade_out(signal: np.ndarray, n_samples: int) -> np.ndarray:
    """Apply raised-cosine fade-out."""
    out = signal.copy()
    n = min(n_samples, len(signal))
    out[-n:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / n))
    return out


def crossfade(signal_a: np.ndarray, signal_b: np.ndarray,
              overlap: int) -> np.ndarray:
    """Crossfade two signals with raised-cosine window."""
    if overlap <= 0:
        return np.concatenate([signal_a, signal_b])
    overlap = min(overlap, len(signal_a), len(signal_b))
    window = 0.5 * (1.0 - np.cos(np.pi * np.arange(overlap) / overlap))
    result = np.zeros(len(signal_a) + len(signal_b) - overlap)
    result[:len(signal_a)] = signal_a
    result[len(signal_a) - overlap:len(signal_a)] *= (1.0 - window)
    result[len(signal_a) - overlap:len(signal_a)] += signal_b[:overlap] * window
    result[len(signal_a):] = signal_b[overlap:]
    return result


def silence_detect(signal: np.ndarray, sr: int, threshold_db: float = -60.0,
                   min_duration: float = 0.1) -> list:
    """Detect silent regions. Returns list of (start_sec, end_sec)."""
    hop = max(1, sr // 100)
    n_frames = len(signal) // hop
    min_frames = int(min_duration * 100)
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    regions = []
    in_silence = False
    start = 0
    for i in range(n_frames):
        chunk = signal[i * hop:(i + 1) * hop]
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < threshold_linear:
            if not in_silence:
                start = i
                in_silence = True
        else:
            if in_silence and (i - start) >= min_frames:
                regions.append((start / 100.0, i / 100.0))
            in_silence = False
    if in_silence and (n_frames - start) >= min_frames:
        regions.append((start / 100.0, n_frames / 100.0))
    return regions
