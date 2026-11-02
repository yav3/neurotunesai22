"""Robust multi-algorithm audio fingerprinting.

Implements five fingerprinting strategies — spectral-peak, chroma-energy,
wavelet, Bark-band, and modulation-spectrum — with configurable hashing,
match scoring via time-offset histograms, and robustness evaluation
against additive noise and low-pass compression.

Pure numpy/scipy — no external ML frameworks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence
from enum import Enum

from scipy.signal import stft as scipy_stft, butter, sosfilt


class HashAlgorithm(Enum):
    SPECTRAL_PEAKS = "spectral_peaks"
    CHROMA_ENERGY = "chroma_energy"
    WAVELET = "wavelet"
    BARK_BAND = "bark_band"
    MODULATION = "modulation"


@dataclass
class FingerprintConfig:
    algorithm: HashAlgorithm = HashAlgorithm.SPECTRAL_PEAKS
    hop_size: int = 512
    n_fft: int = 2048
    n_bands: int = 24
    hash_length: int = 32
    fan_out: int = 5
    sr: int = 22050


@dataclass
class AudioFingerprint:
    hashes: List[int]
    timestamps: List[float]
    algorithm: HashAlgorithm
    duration: float


def _hz_to_bark(f: float) -> float:
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def bark_band_energies(
    signal: np.ndarray, sr: int = 22050, n_fft: int = 2048,
    hop_size: int = 512, n_bands: int = 24,
) -> np.ndarray:
    _, _, Zxx = scipy_stft(signal, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_size)
    mag = np.abs(Zxx) ** 2
    freqs = np.linspace(0, sr / 2, mag.shape[0])
    bark_vals = np.array([_hz_to_bark(f) for f in freqs])
    max_bark = _hz_to_bark(sr / 2)
    band_edges = np.linspace(0, max_bark, n_bands + 1)
    n_frames = mag.shape[1]
    energies = np.zeros((n_bands, n_frames))
    for b in range(n_bands):
        mask = (bark_vals >= band_edges[b]) & (bark_vals < band_edges[b + 1])
        if mask.any():
            energies[b] = mag[mask].sum(axis=0)
    return energies


def _energy_to_hash(energies: np.ndarray, hash_length: int) -> List[int]:
    hashes = []
    n_frames = energies.shape[1]
    for t in range(n_frames):
        col = energies[:, t]
        if col.sum() < 1e-12:
            hashes.append(0)
            continue
        bits = 0
        n_bands = col.shape[0]
        for i in range(min(hash_length, n_bands - 1)):
            if col[i] > col[i + 1]:
                bits |= (1 << i)
        hashes.append(bits)
    return hashes


def _bark_band_fingerprint(
    signal: np.ndarray, cfg: FingerprintConfig,
) -> AudioFingerprint:
    energies = bark_band_energies(signal, cfg.sr, cfg.n_fft, cfg.hop_size, cfg.n_bands)
    hashes = _energy_to_hash(energies, cfg.hash_length)
    n_frames = len(hashes)
    timestamps = [t * cfg.hop_size / cfg.sr for t in range(n_frames)]
    duration = len(signal) / cfg.sr
    return AudioFingerprint(hashes, timestamps, HashAlgorithm.BARK_BAND, duration)


def _ricker(points: int, a: float) -> np.ndarray:
    t = np.arange(points) - (points - 1.0) / 2
    norm = 2.0 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    tsq = (t / a) ** 2
    return norm * (1.0 - tsq) * np.exp(-tsq / 2.0)


def _manual_cwt(signal: np.ndarray, widths: np.ndarray) -> np.ndarray:
    out = np.zeros((len(widths), len(signal)))
    for i, w in enumerate(widths):
        pts = min(10 * int(w) + 1, len(signal))
        wavelet = _ricker(pts, float(w))
        conv = np.convolve(signal, wavelet, mode='same')
        out[i] = conv
    return out


def wavelet_fingerprint(
    signal: np.ndarray, cfg: FingerprintConfig,
) -> AudioFingerprint:
    widths = np.arange(1, cfg.n_bands + 1, dtype=float)
    cwtmatr = _manual_cwt(signal, widths)
    hop = cfg.hop_size
    n_frames = cwtmatr.shape[1] // hop
    hashes: List[int] = []
    timestamps: List[float] = []
    for t in range(n_frames):
        col = np.abs(cwtmatr[:, t * hop])
        bits = 0
        for i in range(min(cfg.hash_length, len(col) - 1)):
            if col[i] > col[i + 1]:
                bits |= (1 << i)
        hashes.append(bits)
        timestamps.append(t * hop / cfg.sr)
    duration = len(signal) / cfg.sr
    return AudioFingerprint(hashes, timestamps, HashAlgorithm.WAVELET, duration)


def modulation_spectrum_hash(
    signal: np.ndarray, cfg: FingerprintConfig,
) -> AudioFingerprint:
    _, _, Zxx = scipy_stft(signal, fs=cfg.sr, nperseg=cfg.n_fft,
                           noverlap=cfg.n_fft - cfg.hop_size)
    mag = np.abs(Zxx)
    n_sub = min(cfg.n_bands, mag.shape[0])
    band_size = mag.shape[0] // n_sub
    hashes: List[int] = []
    timestamps: List[float] = []
    n_frames = mag.shape[1]
    for t in range(1, n_frames):
        bits = 0
        for b in range(n_sub):
            lo = b * band_size
            hi = min(lo + band_size, mag.shape[0])
            env_now = mag[lo:hi, t].mean()
            env_prev = mag[lo:hi, t - 1].mean()
            if env_now > env_prev:
                bits |= (1 << (b % cfg.hash_length))
        hashes.append(bits)
        timestamps.append(t * cfg.hop_size / cfg.sr)
    duration = len(signal) / cfg.sr
    return AudioFingerprint(hashes, timestamps, HashAlgorithm.MODULATION, duration)


def chromaprint_style(
    signal: np.ndarray, cfg: FingerprintConfig,
) -> AudioFingerprint:
    _, _, Zxx = scipy_stft(signal, fs=cfg.sr, nperseg=cfg.n_fft,
                           noverlap=cfg.n_fft - cfg.hop_size)
    mag = np.abs(Zxx)
    freqs = np.linspace(0, cfg.sr / 2, mag.shape[0])
    chroma = np.zeros((12, mag.shape[1]))
    for i, f in enumerate(freqs):
        if f < 20:
            continue
        midi = 69 + 12 * np.log2(f / 440.0)
        pc = int(round(midi)) % 12
        chroma[pc] += mag[i]
    hashes: List[int] = []
    timestamps: List[float] = []
    for t in range(chroma.shape[1]):
        col = chroma[:, t]
        bits = 0
        for i in range(11):
            if col[i] > col[i + 1]:
                bits |= (1 << i)
        hashes.append(bits)
        timestamps.append(t * cfg.hop_size / cfg.sr)
    duration = len(signal) / cfg.sr
    return AudioFingerprint(hashes, timestamps, HashAlgorithm.CHROMA_ENERGY, duration)


def _spectral_peak_fingerprint(
    signal: np.ndarray, cfg: FingerprintConfig,
) -> AudioFingerprint:
    _, _, Zxx = scipy_stft(signal, fs=cfg.sr, nperseg=cfg.n_fft,
                           noverlap=cfg.n_fft - cfg.hop_size)
    mag = np.abs(Zxx)
    hashes: List[int] = []
    timestamps: List[float] = []
    for t in range(mag.shape[1]):
        col = mag[:, t]
        peak_idx = np.argsort(col)[-cfg.fan_out:]
        h = 0
        for i, idx in enumerate(sorted(peak_idx)):
            h ^= (idx * (i + 1) * 2654435761) & ((1 << cfg.hash_length) - 1)
        hashes.append(h)
        timestamps.append(t * cfg.hop_size / cfg.sr)
    duration = len(signal) / cfg.sr
    return AudioFingerprint(hashes, timestamps, HashAlgorithm.SPECTRAL_PEAKS, duration)


def generate_fingerprint(
    signal: np.ndarray, cfg: Optional[FingerprintConfig] = None,
) -> AudioFingerprint:
    if cfg is None:
        cfg = FingerprintConfig()
    dispatch = {
        HashAlgorithm.SPECTRAL_PEAKS: _spectral_peak_fingerprint,
        HashAlgorithm.CHROMA_ENERGY: chromaprint_style,
        HashAlgorithm.WAVELET: wavelet_fingerprint,
        HashAlgorithm.BARK_BAND: _bark_band_fingerprint,
        HashAlgorithm.MODULATION: modulation_spectrum_hash,
    }
    return dispatch[cfg.algorithm](signal, cfg)


def _popcount(x: int) -> int:
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def _hamming(a: int, b: int, bits: int = 32) -> float:
    return _popcount(a ^ b) / max(bits, 1)


def time_offset_histogram(
    fp_a: AudioFingerprint, fp_b: AudioFingerprint,
    tolerance: float = 0.15, max_hamming: float = 0.25,
) -> np.ndarray:
    offsets: List[float] = []
    for i, ha in enumerate(fp_a.hashes):
        ta = fp_a.timestamps[i]
        for j, hb in enumerate(fp_b.hashes):
            if _hamming(ha, hb) <= max_hamming:
                offsets.append(fp_b.timestamps[j] - ta)
    if not offsets:
        return np.array([])
    arr = np.array(offsets)
    if (arr.max() - arr.min()) < 1e-9:
        return arr
    bins = max(10, int((arr.max() - arr.min()) / tolerance))
    hist, _ = np.histogram(arr, bins=bins)
    return hist


def fingerprint_match_score(
    fp_a: AudioFingerprint, fp_b: AudioFingerprint,
    max_hamming: float = 0.25,
) -> Tuple[float, float]:
    hist = time_offset_histogram(fp_a, fp_b, max_hamming=max_hamming)
    if len(hist) == 0:
        return 0.0, 0.0
    peak = float(hist.max())
    total = float(hist.sum())
    score = peak / max(total, 1.0)
    best_bin = int(np.argmax(hist))
    offset = best_bin * (fp_a.duration / max(len(hist), 1))
    return score, offset


def batch_match(
    query: AudioFingerprint,
    database: Sequence[AudioFingerprint],
    top_k: int = 5,
) -> List[Tuple[int, float, float]]:
    results: List[Tuple[int, float, float]] = []
    for idx, ref in enumerate(database):
        score, offset = fingerprint_match_score(query, ref)
        results.append((idx, score, offset))
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def false_positive_rate(
    n_trials: int = 200, signal_len: int = 22050 * 5,
    cfg: Optional[FingerprintConfig] = None, threshold: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(42)
    if cfg is None:
        cfg = FingerprintConfig()
    positives = 0
    for _ in range(n_trials):
        a = rng.standard_normal(signal_len).astype(np.float32)
        b = rng.standard_normal(signal_len).astype(np.float32)
        fp_a = generate_fingerprint(a, cfg)
        fp_b = generate_fingerprint(b, cfg)
        score, _ = fingerprint_match_score(fp_a, fp_b)
        if score >= threshold:
            positives += 1
    return positives / max(n_trials, 1)


def noise_robustness_test(
    signal: np.ndarray,
    snr_levels: Sequence[float] = (30, 20, 10, 5, 0),
    cfg: Optional[FingerprintConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float]]:
    if rng is None:
        rng = np.random.default_rng(0)
    if cfg is None:
        cfg = FingerprintConfig()
    ref_fp = generate_fingerprint(signal, cfg)
    results: List[Tuple[float, float]] = []
    sig_power = float(np.mean(signal ** 2))
    for snr in snr_levels:
        noise_power = sig_power / (10.0 ** (snr / 10.0))
        noise = rng.normal(0, np.sqrt(max(noise_power, 1e-12)), len(signal)).astype(signal.dtype)
        degraded = signal + noise
        deg_fp = generate_fingerprint(degraded, cfg)
        score, _ = fingerprint_match_score(ref_fp, deg_fp)
        results.append((float(snr), score))
    return results


def compression_robustness(
    signal: np.ndarray, sr: int = 22050,
    cutoffs: Sequence[float] = (8000, 4000, 2000, 1000),
    cfg: Optional[FingerprintConfig] = None,
) -> List[Tuple[float, float]]:
    if cfg is None:
        cfg = FingerprintConfig(sr=sr)
    ref_fp = generate_fingerprint(signal, cfg)
    results: List[Tuple[float, float]] = []
    for fc in cutoffs:
        nyq = sr / 2.0
        if fc >= nyq:
            results.append((float(fc), 1.0))
            continue
        sos = butter(4, fc / nyq, btype='low', output='sos')
        filtered = sosfilt(sos, signal).astype(signal.dtype)
        deg_fp = generate_fingerprint(filtered, cfg)
        score, _ = fingerprint_match_score(ref_fp, deg_fp)
        results.append((float(fc), score))
    return results
