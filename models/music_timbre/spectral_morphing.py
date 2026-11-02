"""Spectral morphing: interpolation and cross-synthesis between timbres.

Implements spectral envelope estimation, cross-fade morphing,
formant transplant, and spectral centroid matching — pure numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


def spectral_envelope(audio: np.ndarray, sr: int, order: int = 30,
                      frame_len: int = 2048) -> np.ndarray:
    """LPC-based spectral envelope (one frame)."""
    x = audio[:frame_len] * np.hanning(min(len(audio), frame_len))
    if len(x) < order + 1:
        return np.ones(frame_len // 2 + 1)
    r = np.correlate(x, x, mode='full')
    r = r[len(r) // 2: len(r) // 2 + order + 1]
    R = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            idx = abs(i - j)
            R[i, j] = r[idx]
    rhs = r[1:order + 1]
    try:
        a = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        return np.ones(frame_len // 2 + 1)
    a_full = np.concatenate([[1.0], -a])
    n_fft = frame_len // 2 + 1
    w, h = sp_signal.freqz([1.0], a_full, worN=n_fft, fs=sr)
    env = np.abs(h)
    env /= np.max(env) + 1e-12
    return env


def stft_frames(audio: np.ndarray, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    frames = np.zeros((n_frames, frame_len // 2 + 1), dtype=complex)
    w = np.hanning(frame_len)
    for i in range(n_frames):
        seg = audio[i * hop: i * hop + frame_len]
        if len(seg) < frame_len:
            seg = np.pad(seg, (0, frame_len - len(seg)))
        frames[i] = np.fft.rfft(seg * w)
    return frames


def istft_frames(frames: np.ndarray, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * hop + frame_len
    out = np.zeros(out_len)
    norm = np.zeros(out_len)
    w = np.hanning(frame_len)
    for i in range(n_frames):
        seg = np.fft.irfft(frames[i], n=frame_len) * w
        start = i * hop
        out[start:start + frame_len] += seg
        norm[start:start + frame_len] += w ** 2
    norm = np.maximum(norm, 1e-8)
    return out / norm


def morph_spectral(source: np.ndarray, target: np.ndarray, sr: int,
                   alpha: float = 0.5, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    """Morph between two sounds by interpolating spectral envelopes.

    alpha=0 → pure source, alpha=1 → pure target.
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    min_len = min(len(source), len(target))
    src = source[:min_len]
    tgt = target[:min_len]
    S = stft_frames(src, frame_len, hop)
    T = stft_frames(tgt, frame_len, hop)
    n = min(S.shape[0], T.shape[0])
    out = np.zeros_like(S[:n])
    for i in range(n):
        s_mag, s_phase = np.abs(S[i]), np.angle(S[i])
        t_mag = np.abs(T[i])
        s_env = np.maximum(s_mag, 1e-12)
        t_env = np.maximum(t_mag, 1e-12)
        s_fine = s_mag / s_env
        morphed_env = s_env ** (1 - alpha) * t_env ** alpha
        out[i] = morphed_env * s_fine * np.exp(1j * s_phase)
    return istft_frames(out, frame_len, hop)


def formant_transplant(source: np.ndarray, target: np.ndarray, sr: int,
                       order: int = 30, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    """Replace source's spectral envelope with target's while keeping source excitation."""
    S = stft_frames(source, frame_len, hop)
    n_frames = S.shape[0]
    out = np.zeros_like(S)
    for i in range(n_frames):
        src_start = i * hop
        tgt_start = min(src_start, len(target) - frame_len)
        if tgt_start < 0:
            tgt_start = 0
        src_seg = source[src_start:src_start + frame_len]
        tgt_seg = target[tgt_start:tgt_start + frame_len]
        if len(src_seg) < frame_len:
            src_seg = np.pad(src_seg, (0, frame_len - len(src_seg)))
        if len(tgt_seg) < frame_len:
            tgt_seg = np.pad(tgt_seg, (0, frame_len - len(tgt_seg)))
        src_env = spectral_envelope(src_seg, sr, order, frame_len)
        tgt_env = spectral_envelope(tgt_seg, sr, order, frame_len)
        s_mag = np.abs(S[i])
        s_phase = np.angle(S[i])
        flat = s_mag / (src_env + 1e-12)
        out[i] = flat * tgt_env * np.exp(1j * s_phase)
    return istft_frames(out, frame_len, hop)


def spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> float:
    total = np.sum(mag) + 1e-12
    return float(np.sum(freqs * mag) / total)


def brightness_match(audio: np.ndarray, sr: int, target_centroid: float,
                     frame_len: int = 2048) -> np.ndarray:
    """Tilt the spectrum so its centroid matches the target."""
    n_fft = frame_len // 2 + 1
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)
    X = np.fft.rfft(audio[:frame_len] * np.hanning(min(len(audio), frame_len)))
    mag = np.abs(X)
    current = spectral_centroid(mag, freqs)
    if current < 1.0:
        return audio.copy()
    ratio = target_centroid / current
    tilt = (freqs / (current + 1e-12)) ** (np.log2(ratio))
    tilt /= np.max(tilt) + 1e-12
    return np.fft.irfft(X * tilt, n=frame_len)[:len(audio)]


@dataclass
class TimbreMorphSequence:
    """Time-varying morph trajectory between two sounds."""
    source: np.ndarray
    target: np.ndarray
    sr: int
    alphas: np.ndarray = None

    def __post_init__(self):
        if self.alphas is None:
            self.alphas = np.linspace(0, 1, 11)

    def render(self) -> np.ndarray:
        results = []
        for alpha in self.alphas:
            morphed = morph_spectral(self.source, self.target, self.sr, alpha=alpha)
            results.append(morphed)
        return np.concatenate(results)

    def render_crossfade(self, duration_sec: float = 5.0) -> np.ndarray:
        n = int(self.sr * duration_sec)
        min_len = min(len(self.source), len(self.target), n)
        alpha_curve = np.linspace(0, 1, min_len)
        S = stft_frames(self.source[:min_len])
        T = stft_frames(self.target[:min_len])
        n_fr = min(S.shape[0], T.shape[0])
        out = np.zeros_like(S[:n_fr])
        for i in range(n_fr):
            a = alpha_curve[min(i * 512, min_len - 1)]
            s_mag, s_ph = np.abs(S[i]), np.angle(S[i])
            t_mag = np.abs(T[i])
            mag = s_mag * (1 - a) + t_mag * a
            out[i] = mag * np.exp(1j * s_ph)
        return istft_frames(out)


def timbre_distance(env_a: np.ndarray, env_b: np.ndarray) -> float:
    """Log-spectral distance between two spectral envelopes."""
    la = np.log(env_a + 1e-12)
    lb = np.log(env_b + 1e-12)
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def harmonic_brightness(audio: np.ndarray, sr: int, f0: float,
                        n_harmonics: int = 10) -> np.ndarray:
    """Energy distribution across harmonics as a brightness descriptor."""
    N = len(audio)
    X = np.abs(np.fft.rfft(audio * np.hanning(N)))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    energies = np.zeros(n_harmonics)
    for h in range(n_harmonics):
        f_h = f0 * (h + 1)
        bw = f0 * 0.25
        mask = (freqs >= f_h - bw) & (freqs <= f_h + bw)
        energies[h] = np.sum(X[mask] ** 2)
    total = np.sum(energies) + 1e-12
    return energies / total
