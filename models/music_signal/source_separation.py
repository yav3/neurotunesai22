"""Source separation for music signals.

NMF-based decomposition, harmonic-percussive source separation (HPSS),
Wiener filtering, ideal/soft masks, and component extraction utilities.
All pure numpy — no external ML frameworks.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class SeparationResult:
    """Result of a source separation algorithm."""
    sources: List[np.ndarray]
    masks: Optional[List[np.ndarray]] = None
    cost_history: Optional[List[float]] = None


def _stft(signal: np.ndarray, frame_size: int = 2048,
           hop: int = 512) -> np.ndarray:
    """Short-time Fourier transform."""
    window = np.hanning(frame_size)
    n = len(signal)
    n_frames = (n - frame_size) // hop + 1
    out = np.zeros((frame_size // 2 + 1, n_frames), dtype=complex)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_size] * window
        out[:, i] = np.fft.rfft(frame)
    return out


def _istft(stft_matrix: np.ndarray, frame_size: int = 2048,
            hop: int = 512, length: Optional[int] = None) -> np.ndarray:
    """Inverse STFT with overlap-add."""
    window = np.hanning(frame_size)
    n_frames = stft_matrix.shape[1]
    n = frame_size + (n_frames - 1) * hop
    if length is not None:
        n = length
    output = np.zeros(n)
    norm = np.zeros(n)
    for i in range(n_frames):
        start = i * hop
        frame = np.fft.irfft(stft_matrix[:, i], n=frame_size) * window
        end = min(start + frame_size, n)
        output[start:end] += frame[:end - start]
        norm[start:end] += window[:end - start] ** 2
    norm[norm < 1e-10] = 1.0
    return output / norm


def nmf(V: np.ndarray, n_components: int = 2,
        n_iter: int = 200, beta: float = 2.0,
        eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Non-negative Matrix Factorisation: V ≈ W·H.

    Multiplicative update rules for β-divergence:
      β=2: Euclidean (Frobenius) distance
      β=1: generalised KL divergence
      β=0: Itakura-Saito divergence

    Args:
        V: non-negative (F × T) matrix (e.g. magnitude spectrogram)
        n_components: number of basis vectors (sources)
        n_iter: number of iterations
        beta: β-divergence parameter
        eps: small constant for numerical stability

    Returns:
        W: (F × K) basis matrix
        H: (K × T) activation matrix
        cost_history: list of divergence values per iteration
    """
    F, T = V.shape
    rng = np.random.RandomState(42)
    W = np.abs(rng.randn(F, n_components)) + eps
    H = np.abs(rng.randn(n_components, T)) + eps
    V = np.maximum(V, eps)
    cost_history = []

    for _ in range(n_iter):
        WH = W @ H + eps
        if beta == 2.0:
            # Euclidean
            H *= (W.T @ V) / (W.T @ WH + eps)
            WH = W @ H + eps
            W *= (V @ H.T) / (WH @ H.T + eps)
            cost = 0.5 * np.sum((V - W @ H) ** 2)
        elif beta == 1.0:
            # KL divergence
            H *= (W.T @ (V / WH)) / (np.sum(W, axis=0, keepdims=True).T + eps)
            WH = W @ H + eps
            W *= ((V / WH) @ H.T) / (np.sum(H, axis=1, keepdims=True).T + eps)
            cost = float(np.sum(V * np.log(V / (W @ H + eps) + eps) - V + W @ H))
        else:
            # General β-divergence
            WH_prev = WH
            H *= ((W.T @ (WH_prev ** (beta - 2) * V)) /
                  (W.T @ WH_prev ** (beta - 1) + eps))
            WH = W @ H + eps
            W *= (((WH ** (beta - 2) * V) @ H.T) /
                  (WH ** (beta - 1) @ H.T + eps))
            cost = float(np.sum((V ** beta - (W @ H) ** beta) / (beta * (beta - 1) + eps)))

        cost_history.append(cost)

    return W, H, cost_history


def nmf_separate(signal: np.ndarray, n_sources: int = 2,
                 n_iter: int = 200, frame_size: int = 2048,
                 hop: int = 512) -> SeparationResult:
    """Separate audio into n_sources components using NMF.

    Decomposes the magnitude spectrogram, then applies Wiener-like
    soft masks to reconstruct each source.
    """
    S = _stft(signal, frame_size, hop)
    V = np.abs(S)
    phase = np.angle(S)

    W, H, costs = nmf(V, n_components=n_sources, n_iter=n_iter)
    WH = W @ H + 1e-10

    sources = []
    masks = []
    for k in range(n_sources):
        mask = np.outer(W[:, k], H[k, :]) / WH
        masks.append(mask)
        source_stft = mask * V * np.exp(1j * phase)
        source = _istft(source_stft, frame_size, hop, length=len(signal))
        sources.append(source)

    return SeparationResult(sources=sources, masks=masks, cost_history=costs)


def hpss(signal: np.ndarray, frame_size: int = 2048,
         hop: int = 512, kernel_harmonic: int = 31,
         kernel_percussive: int = 31,
         power: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Harmonic-Percussive Source Separation (Fitzgerald 2010).

    Separates a signal into harmonic (horizontal continuity) and
    percussive (vertical continuity) components via median filtering
    of the spectrogram.

    Args:
        signal: input audio
        frame_size: STFT frame size
        hop: STFT hop size
        kernel_harmonic: median filter length along time axis
        kernel_percussive: median filter length along frequency axis
        power: exponent for the spectrogram (2.0 = power, 1.0 = magnitude)

    Returns:
        (harmonic, percussive) separated signals
    """
    S = _stft(signal, frame_size, hop)
    mag = np.abs(S) ** power
    phase = np.angle(S)

    # Median filter along time (horizontal → harmonic)
    pad_h = kernel_harmonic // 2
    padded_t = np.pad(mag, ((0, 0), (pad_h, pad_h)), mode='reflect')
    H_enhanced = np.zeros_like(mag)
    for f in range(mag.shape[0]):
        for t in range(mag.shape[1]):
            H_enhanced[f, t] = np.median(padded_t[f, t:t + kernel_harmonic])

    # Median filter along frequency (vertical → percussive)
    pad_p = kernel_percussive // 2
    padded_f = np.pad(mag, ((pad_p, pad_p), (0, 0)), mode='reflect')
    P_enhanced = np.zeros_like(mag)
    for f in range(mag.shape[0]):
        for t in range(mag.shape[1]):
            P_enhanced[f, t] = np.median(padded_f[f:f + kernel_percussive, t])

    # Soft masks
    total = H_enhanced + P_enhanced + 1e-10
    mask_h = H_enhanced / total
    mask_p = P_enhanced / total

    mag_orig = np.abs(S)
    harmonic_stft = mask_h * mag_orig * np.exp(1j * phase)
    percussive_stft = mask_p * mag_orig * np.exp(1j * phase)

    harmonic = _istft(harmonic_stft, frame_size, hop, length=len(signal))
    percussive = _istft(percussive_stft, frame_size, hop, length=len(signal))

    return harmonic, percussive


def wiener_filter(mixed: np.ndarray, noise_estimate: np.ndarray,
                  frame_size: int = 2048, hop: int = 512,
                  alpha: float = 1.0, floor: float = 0.01) -> np.ndarray:
    """Wiener filter for noise reduction.

    G(f) = max(1 - α·|N(f)|²/|X(f)|², floor)

    Args:
        mixed: noisy signal
        noise_estimate: noise-only signal (or noise profile)
        frame_size: STFT frame size
        hop: STFT hop size
        alpha: oversubtraction factor (>1 = more aggressive)
        floor: spectral floor to prevent musical noise
    """
    S_mixed = _stft(mixed, frame_size, hop)
    S_noise = _stft(noise_estimate, frame_size, hop)

    P_mixed = np.abs(S_mixed) ** 2 + 1e-20
    P_noise = np.mean(np.abs(S_noise) ** 2, axis=1, keepdims=True) + 1e-20

    gain = np.maximum(1.0 - alpha * P_noise / P_mixed, floor)
    filtered = gain * S_mixed

    return _istft(filtered, frame_size, hop, length=len(mixed))


def spectral_subtraction(mixed: np.ndarray, noise_estimate: np.ndarray,
                         frame_size: int = 2048, hop: int = 512,
                         alpha: float = 1.0, beta: float = 0.01) -> np.ndarray:
    """Power spectral subtraction for noise reduction.

    |S(f)|² = max(|X(f)|² - α·|N(f)|², β·|X(f)|²)
    """
    S_mixed = _stft(mixed, frame_size, hop)
    S_noise = _stft(noise_estimate, frame_size, hop)
    phase = np.angle(S_mixed)

    P_mixed = np.abs(S_mixed) ** 2
    P_noise = np.mean(np.abs(S_noise) ** 2, axis=1, keepdims=True)

    P_clean = np.maximum(P_mixed - alpha * P_noise, beta * P_mixed)
    mag_clean = np.sqrt(P_clean)
    clean_stft = mag_clean * np.exp(1j * phase)

    return _istft(clean_stft, frame_size, hop, length=len(mixed))


def ideal_binary_mask(target: np.ndarray, interference: np.ndarray,
                      frame_size: int = 2048, hop: int = 512,
                      theta_db: float = 0.0) -> np.ndarray:
    """Ideal Binary Mask — oracle separation mask.

    IBM(f,t) = 1 if SNR(f,t) > θ, else 0.
    Requires access to target and interference (oracle scenario).
    Useful as an upper-bound reference for separation quality.
    """
    S_target = _stft(target, frame_size, hop)
    S_interf = _stft(interference, frame_size, hop)

    P_target = np.abs(S_target) ** 2 + 1e-30
    P_interf = np.abs(S_interf) ** 2 + 1e-30
    local_snr = 10.0 * np.log10(P_target / P_interf)

    mask = (local_snr > theta_db).astype(float)
    mixed = target + interference
    S_mixed = _stft(mixed, frame_size, hop)
    separated_stft = mask * S_mixed

    return _istft(separated_stft, frame_size, hop, length=len(target))


def ideal_ratio_mask(target: np.ndarray, mixture: np.ndarray,
                     frame_size: int = 2048, hop: int = 512) -> np.ndarray:
    """Ideal Ratio Mask — soft oracle mask.

    IRM(f,t) = |S_target(f,t)| / |S_mix(f,t)|
    """
    S_target = _stft(target, frame_size, hop)
    S_mix = _stft(mixture, frame_size, hop)

    mask = np.abs(S_target) / (np.abs(S_mix) + 1e-10)
    mask = np.minimum(mask, 1.0)
    separated_stft = mask * S_mix

    return _istft(separated_stft, frame_size, hop, length=len(target))


def sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Signal-to-Distortion Ratio (BSS Eval).

    SDR = 10·log10(||s_target||² / ||e_total||²)
    where e_total = estimated - reference.
    """
    e = estimated[:len(reference)] - reference[:len(estimated)]
    p_ref = np.sum(reference ** 2)
    p_err = np.sum(e ** 2)
    if p_err < 1e-30:
        return 120.0
    return 10.0 * np.log10(p_ref / p_err)


def sir(reference: np.ndarray, estimated: np.ndarray,
        interference: np.ndarray) -> float:
    """Signal-to-Interference Ratio.

    Measures how well the target is separated from interfering sources.
    """
    s_proj = np.dot(estimated, reference) / (np.dot(reference, reference) + 1e-30) * reference
    i_proj = np.dot(estimated, interference) / (np.dot(interference, interference) + 1e-30) * interference
    p_s = np.sum(s_proj ** 2)
    p_i = np.sum(i_proj ** 2)
    if p_i < 1e-30:
        return 120.0
    return 10.0 * np.log10(p_s / p_i)


def sar(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Signal-to-Artifacts Ratio.

    Measures artifacts introduced by the separation algorithm.
    SAR = SDR when there is no interference component.
    """
    s_proj = np.dot(estimated, reference) / (np.dot(reference, reference) + 1e-30) * reference
    artifacts = estimated[:len(reference)] - s_proj[:len(estimated)]
    p_s = np.sum(s_proj ** 2)
    p_a = np.sum(artifacts ** 2)
    if p_a < 1e-30:
        return 120.0
    return 10.0 * np.log10(p_s / p_a)


def harmonic_mask(signal: np.ndarray, f0_track: np.ndarray,
                  sr: float, n_harmonics: int = 10,
                  frame_size: int = 2048, hop: int = 512,
                  bandwidth_hz: float = 50.0) -> np.ndarray:
    """Extract harmonics of a tracked f0 via a comb-like spectral mask.

    Creates a soft mask that passes energy near integer multiples
    of the fundamental frequency at each frame.
    """
    S = _stft(signal, frame_size, hop)
    freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
    n_frames = min(S.shape[1], len(f0_track))
    mask = np.zeros_like(np.abs(S))

    for t in range(n_frames):
        f0 = f0_track[t]
        if f0 < 20.0:
            continue
        for h in range(1, n_harmonics + 1):
            center = f0 * h
            if center > sr / 2:
                break
            width = bandwidth_hz * h * 0.5
            mask[:, t] += np.exp(-0.5 * ((freqs - center) / max(width, 1.0)) ** 2)

    mask = np.minimum(mask, 1.0)
    phase = np.angle(S)
    harmonic_stft = mask * np.abs(S) * np.exp(1j * phase)
    return _istft(harmonic_stft, frame_size, hop, length=len(signal))
