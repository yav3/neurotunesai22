"""Deep signal processing for music: wavelet analysis, gammatone filterbanks,
reassignment spectrograms, multiresolution analysis, and psychoacoustic models.

References:
  - Continuous wavelet transform: Grossmann & Morlet (1984)
  - Gammatone filterbank: Patterson et al. (1992), Equivalent Rectangular Bandwidth
  - Reassignment method: Auger & Flandrin (1995)
  - Equivalent rectangular bandwidth: Glasberg & Moore (1990)
  - Bark scale: Zwicker (1961)
  - Auditory roughness: Vassilakis (2005)
  - Cochlear model: Lyon (1982) cascade of asymmetric resonators

Pure numpy/scipy. No external ML frameworks.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# §1  Auditory frequency scales
# ---------------------------------------------------------------------------

def hz_to_erb(f: np.ndarray) -> np.ndarray:
    """Hz → ERB-rate (Glasberg & Moore, 1990): E = 21.4 log10(0.00437f + 1)."""
    f = np.asarray(f, dtype=np.float64)
    return 21.4 * np.log10(0.00437 * f + 1.0)

def erb_to_hz(e: np.ndarray) -> np.ndarray:
    """ERB-rate → Hz."""
    e = np.asarray(e, dtype=np.float64)
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437

def erb_bandwidth(fc: np.ndarray) -> np.ndarray:
    """Equivalent rectangular bandwidth at centre frequency fc (Hz)."""
    fc = np.asarray(fc, dtype=np.float64)
    return 24.7 * (0.00437 * fc + 1.0)

def hz_to_bark(f: np.ndarray) -> np.ndarray:
    """Hz → Bark (Traunmüller, 1990)."""
    f = np.asarray(f, dtype=np.float64)
    return (26.81 * f / (1960.0 + f)) - 0.53

def bark_to_hz(z: np.ndarray) -> np.ndarray:
    """Bark → Hz."""
    z = np.asarray(z, dtype=np.float64)
    return 1960.0 * (z + 0.53) / (26.28 - z)

def hz_to_cam(f: np.ndarray) -> np.ndarray:
    """Hz → Cam (ERB-rate number, Moore & Glasberg 1983)."""
    return hz_to_erb(f)


# ---------------------------------------------------------------------------
# §2  Gammatone filterbank (Patterson et al., 1992)
# ---------------------------------------------------------------------------

def gammatone_impulse_response(
    fc: float, sr: int = 22050, duration: float = 0.05, order: int = 4
) -> np.ndarray:
    """Gammatone impulse response: g(t) = t^{n-1} e^{-2π b t} cos(2π fc t).

    Args:
        fc: centre frequency (Hz)
        sr: sample rate
        duration: IR length (seconds)
        order: filter order (typically 4)
    Returns:
        1-D impulse response array
    """
    t = np.arange(int(sr * duration)) / sr
    b = 2.0 * np.pi * 1.019 * erb_bandwidth(fc)
    ir = t ** (order - 1) * np.exp(-b * t) * np.cos(2.0 * np.pi * fc * t)
    ir /= np.max(np.abs(ir)) + 1e-12
    return ir

def gammatone_filterbank(
    n_filters: int = 64,
    f_low: float = 50.0,
    f_high: float = 8000.0,
    sr: int = 22050,
    duration: float = 0.05,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a gammatone filterbank.

    Returns:
        (filters, centre_freqs) — filters shape (n_filters, ir_length)
    """
    erb_low, erb_high = hz_to_erb(f_low), hz_to_erb(f_high)
    erb_points = np.linspace(erb_low, erb_high, n_filters)
    cfs = erb_to_hz(erb_points)
    filters = np.array([gammatone_impulse_response(cf, sr, duration, order) for cf in cfs])
    return filters, cfs

def apply_gammatone(
    signal: np.ndarray, filters: np.ndarray
) -> np.ndarray:
    """Apply gammatone filterbank → (n_filters, n_samples) output."""
    from scipy.signal import fftconvolve
    out = np.zeros((filters.shape[0], len(signal)))
    for i, filt in enumerate(filters):
        conv = fftconvolve(signal, filt, mode='full')[:len(signal)]
        out[i] = conv
    return out

def gammatone_spectrogram(
    signal: np.ndarray,
    sr: int = 22050,
    n_filters: int = 64,
    hop_length: int = 256,
    f_low: float = 50.0,
    f_high: float = 8000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gammatone spectrogram: auditory-model time-frequency representation.

    Returns:
        (spectrogram, centre_freqs) — spectrogram shape (n_filters, n_frames)
    """
    filters, cfs = gammatone_filterbank(n_filters, f_low, f_high, sr)
    filtered = apply_gammatone(signal, filters)
    env = np.abs(filtered)
    n_frames = (len(signal) - 1) // hop_length + 1
    spec = np.zeros((n_filters, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + hop_length, len(signal))
        spec[:, i] = np.mean(env[:, start:end], axis=1)
    return spec, cfs


# ---------------------------------------------------------------------------
# §3  Continuous wavelet transform (Morlet)
# ---------------------------------------------------------------------------

def morlet_wavelet(
    omega0: float, s: float, t: np.ndarray
) -> np.ndarray:
    """Morlet wavelet: ψ(t) = π^{-1/4} e^{iω₀t/s} e^{-t²/(2s²)}."""
    norm = np.pi ** (-0.25) / np.sqrt(s)
    return norm * np.exp(1j * omega0 * t / s) * np.exp(-t ** 2 / (2.0 * s ** 2))

def cwt_morlet(
    signal: np.ndarray,
    scales: np.ndarray,
    omega0: float = 6.0,
    sr: int = 22050,
) -> np.ndarray:
    """Continuous wavelet transform with Morlet wavelet.

    Args:
        signal: 1-D input
        scales: array of scale parameters
        omega0: centre frequency of mother wavelet
    Returns:
        Complex CWT matrix (n_scales, n_samples)
    """
    n = len(signal)
    angular_freqs = np.fft.fftfreq(n, d=1.0 / sr) * 2.0 * np.pi
    sig_fft = np.fft.fft(signal)
    cwt = np.zeros((len(scales), n), dtype=np.complex128)
    for i, s in enumerate(scales):
        norm_freqs = s * angular_freqs / (2.0 * np.pi * sr)
        psi_fft = np.sqrt(2.0 * np.pi * s) * np.pi ** (-0.25) * \
                  np.exp(-0.5 * (norm_freqs * 2.0 * np.pi - omega0) ** 2)
        cwt[i] = np.fft.ifft(sig_fft * np.conj(psi_fft))
    return cwt

def cwt_scales_from_frequencies(
    freqs: np.ndarray, omega0: float = 6.0, sr: int = 22050
) -> np.ndarray:
    """Convert target frequencies (Hz) to CWT scale parameters."""
    return omega0 * sr / (2.0 * np.pi * np.asarray(freqs, dtype=np.float64))

def cwt_scalogram(
    signal: np.ndarray,
    f_low: float = 30.0,
    f_high: float = 4000.0,
    n_freqs: int = 128,
    sr: int = 22050,
) -> Tuple[np.ndarray, np.ndarray]:
    """Log-spaced CWT scalogram.

    Returns:
        (power, freqs) — power shape (n_freqs, n_samples)
    """
    freqs = np.geomspace(f_low, f_high, n_freqs)
    scales = cwt_scales_from_frequencies(freqs, sr=sr)
    W = cwt_morlet(signal, scales, sr=sr)
    return np.abs(W) ** 2, freqs


# ---------------------------------------------------------------------------
# §4  Multiresolution analysis
# ---------------------------------------------------------------------------

def haar_wavelet_decompose(
    signal: np.ndarray, levels: int = 4
) -> List[np.ndarray]:
    """Haar wavelet decomposition into detail + approximation coefficients.

    Returns list [d1, d2, ..., d_levels, a_levels] from finest to coarsest.
    """
    coeffs = []
    a = signal.copy().astype(np.float64)
    for _ in range(levels):
        n = len(a)
        if n % 2 != 0:
            a = np.append(a, a[-1])
            n += 1
        half = n // 2
        approx = (a[0::2] + a[1::2]) / np.sqrt(2)
        detail = (a[0::2] - a[1::2]) / np.sqrt(2)
        coeffs.append(detail)
        a = approx
    coeffs.append(a)
    return coeffs

def haar_wavelet_reconstruct(coeffs: List[np.ndarray]) -> np.ndarray:
    """Reconstruct signal from Haar wavelet coefficients."""
    a = coeffs[-1].copy()
    for detail in reversed(coeffs[:-1]):
        n = len(detail)
        recon = np.zeros(2 * n)
        recon[0::2] = (a[:n] + detail) / np.sqrt(2)
        recon[1::2] = (a[:n] - detail) / np.sqrt(2)
        a = recon
    return a

def wavelet_energy_distribution(coeffs: List[np.ndarray]) -> np.ndarray:
    """Energy fraction in each wavelet subband."""
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total = np.sum(energies)
    return energies / (total + 1e-12)

def wavelet_entropy(coeffs: List[np.ndarray]) -> float:
    """Wavelet entropy: Shannon entropy of the energy distribution."""
    p = wavelet_energy_distribution(coeffs)
    p = p[p > 0]
    return -float(np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# §5  Spectral reassignment (Auger & Flandrin, 1995)
# ---------------------------------------------------------------------------

def reassigned_spectrogram(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    sr: int = 22050,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reassigned spectrogram via time and frequency correction.

    Returns:
        (S_reassigned, freqs, times) — S_reassigned shape (n_fft//2+1, n_frames)
    """
    n = len(signal)
    win = np.hanning(n_fft)
    twin = win * np.arange(n_fft)
    dwin = np.zeros(n_fft)
    dwin[1:-1] = (win[2:] - win[:-2]) / 2.0

    n_frames = (n - n_fft) // hop_length + 1
    n_bins = n_fft // 2 + 1
    S = np.zeros((n_bins, n_frames))

    for t_idx in range(n_frames):
        start = t_idx * hop_length
        frame = signal[start:start + n_fft]
        X = np.fft.rfft(frame * win)
        Xt = np.fft.rfft(frame * twin)
        Xd = np.fft.rfft(frame * dwin)

        mag2 = np.abs(X) ** 2 + 1e-20
        t_corr = np.real(Xt * np.conj(X)) / mag2
        f_corr = -np.imag(Xd * np.conj(X)) / (mag2 * 2.0 * np.pi) * sr

        for k in range(n_bins):
            t_new = start + t_corr[k]
            f_new = f_corr[k] + k * sr / n_fft
            t_bin = int(round(t_new / hop_length))
            f_bin = int(round(f_new * n_fft / sr))
            if 0 <= t_bin < n_frames and 0 <= f_bin < n_bins:
                S[f_bin, t_bin] += mag2[k]

    freqs = np.arange(n_bins) * sr / n_fft
    times = np.arange(n_frames) * hop_length / sr
    return S, freqs, times


# ---------------------------------------------------------------------------
# §6  Psychoacoustic loudness model (Zwicker, 1961)
# ---------------------------------------------------------------------------

def specific_loudness(
    spectrum_db: np.ndarray, bark_bands: Optional[np.ndarray] = None
) -> np.ndarray:
    """Approximate specific loudness in sone/Bark.

    Simple power-law model: N'(z) ≈ max(0, (L(z) - LTQ(z)) / 10)^0.23
    where LTQ is the threshold in quiet.
    """
    n_bands = len(spectrum_db)
    if bark_bands is None:
        bark_bands = np.linspace(0.5, 24.0, n_bands)
    ltq = 3.64 * (bark_bands / 24.0) ** (-0.8) - 6.5 * np.exp(
        -0.6 * (bark_bands / 24.0 - 3.3) ** 2
    ) + 1e-3 * (bark_bands / 24.0) ** 4
    ltq = np.clip(ltq, -10, 80)
    exceedance = np.maximum(0, spectrum_db - ltq)
    return (exceedance / 10.0) ** 0.23

def total_loudness(spectrum_db: np.ndarray) -> float:
    """Total loudness in sone (integral of specific loudness)."""
    N_prime = specific_loudness(spectrum_db)
    dz = 24.0 / len(N_prime)
    return float(np.sum(N_prime) * dz)

def loudness_contour(
    signal: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bark: int = 24,
) -> np.ndarray:
    """Frame-wise total loudness contour."""
    n_frames = (len(signal) - n_fft) // hop_length + 1
    contour = np.zeros(n_frames)
    win = np.hanning(n_fft)
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + n_fft]
        S = np.abs(np.fft.rfft(frame * win)) ** 2
        n_bins = len(S)
        bark_spec = np.zeros(n_bark)
        for b in range(n_bark):
            lo = int(b * n_bins / n_bark)
            hi = int((b + 1) * n_bins / n_bark)
            bark_spec[b] = 10.0 * np.log10(np.mean(S[lo:hi]) + 1e-20)
        contour[i] = total_loudness(bark_spec)
    return contour


# ---------------------------------------------------------------------------
# §7  Cochlear model — cascade of asymmetric resonators (Lyon, 1982)
# ---------------------------------------------------------------------------

def lyon_cochlear_model(
    signal: np.ndarray,
    sr: int = 22050,
    n_channels: int = 64,
    f_low: float = 80.0,
    f_high: float = 5000.0,
    q_factor: float = 8.0,
) -> np.ndarray:
    """Simplified Lyon cochlear cascade model.

    Models the basilar membrane as a cascade of second-order resonators
    with decreasing centre frequency. Output: (n_channels, n_samples).
    """
    erb_lo, erb_hi = hz_to_erb(f_low), hz_to_erb(f_high)
    cfs = erb_to_hz(np.linspace(erb_hi, erb_lo, n_channels))

    out = np.zeros((n_channels, len(signal)))
    x = signal.copy().astype(np.float64)

    for ch in range(n_channels):
        omega = 2.0 * np.pi * cfs[ch] / sr
        r = np.exp(-omega / q_factor)
        a1 = -2.0 * r * np.cos(omega)
        a2 = r * r
        b0 = 1.0 - r

        y = np.zeros_like(x)
        y1, y2 = 0.0, 0.0
        for n in range(len(x)):
            y[n] = b0 * x[n] - a1 * y1 - a2 * y2
            y2, y1 = y1, y[n]
        out[ch] = y
        x = y

    return out


# ---------------------------------------------------------------------------
# §8  Spectral flux variants & novelty functions
# ---------------------------------------------------------------------------

def spectral_flux_log(
    signal: np.ndarray, n_fft: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Log-domain spectral flux (Bello et al., 2005)."""
    win = np.hanning(n_fft)
    n_frames = (len(signal) - n_fft) // hop_length + 1
    flux = np.zeros(max(n_frames - 1, 0))
    prev = None
    for i in range(n_frames):
        start = i * hop_length
        S = np.log1p(np.abs(np.fft.rfft(signal[start:start + n_fft] * win)))
        if prev is not None:
            diff = S - prev
            flux[i - 1] = float(np.sum(np.maximum(0, diff)))
        prev = S
    return flux

def spectral_novelty(
    signal: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    kernel_size: int = 64,
) -> np.ndarray:
    """Novelty function via self-similarity matrix checkerboard kernel (Foote, 2000)."""
    win = np.hanning(n_fft)
    n_frames = (len(signal) - n_fft) // hop_length + 1
    features = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        features[:, i] = np.abs(np.fft.rfft(signal[start:start + n_fft] * win))

    features /= np.max(features) + 1e-12
    S = features.T @ features

    half = kernel_size // 2
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    novelty = np.zeros(n_frames)
    for i in range(half, n_frames - half):
        patch = S[i - half:i + half, i - half:i + half]
        if patch.shape == kernel.shape:
            novelty[i] = float(np.sum(patch * kernel))
    return np.maximum(0, novelty)


# ---------------------------------------------------------------------------
# §9  Chroma variants
# ---------------------------------------------------------------------------

def harmonic_chroma(
    signal: np.ndarray,
    sr: int = 22050,
    n_fft: int = 4096,
    hop_length: int = 512,
    n_harmonics: int = 6,
    f_ref: float = 440.0,
) -> np.ndarray:
    """Harmonic-enhanced chromagram: sums energy across n_harmonics for each pitch class.

    Returns shape (12, n_frames).
    """
    win = np.hanning(n_fft)
    n_frames = (len(signal) - n_fft) // hop_length + 1
    chroma = np.zeros((12, max(n_frames, 0)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    for i in range(n_frames):
        start = i * hop_length
        S = np.abs(np.fft.rfft(signal[start:start + n_fft] * win)) ** 2
        for pc in range(12):
            total = 0.0
            for h in range(1, n_harmonics + 1):
                f0 = f_ref * 2.0 ** ((pc - 9) / 12.0)
                for octave in range(-1, 8):
                    f = f0 * 2.0 ** octave * h
                    if f < freqs[1] or f > freqs[-1]:
                        continue
                    idx = int(round(f * n_fft / sr))
                    if 0 <= idx < len(S):
                        total += S[idx] / h
            chroma[pc, i] = total
    norm = np.max(chroma, axis=0, keepdims=True) + 1e-12
    return chroma / norm

def tonnetz_transform(chroma: np.ndarray) -> np.ndarray:
    """Chroma → 6-D Tonnetz coordinates (Harte et al., 2006).

    Maps 12-D chroma to fifths (y1,y2), minor thirds (y3,y4), major thirds (y5,y6).
    Returns shape (6, n_frames).
    """
    n_frames = chroma.shape[1]
    tonnetz = np.zeros((6, n_frames))
    for t in range(n_frames):
        c = chroma[:, t]
        c = c / (np.sum(c) + 1e-12)
        angles_fifth = np.arange(12) * 7 * 2 * np.pi / 12
        angles_minor = np.arange(12) * 3 * 2 * np.pi / 12
        angles_major = np.arange(12) * 4 * 2 * np.pi / 12
        tonnetz[0, t] = np.sum(c * np.sin(angles_fifth))
        tonnetz[1, t] = np.sum(c * np.cos(angles_fifth))
        tonnetz[2, t] = np.sum(c * np.sin(angles_minor))
        tonnetz[3, t] = np.sum(c * np.cos(angles_minor))
        tonnetz[4, t] = np.sum(c * np.sin(angles_major))
        tonnetz[5, t] = np.sum(c * np.cos(angles_major))
    return tonnetz


# ---------------------------------------------------------------------------
# §10  Aggregate deep signal analysis
# ---------------------------------------------------------------------------

@dataclass
class DeepSignalAnalysis:
    gammatone_spec: Optional[np.ndarray] = None
    gammatone_cfs: Optional[np.ndarray] = None
    cwt_power: Optional[np.ndarray] = None
    cwt_freqs: Optional[np.ndarray] = None
    wavelet_coeffs: Optional[List[np.ndarray]] = None
    wavelet_energy: Optional[np.ndarray] = None
    wavelet_entropy: float = 0.0
    loudness_contour: Optional[np.ndarray] = None
    mean_loudness: float = 0.0
    cochlear_output: Optional[np.ndarray] = None
    log_flux: Optional[np.ndarray] = None
    novelty: Optional[np.ndarray] = None
    chroma: Optional[np.ndarray] = None
    tonnetz: Optional[np.ndarray] = None

def analyze_signal_deep(
    signal: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> DeepSignalAnalysis:
    """Run the full deep signal analysis pipeline."""
    result = DeepSignalAnalysis()

    gt_spec, gt_cfs = gammatone_spectrogram(signal, sr, n_filters=64, hop_length=hop_length)
    result.gammatone_spec = gt_spec
    result.gammatone_cfs = gt_cfs

    cwt_pow, cwt_f = cwt_scalogram(signal, n_freqs=64, sr=sr)
    result.cwt_power = cwt_pow
    result.cwt_freqs = cwt_f

    coeffs = haar_wavelet_decompose(signal, levels=5)
    result.wavelet_coeffs = coeffs
    result.wavelet_energy = wavelet_energy_distribution(coeffs)
    result.wavelet_entropy = wavelet_entropy(coeffs)

    result.loudness_contour = loudness_contour(signal, sr, n_fft, hop_length)
    result.mean_loudness = float(np.mean(result.loudness_contour)) if len(result.loudness_contour) > 0 else 0.0

    result.log_flux = spectral_flux_log(signal, n_fft, hop_length)
    result.novelty = spectral_novelty(signal, n_fft, hop_length)

    result.chroma = harmonic_chroma(signal, sr, n_fft=4096, hop_length=hop_length)
    result.tonnetz = tonnetz_transform(result.chroma)

    return result
