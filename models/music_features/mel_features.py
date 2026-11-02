"""Mel-frequency and chroma feature extraction for music signals.

Implements the standard feature extraction pipeline used in music information
retrieval: mel spectrograms, MFCCs (Davis & Mermelstein 1980), chroma features
(Fujishima 1999; Müller 2015), tonnetz representations (Harte et al. 2006),
and auxiliary spectral descriptors.  All functions operate on full signals or
frame-level spectrograms (batch form), complementing the single-frame helpers
in ``spectral_descriptors``.

References
----------
Davis, S. & Mermelstein, P. (1980). Comparison of parametric representations
    for monosyllabic word recognition in continuously spoken sentences.
    *IEEE Trans. ASSP*, 28(4), 357–366.
Furui, S. (1986). Speaker-independent isolated word recognition based on
    emphasized spectral dynamics. *ICASSP*, 1991–1994.
Harte, C., Sandler, M. & Gasser, M. (2006). Detecting harmonic change in
    musical audio. *Proc. 1st ACM Workshop on Audio and Music Computing
    Multimedia*, 21–26.
Jiang, D.-N., Lu, L., Zhang, H.-J., Tao, J.-H. & Cai, L.-H. (2002).
    Music type classification by spectral contrast feature. *IEEE ICME*.
Müller, M. (2015). *Fundamentals of Music Processing*. Springer.
Müller, M. & Ewert, S. (2011). Chroma Toolbox: MATLAB implementations for
    extracting variants of chroma-based audio features. *ISMIR*.
Slaney, M. (1998). Auditory toolbox. *Tech. Report #1998-010*, Interval
    Research Corporation.
Stevens, S. S., Volkmann, J. & Newman, E. B. (1937). A scale for the
    measurement of the psychological magnitude pitch. *JASA*, 8(3), 185–190.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Mel scale conversions
# ---------------------------------------------------------------------------

def hz_to_mel(hz: float | np.ndarray, htk: bool = False) -> float | np.ndarray:
    """Convert Hz to mel.  Slaney (1998) by default; HTK formula if *htk*."""
    hz = np.asarray(hz, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    # Slaney piecewise linear / log
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0.0) / f_sp
    logstep = np.log(6.4) / 27.0
    mel = np.where(
        hz < min_log_hz,
        hz / f_sp,
        min_log_mel + np.log(hz / min_log_hz) / logstep,
    )
    return float(mel) if mel.ndim == 0 else mel


def mel_to_hz(mel: float | np.ndarray, htk: bool = False) -> float | np.ndarray:
    """Convert mel to Hz."""
    mel = np.asarray(mel, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz = np.where(
        mel < min_log_mel,
        mel * f_sp,
        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
    )
    return float(hz) if hz.ndim == 0 else hz


# ---------------------------------------------------------------------------
# Filterbanks
# ---------------------------------------------------------------------------

def mel_filterbank(
    n_filters: int,
    n_fft: int,
    sr: float,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
) -> np.ndarray:
    """Triangular mel filterbank matrix of shape ``(n_filters, n_fft // 2 + 1)``.

    Each row is a triangular filter whose vertices are defined by consecutive
    mel-spaced centre frequencies (Stevens et al. 1937).
    """
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1
    mel_lo = hz_to_mel(fmin, htk=htk)
    mel_hi = hz_to_mel(fmax, htk=htk)
    mel_points = np.linspace(mel_lo, mel_hi, n_filters + 2)
    hz_points = mel_to_hz(mel_points, htk=htk)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_filters, n_freqs))
    for m in range(n_filters):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]
        for k in range(f_left, f_center):
            if f_center != f_left:
                fbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                fbank[m, k] = (f_right - k) / (f_right - f_center)
    return fbank


def chroma_filterbank(
    n_chroma: int,
    n_fft: int,
    sr: float,
    tuning: float = 0.0,
) -> np.ndarray:
    """Chroma filterbank matrix ``(n_chroma, n_fft // 2 + 1)``.

    Maps DFT bins to pitch classes using a log-frequency mapping with
    optional *tuning* offset in semitones.
    """
    n_freqs = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    freqs[0] = 1.0  # avoid log(0)
    # MIDI-like pitch numbering; A4 = 440 Hz = MIDI 69
    midi = 12.0 * np.log2(freqs / 440.0) + 69.0 + tuning
    chroma_map = np.round(midi) % n_chroma
    fbank = np.zeros((n_chroma, n_freqs))
    for c in range(n_chroma):
        fbank[c] = (chroma_map == c).astype(float)
    fbank[:, 0] = 0.0  # DC bin carries no pitch
    return fbank


# ---------------------------------------------------------------------------
# STFT helpers
# ---------------------------------------------------------------------------

def _stft(
    signal: np.ndarray, n_fft: int, hop: int, window: str = "hann",
) -> np.ndarray:
    """Short-time Fourier transform → complex array ``(n_freqs, n_frames)``."""
    from scipy.signal import get_window
    win = get_window(window, n_fft, fftbins=True)
    n_freqs = n_fft // 2 + 1
    pad_len = n_fft // 2
    sig = np.pad(signal, (pad_len, pad_len), mode="reflect")
    n_frames = 1 + (len(sig) - n_fft) // hop
    stft = np.empty((n_freqs, n_frames), dtype=np.complex128)
    for t in range(n_frames):
        frame = sig[t * hop : t * hop + n_fft] * win
        stft[:, t] = np.fft.rfft(frame, n=n_fft)
    return stft


def _power_spectrogram(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Power spectrogram ``(n_freqs, n_frames)``."""
    S = _stft(signal, n_fft, hop)
    return np.abs(S) ** 2


# ---------------------------------------------------------------------------
# Mel spectrogram / log-mel / MFCC
# ---------------------------------------------------------------------------

def mel_spectrogram(
    signal: np.ndarray,
    sr: float,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Mel spectrogram ``(n_mels, n_frames)``.  Linear amplitude scale."""
    S = _power_spectrogram(signal, n_fft, hop)
    fb = mel_filterbank(n_mels, n_fft, sr, fmin=fmin, fmax=fmax)
    return fb @ S


def log_mel_spectrogram(
    signal: np.ndarray,
    sr: float,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: float = 80.0,
) -> np.ndarray:
    """Log-scaled mel spectrogram in dB, floored at *-top_db*."""
    S = mel_spectrogram(signal, sr, n_fft, hop, n_mels, fmin, fmax)
    S_db = 10.0 * np.log10(np.maximum(S, amin) / ref)
    S_db = np.maximum(S_db, S_db.max() - top_db)
    return S_db


def mfcc(
    signal: np.ndarray,
    sr: float,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 2048,
    hop: int = 512,
    lifter: int = 22,
) -> np.ndarray:
    """Mel-frequency cepstral coefficients ``(n_mfcc, n_frames)``.

    DCT-II of log-mel filterbank energies (Davis & Mermelstein 1980).
    Optional sinusoidal liftering (Furui 1986).
    """
    from scipy.fft import dct
    S = mel_spectrogram(signal, sr, n_fft, hop, n_mels)
    log_S = np.log(np.maximum(S, 1e-10))
    cc = dct(log_S, type=2, axis=0, norm="ortho")[:n_mfcc]
    if lifter > 0:
        n = np.arange(n_mfcc)
        lift = 1.0 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
        cc *= lift[:, np.newaxis]
    return cc


# ---------------------------------------------------------------------------
# Delta features
# ---------------------------------------------------------------------------

def delta_features(features: np.ndarray, width: int = 9) -> np.ndarray:
    """First-order regression-based delta features (Furui 1986).

    Parameters
    ----------
    features : (n_features, n_frames)
    width : odd regression window

    Returns
    -------
    deltas : same shape as *features*
    """
    if width % 2 == 0:
        width += 1
    half = width // 2
    padded = np.pad(features, ((0, 0), (half, half)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, half + 1))
    if denom == 0:
        return np.zeros_like(features)
    n_frames = features.shape[1]
    delta = np.zeros_like(features)
    for t in range(n_frames):
        for n in range(1, half + 1):
            delta[:, t] += n * (padded[:, t + half + n] - padded[:, t + half - n])
        delta[:, t] /= denom
    return delta


def delta_delta_features(features: np.ndarray, width: int = 9) -> np.ndarray:
    """Second-order delta (acceleration) features."""
    return delta_features(delta_features(features, width), width)


# ---------------------------------------------------------------------------
# Chroma features
# ---------------------------------------------------------------------------

def chroma_stft(
    signal: np.ndarray,
    sr: float,
    n_fft: int = 4096,
    hop: int = 512,
    n_chroma: int = 12,
    tuning: float = 0.0,
) -> np.ndarray:
    """STFT-based chromagram ``(n_chroma, n_frames)``."""
    S = np.abs(_stft(signal, n_fft, hop)) ** 2
    fb = chroma_filterbank(n_chroma, n_fft, sr, tuning=tuning)
    raw = fb @ S
    norms = np.maximum(raw.sum(axis=0, keepdims=True), 1e-10)
    return raw / norms


def chroma_cqt(
    signal: np.ndarray,
    sr: float,
    hop: int = 512,
    n_chroma: int = 12,
    n_octaves: int = 7,
    bins_per_octave: int = 36,
    fmin: float = 32.7,
) -> np.ndarray:
    """Constant-Q chromagram ``(n_chroma, n_frames)``.

    Builds a CQT kernel per Brown (1991) / Schörkhuber & Klapuri (2010),
    then folds the resulting ``(n_bins, n_frames)`` matrix into *n_chroma*
    pitch classes.
    """
    n_bins = n_octaves * bins_per_octave
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
    # Q factor for each bin
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)

    n_frames = 1 + (len(signal) - 1) // hop
    cqt_mag = np.zeros((n_bins, n_frames))

    for k, fk in enumerate(freqs):
        N_k = int(np.ceil(Q * sr / fk))
        N_k = min(N_k, len(signal))
        if N_k < 2:
            continue
        win = np.hanning(N_k)
        t_indices = np.arange(N_k)
        kernel = win * np.exp(-2j * np.pi * fk * t_indices / sr) / N_k
        for t in range(n_frames):
            start = t * hop
            end = start + N_k
            if end > len(signal):
                break
            cqt_mag[k, t] = np.abs(np.dot(signal[start:end], kernel))

    # Fold into chroma
    chroma = np.zeros((n_chroma, n_frames))
    for k in range(n_bins):
        c = k % n_chroma
        chroma[c] += cqt_mag[k] ** 2
    chroma = np.sqrt(np.maximum(chroma, 0.0))
    norms = np.maximum(chroma.sum(axis=0, keepdims=True), 1e-10)
    return chroma / norms


def chroma_cens(
    chroma: np.ndarray,
    win_len: int = 41,
    quant_steps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Chroma Energy Normalized Statistics (Müller & Ewert 2011).

    1. L¹-normalise each frame.
    2. Quantise into logarithmic bins.
    3. Smooth with a Hann window of length *win_len*.
    4. Re-normalise each frame to unit L².

    Returns ``(n_chroma, n_frames)``.
    """
    if quant_steps is None:
        quant_steps = np.array([0.4, 0.2, 0.1, 0.05])
    n_chroma, n_frames = chroma.shape
    # L1 normalise
    norms = np.maximum(chroma.sum(axis=0, keepdims=True), 1e-10)
    c_norm = chroma / norms
    # Quantise
    c_quant = np.zeros_like(c_norm)
    for step in quant_steps:
        c_quant += (c_norm >= step).astype(float)
    # Smooth with Hann window
    if win_len > 1:
        win = np.hanning(win_len)
        win /= win.sum()
        from scipy.ndimage import convolve1d
        c_quant = convolve1d(c_quant, win, axis=1, mode="constant")
    # L2 normalise
    l2 = np.maximum(np.sqrt((c_quant ** 2).sum(axis=0, keepdims=True)), 1e-10)
    return c_quant / l2


# ---------------------------------------------------------------------------
# Tonnetz
# ---------------------------------------------------------------------------

def tonnetz_features(chroma: np.ndarray) -> np.ndarray:
    """Six-dimensional tonal centroid features (Harte et al. 2006).

    Maps a ``(12, n_frames)`` chromagram to ``(6, n_frames)`` coordinates on
    the Tonnetz: perfect fifths (dim 0–1), minor thirds (dim 2–3), major
    thirds (dim 4–5).
    """
    # Normalise chroma
    norms = np.maximum(chroma.sum(axis=0, keepdims=True), 1e-10)
    c = chroma / norms
    n_frames = chroma.shape[1]
    tonnetz = np.zeros((6, n_frames))
    phi = np.arange(12) * (7.0 / 12.0) * 2.0 * np.pi  # fifths
    tonnetz[0] = np.sin(phi) @ c
    tonnetz[1] = np.cos(phi) @ c
    phi3 = np.arange(12) * (3.0 / 12.0) * 2.0 * np.pi  # minor thirds
    tonnetz[2] = np.sin(phi3) @ c
    tonnetz[3] = np.cos(phi3) @ c
    phi4 = np.arange(12) * (4.0 / 12.0) * 2.0 * np.pi  # major thirds
    tonnetz[4] = np.sin(phi4) @ c
    tonnetz[5] = np.cos(phi4) @ c
    return tonnetz


# ---------------------------------------------------------------------------
# Spectral descriptors (batch / frame-level)
# ---------------------------------------------------------------------------

def spectral_contrast(
    spectrogram: np.ndarray,
    sr: float,
    n_bands: int = 6,
    fmin: float = 200.0,
    quantile: float = 0.02,
) -> np.ndarray:
    """Spectral contrast ``(n_bands + 1, n_frames)`` (Jiang et al. 2002).

    For each sub-band the contrast is the dB difference between the peaks
    (top *quantile*) and the valleys (bottom *quantile*).
    """
    n_freqs, n_frames = spectrogram.shape
    freqs = np.linspace(0, sr / 2.0, n_freqs)
    # Octave-spaced band edges
    edges = fmin * 2.0 ** np.arange(n_bands + 1)
    edges = np.clip(edges, 0, sr / 2.0)
    contrast = np.zeros((n_bands + 1, n_frames))
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[min(b + 1, n_bands)])
        if not mask.any():
            continue
        band = spectrogram[mask]
        q_lo = max(1, int(quantile * band.shape[0]))
        q_hi = max(1, int((1.0 - quantile) * band.shape[0]))
        sorted_band = np.sort(band, axis=0)
        valleys = sorted_band[:q_lo].mean(axis=0)
        peaks = sorted_band[q_hi:].mean(axis=0)
        contrast[b] = np.log1p(peaks) - np.log1p(valleys)
    # Last row: mean of full spectrum as "valley"
    contrast[n_bands] = np.log1p(spectrogram.mean(axis=0))
    return contrast


def spectral_bandwidth(
    spectrogram: np.ndarray,
    freqs: np.ndarray,
    p: int = 2,
) -> np.ndarray:
    """Spectral bandwidth of order *p* for each frame ``(n_frames,)``."""
    centroid = (freqs[:, None] * spectrogram).sum(axis=0) / np.maximum(
        spectrogram.sum(axis=0), 1e-10
    )
    deviation = np.abs(freqs[:, None] - centroid[None, :]) ** p
    bw = ((deviation * spectrogram).sum(axis=0) / np.maximum(
        spectrogram.sum(axis=0), 1e-10
    )) ** (1.0 / p)
    return bw


def spectral_flatness(spectrogram: np.ndarray) -> np.ndarray:
    """Wiener entropy / spectral flatness per frame ``(n_frames,)``."""
    log_spec = np.log(np.maximum(spectrogram, 1e-10))
    geo_mean = np.exp(log_spec.mean(axis=0))
    arith_mean = np.maximum(spectrogram.mean(axis=0), 1e-10)
    return geo_mean / arith_mean


def zero_crossing_rate(
    signal: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Frame-wise zero-crossing rate ``(n_frames,)``."""
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    zcr = np.empty(n_frames)
    for t in range(n_frames):
        start = t * hop_length
        frame = signal[start : start + frame_length]
        zcr[t] = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2.0 * frame_length)
    return zcr


def rms_energy(
    signal: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Frame-wise RMS energy ``(n_frames,)``."""
    n_frames = 1 + (len(signal) - frame_length) // hop_length
    rms = np.empty(n_frames)
    for t in range(n_frames):
        start = t * hop_length
        frame = signal[start : start + frame_length]
        rms[t] = np.sqrt(np.mean(frame ** 2))
    return rms


# ---------------------------------------------------------------------------
# Convenience: full feature extraction pipeline
# ---------------------------------------------------------------------------

@dataclass
class MelFeatureSet:
    """Container for mel/chroma features extracted from a signal."""
    mel_spectrogram: np.ndarray
    log_mel_spectrogram: np.ndarray
    mfcc: np.ndarray
    mfcc_delta: np.ndarray
    mfcc_delta2: np.ndarray
    chroma: np.ndarray
    tonnetz: np.ndarray
    spectral_contrast: np.ndarray
    spectral_flatness: np.ndarray
    spectral_bandwidth: np.ndarray
    zcr: np.ndarray
    rms: np.ndarray


def extract_mel_features(
    signal: np.ndarray,
    sr: float,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 128,
    n_mfcc: int = 13,
    n_chroma: int = 12,
) -> MelFeatureSet:
    """Extract a comprehensive set of mel/chroma features from a mono signal."""
    mel_spec = mel_spectrogram(signal, sr, n_fft, hop, n_mels)
    log_mel = log_mel_spectrogram(signal, sr, n_fft, hop, n_mels)
    cc = mfcc(signal, sr, n_mfcc, n_mels=40, n_fft=n_fft, hop=hop)
    cc_d = delta_features(cc)
    cc_dd = delta_delta_features(cc)
    chroma = chroma_stft(signal, sr, n_fft=max(n_fft, 4096), hop=hop, n_chroma=n_chroma)
    tn = tonnetz_features(chroma)
    S_mag = np.sqrt(np.maximum(mel_spec, 0.0))
    sc = spectral_contrast(np.abs(_stft(signal, n_fft, hop)), sr)
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    S_abs = np.abs(_stft(signal, n_fft, hop))
    sf = spectral_flatness(S_abs)
    sb = spectral_bandwidth(S_abs, freqs)
    z = zero_crossing_rate(signal, n_fft, hop)
    r = rms_energy(signal, n_fft, hop)
    return MelFeatureSet(
        mel_spectrogram=mel_spec,
        log_mel_spectrogram=log_mel,
        mfcc=cc,
        mfcc_delta=cc_d,
        mfcc_delta2=cc_dd,
        chroma=chroma,
        tonnetz=tn,
        spectral_contrast=sc,
        spectral_flatness=sf,
        spectral_bandwidth=sb,
        zcr=z,
        rms=r,
    )
