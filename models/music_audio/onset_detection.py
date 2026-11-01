"""Onset detection and transient analysis for music signals.

Comprehensive onset detection with multiple detection functions (spectral flux,
high-frequency content, complex-domain, SuperFlux), adaptive thresholding,
transient-steady separation, and tempo estimation from onsets.

References
----------
Bello, J. P. et al. (2005). "A Tutorial on Onset Detection in Music Signals."
    IEEE Trans. Speech and Audio Processing, 13(5), 1035-1047.
Dixon, S. (2006). "Onset Detection Revisited." Proc. DAFx, 133-137.
Böck, S. & Widmer, G. (2013). "Maximum Filter Vibrato Suppression for Onset
    Detection." Proc. DAFx.
Böck, S. et al. (2012). "Evaluating the Online Capabilities of Onset Detection
    Methods." Proc. ISMIR.
Masri, P. (1996). "Computer Modelling of Sound for Transformation and Synthesis
    of Musical Signals." PhD thesis, University of Bristol.
Fitzgerald, D. (2010). "Harmonic/Percussive Separation using Median Filtering."
    Proc. DAFx.

Pure numpy/scipy — no external ML framework dependencies.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class OnsetType(Enum):
    """Classification of onset transient character."""
    PERCUSSIVE = "percussive"
    PITCHED = "pitched"
    MIXED = "mixed"
    SPECTRAL_FLUX = "spectral_flux"
    ENERGY = "energy"


@dataclass
class OnsetResult:
    """Result of full onset detection pipeline."""
    times: np.ndarray
    strengths: np.ndarray
    onset_types: List[OnsetType]
    tempo_estimate: float


# ---------------------------------------------------------------------------
# Core onset detection functions (ODFs)
# ---------------------------------------------------------------------------

def spectral_flux(spectrogram: np.ndarray, rectify: bool = True) -> np.ndarray:
    """Half-wave rectified spectral flux (Böck et al. 2012).

    Parameters
    ----------
    spectrogram : (n_bins, n_frames) magnitude spectrogram
    rectify : if True, apply half-wave rectification (only positive changes)

    Returns
    -------
    odf : (n_frames,) onset detection function
    """
    if spectrogram.ndim != 2 or spectrogram.shape[1] < 2:
        return np.zeros(max(spectrogram.shape[-1] if spectrogram.ndim == 2 else 1, 1))
    diff = np.diff(spectrogram, axis=1)
    if rectify:
        diff = np.maximum(diff, 0.0)
    odf = np.sum(diff, axis=0)
    return np.concatenate([[0.0], odf])


def high_frequency_content(spectrogram: np.ndarray, sr: int = 22050) -> np.ndarray:
    """High-frequency content onset detection function (Masri 1996).

    Weights each frequency bin by its frequency, emphasising transients which
    have broadband energy concentrated at high frequencies.

    Parameters
    ----------
    spectrogram : (n_bins, n_frames) magnitude spectrogram
    sr : sample rate (used to compute physical frequency weights)

    Returns
    -------
    hfc : (n_frames,) HFC values per frame
    """
    n_bins, n_frames = spectrogram.shape
    freqs = np.linspace(0, sr / 2, n_bins)
    weights = freqs / max(freqs[-1], 1e-12)
    hfc = spectrogram.T @ weights
    return hfc.ravel()


def complex_domain(stft: np.ndarray) -> np.ndarray:
    """Complex-domain onset detection (Bello et al. 2005).

    Combines magnitude and phase information by comparing the predicted
    complex spectrum (from linear phase extrapolation) against the actual.

    Parameters
    ----------
    stft : (n_bins, n_frames) complex STFT

    Returns
    -------
    odf : (n_frames,) complex-domain ODF
    """
    n_bins, n_frames = stft.shape
    if n_frames < 3:
        return np.zeros(n_frames)
    mag = np.abs(stft)
    phase = np.angle(stft)
    # Phase difference (instantaneous frequency)
    dphi = np.diff(phase, axis=1)
    # Second-order phase difference (phase deviation)
    ddphi = np.diff(dphi, axis=1)
    # Predicted magnitude: assume steady
    mag_pred = mag[:, 1:-1]
    # Target for deviation: expected phase = phase[t-1] + dphi[t-1]
    predicted_phase = phase[:, 1:-1] + dphi[:, :-1]
    predicted = mag_pred * np.exp(1j * predicted_phase)
    actual = stft[:, 2:]
    deviation = np.abs(actual - predicted)
    odf = np.sum(deviation, axis=0)
    return np.concatenate([np.zeros(2), odf])


def superflux(
    spectrogram: np.ndarray,
    n_bands: int = 24,
    lag: int = 2,
    max_filter: int = 3,
) -> np.ndarray:
    """Vibrato-robust onset detection — SuperFlux (Böck & Widmer 2013).

    Applies a maximum filter along frequency before computing flux, which
    suppresses spectral modulations from vibrato that cause false onsets.

    Parameters
    ----------
    spectrogram : (n_bins, n_frames) magnitude spectrogram
    n_bands : number of frequency bands for grouping (Mel-like)
    lag : frame lag for difference computation
    max_filter : size of maximum filter along frequency axis

    Returns
    -------
    odf : (n_frames,) SuperFlux ODF
    """
    from scipy.ndimage import maximum_filter1d

    n_bins, n_frames = spectrogram.shape
    if n_frames < lag + 1:
        return np.zeros(n_frames)

    # Apply max-filter along frequency axis to suppress vibrato
    filtered = maximum_filter1d(spectrogram, size=max_filter, axis=0)

    # Logarithmic compression
    log_spec = np.log1p(filtered * 100)

    # Compute difference with specified lag
    if lag >= n_frames:
        return np.zeros(n_frames)
    diff = log_spec[:, lag:] - log_spec[:, :-lag]
    hw = np.maximum(diff, 0.0)
    odf = np.sum(hw, axis=0)

    # Pad to match original length
    return np.concatenate([np.zeros(lag), odf])


def _mel_filterbank(n_bins: int, n_bands: int, sr: int) -> np.ndarray:
    """Triangular Mel filterbank matrix (n_bands, n_bins)."""
    f_min, f_max = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_bands + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.round(hz_points * n_bins / (sr / 2.0)).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    fb = np.zeros((n_bands, n_bins))
    for m in range(n_bands):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        if center > left:
            fb[m, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        if right > center:
            fb[m, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    return fb


def onset_strength_envelope(
    signal: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_bands: int = 80,
) -> np.ndarray:
    """Combined multi-band onset strength envelope.

    Computes a Mel-band spectrogram, takes first-order differences,
    half-wave rectifies per band, sums across bands, and normalises.

    Parameters
    ----------
    signal : mono audio signal
    sr : sample rate
    hop_length : hop in samples
    n_fft : FFT size
    n_bands : number of Mel bands

    Returns
    -------
    envelope : (n_frames,) onset strength over time
    """
    if len(signal) < n_fft:
        return np.array([0.0])

    n_frames = 1 + (len(signal) - n_fft) // hop_length
    window = np.hanning(n_fft)
    mag = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start: start + n_fft] * window
        mag[:, i] = np.abs(np.fft.rfft(frame))

    # Mel filterbank
    fb = _mel_filterbank(mag.shape[0], n_bands, sr)
    mel_spec = fb @ mag  # (n_bands, n_frames)

    # Log compression
    mel_spec = np.log1p(mel_spec * 1000)

    # First-order difference + half-wave rectification + sum
    if mel_spec.shape[1] < 2:
        return np.array([0.0])
    diff = np.diff(mel_spec, axis=1)
    diff = np.maximum(diff, 0.0)
    envelope = np.sum(diff, axis=0)

    # Normalise to [0, 1]
    mx = envelope.max()
    if mx > 0:
        envelope = envelope / mx

    return np.concatenate([[0.0], envelope])


# ---------------------------------------------------------------------------
# Thresholding and peak picking
# ---------------------------------------------------------------------------

def adaptive_threshold(
    odf: np.ndarray, window: int = 17, delta: float = 0.05
) -> np.ndarray:
    """Local-mean adaptive threshold (Dixon 2006).

    threshold[n] = delta + mean(odf[n - w//2 : n + w//2 + 1])

    Parameters
    ----------
    odf : onset detection function
    window : window size for local mean (odd recommended)
    delta : offset added to the local mean

    Returns
    -------
    threshold : (len(odf),) adaptive threshold curve
    """
    n = len(odf)
    if n == 0:
        return np.array([])
    half_w = window // 2
    padded = np.pad(odf, (half_w, half_w), mode='reflect')
    cumsum = np.concatenate([[0.0], np.cumsum(padded)])
    # Local mean via running sum
    local_mean = (cumsum[window:] - cumsum[:-window]) / window
    # Ensure exact same length as input
    if len(local_mean) > n:
        local_mean = local_mean[:n]
    elif len(local_mean) < n:
        local_mean = np.pad(local_mean, (0, n - len(local_mean)), mode='edge')
    return local_mean + delta


def pick_onsets(
    odf: np.ndarray,
    threshold: float | np.ndarray = 0.3,
    min_interval_ms: float = 50.0,
    hop_length: int = 512,
    sr: int = 22050,
) -> np.ndarray:
    """Pick onset times from an ODF by thresholding + peak picking.

    Parameters
    ----------
    odf : onset detection function values
    threshold : scalar or adaptive threshold array
    min_interval_ms : minimum inter-onset interval in milliseconds
    hop_length : hop length used to compute the ODF
    sr : sample rate

    Returns
    -------
    onset_times : (n_onsets,) array of onset times in seconds
    """
    if len(odf) == 0:
        return np.array([], dtype=float)

    if isinstance(threshold, (int, float)):
        thresh = np.full_like(odf, threshold)
    else:
        thresh = threshold

    # Minimum distance in frames
    frame_time = hop_length / sr
    min_dist_frames = max(1, int(min_interval_ms / 1000.0 / frame_time))

    above = odf > thresh
    peaks = []
    for i in range(1, len(odf) - 1):
        if above[i] and odf[i] >= odf[i - 1] and odf[i] >= odf[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist_frames:
                peaks.append(i)

    return np.array(peaks, dtype=float) * frame_time


def onset_backtrack(
    signal: np.ndarray,
    onsets: np.ndarray,
    sr: int = 22050,
    max_backtrack_ms: float = 30.0,
) -> np.ndarray:
    """Backtrack onset times to the nearest energy minimum before each onset.

    Corrects for the delay in spectral-domain ODFs by finding the local
    energy minimum in a short window before each detected onset.

    Parameters
    ----------
    signal : mono audio
    onsets : onset times in seconds
    sr : sample rate
    max_backtrack_ms : maximum backtrack window in ms

    Returns
    -------
    corrected : (n_onsets,) backtracked onset times in seconds
    """
    if len(onsets) == 0:
        return onsets.copy()

    max_samples = int(max_backtrack_ms / 1000.0 * sr)
    # Short-time energy
    hop = 64
    n_frames = max(1, len(signal) // hop)
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(signal))
        energy[i] = np.sum(signal[start:end] ** 2)

    corrected = np.zeros_like(onsets)
    for j, t in enumerate(onsets):
        frame_idx = int(t * sr / hop)
        search_start = max(0, frame_idx - max_samples // hop)
        if search_start >= frame_idx or frame_idx >= n_frames:
            corrected[j] = t
            continue
        segment = energy[search_start:frame_idx + 1]
        local_min = search_start + np.argmin(segment)
        corrected[j] = local_min * hop / sr

    return corrected


# ---------------------------------------------------------------------------
# Onset classification
# ---------------------------------------------------------------------------

def classify_onset_type(
    signal: np.ndarray, onset_idx: int, sr: int = 22050, n_fft: int = 2048
) -> OnsetType:
    """Classify an onset as percussive, pitched, or mixed.

    Uses the spectral centroid trajectory immediately after the onset:
    - High centroid + rapid decay → percussive
    - Low, stable centroid → pitched
    - Otherwise → mixed

    Parameters
    ----------
    signal : mono audio
    onset_idx : sample index of the onset
    sr : sample rate
    n_fft : FFT window size

    Returns
    -------
    OnsetType classification
    """
    # Extract a short window after onset (~50 ms chunks)
    window_samples = int(0.05 * sr)
    n_chunks = 4
    centroids = []

    for c in range(n_chunks):
        start = onset_idx + c * window_samples
        end = start + min(n_fft, window_samples)
        if end > len(signal):
            break
        frame = signal[start:end]
        if len(frame) < 64:
            break
        # Zero-pad if needed
        padded = np.zeros(n_fft)
        padded[:len(frame)] = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.linspace(0, sr / 2, len(spectrum))
        total = spectrum.sum()
        if total > 0:
            centroids.append(np.sum(freqs * spectrum) / total)
        else:
            centroids.append(0.0)

    if len(centroids) < 2:
        return OnsetType.MIXED

    centroids = np.array(centroids)
    mean_centroid = centroids.mean()
    centroid_std = centroids.std()
    nyquist = sr / 2.0

    # Decision boundaries
    if mean_centroid > 0.35 * nyquist and centroid_std > 0.05 * nyquist:
        return OnsetType.PERCUSSIVE
    elif centroid_std < 0.02 * nyquist and mean_centroid < 0.3 * nyquist:
        return OnsetType.PITCHED
    else:
        return OnsetType.MIXED


# ---------------------------------------------------------------------------
# Transient/steady separation
# ---------------------------------------------------------------------------

def transient_steady_separation(
    signal: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    kernel_size: int = 31,
) -> Tuple[np.ndarray, np.ndarray]:
    """Harmonic/percussive (steady/transient) separation via median filtering.

    Fitzgerald (2010): apply median filter along time (→ harmonic mask) and
    along frequency (→ percussive mask) to a magnitude spectrogram, then
    soft-mask and invert.

    Parameters
    ----------
    signal : mono audio
    sr : sample rate
    n_fft : FFT size
    hop_length : hop length
    kernel_size : median filter kernel size

    Returns
    -------
    transient : transient component (same length as signal)
    steady : steady/harmonic component
    """
    from scipy.ndimage import median_filter

    if len(signal) < n_fft:
        return signal.copy(), np.zeros_like(signal)

    # STFT
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    window = np.hanning(n_fft)
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    for i in range(n_frames):
        start = i * hop_length
        frame = np.zeros(n_fft)
        seg = signal[start: start + n_fft]
        frame[:len(seg)] = seg
        frame[:len(seg)] *= window[:len(seg)]
        stft_matrix[:, i] = np.fft.rfft(frame)

    mag = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # Median filtering
    harmonic_mag = median_filter(mag, size=(1, kernel_size))
    percussive_mag = median_filter(mag, size=(kernel_size, 1))

    # Soft masks (Wiener-style)
    eps = 1e-10
    total = harmonic_mag + percussive_mag + eps
    harmonic_mask = harmonic_mag / total
    percussive_mask = percussive_mag / total

    # Apply masks
    harmonic_stft = harmonic_mask * mag * np.exp(1j * phase)
    percussive_stft = percussive_mask * mag * np.exp(1j * phase)

    # Inverse STFT (overlap-add)
    def istft(stft_data: np.ndarray) -> np.ndarray:
        out_len = n_fft + (n_frames - 1) * hop_length
        output = np.zeros(out_len)
        window_sum = np.zeros(out_len)
        for i in range(n_frames):
            start = i * hop_length
            frame = np.fft.irfft(stft_data[:, i], n=n_fft)
            frame *= window
            end = start + n_fft
            output[start:end] += frame
            window_sum[start:end] += window ** 2
        # Normalise by window overlap
        nonzero = window_sum > 1e-10
        output[nonzero] /= window_sum[nonzero]
        return output[:len(signal)]

    transient = istft(percussive_stft)
    steady = istft(harmonic_stft)

    return transient, steady


# ---------------------------------------------------------------------------
# Attack and tempo
# ---------------------------------------------------------------------------

def attack_time(
    signal: np.ndarray,
    onset_idx: int,
    sr: int = 22050,
    max_attack_ms: float = 200.0,
) -> float:
    """Measure attack time: duration from onset to local peak, in milliseconds.

    Parameters
    ----------
    signal : mono audio
    onset_idx : sample index of the onset
    sr : sample rate
    max_attack_ms : maximum search window for the peak

    Returns
    -------
    attack_ms : attack duration in milliseconds
    """
    max_samples = int(max_attack_ms / 1000.0 * sr)
    end = min(onset_idx + max_samples, len(signal))
    if onset_idx >= end:
        return 0.0

    segment = np.abs(signal[onset_idx:end])
    if len(segment) == 0:
        return 0.0

    # Smooth with short window to avoid spurious peaks
    win = min(64, len(segment) // 2)
    if win > 1:
        kernel = np.ones(win) / win
        smoothed = np.convolve(segment, kernel, mode='same')
    else:
        smoothed = segment

    peak_idx = np.argmax(smoothed)
    return float(peak_idx / sr * 1000.0)


def compute_tempo_from_onsets(
    onset_times: np.ndarray,
    min_bpm: float = 40.0,
    max_bpm: float = 240.0,
) -> float:
    """Estimate tempo (BPM) from onset times via autocorrelation.

    Computes the autocorrelation of the onset impulse train and finds the
    peak corresponding to the most prominent periodicity within [min_bpm, max_bpm].

    Parameters
    ----------
    onset_times : array of onset times in seconds
    min_bpm : minimum tempo to consider
    max_bpm : maximum tempo to consider

    Returns
    -------
    bpm : estimated tempo in beats per minute
    """
    if len(onset_times) < 2:
        return 0.0

    # Create an impulse train at ~200 Hz resolution
    resolution = 0.005  # 5 ms
    duration = onset_times[-1] - onset_times[0]
    if duration <= 0:
        return 0.0
    n_bins = int(duration / resolution) + 1
    impulse = np.zeros(n_bins)
    for t in onset_times:
        idx = int((t - onset_times[0]) / resolution)
        if 0 <= idx < n_bins:
            impulse[idx] = 1.0

    # Autocorrelation
    acf = np.correlate(impulse, impulse, mode='full')
    acf = acf[len(acf) // 2:]  # Take positive lags only

    # Convert lag range to BPM range
    min_lag = int(60.0 / max_bpm / resolution)
    max_lag = int(60.0 / min_bpm / resolution)
    min_lag = max(1, min_lag)
    max_lag = min(len(acf) - 1, max_lag)

    if min_lag >= max_lag:
        return 0.0

    segment = acf[min_lag:max_lag + 1]
    if len(segment) == 0:
        return 0.0

    best_lag = min_lag + np.argmax(segment)
    bpm = 60.0 / (best_lag * resolution)
    return float(bpm)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def detect_onsets(
    signal: np.ndarray,
    sr: int = 22050,
    method: str = "superflux",
    hop_length: int = 512,
    n_fft: int = 2048,
    threshold_window: int = 17,
    threshold_delta: float = 0.07,
    min_interval_ms: float = 50.0,
) -> OnsetResult:
    """Full onset detection pipeline.

    Steps:
    1. Compute STFT / magnitude spectrogram
    2. Apply selected ODF method
    3. Adaptive thresholding (Dixon 2006)
    4. Peak picking with minimum inter-onset interval
    5. Backtrack onsets to energy minima
    6. Classify each onset type
    7. Estimate tempo from onset times

    Parameters
    ----------
    signal : mono audio
    sr : sample rate
    method : one of 'spectral_flux', 'hfc', 'complex', 'superflux', 'envelope'
    hop_length : hop length for STFT
    n_fft : FFT size
    threshold_window : adaptive threshold window
    threshold_delta : adaptive threshold offset
    min_interval_ms : minimum inter-onset interval

    Returns
    -------
    OnsetResult with times, strengths, types, and tempo
    """
    if len(signal) < n_fft:
        return OnsetResult(
            times=np.array([]),
            strengths=np.array([]),
            onset_types=[],
            tempo_estimate=0.0,
        )

    # Compute STFT
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    window = np.hanning(n_fft)
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    for i in range(n_frames):
        start = i * hop_length
        frame = np.zeros(n_fft)
        seg = signal[start: start + n_fft]
        frame[:len(seg)] = seg
        frame[:len(seg)] *= window[:len(seg)]
        stft_matrix[:, i] = np.fft.rfft(frame)
    mag = np.abs(stft_matrix)

    # Select ODF
    if method == "spectral_flux":
        odf = spectral_flux(mag, rectify=True)
    elif method == "hfc":
        odf = high_frequency_content(mag, sr)
    elif method == "complex":
        odf = complex_domain(stft_matrix)
    elif method == "superflux":
        odf = superflux(mag)
    elif method == "envelope":
        odf = onset_strength_envelope(signal, sr, hop_length, n_fft)
    else:
        odf = superflux(mag)

    # Normalise ODF
    mx = odf.max()
    if mx > 0:
        odf_norm = odf / mx
    else:
        odf_norm = odf

    # Adaptive threshold
    thresh = adaptive_threshold(odf_norm, window=threshold_window, delta=threshold_delta)

    # Pick onsets
    onset_times = pick_onsets(odf_norm, thresh, min_interval_ms, hop_length, sr)

    if len(onset_times) == 0:
        return OnsetResult(
            times=np.array([]),
            strengths=np.array([]),
            onset_types=[],
            tempo_estimate=0.0,
        )

    # Backtrack
    onset_times = onset_backtrack(signal, onset_times, sr)

    # Strengths: interpolate ODF at onset positions
    frame_times = np.arange(len(odf)) * hop_length / sr
    strengths = np.interp(onset_times, frame_times, odf)

    # Classify each onset
    onset_types = []
    for t in onset_times:
        sample_idx = int(t * sr)
        sample_idx = max(0, min(sample_idx, len(signal) - 1))
        onset_types.append(classify_onset_type(signal, sample_idx, sr, n_fft))

    # Tempo
    tempo = compute_tempo_from_onsets(onset_times)

    return OnsetResult(
        times=onset_times,
        strengths=strengths,
        onset_types=onset_types,
        tempo_estimate=tempo,
    )


# ---------------------------------------------------------------------------
# Legacy class (kept for backward compat with existing __init__.py import)
# ---------------------------------------------------------------------------

class OnsetDetector:
    """Frame-level onset detection functions and peak picking."""

    @staticmethod
    def spectral_flux(spectrogram: np.ndarray) -> np.ndarray:
        """Half-wave rectified spectral flux (Bello et al. 2005)."""
        if spectrogram.shape[1] < 2:
            return np.zeros(spectrogram.shape[1])
        diff = np.diff(spectrogram, axis=1)
        hw = np.maximum(diff, 0.0)
        return np.sum(hw, axis=0)

    @staticmethod
    def high_frequency_content(
        spectrogram: np.ndarray, freqs: np.ndarray | None = None
    ) -> np.ndarray:
        """Energy weighted toward high frequencies (Masri 1996)."""
        n_bins, n_frames = spectrogram.shape
        if freqs is None:
            freqs = np.arange(n_bins, dtype=float)
        weights = freqs[:n_bins] / max(freqs.max(), 1e-12)
        return (spectrogram.T @ weights).ravel()

    @staticmethod
    def complex_domain(stft_frames: np.ndarray) -> np.ndarray:
        """Complex-domain onset detection (Bello 2005)."""
        if stft_frames.shape[1] < 3:
            return np.zeros(stft_frames.shape[1])
        mag = np.abs(stft_frames)
        phase = np.angle(stft_frames)
        phase_diff = np.diff(phase, axis=1)
        predicted_phase = phase[:, 1:-1] + phase_diff[:, :-1]
        predicted = mag[:, 1:-1] * np.exp(1j * predicted_phase)
        actual = stft_frames[:, 2:]
        deviation = np.abs(actual - predicted)
        odf = np.sum(deviation, axis=0)
        return np.concatenate([np.zeros(2), odf])

    @staticmethod
    def superflux(
        spectrogram: np.ndarray, max_filter_size: int = 3
    ) -> np.ndarray:
        """Vibrato-robust onset detection (Böck & Widmer 2013)."""
        from scipy.ndimage import maximum_filter1d

        if spectrogram.shape[1] < 2:
            return np.zeros(spectrogram.shape[1])
        filtered = maximum_filter1d(spectrogram, size=max_filter_size, axis=0)
        diff = filtered[:, 1:] - spectrogram[:, :-1]
        hw = np.maximum(diff, 0.0)
        odf = np.sum(hw, axis=0)
        return np.concatenate([[0.0], odf])

    @staticmethod
    def pick_peaks(
        detection_fn: np.ndarray,
        threshold: float = 0.0,
        min_distance: int = 1,
    ) -> np.ndarray:
        """Adaptive peak picking with minimum inter-onset interval."""
        if len(detection_fn) == 0:
            return np.array([], dtype=int)
        above = detection_fn > threshold
        peaks = []
        for i in range(1, len(detection_fn) - 1):
            if (
                above[i]
                and detection_fn[i] >= detection_fn[i - 1]
                and detection_fn[i] >= detection_fn[i + 1]
            ):
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        return np.array(peaks, dtype=int)

    @staticmethod
    def onset_strength_envelope(
        signal: np.ndarray, sr: int = 22050, hop_length: int = 512
    ) -> np.ndarray:
        """Aggregate onset strength from magnitude spectrogram."""
        n_fft = 2048
        if len(signal) < n_fft:
            return np.array([0.0])
        step = hop_length
        n_frames = 1 + (len(signal) - n_fft) // step
        window = np.hanning(n_fft)
        mag = np.zeros((n_fft // 2 + 1, n_frames))
        for i in range(n_frames):
            frame = signal[i * step: i * step + n_fft] * window
            mag[:, i] = np.abs(np.fft.rfft(frame))
        return OnsetDetector.spectral_flux(mag)

    @staticmethod
    def tempo_from_onsets(onset_times: np.ndarray) -> float:
        """Estimate BPM from inter-onset intervals via histogram peak."""
        if len(onset_times) < 2:
            return 0.0
        ioi = np.diff(onset_times)
        ioi = ioi[ioi > 0.1]
        if len(ioi) == 0:
            return 0.0
        bpm_candidates = 60.0 / ioi
        bpm_candidates = bpm_candidates[
            (bpm_candidates > 30) & (bpm_candidates < 300)
        ]
        if len(bpm_candidates) == 0:
            return 0.0
        hist, edges = np.histogram(bpm_candidates, bins=50)
        peak_idx = np.argmax(hist)
        return float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)
