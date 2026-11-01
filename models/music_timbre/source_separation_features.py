"""Source separation features: harmonic/percussive masks, singing voice detection, drum isolation.

Provides median-filter HPSS, vibrato detection for singing voice,
spectral template initialization for NMF, and drum-pattern spectral peak isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import signal as sig


# ---------------------------------------------------------------------------
# Harmonic/percussive source separation (HPSS)
# ---------------------------------------------------------------------------

def median_filter_horizontal(S: np.ndarray, k: int = 31) -> np.ndarray:
    """Median filter along time axis (horizontal) for harmonic enhancement."""
    out = np.zeros_like(S)
    for f in range(S.shape[0]):
        out[f, :] = sig.medfilt(S[f, :], kernel_size=k)
    return out


def median_filter_vertical(S: np.ndarray, k: int = 31) -> np.ndarray:
    """Median filter along frequency axis (vertical) for percussive enhancement."""
    out = np.zeros_like(S)
    for t in range(S.shape[1]):
        out[:, t] = sig.medfilt(S[:, t], kernel_size=k)
    return out


@dataclass
class HPSSResult:
    harmonic_mask: np.ndarray
    percussive_mask: np.ndarray
    residual_mask: np.ndarray
    harmonic_energy_ratio: float
    percussive_energy_ratio: float


def hpss(spectrogram: np.ndarray,
         k_harmonic: int = 31,
         k_percussive: int = 31,
         margin: float = 1.0) -> HPSSResult:
    """Harmonic-percussive separation via median filtering with margin.

    Args:
        spectrogram: magnitude spectrogram (freq × time)
        k_harmonic: median filter kernel for harmonic (along time)
        k_percussive: median filter kernel for percussive (along freq)
        margin: soft mask exponent (1.0 = standard Wiener, >1 = harder)
    """
    H = median_filter_horizontal(spectrogram, k_harmonic)
    P = median_filter_vertical(spectrogram, k_percussive)

    if margin != 1.0:
        H = H ** margin
        P = P ** margin

    total = H + P + 1e-10
    h_mask = H / total
    p_mask = P / total
    r_mask = np.clip(1.0 - h_mask - p_mask, 0.0, 1.0)

    S2 = spectrogram ** 2
    total_energy = float(np.sum(S2)) + 1e-30
    h_ratio = float(np.sum(S2 * h_mask)) / total_energy
    p_ratio = float(np.sum(S2 * p_mask)) / total_energy

    return HPSSResult(
        harmonic_mask=h_mask,
        percussive_mask=p_mask,
        residual_mask=r_mask,
        harmonic_energy_ratio=h_ratio,
        percussive_energy_ratio=p_ratio,
    )


def iterative_hpss(spectrogram: np.ndarray,
                   k_harmonic: int = 31,
                   k_percussive: int = 31,
                   n_iter: int = 3) -> HPSSResult:
    """Multi-pass HPSS for cleaner separation."""
    current = spectrogram.copy()
    h_acc = np.zeros_like(spectrogram)

    for i in range(n_iter):
        result = hpss(current, k_harmonic, k_percussive)
        h_part = current * result.harmonic_mask
        h_acc += h_part
        current = current * result.percussive_mask

    total_energy = float(np.sum(spectrogram ** 2)) + 1e-30
    h_energy = float(np.sum(h_acc ** 2))

    h_mask_final = h_acc / (spectrogram + 1e-10)
    h_mask_final = np.clip(h_mask_final, 0.0, 1.0)
    p_mask_final = 1.0 - h_mask_final

    return HPSSResult(
        harmonic_mask=h_mask_final,
        percussive_mask=p_mask_final,
        residual_mask=np.zeros_like(spectrogram),
        harmonic_energy_ratio=h_energy / total_energy,
        percussive_energy_ratio=1.0 - h_energy / total_energy,
    )


# ---------------------------------------------------------------------------
# Singing voice detection features
# ---------------------------------------------------------------------------

@dataclass
class VoiceDetectionFeatures:
    vibrato_rate: float         # Hz (typically 5-7 Hz for singing)
    vibrato_extent: float       # semitones
    formant_prominence: float   # relative energy in formant regions
    harmonic_stability: float   # consistency of harmonic structure
    has_voice_likelihood: float # 0..1 estimate


def detect_vibrato(f0_trajectory: np.ndarray, sr_f0: float,
                   min_rate: float = 4.0,
                   max_rate: float = 8.0) -> tuple[float, float]:
    """Detect vibrato rate and extent from an f0 trajectory.

    Args:
        f0_trajectory: fundamental frequency over time (Hz), 0 = unvoiced
        sr_f0: sample rate of f0 trajectory (frames/sec)

    Returns:
        (rate_hz, extent_semitones)
    """
    voiced = f0_trajectory[f0_trajectory > 0]
    if len(voiced) < 10:
        return 0.0, 0.0

    # Detrend (remove slow pitch drift)
    detrended = voiced - np.convolve(voiced, np.ones(5) / 5, mode="same")

    # FFT to find vibrato rate
    n = len(detrended)
    freqs = np.fft.rfftfreq(n, 1.0 / sr_f0)
    spectrum = np.abs(np.fft.rfft(detrended))

    # Look for peak in vibrato range
    mask = (freqs >= min_rate) & (freqs <= max_rate)
    if not np.any(mask):
        return 0.0, 0.0

    vibrato_spectrum = spectrum[mask]
    vibrato_freqs = freqs[mask]
    peak_idx = int(np.argmax(vibrato_spectrum))
    rate = float(vibrato_freqs[peak_idx])

    # Extent in semitones
    extent_hz = float(np.std(detrended)) * 2.0  # ~peak-to-peak
    if np.mean(voiced) > 0:
        extent_semitones = 12.0 * np.log2(1.0 + extent_hz / np.mean(voiced))
    else:
        extent_semitones = 0.0

    return rate, float(extent_semitones)


def formant_prominence(spectrogram: np.ndarray, freqs: np.ndarray,
                       formant_ranges: list[tuple[float, float]] | None = None,
                       ) -> float:
    """Measure energy concentration in typical vocal formant regions.

    Default ranges cover F1-F4 of adult singing voice.
    """
    if formant_ranges is None:
        formant_ranges = [
            (250.0, 900.0),    # F1
            (900.0, 2500.0),   # F2
            (2500.0, 3500.0),  # F3
            (3500.0, 5000.0),  # F4
        ]

    total_energy = float(np.sum(spectrogram ** 2)) + 1e-30
    formant_energy = 0.0

    for lo, hi in formant_ranges:
        mask = (freqs >= lo) & (freqs < hi)
        formant_energy += float(np.sum(spectrogram[mask, :] ** 2))

    return formant_energy / total_energy


def voice_detection_features(spectrogram: np.ndarray,
                             freqs: np.ndarray,
                             f0_trajectory: np.ndarray | None = None,
                             sr_f0: float = 100.0) -> VoiceDetectionFeatures:
    """Extract features indicative of singing voice presence."""
    fp = formant_prominence(spectrogram, freqs)

    if f0_trajectory is not None:
        vib_rate, vib_extent = detect_vibrato(f0_trajectory, sr_f0)
    else:
        vib_rate, vib_extent = 0.0, 0.0

    # Harmonic stability: variance of spectral centroid across frames
    centroids = np.zeros(spectrogram.shape[1])
    for t in range(spectrogram.shape[1]):
        col = spectrogram[:, t]
        total = np.sum(col) + 1e-30
        centroids[t] = np.sum(freqs * col) / total
    stability = 1.0 / (1.0 + np.std(centroids) / (np.mean(centroids) + 1e-30))

    # Heuristic voice likelihood
    voice_score = 0.0
    if 4.5 <= vib_rate <= 7.5:
        voice_score += 0.3
    if vib_extent > 0.1:
        voice_score += 0.2
    if fp > 0.3:
        voice_score += 0.3
    if stability > 0.5:
        voice_score += 0.2

    return VoiceDetectionFeatures(
        vibrato_rate=vib_rate,
        vibrato_extent=vib_extent,
        formant_prominence=fp,
        harmonic_stability=float(stability),
        has_voice_likelihood=float(np.clip(voice_score, 0.0, 1.0)),
    )


# ---------------------------------------------------------------------------
# NMF basis initialization for source separation
# ---------------------------------------------------------------------------

def spectral_template_basis(spectrogram: np.ndarray,
                            n_bases: int = 8,
                            n_iter: int = 50,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Initialize NMF bases from spectrogram via multiplicative updates.

    V ≈ W H where W = (freq × n_bases), H = (n_bases × time).

    Returns (W, H).
    """
    rng = np.random.RandomState(seed)
    F, T = spectrogram.shape
    V = np.maximum(spectrogram, 1e-10)

    W = rng.rand(F, n_bases).astype(np.float64) + 0.1
    H = rng.rand(n_bases, T).astype(np.float64) + 0.1

    for _ in range(n_iter):
        WH = W @ H + 1e-10
        H *= (W.T @ (V / WH)) / (W.T @ np.ones_like(V) + 1e-10)
        WH = W @ H + 1e-10
        W *= ((V / WH) @ H.T) / (np.ones_like(V) @ H.T + 1e-10)

    return W, H


# ---------------------------------------------------------------------------
# Drum pattern isolation
# ---------------------------------------------------------------------------

@dataclass
class DrumHit:
    time_frame: int
    frequency_peak_hz: float
    energy: float
    drum_type: str   # "kick", "snare", "hihat", "other"


def classify_drum_hit(freq_peak: float) -> str:
    """Heuristic drum classification from spectral peak frequency."""
    if freq_peak < 150.0:
        return "kick"
    elif freq_peak < 500.0:
        return "snare"
    elif freq_peak > 5000.0:
        return "hihat"
    else:
        return "other"


def detect_drum_onsets(percussive_spectrogram: np.ndarray,
                       freqs: np.ndarray,
                       threshold: float = 0.3) -> list[DrumHit]:
    """Detect drum onsets from percussive-separated spectrogram.

    Args:
        percussive_spectrogram: output of HPSS percussive component
        freqs: frequency bin centers
        threshold: onset detection threshold (fraction of max energy)
    """
    # Onset strength: spectral flux of percussive component
    flux = np.zeros(percussive_spectrogram.shape[1])
    for t in range(1, percussive_spectrogram.shape[1]):
        diff = percussive_spectrogram[:, t] - percussive_spectrogram[:, t - 1]
        flux[t] = float(np.sum(np.maximum(diff, 0.0) ** 2))

    if np.max(flux) < 1e-10:
        return []

    flux_norm = flux / (np.max(flux) + 1e-30)
    onset_frames = np.where(flux_norm > threshold)[0]

    # Suppress close onsets (within 3 frames)
    if len(onset_frames) > 1:
        filtered = [onset_frames[0]]
        for f in onset_frames[1:]:
            if f - filtered[-1] >= 3:
                filtered.append(f)
        onset_frames = np.array(filtered)

    hits = []
    for frame in onset_frames:
        col = percussive_spectrogram[:, frame]
        peak_bin = int(np.argmax(col))
        peak_freq = float(freqs[peak_bin])
        energy = float(np.sum(col ** 2))
        hits.append(DrumHit(
            time_frame=int(frame),
            frequency_peak_hz=peak_freq,
            energy=energy,
            drum_type=classify_drum_hit(peak_freq),
        ))

    return hits


def drum_pattern_density(hits: list[DrumHit],
                         n_frames: int,
                         frame_rate: float) -> dict[str, float]:
    """Compute onset density per drum type (hits/second)."""
    duration = n_frames / (frame_rate + 1e-30)
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.drum_type] = counts.get(h.drum_type, 0) + 1
    return {k: v / duration for k, v in counts.items()}
