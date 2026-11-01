"""Signal quality metrics for audio and music signals.

Measures: SNR, segmental SNR, THD, THD+N, SINAD, ENOB,
noise floor estimation, dynamic range, crest factor,
inter-channel phase, and perceptual quality proxies.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class QualityReport:
    """Aggregated signal quality report."""
    snr_db: float
    thd_percent: float
    thd_plus_n_db: float
    sinad_db: float
    enob: float
    noise_floor_db: float
    dynamic_range_db: float
    crest_factor_db: float
    dc_offset: float


def snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio in dB given separate signal and noise."""
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise < 1e-30:
        return 120.0
    return 10.0 * np.log10(p_signal / p_noise)


def snr_from_noisy(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR from clean reference and noisy observation."""
    noise = noisy - clean
    return snr(clean, noise)


def segmental_snr(clean: np.ndarray, noisy: np.ndarray,
                  frame_size: int = 1024, hop: int = 512) -> float:
    """Segmental SNR — averaged per-frame SNR, more perceptually relevant.

    Frames with very low energy (< -60 dB re max) are excluded to
    avoid silence frames dominating the average.
    """
    n = len(clean)
    noise = noisy - clean
    max_energy = np.max(np.mean(clean ** 2) + 1e-30)
    threshold = max_energy * 1e-6  # -60 dB
    snrs = []
    for start in range(0, n - frame_size + 1, hop):
        s_frame = clean[start:start + frame_size]
        n_frame = noise[start:start + frame_size]
        p_s = np.mean(s_frame ** 2)
        p_n = np.mean(n_frame ** 2)
        if p_s < threshold:
            continue
        if p_n < 1e-30:
            snrs.append(60.0)
        else:
            val = 10.0 * np.log10(p_s / p_n)
            snrs.append(np.clip(val, -10.0, 60.0))
    if not snrs:
        return 0.0
    return float(np.mean(snrs))


def thd(signal: np.ndarray, sr: float, fundamental_hz: float,
        n_harmonics: int = 6) -> float:
    """Total Harmonic Distortion as a fraction (0–1).

    Measures the ratio of harmonic power to fundamental power.
    Uses DFT bin magnitudes with ±2 bin tolerance for each harmonic.
    """
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    bin_tol = 2

    def peak_power(target_hz: float) -> float:
        idx = np.argmin(np.abs(freqs - target_hz))
        lo = max(0, idx - bin_tol)
        hi = min(len(spectrum), idx + bin_tol + 1)
        return float(np.max(spectrum[lo:hi]) ** 2)

    p_fund = peak_power(fundamental_hz)
    if p_fund < 1e-30:
        return 0.0
    p_harm = 0.0
    for k in range(2, n_harmonics + 2):
        p_harm += peak_power(fundamental_hz * k)
    return float(np.sqrt(p_harm / p_fund))


def thd_plus_n(signal: np.ndarray, sr: float,
               fundamental_hz: float) -> float:
    """THD+N in dB: ratio of (harmonics + noise) to fundamental."""
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    bin_tol = 3
    fund_idx = np.argmin(np.abs(freqs - fundamental_hz))
    lo = max(0, fund_idx - bin_tol)
    hi = min(len(spectrum), fund_idx + bin_tol + 1)
    p_fund = float(np.sum(spectrum[lo:hi] ** 2))
    p_total = float(np.sum(spectrum ** 2))
    p_rest = p_total - p_fund
    if p_rest < 1e-30:
        return -120.0
    return 10.0 * np.log10(p_rest / max(p_fund, 1e-30))


def sinad(signal: np.ndarray, sr: float,
          fundamental_hz: float) -> float:
    """SINAD (Signal to Noise And Distortion) in dB.

    SINAD = 10·log10(P_signal / P_{noise+distortion}).
    """
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    bin_tol = 3
    fund_idx = np.argmin(np.abs(freqs - fundamental_hz))
    lo = max(0, fund_idx - bin_tol)
    hi = min(len(spectrum), fund_idx + bin_tol + 1)
    p_fund = float(np.sum(spectrum[lo:hi] ** 2))
    p_total = float(np.sum(spectrum ** 2))
    p_nad = p_total - p_fund
    if p_nad < 1e-30:
        return 120.0
    return 10.0 * np.log10(p_fund / p_nad)


def enob(sinad_db: float) -> float:
    """Effective Number of Bits from SINAD (IEEE 1241)."""
    return (sinad_db - 1.76) / 6.02


def noise_floor_estimate(signal: np.ndarray, sr: float,
                         percentile: float = 5.0) -> float:
    """Estimate noise floor in dB from the lowest spectral bins.

    Uses the given percentile of the magnitude spectrum as
    the noise floor proxy.
    """
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
    power_db = 20.0 * np.log10(spectrum + 1e-30)
    return float(np.percentile(power_db, percentile))


def dynamic_range_from_signal(signal: np.ndarray,
                              frame_size: int = 2048,
                              hop: int = 1024) -> float:
    """Dynamic range in dB: ratio of loudest to quietest frame RMS."""
    n = len(signal)
    rms_vals = []
    for start in range(0, n - frame_size + 1, hop):
        frame = signal[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > 1e-10:
            rms_vals.append(rms)
    if len(rms_vals) < 2:
        return 0.0
    return 20.0 * np.log10(max(rms_vals) / min(rms_vals))


def crest_factor(signal: np.ndarray) -> float:
    """Crest factor in dB: peak / RMS."""
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    if rms < 1e-30:
        return 0.0
    return 20.0 * np.log10(peak / rms)


def dc_offset(signal: np.ndarray) -> float:
    """DC offset as a fraction of peak amplitude."""
    peak = np.max(np.abs(signal))
    if peak < 1e-30:
        return 0.0
    return float(np.mean(signal) / peak)


def inter_channel_phase(left: np.ndarray, right: np.ndarray,
                        frame_size: int = 2048,
                        hop: int = 1024) -> np.ndarray:
    """Per-frame inter-channel phase difference (radians).

    Computed from the cross-spectrum of left and right channels.
    Useful for stereo quality assessment and phase correlation.
    """
    n = min(len(left), len(right))
    phases = []
    window = np.hanning(frame_size)
    for start in range(0, n - frame_size + 1, hop):
        l_frame = left[start:start + frame_size] * window
        r_frame = right[start:start + frame_size] * window
        L = np.fft.rfft(l_frame)
        R = np.fft.rfft(r_frame)
        cross = L * np.conj(R)
        mag = np.abs(cross)
        total = np.sum(mag)
        if total < 1e-30:
            phases.append(0.0)
        else:
            weighted_phase = np.sum(np.angle(cross) * mag) / total
            phases.append(float(weighted_phase))
    return np.array(phases)


def phase_correlation(left: np.ndarray, right: np.ndarray,
                      frame_size: int = 2048,
                      hop: int = 1024) -> np.ndarray:
    """Per-frame Pearson correlation between stereo channels.

    +1 = identical, 0 = uncorrelated, -1 = inverted.
    """
    n = min(len(left), len(right))
    corrs = []
    for start in range(0, n - frame_size + 1, hop):
        l_f = left[start:start + frame_size]
        r_f = right[start:start + frame_size]
        l_std = np.std(l_f)
        r_std = np.std(r_f)
        if l_std < 1e-10 or r_std < 1e-10:
            corrs.append(0.0)
        else:
            c = np.mean((l_f - np.mean(l_f)) * (r_f - np.mean(r_f))) / (l_std * r_std)
            corrs.append(float(c))
    return np.array(corrs)


def spectral_distortion(clean: np.ndarray, degraded: np.ndarray,
                        frame_size: int = 2048,
                        hop: int = 1024) -> float:
    """Log-spectral distance (dB) between clean and degraded signals.

    Average across frames of the RMS of log-spectral differences.
    Standard measure in speech/audio codec evaluation.
    """
    n = min(len(clean), len(degraded))
    window = np.hanning(frame_size)
    dists = []
    for start in range(0, n - frame_size + 1, hop):
        c_frame = clean[start:start + frame_size] * window
        d_frame = degraded[start:start + frame_size] * window
        C = np.abs(np.fft.rfft(c_frame)) + 1e-10
        D = np.abs(np.fft.rfft(d_frame)) + 1e-10
        log_diff = 20.0 * np.log10(C) - 20.0 * np.log10(D)
        dists.append(float(np.sqrt(np.mean(log_diff ** 2))))
    if not dists:
        return 0.0
    return float(np.mean(dists))


def itakura_saito_distance(clean: np.ndarray, degraded: np.ndarray,
                           frame_size: int = 2048,
                           hop: int = 1024) -> float:
    """Itakura-Saito divergence — perceptually motivated spectral distance.

    IS(P,Q) = P/Q - log(P/Q) - 1, averaged over bins and frames.
    Asymmetric; penalises underestimation more than overestimation.
    """
    n = min(len(clean), len(degraded))
    window = np.hanning(frame_size)
    vals = []
    for start in range(0, n - frame_size + 1, hop):
        c_frame = clean[start:start + frame_size] * window
        d_frame = degraded[start:start + frame_size] * window
        P = np.abs(np.fft.rfft(c_frame)) ** 2 + 1e-20
        Q = np.abs(np.fft.rfft(d_frame)) ** 2 + 1e-20
        ratio = P / Q
        is_val = ratio - np.log(ratio) - 1.0
        vals.append(float(np.mean(is_val)))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def clipping_detection(signal: np.ndarray,
                       threshold: float = 0.99) -> Tuple[int, float]:
    """Detect clipping: count of samples above threshold and percentage."""
    n = len(signal)
    peak = np.max(np.abs(signal))
    if peak < 1e-10:
        return 0, 0.0
    normed = np.abs(signal) / peak
    count = int(np.sum(normed >= threshold))
    return count, count / n


def quality_report(signal: np.ndarray, sr: float,
                   fundamental_hz: Optional[float] = None,
                   noise: Optional[np.ndarray] = None) -> QualityReport:
    """Generate a comprehensive signal quality report.

    If fundamental_hz is not provided, estimates it from the
    peak of the magnitude spectrum.
    """
    if fundamental_hz is None:
        n = len(signal)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        min_bin = max(1, int(20.0 * n / sr))
        fundamental_hz = float(freqs[min_bin + np.argmax(spectrum[min_bin:])])

    if noise is not None:
        snr_val = snr(signal, noise)
    else:
        snr_val = 0.0

    thd_val = thd(signal, sr, fundamental_hz)
    thdn_val = thd_plus_n(signal, sr, fundamental_hz)
    sinad_val = sinad(signal, sr, fundamental_hz)
    enob_val = enob(sinad_val)
    floor_val = noise_floor_estimate(signal, sr)
    dr_val = dynamic_range_from_signal(signal)
    cf_val = crest_factor(signal)
    dc_val = dc_offset(signal)

    return QualityReport(
        snr_db=snr_val,
        thd_percent=thd_val * 100.0,
        thd_plus_n_db=thdn_val,
        sinad_db=sinad_val,
        enob=enob_val,
        noise_floor_db=floor_val,
        dynamic_range_db=dr_val,
        crest_factor_db=cf_val,
        dc_offset=dc_val,
    )
