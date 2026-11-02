"""Quality-gating pipeline for audio ingestion.

Multi-stage validation with configurable thresholds that accepts,
flags, or rejects audio before it enters the feature-extraction
pipeline.  Each gate is a predicate on the signal or its summary
statistics; the pipeline short-circuits on the first hard rejection
unless ``collect_all=True``.

Gates cover:
- Format validation (sample rate, bit depth, channel count, duration)
- Signal integrity (clipping, DC offset, silence fraction, SNR)
- Spectral sanity (bandwidth floor, spectral flatness ceiling, energy)
- Loudness compliance (EBU R 128 range approximation, true peak)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Verdict taxonomy
# ---------------------------------------------------------------------------

class Verdict(enum.Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class GateCategory(enum.Enum):
    FORMAT = "format"
    INTEGRITY = "integrity"
    SPECTRAL = "spectral"
    LOUDNESS = "loudness"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Gate results
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    name: str
    category: GateCategory
    verdict: Verdict
    measured: float
    threshold: float
    message: str = ""


@dataclass
class GatingReport:
    results: List[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.verdict != Verdict.FAIL for r in self.results)

    @property
    def n_warnings(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.WARN)

    @property
    def n_failures(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.FAIL)

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "pass": sum(1 for r in self.results if r.verdict == Verdict.PASS),
            "warn": self.n_warnings,
            "fail": self.n_failures,
        }

    def failed_gates(self) -> List[GateResult]:
        return [r for r in self.results if r.verdict == Verdict.FAIL]

    def warning_gates(self) -> List[GateResult]:
        return [r for r in self.results if r.verdict == Verdict.WARN]


# ---------------------------------------------------------------------------
# Individual gate functions
# ---------------------------------------------------------------------------

def gate_sample_rate(
    signal: np.ndarray,
    sr: float,
    min_sr: float = 8000.0,
    max_sr: float = 192000.0,
) -> GateResult:
    """Reject signals with sample rates outside the valid range."""
    if sr < min_sr:
        return GateResult("sample_rate", GateCategory.FORMAT, Verdict.FAIL,
                          sr, min_sr, f"Sample rate {sr} Hz below minimum {min_sr} Hz")
    if sr > max_sr:
        return GateResult("sample_rate", GateCategory.FORMAT, Verdict.FAIL,
                          sr, max_sr, f"Sample rate {sr} Hz above maximum {max_sr} Hz")
    return GateResult("sample_rate", GateCategory.FORMAT, Verdict.PASS, sr, min_sr)


def gate_duration(
    signal: np.ndarray,
    sr: float,
    min_seconds: float = 0.5,
    max_seconds: float = 3600.0,
) -> GateResult:
    """Reject signals too short or too long."""
    dur = len(signal) / max(sr, 1.0)
    if dur < min_seconds:
        return GateResult("duration", GateCategory.FORMAT, Verdict.FAIL,
                          dur, min_seconds, f"Duration {dur:.2f}s below minimum {min_seconds}s")
    if dur > max_seconds:
        return GateResult("duration", GateCategory.FORMAT, Verdict.FAIL,
                          dur, max_seconds, f"Duration {dur:.2f}s above maximum {max_seconds}s")
    return GateResult("duration", GateCategory.FORMAT, Verdict.PASS, dur, min_seconds)


def gate_clipping(
    signal: np.ndarray,
    sr: float,
    max_clip_ratio: float = 0.01,
    warn_clip_ratio: float = 0.001,
) -> GateResult:
    """Check the fraction of samples at or near digital full scale."""
    threshold = 0.999
    n_clipped = int(np.sum(np.abs(signal) >= threshold))
    ratio = n_clipped / max(len(signal), 1)
    if ratio > max_clip_ratio:
        return GateResult("clipping", GateCategory.INTEGRITY, Verdict.FAIL,
                          ratio, max_clip_ratio,
                          f"Clipping ratio {ratio:.4f} exceeds {max_clip_ratio}")
    if ratio > warn_clip_ratio:
        return GateResult("clipping", GateCategory.INTEGRITY, Verdict.WARN,
                          ratio, warn_clip_ratio,
                          f"Clipping ratio {ratio:.4f} above warning threshold")
    return GateResult("clipping", GateCategory.INTEGRITY, Verdict.PASS,
                      ratio, max_clip_ratio)


def gate_dc_offset(
    signal: np.ndarray,
    sr: float,
    max_offset: float = 0.05,
) -> GateResult:
    """Warn if the DC offset exceeds a fraction of full scale."""
    dc = abs(float(np.mean(signal)))
    verdict = Verdict.WARN if dc > max_offset else Verdict.PASS
    return GateResult("dc_offset", GateCategory.INTEGRITY, verdict,
                      dc, max_offset,
                      f"DC offset {dc:.4f}" if dc > max_offset else "")


def gate_silence_fraction(
    signal: np.ndarray,
    sr: float,
    max_silence: float = 0.90,
    threshold_db: float = -50.0,
    frame_ms: float = 20.0,
) -> GateResult:
    """Reject if more than *max_silence* fraction of frames are silent."""
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    n_frames = max(1, len(signal) // frame_len)
    threshold_amp = 10.0 ** (threshold_db / 20.0)

    n_silent = 0
    for i in range(n_frames):
        frame = signal[i * frame_len:(i + 1) * frame_len]
        if np.sqrt(np.mean(frame ** 2)) < threshold_amp:
            n_silent += 1

    ratio = n_silent / n_frames
    if ratio > max_silence:
        return GateResult("silence", GateCategory.INTEGRITY, Verdict.FAIL,
                          ratio, max_silence,
                          f"Silence fraction {ratio:.2%} exceeds {max_silence:.0%}")
    return GateResult("silence", GateCategory.INTEGRITY, Verdict.PASS,
                      ratio, max_silence)


def gate_snr(
    signal: np.ndarray,
    sr: float,
    min_snr_db: float = 10.0,
    frame_ms: float = 20.0,
) -> GateResult:
    """Estimate SNR via the ratio of top-decile to bottom-decile frame energy."""
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    n_frames = max(1, len(signal) // frame_len)
    energies = np.array([
        np.mean(signal[i * frame_len:(i + 1) * frame_len] ** 2)
        for i in range(n_frames)
    ])
    energies = np.sort(energies)
    noise_floor = np.mean(energies[: max(1, n_frames // 10)]) + 1e-10
    signal_level = np.mean(energies[-max(1, n_frames // 10):]) + 1e-10
    snr_db = 10.0 * np.log10(signal_level / noise_floor)

    if snr_db < min_snr_db:
        return GateResult("snr", GateCategory.INTEGRITY, Verdict.FAIL,
                          snr_db, min_snr_db,
                          f"Estimated SNR {snr_db:.1f} dB below minimum {min_snr_db} dB")
    return GateResult("snr", GateCategory.INTEGRITY, Verdict.PASS,
                      snr_db, min_snr_db)


def gate_spectral_bandwidth(
    signal: np.ndarray,
    sr: float,
    min_bw_hz: float = 2000.0,
    frame_size: int = 2048,
) -> GateResult:
    """Reject signals with very narrow spectral bandwidth (e.g. pure tones, hum)."""
    window = np.hanning(frame_size)
    n_frames = max(1, (len(signal) - frame_size) // (frame_size // 2))
    bws = []
    for i in range(n_frames):
        start = i * (frame_size // 2)
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            break
        mag = np.abs(np.fft.rfft(frame * window))
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)
        total = np.sum(mag) + 1e-10
        centroid = np.sum(freqs * mag) / total
        bw = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / total)
        bws.append(bw)
    mean_bw = float(np.mean(bws)) if bws else 0.0
    if mean_bw < min_bw_hz:
        return GateResult("spectral_bw", GateCategory.SPECTRAL, Verdict.WARN,
                          mean_bw, min_bw_hz,
                          f"Mean spectral bandwidth {mean_bw:.0f} Hz below {min_bw_hz:.0f} Hz")
    return GateResult("spectral_bw", GateCategory.SPECTRAL, Verdict.PASS,
                      mean_bw, min_bw_hz)


def gate_spectral_flatness(
    signal: np.ndarray,
    sr: float,
    max_flatness: float = 0.95,
    frame_size: int = 2048,
) -> GateResult:
    """Warn if spectral flatness is very high (≈white noise)."""
    window = np.hanning(frame_size)
    n_frames = max(1, (len(signal) - frame_size) // (frame_size // 2))
    flatness_vals = []
    for i in range(n_frames):
        start = i * (frame_size // 2)
        frame = signal[start:start + frame_size]
        if len(frame) < frame_size:
            break
        mag = np.abs(np.fft.rfft(frame * window)) + 1e-10
        geometric = np.exp(np.mean(np.log(mag)))
        arithmetic = np.mean(mag)
        flatness_vals.append(geometric / arithmetic)
    mean_flat = float(np.mean(flatness_vals)) if flatness_vals else 0.0
    if mean_flat > max_flatness:
        return GateResult("spectral_flatness", GateCategory.SPECTRAL, Verdict.WARN,
                          mean_flat, max_flatness,
                          f"Spectral flatness {mean_flat:.3f} suggests noise-like content")
    return GateResult("spectral_flatness", GateCategory.SPECTRAL, Verdict.PASS,
                      mean_flat, max_flatness)


def gate_loudness_range(
    signal: np.ndarray,
    sr: float,
    min_lufs: float = -40.0,
    max_lufs: float = -5.0,
) -> GateResult:
    """Check RMS-approximated integrated loudness against an EBU R 128-like range."""
    rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
    lufs_approx = 20.0 * np.log10(rms)
    if lufs_approx < min_lufs:
        return GateResult("loudness", GateCategory.LOUDNESS, Verdict.FAIL,
                          lufs_approx, min_lufs,
                          f"Loudness {lufs_approx:.1f} dB below floor {min_lufs} dB")
    if lufs_approx > max_lufs:
        return GateResult("loudness", GateCategory.LOUDNESS, Verdict.WARN,
                          lufs_approx, max_lufs,
                          f"Loudness {lufs_approx:.1f} dB above ceiling {max_lufs} dB")
    return GateResult("loudness", GateCategory.LOUDNESS, Verdict.PASS,
                      lufs_approx, min_lufs)


def gate_true_peak(
    signal: np.ndarray,
    sr: float,
    max_dbtp: float = -1.0,
    oversample: int = 4,
) -> GateResult:
    """Estimate true peak via zero-order-hold oversampling."""
    n = len(signal)
    n_up = n * oversample
    x_old = np.arange(n)
    x_new = np.linspace(0, n - 1, n_up)
    upsampled = np.interp(x_new, x_old, signal)
    tp = float(np.max(np.abs(upsampled)))
    tp_db = 20.0 * np.log10(tp + 1e-10)
    if tp_db > max_dbtp:
        return GateResult("true_peak", GateCategory.LOUDNESS, Verdict.WARN,
                          tp_db, max_dbtp,
                          f"True peak {tp_db:.1f} dBTP exceeds {max_dbtp} dBTP")
    return GateResult("true_peak", GateCategory.LOUDNESS, Verdict.PASS,
                      tp_db, max_dbtp)


# ---------------------------------------------------------------------------
# Quality gate pipeline
# ---------------------------------------------------------------------------

GateFn = Callable[[np.ndarray, float], GateResult]


@dataclass
class QualityGateConfig:
    """Configuration for the quality-gating pipeline."""
    min_sr: float = 8000.0
    max_sr: float = 192000.0
    min_duration: float = 0.5
    max_duration: float = 3600.0
    max_clip_ratio: float = 0.01
    max_dc_offset: float = 0.05
    max_silence: float = 0.90
    min_snr_db: float = 10.0
    min_spectral_bw: float = 2000.0
    max_spectral_flatness: float = 0.95
    min_lufs: float = -40.0
    max_lufs: float = -5.0
    max_true_peak_dbtp: float = -1.0
    collect_all: bool = False


def build_gates(config: QualityGateConfig) -> List[GateFn]:
    """Build the ordered list of gate functions from a config."""
    return [
        lambda s, sr, c=config: gate_sample_rate(s, sr, c.min_sr, c.max_sr),
        lambda s, sr, c=config: gate_duration(s, sr, c.min_duration, c.max_duration),
        lambda s, sr, c=config: gate_clipping(s, sr, c.max_clip_ratio),
        lambda s, sr, c=config: gate_dc_offset(s, sr, c.max_dc_offset),
        lambda s, sr, c=config: gate_silence_fraction(s, sr, c.max_silence),
        lambda s, sr, c=config: gate_snr(s, sr, c.min_snr_db),
        lambda s, sr, c=config: gate_spectral_bandwidth(s, sr, c.min_spectral_bw),
        lambda s, sr, c=config: gate_spectral_flatness(s, sr, c.max_spectral_flatness),
        lambda s, sr, c=config: gate_loudness_range(s, sr, c.min_lufs, c.max_lufs),
        lambda s, sr, c=config: gate_true_peak(s, sr, c.max_true_peak_dbtp),
    ]


def run_quality_gates(
    signal: np.ndarray,
    sr: float = 22050,
    config: Optional[QualityGateConfig] = None,
    extra_gates: Optional[List[GateFn]] = None,
) -> GatingReport:
    """Run the full quality-gating pipeline.

    If ``config.collect_all`` is False (default), the pipeline
    short-circuits on the first FAIL verdict.
    """
    if config is None:
        config = QualityGateConfig()

    gates = build_gates(config)
    if extra_gates:
        gates.extend(extra_gates)

    report = GatingReport()
    for gate_fn in gates:
        result = gate_fn(signal, sr)
        report.results.append(result)
        if result.verdict == Verdict.FAIL and not config.collect_all:
            break
    return report


def batch_quality_gate(
    signals: List[Tuple[str, np.ndarray, float]],
    config: Optional[QualityGateConfig] = None,
) -> Dict[str, GatingReport]:
    """Run quality gates on a batch of named signals.

    Args:
        signals: list of (track_id, signal, sr) tuples
    Returns:
        dict mapping track_id → GatingReport
    """
    if config is None:
        config = QualityGateConfig(collect_all=True)
    results = {}
    for track_id, signal, sr in signals:
        results[track_id] = run_quality_gates(signal, sr, config)
    return results


def rejection_summary(reports: Dict[str, GatingReport]) -> Dict[str, object]:
    """Summarize batch quality gating: acceptance rate, common failure modes."""
    n_total = len(reports)
    n_passed = sum(1 for r in reports.values() if r.passed)
    n_warned = sum(1 for r in reports.values() if r.n_warnings > 0 and r.passed)

    failure_counts: Dict[str, int] = {}
    for report in reports.values():
        for gate in report.failed_gates():
            failure_counts[gate.name] = failure_counts.get(gate.name, 0) + 1

    return {
        "total": n_total,
        "passed": n_passed,
        "warned": n_warned,
        "rejected": n_total - n_passed,
        "acceptance_rate": n_passed / max(n_total, 1),
        "failure_modes": dict(sorted(failure_counts.items(),
                                     key=lambda x: x[1], reverse=True)),
    }
