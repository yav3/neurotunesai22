"""Spectrogram annotation: overlay musical events, harmonic partials, formants,
and structural boundaries onto time-frequency representations.

Pure numpy — generates annotation coordinate data suitable for any renderer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np


class AnnotationType(Enum):
    HARMONIC_PARTIAL = "harmonic_partial"
    FUNDAMENTAL = "fundamental"
    ONSET = "onset"
    OFFSET = "offset"
    FORMANT = "formant"
    PITCH_CONTOUR = "pitch_contour"
    SECTION_BOUNDARY = "section_boundary"
    NOTE_REGION = "note_region"
    SPECTRAL_PEAK = "spectral_peak"
    NOISE_BAND = "noise_band"


@dataclass
class Annotation:
    kind: AnnotationType
    time_start: float
    time_end: float
    freq_low: float
    freq_high: float
    label: str = ""
    confidence: float = 1.0
    color_idx: int = 0


@dataclass
class HarmonicSeries:
    fundamental_hz: float
    n_partials: int
    partial_amplitudes: List[float] = field(default_factory=list)
    time_start: float = 0.0
    time_end: float = 1.0


@dataclass
class PitchContour:
    times: np.ndarray
    frequencies: np.ndarray
    confidences: np.ndarray

    def duration(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return float(self.times[-1] - self.times[0])

    def mean_pitch(self) -> float:
        mask = self.confidences > 0.5
        if not np.any(mask):
            return 0.0
        return float(np.mean(self.frequencies[mask]))

    def vibrato_rate(self, min_rate: float = 4.0, max_rate: float = 8.0) -> float:
        mask = self.confidences > 0.5
        if np.sum(mask) < 8:
            return 0.0
        freqs = self.frequencies[mask]
        ts = self.times[mask]
        if ts[-1] - ts[0] < 0.2:
            return 0.0
        detrended = freqs - np.polyval(np.polyfit(ts, freqs, 1), ts)
        dt = np.mean(np.diff(ts))
        if dt <= 0:
            return 0.0
        fft_mag = np.abs(np.fft.rfft(detrended))
        fft_freqs = np.fft.rfftfreq(len(detrended), d=dt)
        band = (fft_freqs >= min_rate) & (fft_freqs <= max_rate)
        if not np.any(band):
            return 0.0
        peak_idx = np.argmax(fft_mag[band])
        return float(fft_freqs[band][peak_idx])

    def vibrato_extent_cents(self) -> float:
        mask = self.confidences > 0.5
        if np.sum(mask) < 4:
            return 0.0
        freqs = self.frequencies[mask]
        ts = self.times[mask]
        trend = np.polyval(np.polyfit(ts, freqs, 1), ts)
        deviations = 1200.0 * np.log2(np.maximum(freqs, 1e-6) / np.maximum(trend, 1e-6))
        return float(np.std(deviations) * 2.0)


@dataclass
class FormantTrack:
    times: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    bandwidths: Optional[np.ndarray] = None


def harmonic_series_annotations(
    series: HarmonicSeries,
) -> List[Annotation]:
    annotations = []
    f0 = series.fundamental_hz
    amps = series.partial_amplitudes or [1.0] * series.n_partials
    for i in range(series.n_partials):
        freq = f0 * (i + 1)
        amp = amps[i] if i < len(amps) else 0.0
        kind = AnnotationType.FUNDAMENTAL if i == 0 else AnnotationType.HARMONIC_PARTIAL
        bw = max(freq * 0.02, 5.0)
        annotations.append(Annotation(
            kind=kind,
            time_start=series.time_start,
            time_end=series.time_end,
            freq_low=freq - bw,
            freq_high=freq + bw,
            label=f"H{i+1}" if i > 0 else "F0",
            confidence=min(amp, 1.0),
            color_idx=i,
        ))
    return annotations


def onset_annotations(
    onset_times: Sequence[float],
    freq_range: Tuple[float, float] = (20.0, 20000.0),
    strengths: Optional[Sequence[float]] = None,
) -> List[Annotation]:
    out = []
    for i, t in enumerate(onset_times):
        s = strengths[i] if strengths and i < len(strengths) else 1.0
        out.append(Annotation(
            kind=AnnotationType.ONSET,
            time_start=t,
            time_end=t,
            freq_low=freq_range[0],
            freq_high=freq_range[1],
            label=f"onset_{i}",
            confidence=float(s),
        ))
    return out


def section_boundary_annotations(
    boundary_times: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    freq_range: Tuple[float, float] = (20.0, 20000.0),
) -> List[Annotation]:
    out = []
    for i, t in enumerate(boundary_times):
        lbl = labels[i] if labels and i < len(labels) else f"S{i}"
        out.append(Annotation(
            kind=AnnotationType.SECTION_BOUNDARY,
            time_start=t,
            time_end=t,
            freq_low=freq_range[0],
            freq_high=freq_range[1],
            label=lbl,
            confidence=1.0,
            color_idx=i % 8,
        ))
    return out


def pitch_contour_annotations(
    contour: PitchContour,
    confidence_threshold: float = 0.5,
    cents_bandwidth: float = 50.0,
) -> List[Annotation]:
    out = []
    mask = contour.confidences >= confidence_threshold
    if not np.any(mask):
        return out
    indices = np.where(mask)[0]
    segments: list[list[int]] = []
    current: list[int] = [indices[0]]
    for j in range(1, len(indices)):
        if indices[j] - indices[j - 1] <= 2:
            current.append(indices[j])
        else:
            segments.append(current)
            current = [indices[j]]
    segments.append(current)

    for seg in segments:
        ts = contour.times[seg]
        fs = contour.frequencies[seg]
        ratio = 2.0 ** (cents_bandwidth / 1200.0)
        out.append(Annotation(
            kind=AnnotationType.PITCH_CONTOUR,
            time_start=float(ts[0]),
            time_end=float(ts[-1]),
            freq_low=float(np.min(fs) / ratio),
            freq_high=float(np.max(fs) * ratio),
            label=f"{np.mean(fs):.1f}Hz",
            confidence=float(np.mean(contour.confidences[seg])),
        ))
    return out


def formant_annotations(
    track: FormantTrack,
    bandwidth_hz: float = 100.0,
) -> List[Annotation]:
    out = []
    for name, freqs in [("F1", track.f1), ("F2", track.f2), ("F3", track.f3)]:
        valid = freqs > 0
        if not np.any(valid):
            continue
        ts_valid = track.times[valid]
        fs_valid = freqs[valid]
        bw = bandwidth_hz / 2.0
        out.append(Annotation(
            kind=AnnotationType.FORMANT,
            time_start=float(ts_valid[0]),
            time_end=float(ts_valid[-1]),
            freq_low=float(np.min(fs_valid) - bw),
            freq_high=float(np.max(fs_valid) + bw),
            label=name,
            confidence=1.0,
        ))
    return out


def note_region_annotations(
    pitches_hz: Sequence[float],
    onsets: Sequence[float],
    offsets: Sequence[float],
    velocities: Optional[Sequence[float]] = None,
    cents_margin: float = 100.0,
) -> List[Annotation]:
    out = []
    ratio = 2.0 ** (cents_margin / 1200.0)
    for i, (p, on, off) in enumerate(zip(pitches_hz, onsets, offsets)):
        vel = velocities[i] if velocities and i < len(velocities) else 1.0
        out.append(Annotation(
            kind=AnnotationType.NOTE_REGION,
            time_start=float(on),
            time_end=float(off),
            freq_low=float(p / ratio),
            freq_high=float(p * ratio),
            label=_hz_to_note_name(p),
            confidence=float(vel),
            color_idx=int(round(12.0 * math.log2(p / 440.0))) % 12,
        ))
    return out


def spectral_peak_annotations(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    time_center: float,
    n_peaks: int = 10,
    min_prominence_db: float = 6.0,
) -> List[Annotation]:
    mag_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    peaks = _find_spectral_peaks(mag_db, min_prominence_db)
    peaks = sorted(peaks, key=lambda i: mag_db[i], reverse=True)[:n_peaks]
    out = []
    for idx in peaks:
        f = float(freqs[idx])
        bw = max(f * 0.01, 2.0)
        out.append(Annotation(
            kind=AnnotationType.SPECTRAL_PEAK,
            time_start=time_center,
            time_end=time_center,
            freq_low=f - bw,
            freq_high=f + bw,
            label=f"{f:.0f}Hz",
            confidence=float((mag_db[idx] - np.min(mag_db)) / max(np.ptp(mag_db), 1e-6)),
        ))
    return out


def noise_band_annotations(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    time_start: float,
    time_end: float,
    flatness_threshold: float = 0.8,
    band_width_bins: int = 32,
) -> List[Annotation]:
    mag = np.abs(spectrum)
    out = []
    for start in range(0, len(mag) - band_width_bins, band_width_bins // 2):
        band = mag[start:start + band_width_bins]
        if np.max(band) < 1e-12:
            continue
        geo_mean = np.exp(np.mean(np.log(np.maximum(band, 1e-12))))
        arith_mean = np.mean(band)
        flatness = geo_mean / max(arith_mean, 1e-12)
        if flatness >= flatness_threshold:
            out.append(Annotation(
                kind=AnnotationType.NOISE_BAND,
                time_start=time_start,
                time_end=time_end,
                freq_low=float(freqs[start]),
                freq_high=float(freqs[min(start + band_width_bins, len(freqs) - 1)]),
                label=f"noise ({flatness:.2f})",
                confidence=float(flatness),
            ))
    return out


def merge_annotations(
    *annotation_lists: List[Annotation],
) -> List[Annotation]:
    merged: list[Annotation] = []
    for al in annotation_lists:
        merged.extend(al)
    merged.sort(key=lambda a: (a.time_start, a.freq_low))
    return merged


def filter_annotations(
    annotations: List[Annotation],
    kinds: Optional[set] = None,
    time_range: Optional[Tuple[float, float]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    min_confidence: float = 0.0,
) -> List[Annotation]:
    out = []
    for a in annotations:
        if kinds and a.kind not in kinds:
            continue
        if time_range and (a.time_end < time_range[0] or a.time_start > time_range[1]):
            continue
        if freq_range and (a.freq_high < freq_range[0] or a.freq_low > freq_range[1]):
            continue
        if a.confidence < min_confidence:
            continue
        out.append(a)
    return out


def annotation_density(
    annotations: List[Annotation],
    time_bins: int = 100,
    freq_bins: int = 64,
    time_range: Optional[Tuple[float, float]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    if not annotations:
        return np.zeros((freq_bins, time_bins))
    if time_range is None:
        t0 = min(a.time_start for a in annotations)
        t1 = max(a.time_end for a in annotations)
        if t1 <= t0:
            t1 = t0 + 1.0
        time_range = (t0, t1)
    if freq_range is None:
        f0 = min(a.freq_low for a in annotations)
        f1 = max(a.freq_high for a in annotations)
        if f1 <= f0:
            f1 = f0 + 1.0
        freq_range = (f0, f1)

    density = np.zeros((freq_bins, time_bins))
    t_edges = np.linspace(time_range[0], time_range[1], time_bins + 1)
    f_edges = np.linspace(freq_range[0], freq_range[1], freq_bins + 1)

    for a in annotations:
        t_start_idx = max(0, np.searchsorted(t_edges, a.time_start) - 1)
        t_end_idx = min(time_bins, np.searchsorted(t_edges, a.time_end))
        f_start_idx = max(0, np.searchsorted(f_edges, a.freq_low) - 1)
        f_end_idx = min(freq_bins, np.searchsorted(f_edges, a.freq_high))
        density[f_start_idx:f_end_idx, t_start_idx:t_end_idx] += a.confidence

    return density


def _find_spectral_peaks(mag_db: np.ndarray, min_prom: float) -> List[int]:
    peaks = []
    for i in range(1, len(mag_db) - 1):
        if mag_db[i] > mag_db[i - 1] and mag_db[i] > mag_db[i + 1]:
            left_min = np.min(mag_db[max(0, i - 10):i])
            right_min = np.min(mag_db[i + 1:min(len(mag_db), i + 11)])
            prom = mag_db[i] - max(left_min, right_min)
            if prom >= min_prom:
                peaks.append(i)
    return peaks


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_note_name(hz: float) -> str:
    if hz <= 0:
        return "?"
    midi = 69.0 + 12.0 * math.log2(hz / 440.0)
    midi_round = int(round(midi))
    octave = (midi_round // 12) - 1
    note = midi_round % 12
    cents = (midi - midi_round) * 100.0
    name = _NOTE_NAMES[note]
    if abs(cents) < 5:
        return f"{name}{octave}"
    return f"{name}{octave}{cents:+.0f}c"
