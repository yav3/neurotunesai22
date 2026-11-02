"""Performance expression analysis: rubato, dynamics, articulation, vibrato."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ArticulationType(str, Enum):
    LEGATO = "legato"
    STACCATO = "staccato"
    ACCENT = "accent"
    TENUTO = "tenuto"
    MARCATO = "marcato"
    PORTATO = "portato"
    NORMAL = "normal"


@dataclass
class NoteEvent:
    onset: float
    offset: float
    pitch: int
    velocity: int
    score_onset: Optional[float] = None
    score_duration: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    @property
    def timing_deviation(self) -> Optional[float]:
        if self.score_onset is None:
            return None
        return self.onset - self.score_onset


@dataclass
class RubatoSegment:
    start_time: float
    end_time: float
    tempo_curve: np.ndarray
    mean_tempo: float
    tempo_variance: float
    rubato_intensity: float


def detect_rubato(onsets: np.ndarray, score_onsets: np.ndarray,
                  window_size: int = 8) -> List[RubatoSegment]:
    if len(onsets) < window_size + 1 or len(score_onsets) < window_size + 1:
        return []

    deviations = onsets[:len(score_onsets)] - score_onsets[:len(onsets)]
    ioi_perf = np.diff(onsets)
    ioi_score = np.diff(score_onsets[:len(onsets)])
    mask = ioi_score > 1e-6
    tempo_ratio = np.ones_like(ioi_perf)
    tempo_ratio[mask] = ioi_score[mask] / ioi_perf[mask]

    segments = []
    half_w = window_size // 2
    for i in range(half_w, len(tempo_ratio) - half_w):
        window = tempo_ratio[i - half_w:i + half_w]
        mean_t = float(window.mean())
        var_t = float(window.var())
        intensity = math.sqrt(var_t) / mean_t if mean_t > 1e-6 else 0.0

        if intensity > 0.05:
            segments.append(RubatoSegment(
                start_time=float(onsets[i - half_w]),
                end_time=float(onsets[i + half_w]),
                tempo_curve=window.copy(),
                mean_tempo=mean_t,
                tempo_variance=var_t,
                rubato_intensity=intensity,
            ))

    return _merge_rubato_segments(segments)


def _merge_rubato_segments(segments: List[RubatoSegment]) -> List[RubatoSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.start_time <= merged[-1].end_time:
            merged[-1] = RubatoSegment(
                start_time=merged[-1].start_time,
                end_time=max(merged[-1].end_time, seg.end_time),
                tempo_curve=np.concatenate([merged[-1].tempo_curve, seg.tempo_curve]),
                mean_tempo=(merged[-1].mean_tempo + seg.mean_tempo) / 2,
                tempo_variance=max(merged[-1].tempo_variance, seg.tempo_variance),
                rubato_intensity=max(merged[-1].rubato_intensity, seg.rubato_intensity),
            )
        else:
            merged.append(seg)
    return merged


def rubato_amount(onsets: np.ndarray, score_onsets: np.ndarray) -> float:
    if len(onsets) < 2 or len(score_onsets) < 2:
        return 0.0
    n = min(len(onsets), len(score_onsets))
    deviations = onsets[:n] - score_onsets[:n]
    return float(np.std(deviations))


@dataclass
class DynamicShape:
    start_time: float
    end_time: float
    start_level: float
    end_level: float
    shape_type: str
    fit_r_squared: float

    @property
    def range_db(self) -> float:
        return abs(self.end_level - self.start_level)


def detect_dynamic_shapes(velocities: np.ndarray, times: np.ndarray,
                          min_length: int = 4,
                          min_change_db: float = 6.0) -> List[DynamicShape]:
    if len(velocities) < min_length:
        return []

    vel_db = 20.0 * np.log10(np.clip(velocities / 127.0, 1e-6, 1.0))
    shapes = []

    i = 0
    while i < len(vel_db) - min_length:
        for length in range(min_length, min(len(vel_db) - i + 1, 32)):
            segment = vel_db[i:i + length]
            x = np.arange(length, dtype=float)
            coeffs = np.polyfit(x, segment, 1)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((segment - y_pred) ** 2)
            ss_tot = np.sum((segment - segment.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            total_change = coeffs[0] * (length - 1)
            if abs(total_change) >= min_change_db and r2 > 0.5:
                shape_type = "crescendo" if coeffs[0] > 0 else "decrescendo"
                shapes.append(DynamicShape(
                    start_time=float(times[i]),
                    end_time=float(times[i + length - 1]),
                    start_level=float(segment[0]),
                    end_level=float(segment[-1]),
                    shape_type=shape_type,
                    fit_r_squared=float(r2),
                ))
                i += length
                break
        else:
            i += 1

    return shapes


def dynamic_range(velocities: np.ndarray) -> float:
    if len(velocities) == 0:
        return 0.0
    v_clip = np.clip(velocities / 127.0, 1e-6, 1.0)
    db = 20.0 * np.log10(v_clip)
    return float(db.max() - db.min())


def classify_articulation(note: NoteEvent, next_note: Optional[NoteEvent] = None,
                          nominal_duration: Optional[float] = None) -> ArticulationType:
    if nominal_duration is None:
        nominal_duration = note.duration

    ratio = note.duration / nominal_duration if nominal_duration > 0 else 1.0

    if note.velocity > 110:
        if ratio < 0.6:
            return ArticulationType.MARCATO
        return ArticulationType.ACCENT

    if ratio < 0.4:
        return ArticulationType.STACCATO

    if ratio > 1.1:
        return ArticulationType.TENUTO

    if next_note is not None:
        gap = next_note.onset - note.offset
        if gap < 0.02:
            return ArticulationType.LEGATO
        if 0.6 < ratio < 0.85:
            return ArticulationType.PORTATO

    return ArticulationType.NORMAL


def articulation_profile(notes: List[NoteEvent]) -> Dict[str, float]:
    if not notes:
        return {}
    types = []
    for i, note in enumerate(notes):
        next_n = notes[i + 1] if i + 1 < len(notes) else None
        art = classify_articulation(note, next_n)
        types.append(art.value)
    counts: Dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    return {k: v / len(types) for k, v in counts.items()}


@dataclass
class VibratoAnalysis:
    rate_hz: float
    extent_cents: float
    onset_delay: float
    regularity: float


def analyze_vibrato(pitch_curve: np.ndarray, sr: float,
                    min_rate: float = 4.0, max_rate: float = 8.0) -> Optional[VibratoAnalysis]:
    if len(pitch_curve) < int(sr * 0.25):
        return None

    centered = pitch_curve - np.mean(pitch_curve)
    fft = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sr)
    magnitudes = np.abs(fft)

    mask = (freqs >= min_rate) & (freqs <= max_rate)
    if not np.any(mask):
        return None

    vib_magnitudes = magnitudes.copy()
    vib_magnitudes[~mask] = 0
    peak_idx = np.argmax(vib_magnitudes)
    peak_mag = magnitudes[peak_idx]

    total_energy = np.sum(magnitudes[mask] ** 2)
    if total_energy < 1e-12:
        return None

    rate = float(freqs[peak_idx])
    extent = float(np.std(centered) * 2.0 * 100.0)

    threshold = 0.1 * np.max(np.abs(centered))
    onset_samples = 0
    for i, v in enumerate(np.abs(centered)):
        if v > threshold:
            onset_samples = i
            break
    onset_delay = onset_samples / sr

    cycle_len = int(sr / rate) if rate > 0 else len(centered)
    if cycle_len > 0:
        n_cycles = len(centered) // cycle_len
        if n_cycles >= 2:
            cycle_peaks = []
            for c in range(n_cycles):
                seg = centered[c * cycle_len:(c + 1) * cycle_len]
                cycle_peaks.append(float(np.max(np.abs(seg))))
            regularity = 1.0 - (np.std(cycle_peaks) / np.mean(cycle_peaks)
                                if np.mean(cycle_peaks) > 0 else 0.0)
        else:
            regularity = 0.5
    else:
        regularity = 0.5

    return VibratoAnalysis(
        rate_hz=rate,
        extent_cents=extent,
        onset_delay=onset_delay,
        regularity=float(max(0.0, min(1.0, regularity))),
    )


def expressive_timing_deviation(notes: List[NoteEvent]) -> Dict[str, float]:
    deviations = [n.timing_deviation for n in notes if n.timing_deviation is not None]
    if not deviations:
        return {"mean": 0.0, "std": 0.0, "max_early": 0.0, "max_late": 0.0}
    arr = np.array(deviations)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max_early": float(arr.min()),
        "max_late": float(arr.max()),
    }
