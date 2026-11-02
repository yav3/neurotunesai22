"""Texture analysis: classification, voice density, and textural complexity.

Quantifies musical texture using:
- Polyphonic/homophonic/monophonic classification
- Voice density curves and voice-count estimation
- Textural complexity via information-theoretic measures
- Onset synchrony for texture type discrimination
- Registral spread and voice spacing analysis
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class TextureType(Enum):
    """Basic texture classifications following standard music theory."""
    MONOPHONIC = "monophonic"
    HOMOPHONIC = "homophonic"
    POLYPHONIC = "polyphonic"
    HETEROPHONIC = "heterophonic"
    HOMORHYTHMIC = "homorhythmic"


@dataclass
class TextureSegment:
    """A segment of music with a classified texture."""
    start_time: float
    end_time: float
    texture_type: TextureType
    voice_count: int
    complexity: float
    registral_spread: float


@dataclass
class TextureProfile:
    """Complete texture analysis of a piece."""
    segments: List[TextureSegment] = field(default_factory=list)
    voice_density_curve: Optional[np.ndarray] = None
    complexity_curve: Optional[np.ndarray] = None
    mean_voices: float = 0.0
    dominant_texture: Optional[TextureType] = None
    texture_change_rate: float = 0.0


def estimate_voice_count(pitches_at_time: np.ndarray,
                          min_separation: float = 2.0) -> int:
    """Estimate the number of concurrent voices at a single time point.

    Clusters simultaneous pitches; notes within min_separation semitones
    are considered the same voice (unison/doubling).
    """
    if len(pitches_at_time) == 0:
        return 0
    sorted_p = np.sort(pitches_at_time)
    voices = 1
    for i in range(1, len(sorted_p)):
        if sorted_p[i] - sorted_p[i - 1] >= min_separation:
            voices += 1
    return voices


def voice_density_curve(pitch_matrix: np.ndarray,
                         min_separation: float = 2.0) -> np.ndarray:
    """Compute voice count over time from a pitch matrix.

    pitch_matrix: (n_voices, n_frames) — 0 or NaN indicates silence.
    Returns array of voice counts per frame.
    """
    n_voices, n_frames = pitch_matrix.shape
    density = np.zeros(n_frames)
    for t in range(n_frames):
        active = pitch_matrix[:, t]
        active = active[~np.isnan(active)] if np.any(np.isnan(active)) else active[active != 0]
        density[t] = estimate_voice_count(active, min_separation)
    return density


def onset_synchrony(onset_times: List[np.ndarray],
                     tolerance: float = 0.05) -> np.ndarray:
    """Measure onset synchrony between voices over time.

    onset_times: list of onset-time arrays, one per voice.
    Returns pairwise synchrony ratios.

    High synchrony → homophonic/homorhythmic; low → polyphonic.
    """
    n_voices = len(onset_times)
    if n_voices < 2:
        return np.array([1.0])
    sync_ratios = []
    for i in range(n_voices):
        for j in range(i + 1, n_voices):
            oi = onset_times[i]
            oj = onset_times[j]
            if len(oi) == 0 or len(oj) == 0:
                sync_ratios.append(0.0)
                continue
            matched = 0
            for t in oi:
                if np.any(np.abs(oj - t) <= tolerance):
                    matched += 1
            ratio = matched / max(len(oi), len(oj))
            sync_ratios.append(ratio)
    return np.array(sync_ratios)


def registral_spread(pitches_at_time: np.ndarray) -> float:
    """Range in semitones of all sounding pitches at a time point."""
    active = pitches_at_time[pitches_at_time > 0] if np.any(pitches_at_time > 0) else pitches_at_time[~np.isnan(pitches_at_time)]
    if len(active) < 2:
        return 0.0
    return float(np.max(active) - np.min(active))


def registral_spread_curve(pitch_matrix: np.ndarray) -> np.ndarray:
    """Compute registral spread over time."""
    _, n_frames = pitch_matrix.shape
    spread = np.zeros(n_frames)
    for t in range(n_frames):
        col = pitch_matrix[:, t]
        spread[t] = registral_spread(col)
    return spread


def voice_spacing(pitches_at_time: np.ndarray) -> np.ndarray:
    """Compute intervals between adjacent voices (bottom-up).

    Useful for detecting close/open voicing and spacing regularity.
    """
    active = np.sort(pitches_at_time[pitches_at_time > 0])
    if len(active) < 2:
        return np.array([])
    return np.diff(active)


def spacing_regularity(pitches_at_time: np.ndarray) -> float:
    """Measure how evenly spaced the voices are.

    Returns 1.0 for perfectly even spacing, 0.0 for maximally irregular.
    Based on coefficient of variation of inter-voice intervals.
    """
    intervals = voice_spacing(pitches_at_time)
    if len(intervals) < 2:
        return 1.0
    mean_iv = np.mean(intervals)
    if mean_iv < 1e-10:
        return 1.0
    cv = np.std(intervals) / mean_iv
    return float(np.exp(-cv))


def classify_texture_frame(voice_count: int,
                            sync_ratio: float,
                            melody_prominence: float = 0.0) -> TextureType:
    """Classify texture type for a single analysis frame.

    voice_count: estimated number of concurrent voices
    sync_ratio: mean onset synchrony across voice pairs [0,1]
    melody_prominence: ratio of melodic activity in top voice [0,1]
    """
    if voice_count <= 1:
        return TextureType.MONOPHONIC
    if sync_ratio > 0.85:
        return TextureType.HOMORHYTHMIC
    if sync_ratio > 0.6 and melody_prominence > 0.5:
        return TextureType.HOMOPHONIC
    if sync_ratio < 0.3:
        return TextureType.POLYPHONIC
    if voice_count == 2 and 0.3 <= sync_ratio <= 0.6:
        return TextureType.HETEROPHONIC
    if melody_prominence > 0.5:
        return TextureType.HOMOPHONIC
    return TextureType.POLYPHONIC


def pitch_class_entropy(pitches_at_time: np.ndarray) -> float:
    """Shannon entropy of pitch-class distribution at a time point.

    Higher entropy → more complex texture (more distinct pitch classes).
    """
    active = pitches_at_time[pitches_at_time > 0] if np.any(pitches_at_time > 0) else pitches_at_time[~np.isnan(pitches_at_time)]
    if len(active) == 0:
        return 0.0
    pcs = active.astype(int) % 12
    hist = np.zeros(12)
    for pc in pcs:
        hist[pc] += 1
    hist = hist / (hist.sum() + 1e-10)
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def textural_complexity(voice_count: int,
                         pc_entropy: float,
                         reg_spread: float,
                         sync_ratio: float) -> float:
    """Composite textural complexity measure.

    Combines voice count, pitch-class entropy, registral spread,
    and rhythmic independence (inverse synchrony).
    """
    norm_voices = min(voice_count / 6.0, 1.0)
    norm_entropy = min(pc_entropy / 3.58, 1.0)  # log2(12) ≈ 3.58
    norm_spread = min(reg_spread / 48.0, 1.0)  # 4 octaves
    independence = 1.0 - sync_ratio
    return float(
        0.25 * norm_voices +
        0.25 * norm_entropy +
        0.20 * norm_spread +
        0.30 * independence
    )


def analyze_texture(pitch_matrix: np.ndarray,
                     onset_times: Optional[List[np.ndarray]] = None,
                     hop_time: float = 0.05,
                     segment_min_frames: int = 20) -> TextureProfile:
    """Full texture analysis of a multi-voice pitch matrix.

    pitch_matrix: (n_voices, n_frames); 0 or NaN = silence.
    onset_times: optional per-voice onset arrays for synchrony.
    """
    n_voices, n_frames = pitch_matrix.shape
    density = voice_density_curve(pitch_matrix)

    if onset_times is not None and len(onset_times) >= 2:
        sync = onset_synchrony(onset_times)
        mean_sync = float(np.mean(sync))
    else:
        mean_sync = 0.5

    complexity_arr = np.zeros(n_frames)
    spread_arr = registral_spread_curve(pitch_matrix)
    frame_types: List[TextureType] = []

    for t in range(n_frames):
        col = pitch_matrix[:, t]
        pc_ent = pitch_class_entropy(col)
        vc = int(density[t])
        complexity_arr[t] = textural_complexity(vc, pc_ent, spread_arr[t], mean_sync)
        tt = classify_texture_frame(vc, mean_sync)
        frame_types.append(tt)

    segments: List[TextureSegment] = []
    if frame_types:
        seg_start = 0
        current_type = frame_types[0]
        for t in range(1, n_frames):
            if frame_types[t] != current_type or t == n_frames - 1:
                end = t if frame_types[t] != current_type else t + 1
                if end - seg_start >= segment_min_frames or t == n_frames - 1:
                    seg_density = density[seg_start:end]
                    seg_spread = spread_arr[seg_start:end]
                    seg_complex = complexity_arr[seg_start:end]
                    segments.append(TextureSegment(
                        start_time=seg_start * hop_time,
                        end_time=end * hop_time,
                        texture_type=current_type,
                        voice_count=int(round(np.mean(seg_density))),
                        complexity=float(np.mean(seg_complex)),
                        registral_spread=float(np.mean(seg_spread)),
                    ))
                    seg_start = t
                    current_type = frame_types[t]

    type_counts: dict = {}
    for tt in frame_types:
        type_counts[tt] = type_counts.get(tt, 0) + 1
    dominant = max(type_counts, key=type_counts.get) if type_counts else None

    changes = sum(1 for i in range(1, len(frame_types)) if frame_types[i] != frame_types[i - 1])
    change_rate = changes / (n_frames * hop_time) if n_frames > 0 else 0.0

    return TextureProfile(
        segments=segments,
        voice_density_curve=density,
        complexity_curve=complexity_arr,
        mean_voices=float(np.mean(density)),
        dominant_texture=dominant,
        texture_change_rate=change_rate,
    )


def melodic_prominence(pitch_matrix: np.ndarray,
                        voice_index: int = 0) -> np.ndarray:
    """Measure how prominent a single voice is relative to others.

    Returns per-frame prominence in [0, 1]. Voice at voice_index
    is prominent when it has the largest pitch range or most activity.
    """
    n_voices, n_frames = pitch_matrix.shape
    prominence = np.zeros(n_frames)
    for t in range(n_frames):
        target = pitch_matrix[voice_index, t]
        if target == 0 or np.isnan(target):
            continue
        all_active = pitch_matrix[:, t]
        active = all_active[(all_active > 0) & ~np.isnan(all_active)]
        if len(active) <= 1:
            prominence[t] = 1.0
            continue
        prominence[t] = float(target == np.max(active))
    return prominence


def texture_transition_matrix(frame_types: List[TextureType]) -> np.ndarray:
    """Compute transition probabilities between texture types.

    Returns a 5x5 matrix indexed by TextureType enum order.
    """
    types = list(TextureType)
    n = len(types)
    idx_map = {t: i for i, t in enumerate(types)}
    counts = np.zeros((n, n))
    for i in range(len(frame_types) - 1):
        a = idx_map.get(frame_types[i])
        b = idx_map.get(frame_types[i + 1])
        if a is not None and b is not None:
            counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums
