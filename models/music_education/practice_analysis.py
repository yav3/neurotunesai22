"""Practice session analysis: repetition detection, spaced repetition, error patterns, difficulty estimation.

Provides SM-2-derived scheduling for musical passages, tempo consistency
metrics, practice efficiency curves, and feature-based difficulty scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal as sig


# ---------------------------------------------------------------------------
# Spaced repetition (SM-2 variant for musical passages)
# ---------------------------------------------------------------------------

@dataclass
class SM2State:
    easiness: float = 2.5
    interval_days: float = 1.0
    repetitions: int = 0
    next_review_day: float = 0.0


def sm2_update(state: SM2State, quality: int) -> SM2State:
    """Update SM-2 state given a quality grade 0..5.

    quality: 0 = complete blackout, 5 = perfect recall.
    Adapted from Wozniak's SM-2 with a minimum easiness of 1.3.
    """
    q = max(0, min(5, quality))
    ef = state.easiness + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02)
    ef = max(1.3, ef)

    if q < 3:
        reps = 0
        interval = 1.0
    else:
        reps = state.repetitions + 1
        if reps == 1:
            interval = 1.0
        elif reps == 2:
            interval = 6.0
        else:
            interval = state.interval_days * ef

    return SM2State(
        easiness=ef,
        interval_days=interval,
        repetitions=reps,
        next_review_day=state.next_review_day + interval,
    )


def schedule_passages(states: list[SM2State], current_day: float) -> list[int]:
    """Return indices of passages due for review on or before *current_day*."""
    return [i for i, s in enumerate(states) if s.next_review_day <= current_day]


# ---------------------------------------------------------------------------
# Error pattern detection
# ---------------------------------------------------------------------------

class ErrorType(Enum):
    WRONG_PITCH = "wrong_pitch"
    TIMING_EARLY = "timing_early"
    TIMING_LATE = "timing_late"
    DYNAMIC_OVER = "dynamic_over"
    DYNAMIC_UNDER = "dynamic_under"
    MISSED_NOTE = "missed_note"
    EXTRA_NOTE = "extra_note"


@dataclass
class NoteError:
    position: int
    error_type: ErrorType
    magnitude: float  # semitones for pitch, seconds for timing, dB for dynamics


def detect_pitch_errors(performed: np.ndarray, reference: np.ndarray,
                        tolerance_semitones: float = 0.5) -> list[NoteError]:
    """Compare performed pitches (MIDI) against reference, flag deviations."""
    n = min(len(performed), len(reference))
    errors: list[NoteError] = []
    for i in range(n):
        diff = performed[i] - reference[i]
        if abs(diff) > tolerance_semitones:
            errors.append(NoteError(i, ErrorType.WRONG_PITCH, float(diff)))
    for i in range(n, len(reference)):
        errors.append(NoteError(i, ErrorType.MISSED_NOTE, 0.0))
    for i in range(n, len(performed)):
        errors.append(NoteError(i, ErrorType.EXTRA_NOTE, 0.0))
    return errors


def detect_timing_errors(performed_onsets: np.ndarray,
                         reference_onsets: np.ndarray,
                         tolerance_s: float = 0.05) -> list[NoteError]:
    """Flag notes that arrive too early or too late relative to reference."""
    n = min(len(performed_onsets), len(reference_onsets))
    errors: list[NoteError] = []
    for i in range(n):
        diff = float(performed_onsets[i] - reference_onsets[i])
        if abs(diff) > tolerance_s:
            etype = ErrorType.TIMING_EARLY if diff < 0 else ErrorType.TIMING_LATE
            errors.append(NoteError(i, etype, abs(diff)))
    return errors


def detect_dynamic_errors(performed_vel: np.ndarray,
                          reference_vel: np.ndarray,
                          tolerance_db: float = 6.0) -> list[NoteError]:
    """Flag velocity/dynamic deviations beyond tolerance (in dB-like units)."""
    n = min(len(performed_vel), len(reference_vel))
    errors: list[NoteError] = []
    for i in range(n):
        p = max(performed_vel[i], 1e-10)
        r = max(reference_vel[i], 1e-10)
        diff_db = 20.0 * np.log10(p / r)
        if abs(diff_db) > tolerance_db:
            etype = ErrorType.DYNAMIC_OVER if diff_db > 0 else ErrorType.DYNAMIC_UNDER
            errors.append(NoteError(i, etype, float(abs(diff_db))))
    return errors


# ---------------------------------------------------------------------------
# Tempo consistency
# ---------------------------------------------------------------------------

@dataclass
class TempoAnalysis:
    mean_bpm: float
    std_bpm: float
    drift_bpm_per_beat: float  # linear trend slope
    rubato_index: float        # CV of local tempo
    metronome_score: float     # 0..1 adherence


def analyze_tempo_consistency(onset_times: np.ndarray,
                              target_bpm: float | None = None) -> TempoAnalysis:
    """Analyze tempo consistency from onset times (seconds)."""
    if len(onset_times) < 3:
        return TempoAnalysis(0.0, 0.0, 0.0, 0.0, 0.0)

    ioi = np.diff(onset_times)
    ioi = ioi[ioi > 0.01]  # filter artifacts
    if len(ioi) < 2:
        return TempoAnalysis(0.0, 0.0, 0.0, 0.0, 0.0)

    local_bpm = 60.0 / ioi
    mean_bpm = float(np.mean(local_bpm))
    std_bpm = float(np.std(local_bpm))

    # Linear drift
    x = np.arange(len(local_bpm), dtype=np.float64)
    if len(x) > 1:
        coeffs = np.polyfit(x, local_bpm, 1)
        drift = float(coeffs[0])
    else:
        drift = 0.0

    rubato = std_bpm / (mean_bpm + 1e-10)

    if target_bpm is not None and target_bpm > 0:
        deviations = np.abs(local_bpm - target_bpm) / target_bpm
        metronome = float(1.0 - np.clip(np.mean(deviations), 0.0, 1.0))
    else:
        metronome = float(1.0 - np.clip(rubato, 0.0, 1.0))

    return TempoAnalysis(
        mean_bpm=mean_bpm,
        std_bpm=std_bpm,
        drift_bpm_per_beat=drift,
        rubato_index=rubato,
        metronome_score=metronome,
    )


# ---------------------------------------------------------------------------
# Practice efficiency
# ---------------------------------------------------------------------------

@dataclass
class PracticeBlock:
    start_idx: int
    end_idx: int
    repetitions: int
    error_rate_start: float
    error_rate_end: float
    improvement: float  # positive = better


def segment_practice_blocks(error_flags: np.ndarray,
                            window: int = 8,
                            overlap: int = 4) -> list[PracticeBlock]:
    """Segment a practice session into repetition blocks based on error density."""
    n = len(error_flags)
    if n < window:
        rate = float(np.mean(error_flags))
        return [PracticeBlock(0, n, 1, rate, rate, 0.0)]

    step = max(1, window - overlap)
    blocks: list[PracticeBlock] = []
    i = 0
    while i + window <= n:
        chunk = error_flags[i : i + window]
        rate = float(np.mean(chunk))
        blocks.append(PracticeBlock(
            start_idx=i, end_idx=i + window, repetitions=1,
            error_rate_start=rate, error_rate_end=rate, improvement=0.0,
        ))
        i += step

    # Merge adjacent blocks with similar error rates → repetition groups
    merged: list[PracticeBlock] = []
    for b in blocks:
        if merged and abs(b.error_rate_start - merged[-1].error_rate_end) < 0.15:
            merged[-1].end_idx = b.end_idx
            merged[-1].repetitions += 1
            merged[-1].error_rate_end = b.error_rate_end
            merged[-1].improvement = merged[-1].error_rate_start - b.error_rate_end
        else:
            merged.append(b)

    return merged


def improvement_curve(error_rates: np.ndarray) -> np.ndarray:
    """Fit a learning curve: error(t) ≈ a · t^(-b) + c.

    Returns smoothed error rates (same length as input).
    """
    n = len(error_rates)
    if n < 3:
        return error_rates.copy()
    kernel = np.ones(min(5, n)) / min(5, n)
    return np.convolve(error_rates, kernel, mode="same")


def diminishing_returns_point(error_rates: np.ndarray,
                              threshold: float = 0.01) -> int:
    """Find the repetition index where improvement per rep drops below threshold."""
    if len(error_rates) < 3:
        return len(error_rates)
    smoothed = improvement_curve(error_rates)
    diffs = -np.diff(smoothed)  # positive = improving
    for i, d in enumerate(diffs):
        if d < threshold:
            return i
    return len(error_rates)


# ---------------------------------------------------------------------------
# Difficulty estimation from musical features
# ---------------------------------------------------------------------------

@dataclass
class DifficultyEstimate:
    note_density: float        # notes per second
    interval_complexity: float # mean absolute interval in semitones
    rhythmic_irregularity: float  # entropy of quantised durations
    key_difficulty: float      # 0..1 based on accidental count
    overall: float             # weighted composite 0..1


def estimate_difficulty(pitches: np.ndarray,
                        onsets: np.ndarray,
                        durations: np.ndarray | None = None,
                        key_accidentals: int = 0) -> DifficultyEstimate:
    """Estimate passage difficulty from note-level features.

    pitches: MIDI note numbers
    onsets: onset times in seconds
    durations: note durations in seconds (optional)
    key_accidentals: number of sharps/flats in key signature (0..7)
    """
    if len(pitches) < 2 or len(onsets) < 2:
        return DifficultyEstimate(0.0, 0.0, 0.0, 0.0, 0.0)

    span = float(onsets[-1] - onsets[0])
    note_density = len(pitches) / max(span, 0.1)
    # Normalize: 8 notes/s = 1.0
    nd_norm = min(note_density / 8.0, 1.0)

    intervals = np.abs(np.diff(pitches))
    interval_complexity = float(np.mean(intervals))
    # Normalize: mean interval 12 semitones = 1.0
    ic_norm = min(interval_complexity / 12.0, 1.0)

    # Rhythmic irregularity via entropy
    if durations is not None and len(durations) > 1:
        quantised = np.round(durations * 8) / 8  # quantise to 32nd notes
        unique, counts = np.unique(quantised, return_counts=True)
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = np.log2(max(len(unique), 2))
        ri_norm = entropy / max(max_entropy, 1e-10)
    else:
        ioi = np.diff(onsets)
        cv = float(np.std(ioi) / (np.mean(ioi) + 1e-10))
        ri_norm = min(cv, 1.0)

    key_diff = min(key_accidentals / 7.0, 1.0)

    overall = 0.3 * nd_norm + 0.25 * ic_norm + 0.25 * ri_norm + 0.2 * key_diff

    return DifficultyEstimate(
        note_density=note_density,
        interval_complexity=interval_complexity,
        rhythmic_irregularity=ri_norm,
        key_difficulty=key_diff,
        overall=float(np.clip(overall, 0.0, 1.0)),
    )
