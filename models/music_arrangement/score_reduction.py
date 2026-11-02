"""Score reduction: piano reduction, lead sheet extraction, melody-accompaniment separation.

Algorithms for reducing complex multi-voice scores to simpler representations
while preserving harmonic and melodic content. Based on Huron's voice-leading
principles and Lerdahl & Jackendoff's reductional hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class ReductionType(Enum):
    PIANO = "piano"
    LEAD_SHEET = "lead_sheet"
    MELODY_ONLY = "melody_only"
    BASS_MELODY = "bass_melody"


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset: float
    duration: float
    velocity: int = 64
    voice: int = 0


@dataclass
class ReducedScore:
    notes: list[NoteEvent]
    reduction_type: ReductionType
    original_voice_count: int
    reduced_voice_count: int
    information_retained: float


def extract_melody(notes: Sequence[NoteEvent]) -> list[NoteEvent]:
    if not notes:
        return []
    time_slots: dict[float, list[NoteEvent]] = {}
    for n in notes:
        t = round(n.onset, 4)
        if t not in time_slots:
            time_slots[t] = []
        time_slots[t].append(n)
    melody = []
    for t in sorted(time_slots.keys()):
        group = time_slots[t]
        top = max(group, key=lambda n: n.pitch)
        melody.append(top)
    return melody


def extract_bass(notes: Sequence[NoteEvent]) -> list[NoteEvent]:
    if not notes:
        return []
    time_slots: dict[float, list[NoteEvent]] = {}
    for n in notes:
        t = round(n.onset, 4)
        if t not in time_slots:
            time_slots[t] = []
        time_slots[t].append(n)
    bass = []
    for t in sorted(time_slots.keys()):
        group = time_slots[t]
        bottom = min(group, key=lambda n: n.pitch)
        bass.append(bottom)
    return bass


def piano_reduction(notes: Sequence[NoteEvent], max_voices: int = 4) -> ReducedScore:
    if not notes:
        return ReducedScore([], ReductionType.PIANO, 0, 0, 0.0)
    original_voices = len(set(n.voice for n in notes))
    time_slots: dict[float, list[NoteEvent]] = {}
    for n in notes:
        t = round(n.onset, 4)
        if t not in time_slots:
            time_slots[t] = []
        time_slots[t].append(n)
    reduced = []
    for t in sorted(time_slots.keys()):
        group = time_slots[t]
        if len(group) <= max_voices:
            reduced.extend(group)
        else:
            sorted_group = sorted(group, key=lambda n: n.pitch)
            kept = set()
            kept.add(0)
            kept.add(len(sorted_group) - 1)
            remaining = max_voices - len(kept)
            if remaining > 0:
                middle = sorted_group[1:-1]
                middle_sorted = sorted(middle, key=lambda n: n.velocity, reverse=True)
                for m in middle_sorted[:remaining]:
                    idx = sorted_group.index(m)
                    kept.add(idx)
            for idx in sorted(kept):
                reduced.append(sorted_group[idx])
    info_retained = len(reduced) / max(len(notes), 1)
    return ReducedScore(
        notes=reduced,
        reduction_type=ReductionType.PIANO,
        original_voice_count=original_voices,
        reduced_voice_count=min(max_voices, original_voices),
        information_retained=float(info_retained),
    )


def lead_sheet_extraction(notes: Sequence[NoteEvent],
                           beat_duration: float = 0.5) -> ReducedScore:
    if not notes:
        return ReducedScore([], ReductionType.LEAD_SHEET, 0, 0, 0.0)
    original_voices = len(set(n.voice for n in notes))
    melody = extract_melody(notes)
    chord_times: dict[int, list[int]] = {}
    for n in notes:
        beat = int(n.onset / beat_duration)
        if beat not in chord_times:
            chord_times[beat] = []
        chord_times[beat].append(n.pitch % 12)
    result = list(melody)
    for beat, pcs in sorted(chord_times.items()):
        unique_pcs = list(set(pcs))
        if len(unique_pcs) >= 3:
            root = min(unique_pcs)
            chord_note = NoteEvent(
                pitch=root + 48,
                onset=beat * beat_duration,
                duration=beat_duration,
                velocity=40,
                voice=99,
            )
            result.append(chord_note)
    return ReducedScore(
        notes=result,
        reduction_type=ReductionType.LEAD_SHEET,
        original_voice_count=original_voices,
        reduced_voice_count=2,
        information_retained=len(result) / max(len(notes), 1),
    )


def voice_importance(notes: Sequence[NoteEvent]) -> dict[int, float]:
    if not notes:
        return {}
    voices: dict[int, list[NoteEvent]] = {}
    for n in notes:
        if n.voice not in voices:
            voices[n.voice] = []
        voices[n.voice].append(n)
    scores = {}
    all_pitches = [n.pitch for n in notes]
    max_pitch = max(all_pitches)
    min_pitch = min(all_pitches)
    pitch_range = max(max_pitch - min_pitch, 1)
    for v, v_notes in voices.items():
        mean_vel = np.mean([n.velocity for n in v_notes])
        mean_pitch = np.mean([n.pitch for n in v_notes])
        is_outer = (mean_pitch - min_pitch) / pitch_range
        outer_bonus = max(is_outer, 1.0 - is_outer) * 0.3
        scores[v] = float(mean_vel / 127.0 * 0.7 + outer_bonus)
    return scores


def texture_density(notes: Sequence[NoteEvent],
                     window_size: float = 1.0) -> np.ndarray:
    if not notes:
        return np.array([], dtype=np.float64)
    onsets = np.array([n.onset for n in notes])
    start = onsets.min()
    end = onsets.max() + window_size
    n_windows = max(1, int((end - start) / window_size))
    density = np.zeros(n_windows, dtype=np.float64)
    for i in range(n_windows):
        w_start = start + i * window_size
        w_end = w_start + window_size
        mask = (onsets >= w_start) & (onsets < w_end)
        density[i] = np.sum(mask)
    return density


def adaptive_thinning(notes: Sequence[NoteEvent],
                       target_density: float = 4.0,
                       window_size: float = 1.0) -> list[NoteEvent]:
    if not notes:
        return []
    density = texture_density(notes, window_size)
    onsets = np.array([n.onset for n in notes])
    start = onsets.min()
    result = []
    for i, d in enumerate(density):
        w_start = start + i * window_size
        w_end = w_start + window_size
        window_notes = [n for n in notes if w_start <= n.onset < w_end]
        if d <= target_density:
            result.extend(window_notes)
        else:
            keep_ratio = target_density / d
            sorted_notes = sorted(window_notes, key=lambda n: n.velocity, reverse=True)
            n_keep = max(1, int(len(sorted_notes) * keep_ratio))
            result.extend(sorted_notes[:n_keep])
    return result
