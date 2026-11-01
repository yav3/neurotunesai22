"""Notation intelligence: beam grouping, tie/slur inference, spacing, voice separation.

Implements music engraving heuristics drawn from Gould (2011) *Behind Bars*
and Read (1979) *Music Notation*: automatic beam grouping by meter, tie
inference across bar lines, voice allocation in polyphonic staves, and
proportional horizontal spacing.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Sequence
from enum import Enum


class NoteValue(Enum):
    WHOLE = 4.0
    HALF = 2.0
    QUARTER = 1.0
    EIGHTH = 0.5
    SIXTEENTH = 0.25
    THIRTY_SECOND = 0.125


class ClefType(Enum):
    TREBLE = "treble"
    BASS = "bass"
    ALTO = "alto"
    TENOR = "tenor"


@dataclass
class NoteEvent:
    pitch: int          # MIDI
    onset_beat: float
    duration_beats: float
    voice: int = 0
    tied_to_next: bool = False
    beam_group: int = -1


@dataclass
class BeamGroup:
    notes: List[int]    # indices into a note list
    level: int          # 1 = single beam (eighths), 2 = double (sixteenths), etc.


@dataclass
class SpacingResult:
    x_positions: List[float]
    bar_widths: List[float]
    total_width: float


def quantise_duration(beats: float, grid: float = 0.25) -> float:
    return round(beats / grid) * grid


def duration_to_note_value(beats: float) -> Optional[NoteValue]:
    for nv in NoteValue:
        if abs(beats - nv.value) < 0.01:
            return nv
    return None


def needs_tie(onset: float, duration: float, bar_length: float) -> bool:
    bar_start = (onset // bar_length) * bar_length
    bar_end = bar_start + bar_length
    note_end = onset + duration
    return note_end > bar_end + 0.01


def split_tied_note(onset: float, duration: float, bar_length: float) -> List[Tuple[float, float]]:
    segments = []
    remaining = duration
    pos = onset
    while remaining > 0.01:
        bar_start = (pos // bar_length) * bar_length
        bar_end = bar_start + bar_length
        space = bar_end - pos
        seg_dur = min(remaining, space) if space > 0.01 else remaining
        segments.append((pos, quantise_duration(seg_dur)))
        pos += seg_dur
        remaining -= seg_dur
    return segments


def infer_ties(notes: List[NoteEvent], bar_length: float = 4.0) -> List[NoteEvent]:
    result = []
    for note in notes:
        if needs_tie(note.onset_beat, note.duration_beats, bar_length):
            segments = split_tied_note(note.onset_beat, note.duration_beats, bar_length)
            for i, (onset, dur) in enumerate(segments):
                result.append(NoteEvent(
                    pitch=note.pitch, onset_beat=onset, duration_beats=dur,
                    voice=note.voice, tied_to_next=(i < len(segments) - 1),
                ))
        else:
            result.append(note)
    return result


def beam_group_notes(notes: List[NoteEvent], beats_per_bar: int = 4,
                     subdivision: float = 1.0) -> List[BeamGroup]:
    beamable = [(i, n) for i, n in enumerate(notes) if n.duration_beats < 1.0]
    groups: List[BeamGroup] = []
    current_group: List[int] = []
    current_beat_group = -1
    for idx, note in beamable:
        beat_group = int(note.onset_beat // subdivision)
        if beat_group != current_beat_group:
            if len(current_group) >= 2:
                level = 1 if all(notes[i].duration_beats >= 0.25 for i in current_group) else 2
                groups.append(BeamGroup(notes=current_group, level=level))
            current_group = [idx]
            current_beat_group = beat_group
        else:
            current_group.append(idx)
    if len(current_group) >= 2:
        level = 1 if all(notes[i].duration_beats >= 0.25 for i in current_group) else 2
        groups.append(BeamGroup(notes=current_group, level=level))
    return groups


def assign_beam_groups(notes: List[NoteEvent], beats_per_bar: int = 4) -> List[NoteEvent]:
    groups = beam_group_notes(notes, beats_per_bar)
    result = [NoteEvent(n.pitch, n.onset_beat, n.duration_beats, n.voice, n.tied_to_next, -1)
              for n in notes]
    for gid, grp in enumerate(groups):
        for idx in grp.notes:
            result[idx].beam_group = gid
    return result


def voice_separate(notes: List[NoteEvent], max_voices: int = 4) -> List[NoteEvent]:
    sorted_notes = sorted(range(len(notes)), key=lambda i: (notes[i].onset_beat, -notes[i].pitch))
    voice_ends = [0.0] * max_voices
    result = [NoteEvent(n.pitch, n.onset_beat, n.duration_beats, 0, n.tied_to_next, n.beam_group)
              for n in notes]
    for idx in sorted_notes:
        note = notes[idx]
        best_voice = 0
        best_gap = float('inf')
        for v in range(max_voices):
            if note.onset_beat >= voice_ends[v] - 0.01:
                gap = note.onset_beat - voice_ends[v]
                if gap < best_gap:
                    best_gap = gap
                    best_voice = v
        result[idx].voice = best_voice
        voice_ends[best_voice] = note.onset_beat + note.duration_beats
    return result


def proportional_spacing(durations: List[float], min_width: float = 10.0,
                          base_width: float = 40.0) -> SpacingResult:
    if not durations:
        return SpacingResult([], [], 0.0)
    log_durs = [np.log2(max(d, 0.125)) + 4.0 for d in durations]
    max_log = max(log_durs) if log_durs else 1.0
    widths = [max(min_width, base_width * (ld / max(max_log, 1e-10))) for ld in log_durs]
    positions = []
    x = 0.0
    for w in widths:
        positions.append(x)
        x += w
    return SpacingResult(x_positions=positions, bar_widths=widths, total_width=x)


def optimal_clef(pitches: List[int]) -> ClefType:
    if not pitches:
        return ClefType.TREBLE
    median_pitch = float(np.median(pitches))
    if median_pitch >= 60:
        return ClefType.TREBLE
    elif median_pitch >= 48:
        return ClefType.ALTO
    else:
        return ClefType.BASS


def detect_clef_changes(pitches: List[int], onsets: List[float],
                         window_beats: float = 8.0) -> List[Tuple[float, ClefType]]:
    if not pitches:
        return [(0.0, ClefType.TREBLE)]
    changes = []
    current_clef = optimal_clef(pitches[:1])
    changes.append((0.0, current_clef))
    i = 0
    while i < len(pitches):
        window_pitches = []
        j = i
        while j < len(pitches) and onsets[j] < onsets[i] + window_beats:
            window_pitches.append(pitches[j])
            j += 1
        new_clef = optimal_clef(window_pitches)
        if new_clef != current_clef:
            changes.append((onsets[i], new_clef))
            current_clef = new_clef
        i = j if j > i else i + 1
    return changes


def note_collision_detect(notes: List[NoteEvent]) -> List[Tuple[int, int]]:
    collisions = []
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            if notes[i].voice == notes[j].voice:
                continue
            time_overlap = (notes[i].onset_beat < notes[j].onset_beat + notes[j].duration_beats and
                           notes[j].onset_beat < notes[i].onset_beat + notes[i].duration_beats)
            pitch_close = abs(notes[i].pitch - notes[j].pitch) <= 1
            if time_overlap and pitch_close:
                collisions.append((i, j))
    return collisions


def stem_direction(pitch: int, clef: ClefType = ClefType.TREBLE) -> int:
    mid = 71 if clef == ClefType.TREBLE else 50
    return -1 if pitch >= mid else 1


def accidental_needed(pitch: int, key_sig_sharps: int = 0) -> bool:
    pc = pitch % 12
    white_keys = {0, 2, 4, 5, 7, 9, 11}
    sharp_order = [6, 1, 8, 3, 10, 5, 0]
    sharped = set()
    for i in range(abs(key_sig_sharps)):
        if key_sig_sharps > 0 and i < len(sharp_order):
            sharped.add(sharp_order[i])
        elif key_sig_sharps < 0 and i < len(sharp_order):
            sharped.add((sharp_order[i] - 1) % 12)
    if pc in sharped:
        return False
    return pc not in white_keys
