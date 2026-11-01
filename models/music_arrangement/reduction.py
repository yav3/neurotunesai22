"""Score reduction — full score to piano reduction, lead-sheet extraction, part condensation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ScoreNote:
    pitch: int
    onset: float
    duration: float
    voice: int = 0
    dynamic: float = 0.7


@dataclass
class PianoReduction:
    right_hand: list[ScoreNote]
    left_hand: list[ScoreNote]
    hand_span_right: int = 0
    hand_span_left: int = 0


def piano_reduce(notes: list[ScoreNote], split_point: int = 60) -> PianoReduction:
    if not notes:
        return PianoReduction([], [])
    rh = [n for n in notes if n.pitch >= split_point]
    lh = [n for n in notes if n.pitch < split_point]
    rh = _limit_span(rh, max_span=12)
    lh = _limit_span(lh, max_span=14)
    span_r = _hand_span(rh)
    span_l = _hand_span(lh)
    return PianoReduction(rh, lh, span_r, span_l)


def _hand_span(notes: list[ScoreNote]) -> int:
    if not notes:
        return 0
    groups: dict[float, list[int]] = {}
    for n in notes:
        groups.setdefault(n.onset, []).append(n.pitch)
    return max((max(ps) - min(ps) for ps in groups.values()), default=0)


def _limit_span(notes: list[ScoreNote], max_span: int = 12) -> list[ScoreNote]:
    groups: dict[float, list[ScoreNote]] = {}
    for n in notes:
        groups.setdefault(n.onset, []).append(n)
    result: list[ScoreNote] = []
    for onset, group in groups.items():
        group.sort(key=lambda x: x.pitch)
        while group and (group[-1].pitch - group[0].pitch) > max_span:
            group.pop(0) if len(group) > 1 else None
        result.extend(group)
    return result


def extract_melody(notes: list[ScoreNote]) -> list[ScoreNote]:
    groups: dict[float, list[ScoreNote]] = {}
    for n in notes:
        groups.setdefault(n.onset, []).append(n)
    melody: list[ScoreNote] = []
    for onset in sorted(groups.keys()):
        top = max(groups[onset], key=lambda n: n.pitch)
        melody.append(top)
    return melody


def extract_bass_line(notes: list[ScoreNote]) -> list[ScoreNote]:
    groups: dict[float, list[ScoreNote]] = {}
    for n in notes:
        groups.setdefault(n.onset, []).append(n)
    bass: list[ScoreNote] = []
    for onset in sorted(groups.keys()):
        bottom = min(groups[onset], key=lambda n: n.pitch)
        bass.append(bottom)
    return bass


def condense_parts(notes: list[ScoreNote], max_voices: int = 4) -> list[ScoreNote]:
    groups: dict[float, list[ScoreNote]] = {}
    for n in notes:
        groups.setdefault(n.onset, []).append(n)
    result: list[ScoreNote] = []
    for onset in sorted(groups.keys()):
        group = groups[onset]
        if len(group) <= max_voices:
            result.extend(group)
        else:
            group.sort(key=lambda n: n.pitch)
            keep = [group[0], group[-1]]
            inner = group[1:-1]
            step = max(1, len(inner) // (max_voices - 2)) if max_voices > 2 else 1
            keep.extend(inner[::step][:max_voices - 2])
            result.extend(keep)
    return result


def lead_sheet(notes: list[ScoreNote]) -> tuple[list[ScoreNote], list[ScoreNote]]:
    mel = extract_melody(notes)
    bass = extract_bass_line(notes)
    return mel, bass


def reduction_quality(original: list[ScoreNote], reduced: list[ScoreNote]) -> float:
    if not original:
        return 1.0 if not reduced else 0.0
    orig_pitches = {(n.onset, n.pitch) for n in original}
    red_pitches = {(n.onset, n.pitch) for n in reduced}
    preserved = len(orig_pitches & red_pitches)
    return preserved / len(orig_pitches) if orig_pitches else 0.0


def voice_count_over_time(notes: list[ScoreNote], resolution: float = 0.5) -> list[int]:
    if not notes:
        return []
    max_t = max(n.onset + n.duration for n in notes)
    steps = int(math.ceil(max_t / resolution))
    counts = [0] * steps
    for n in notes:
        start = int(n.onset / resolution)
        end = int((n.onset + n.duration) / resolution)
        for i in range(start, min(end, steps)):
            counts[i] += 1
    return counts
