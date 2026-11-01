"""Brass section voicing: close/drop-2/spread techniques for big band and orchestral writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

__all__ = [
    "BrassInstrument", "BRASS_RANGES", "VoicingTechnique", "BrassVoicing",
    "close_voicing", "drop2", "drop24", "spread_voicing",
    "assign_instruments", "range_check", "voicing_tension",
    "parallel_motion",
]


class BrassInstrument(Enum):
    TRUMPET = "trumpet"
    FLUGELHORN = "flugelhorn"
    FRENCH_HORN = "french_horn"
    TROMBONE = "trombone"
    BASS_TROMBONE = "bass_trombone"
    TUBA = "tuba"


BRASS_RANGES: Dict[BrassInstrument, Tuple[int, int]] = {
    BrassInstrument.TRUMPET: (54, 82),
    BrassInstrument.FLUGELHORN: (52, 80),
    BrassInstrument.FRENCH_HORN: (41, 77),
    BrassInstrument.TROMBONE: (40, 72),
    BrassInstrument.BASS_TROMBONE: (34, 67),
    BrassInstrument.TUBA: (28, 58),
}


class VoicingTechnique(Enum):
    CLOSE = "close"
    DROP2 = "drop2"
    DROP24 = "drop24"
    SPREAD = "spread"
    CLUSTER = "cluster"
    UNISON_OCTAVE = "unison_octave"


@dataclass
class BrassVoicing:
    pitches: List[int] = field(default_factory=list)
    instruments: List[BrassInstrument] = field(default_factory=list)
    technique: VoicingTechnique = VoicingTechnique.CLOSE


def close_voicing(chord_tones: List[int], lead_pitch: int) -> List[int]:
    if not chord_tones:
        return []
    pcs = sorted(set(p % 12 for p in chord_tones))
    lead_pc = lead_pitch % 12
    result = [lead_pitch]

    current = lead_pitch
    for _ in range(len(pcs) - 1):
        current_pc = current % 12
        candidates = [pc for pc in pcs if pc != current_pc]
        if not candidates:
            break
        below = [(current_pc - pc) % 12 for pc in candidates]
        best_idx = below.index(min(below))
        next_pitch = current - min(below)
        result.append(next_pitch)
        current = next_pitch
        pcs = [pc for pc in pcs if pc != (next_pitch % 12)]

    return result


def drop2(close: List[int]) -> List[int]:
    if len(close) < 2:
        return list(close)
    result = list(close)
    result[1] -= 12
    return result


def drop24(close: List[int]) -> List[int]:
    if len(close) < 4:
        return drop2(close)
    result = list(close)
    result[1] -= 12
    result[3] -= 12
    return result


def spread_voicing(chord_tones: List[int], bass_pitch: int) -> List[int]:
    if not chord_tones:
        return [bass_pitch]
    pcs = sorted(set(p % 12 for p in chord_tones))
    bass_pc = bass_pitch % 12

    result = [bass_pitch]
    remaining = [pc for pc in pcs if pc != bass_pc]

    octave = bass_pitch + 12
    for pc in remaining:
        while pc + (octave // 12) * 12 < octave:
            octave += 12
        target = pc + (octave // 12) * 12
        while target < result[-1] + 3:
            target += 12
        result.append(target)

    return result


def assign_instruments(
    pitches: List[int],
    section: Optional[List[BrassInstrument]] = None,
) -> BrassVoicing:
    if section is None:
        section = [
            BrassInstrument.TRUMPET,
            BrassInstrument.TRUMPET,
            BrassInstrument.FRENCH_HORN,
            BrassInstrument.TROMBONE,
            BrassInstrument.BASS_TROMBONE,
        ]

    sorted_pitches = sorted(pitches, reverse=True)
    sorted_section = sorted(
        section, key=lambda inst: BRASS_RANGES[inst][1], reverse=True
    )

    assignments = []
    used = [False] * len(sorted_section)
    for p in sorted_pitches:
        best_idx = -1
        best_fit = float("inf")
        for i, inst in enumerate(sorted_section):
            if used[i]:
                continue
            low, high = BRASS_RANGES[inst]
            if low <= p <= high:
                mid = (low + high) / 2
                fit = abs(p - mid)
                if fit < best_fit:
                    best_fit = fit
                    best_idx = i
        if best_idx >= 0:
            assignments.append((p, sorted_section[best_idx]))
            used[best_idx] = True
        else:
            assignments.append((p, sorted_section[0]))

    return BrassVoicing(
        pitches=[a[0] for a in assignments],
        instruments=[a[1] for a in assignments],
        technique=VoicingTechnique.CLOSE,
    )


def range_check(voicing: BrassVoicing) -> List[Tuple[int, BrassInstrument, str]]:
    issues = []
    for pitch, inst in zip(voicing.pitches, voicing.instruments):
        low, high = BRASS_RANGES[inst]
        if pitch < low:
            issues.append((pitch, inst, "below_range"))
        elif pitch > high:
            issues.append((pitch, inst, "above_range"))
    return issues


_IC_DISSONANCE = [0.0, 1.0, 0.8, 0.3, 0.2, 0.1, 0.9]


def voicing_tension(pitches: List[int]) -> float:
    if len(pitches) < 2:
        return 0.0
    tension = 0.0
    count = 0
    for i in range(len(pitches)):
        for j in range(i + 1, len(pitches)):
            ic = abs(pitches[i] - pitches[j]) % 12
            if ic > 6:
                ic = 12 - ic
            tension += _IC_DISSONANCE[ic]
            count += 1
    return tension / count if count else 0.0


def parallel_motion(voicings: List[BrassVoicing]) -> float:
    if len(voicings) < 2:
        return 0.0
    total_pairs = 0
    parallel_count = 0
    for v_idx in range(1, len(voicings)):
        prev = voicings[v_idx - 1].pitches
        curr = voicings[v_idx].pitches
        n_voices = min(len(prev), len(curr))
        for i in range(n_voices):
            for j in range(i + 1, n_voices):
                motion_i = curr[i] - prev[i]
                motion_j = curr[j] - prev[j]
                total_pairs += 1
                if motion_i != 0 and motion_i == motion_j:
                    parallel_count += 1
    return parallel_count / total_pairs if total_pairs else 0.0
