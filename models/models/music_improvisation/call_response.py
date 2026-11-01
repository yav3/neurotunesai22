"""Call-and-response pattern analysis for improvisation."""

from __future__ import annotations

import math


def call_response_score(
    call_pitches: list[int],
    response_pitches: list[int],
    call_durations: list[float],
    response_durations: list[float],
) -> float:
    if not call_pitches or not response_pitches:
        return 0.0

    len_ratio = min(len(call_pitches), len(response_pitches)) / max(len(call_pitches), len(response_pitches))

    call_range = max(call_pitches) - min(call_pitches) if call_pitches else 0
    resp_range = max(response_pitches) - min(response_pitches) if response_pitches else 0
    range_sim = 1.0 - abs(call_range - resp_range) / max(call_range, resp_range, 1)

    call_dur = sum(call_durations)
    resp_dur = sum(response_durations)
    dur_ratio = min(call_dur, resp_dur) / max(call_dur, resp_dur) if max(call_dur, resp_dur) > 0 else 0

    return 0.3 * len_ratio + 0.3 * range_sim + 0.4 * dur_ratio


def rhythmic_complement(pattern: list[float], total_beats: float) -> list[float]:
    used = sum(pattern)
    remainder = total_beats - used
    if remainder <= 0:
        return []
    n_notes = max(1, len(pattern))
    note_dur = remainder / n_notes
    return [note_dur] * n_notes


def melodic_answer(
    call_pitches: list[int],
    target_root: int,
    scale_pcs: list[int],
) -> list[int]:
    if not call_pitches:
        return []
    intervals = [call_pitches[i + 1] - call_pitches[i] for i in range(len(call_pitches) - 1)]
    answer = [target_root]
    for iv in intervals:
        next_pitch = answer[-1] - iv
        pc = next_pitch % 12
        if pc not in scale_pcs and scale_pcs:
            dists = [(abs(pc - s) % 12, s) for s in scale_pcs]
            dists.sort()
            oct = (next_pitch // 12) * 12
            next_pitch = oct + dists[0][1]
        answer.append(next_pitch)
    return answer


def phrase_balance(
    phrase_lengths: list[int],
) -> float:
    if len(phrase_lengths) < 2:
        return 1.0
    mean_len = sum(phrase_lengths) / len(phrase_lengths)
    if mean_len == 0:
        return 0.0
    variance = sum((l - mean_len) ** 2 for l in phrase_lengths) / len(phrase_lengths)
    cv = math.sqrt(variance) / mean_len
    return max(0.0, 1.0 - cv)
