"""Sight-reading: exercise generation and difficulty estimation."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SightReadingExercise:
    notes: list[int]
    durations: list[float]
    key_signature: int
    time_signature: tuple[int, int]
    difficulty: float


class SightReadingGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(
        self,
        n_bars: int = 4,
        difficulty: int = 1,
        key_sig: int = 0,
        time_sig: tuple[int, int] = (4, 4),
    ) -> SightReadingExercise:
        beats_per_bar = time_sig[0]
        total_beats = n_bars * beats_per_bar
        range_map = {1: (60, 72), 2: (55, 77), 3: (48, 84), 4: (36, 96)}
        lo, hi = range_map.get(min(difficulty, 4), (60, 72))

        notes: list[int] = []
        durations: list[float] = []
        beat_pool = [1.0, 0.5, 2.0] if difficulty >= 2 else [1.0]
        if difficulty >= 3:
            beat_pool.extend([0.25, 1.5])

        remaining = float(total_beats)
        prev_note = self._rng.randint(lo, hi)
        while remaining > 0.01:
            dur = self._rng.choice(beat_pool)
            dur = min(dur, remaining)
            step = self._rng.randint(-4, 4) if difficulty >= 2 else self._rng.randint(-2, 2)
            note = max(lo, min(hi, prev_note + step))
            notes.append(note)
            durations.append(dur)
            remaining -= dur
            prev_note = note

        est = DifficultyEstimator()
        diff_score = est.estimate(notes, durations, key_sig, time_sig)
        return SightReadingExercise(
            notes=notes, durations=durations,
            key_signature=key_sig, time_signature=time_sig,
            difficulty=diff_score,
        )


class DifficultyEstimator:
    def estimate(
        self,
        notes: list[int],
        durations: list[float],
        key_sig: int = 0,
        time_sig: tuple[int, int] = (4, 4),
    ) -> float:
        if not notes:
            return 0.0

        pitch_range = max(notes) - min(notes)
        range_score = min(pitch_range / 36.0, 1.0)

        intervals = [abs(notes[i + 1] - notes[i]) for i in range(len(notes) - 1)]
        avg_interval = sum(intervals) / max(len(intervals), 1)
        interval_score = min(avg_interval / 12.0, 1.0)

        unique_durs = len(set(durations))
        rhythm_score = min(unique_durs / 5.0, 1.0)

        accidental_score = min(abs(key_sig) / 7.0, 1.0)

        return 0.3 * range_score + 0.3 * interval_score + 0.25 * rhythm_score + 0.15 * accidental_score
