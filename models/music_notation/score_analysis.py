"""Score analysis: statistics, voice leading, melodic analysis, harmonic rhythm."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .musicxml import MXLScore, MXLNote, MXLMeasure, STEP_TO_SEMITONE


def _midi(note: MXLNote) -> int:
    return note.midi_number()


def _interval(a: MXLNote, b: MXLNote) -> int:
    return _midi(b) - _midi(a)


@dataclass
class ScoreStatistics:
    note_count: int = 0
    rest_count: int = 0
    pitch_min: int = 0
    pitch_max: int = 0
    duration_distribution: dict[str, int] = field(default_factory=dict)
    rest_fraction: float = 0.0
    density_per_measure: list[float] = field(default_factory=list)

    @classmethod
    def from_score(cls, score: MXLScore) -> ScoreStatistics:
        stats = cls()
        all_notes: list[MXLNote] = []
        for part in score.parts:
            for measure in part.measures:
                all_notes.extend(measure.notes)

        pitches = [n.midi_number() for n in all_notes if not n.is_rest]
        rests = [n for n in all_notes if n.is_rest]
        non_rests = [n for n in all_notes if not n.is_rest]

        stats.note_count = len(non_rests)
        stats.rest_count = len(rests)
        total = len(all_notes)
        stats.rest_fraction = len(rests) / total if total > 0 else 0.0

        if pitches:
            stats.pitch_min = min(pitches)
            stats.pitch_max = max(pitches)

        dur_counter: Counter[str] = Counter()
        for n in all_notes:
            dur_counter[n.note_type] += 1
        stats.duration_distribution = dict(dur_counter)

        for part in score.parts:
            for measure in part.measures:
                notes_in = [n for n in measure.notes if not n.is_rest]
                stats.density_per_measure.append(float(len(notes_in)))

        return stats

    def pitch_range(self) -> int:
        return self.pitch_max - self.pitch_min


class VoiceLeading:
    @staticmethod
    def detect_parallel_fifths(
        voice1: list[MXLNote], voice2: list[MXLNote]
    ) -> list[tuple[int, int]]:
        parallels = []
        pairs = list(zip(voice1, voice2))
        for i in range(len(pairs) - 1):
            n1a, n2a = pairs[i]
            n1b, n2b = pairs[i + 1]
            if any(n.is_rest for n in (n1a, n2a, n1b, n2b)):
                continue
            iv_a = abs(_interval(n1a, n2a)) % 12
            iv_b = abs(_interval(n1b, n2b)) % 12
            if iv_a == 7 and iv_b == 7:
                if _midi(n1a) != _midi(n1b):
                    parallels.append((i, i + 1))
        return parallels

    @staticmethod
    def detect_parallel_octaves(
        voice1: list[MXLNote], voice2: list[MXLNote]
    ) -> list[tuple[int, int]]:
        parallels = []
        pairs = list(zip(voice1, voice2))
        for i in range(len(pairs) - 1):
            n1a, n2a = pairs[i]
            n1b, n2b = pairs[i + 1]
            if any(n.is_rest for n in (n1a, n2a, n1b, n2b)):
                continue
            iv_a = abs(_interval(n1a, n2a)) % 12
            iv_b = abs(_interval(n1b, n2b)) % 12
            if iv_a == 0 and iv_b == 0:
                if _midi(n1a) != _midi(n1b) and _midi(n1a) != _midi(n2a):
                    parallels.append((i, i + 1))
        return parallels

    @staticmethod
    def detect_voice_crossing(
        upper: list[MXLNote], lower: list[MXLNote]
    ) -> list[int]:
        crossings = []
        for i, (u, lo) in enumerate(zip(upper, lower)):
            if u.is_rest or lo.is_rest:
                continue
            if _midi(u) < _midi(lo):
                crossings.append(i)
        return crossings

    @staticmethod
    def spacing_analysis(
        voices: list[list[MXLNote]],
    ) -> list[list[int]]:
        if len(voices) < 2:
            return []
        length = min(len(v) for v in voices)
        result = []
        for i in range(length):
            intervals = []
            for v in range(len(voices) - 1):
                a, b = voices[v][i], voices[v + 1][i]
                if a.is_rest or b.is_rest:
                    intervals.append(0)
                else:
                    intervals.append(abs(_interval(a, b)))
            result.append(intervals)
        return result


class MelodicAnalysis:
    @staticmethod
    def interval_histogram(notes: list[MXLNote]) -> dict[int, int]:
        hist: Counter[int] = Counter()
        pitched = [n for n in notes if not n.is_rest]
        for i in range(len(pitched) - 1):
            iv = _interval(pitched[i], pitched[i + 1])
            hist[iv] += 1
        return dict(hist)

    @staticmethod
    def contour(notes: list[MXLNote]) -> str:
        pitched = [n for n in notes if not n.is_rest]
        if len(pitched) < 3:
            return "static"
        midis = [_midi(n) for n in pitched]
        n = len(midis)
        peak_pos = np.argmax(midis) / max(n - 1, 1)
        valley_pos = np.argmin(midis) / max(n - 1, 1)
        diffs = np.diff(midis)
        ascending_ratio = np.sum(diffs > 0) / max(len(diffs), 1)
        descending_ratio = np.sum(diffs < 0) / max(len(diffs), 1)

        if ascending_ratio > 0.7:
            return "ascending"
        if descending_ratio > 0.7:
            return "descending"
        if peak_pos > 0.2 and peak_pos < 0.7:
            return "arch"
        if valley_pos > 0.2 and valley_pos < 0.7:
            return "valley"
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
        if sign_changes >= len(diffs) * 0.4:
            return "wave"
        return "arch"

    @staticmethod
    def motivic_patterns(
        notes: list[MXLNote], n_gram: int = 3
    ) -> list[tuple[tuple[int, ...], int]]:
        pitched = [n for n in notes if not n.is_rest]
        if len(pitched) < n_gram + 1:
            return []
        intervals = []
        for i in range(len(pitched) - 1):
            intervals.append(_interval(pitched[i], pitched[i + 1]))

        ngrams: Counter[tuple[int, ...]] = Counter()
        for i in range(len(intervals) - n_gram + 1):
            gram = tuple(intervals[i : i + n_gram])
            ngrams[gram] += 1

        repeated = [(g, c) for g, c in ngrams.items() if c > 1]
        repeated.sort(key=lambda x: -x[1])
        return repeated

    @staticmethod
    def leap_step_ratio(notes: list[MXLNote], leap_threshold: int = 3) -> float:
        pitched = [n for n in notes if not n.is_rest]
        if len(pitched) < 2:
            return 0.0
        leaps = 0
        steps = 0
        for i in range(len(pitched) - 1):
            iv = abs(_interval(pitched[i], pitched[i + 1]))
            if iv >= leap_threshold:
                leaps += 1
            elif iv > 0:
                steps += 1
        total = leaps + steps
        if total == 0:
            return 0.0
        return leaps / total


class HarmonicRhythm:
    @staticmethod
    def chord_change_rate(measures: list[MXLMeasure]) -> list[float]:
        rates = []
        for measure in measures:
            pitched = [n for n in measure.notes if not n.is_rest and not n.is_chord]
            if len(pitched) < 2:
                rates.append(0.0)
                continue
            changes = 0
            prev_pc = _midi(pitched[0]) % 12
            for n in pitched[1:]:
                pc = _midi(n) % 12
                if pc != prev_pc:
                    changes += 1
                prev_pc = pc
            rates.append(float(changes))
        return rates

    @staticmethod
    def harmonic_density(measures: list[MXLMeasure]) -> float:
        if not measures:
            return 0.0
        total_changes = 0.0
        for measure in measures:
            pitched = [n for n in measure.notes if not n.is_rest]
            if len(pitched) < 2:
                continue
            pcs = set()
            for n in pitched:
                pcs.add(_midi(n) % 12)
            total_changes += len(pcs)
        return total_changes / len(measures)

    @staticmethod
    def average_chord_duration(measures: list[MXLMeasure]) -> float:
        all_durations = []
        for measure in measures:
            pitched = [n for n in measure.notes if not n.is_rest and not n.is_chord]
            if not pitched:
                continue
            current_dur = pitched[0].duration
            current_pc = _midi(pitched[0]) % 12
            for n in pitched[1:]:
                pc = _midi(n) % 12
                if pc == current_pc:
                    current_dur += n.duration
                else:
                    all_durations.append(current_dur)
                    current_dur = n.duration
                    current_pc = pc
            all_durations.append(current_dur)
        if not all_durations:
            return 0.0
        return float(np.mean(all_durations))
