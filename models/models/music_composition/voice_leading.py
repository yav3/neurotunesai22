"""Voice leading: minimal-cost transitions between chords, neo-Riemannian ops."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VoiceLeading:
    source: list[int]
    target: list[int]
    motion: list[int]
    cost: int


class VoiceLeader:
    def minimal_cost(self, chord_a: list[int], chord_b: list[int]) -> VoiceLeading:
        if len(chord_a) != len(chord_b):
            return self._unequal_voices(chord_a, chord_b)
        a_sorted = sorted(chord_a)
        b_sorted = sorted(chord_b)
        motion = [b - a for a, b in zip(a_sorted, b_sorted)]
        cost = sum(abs(m) for m in motion)
        return VoiceLeading(a_sorted, b_sorted, motion, cost)

    def _unequal_voices(self, chord_a: list[int], chord_b: list[int]) -> VoiceLeading:
        if len(chord_a) > len(chord_b):
            chord_b = chord_b + [chord_b[-1]] * (len(chord_a) - len(chord_b))
        else:
            chord_a = chord_a + [chord_a[-1]] * (len(chord_b) - len(chord_a))
        return self.minimal_cost(chord_a, chord_b)

    def smoothness(self, chord_a: list[int], chord_b: list[int]) -> float:
        vl = self.minimal_cost(chord_a, chord_b)
        if not vl.motion:
            return 1.0
        max_cost = 12 * len(vl.motion)
        return 1.0 - min(vl.cost / max_cost, 1.0)

    def common_tones(self, chord_a: list[int], chord_b: list[int]) -> list[int]:
        pc_a = {p % 12 for p in chord_a}
        pc_b = {p % 12 for p in chord_b}
        return sorted(pc_a & pc_b)

    def parallel_motion(self, motions: list[int]) -> bool:
        if len(motions) < 2:
            return False
        return all(m == motions[0] for m in motions) and motions[0] != 0

    def contrary_motion(self, motions: list[int]) -> bool:
        if len(motions) < 2:
            return False
        pos = any(m > 0 for m in motions)
        neg = any(m < 0 for m in motions)
        return pos and neg


class NeoRiemannian:
    def leading(self, triad: list[int]) -> list[int]:
        pcs = [p % 12 for p in triad]
        intervals = sorted(pcs)
        if len(intervals) < 3:
            return triad
        i01 = (intervals[1] - intervals[0]) % 12
        if i01 == 4:
            return [intervals[0], intervals[1], (intervals[2] - 1) % 12]
        elif i01 == 3:
            return [(intervals[0] + 1) % 12, intervals[1], intervals[2]]
        return triad

    def parallel(self, triad: list[int]) -> list[int]:
        pcs = [p % 12 for p in triad]
        intervals = sorted(pcs)
        if len(intervals) < 3:
            return triad
        i01 = (intervals[1] - intervals[0]) % 12
        if i01 == 4:
            return [intervals[0], (intervals[1] - 1) % 12, intervals[2]]
        elif i01 == 3:
            return [intervals[0], (intervals[1] + 1) % 12, intervals[2]]
        return triad

    def relative(self, triad: list[int]) -> list[int]:
        pcs = [p % 12 for p in triad]
        intervals = sorted(pcs)
        if len(intervals) < 3:
            return triad
        i01 = (intervals[1] - intervals[0]) % 12
        if i01 == 4:
            return [(intervals[0] - 1) % 12, intervals[1], intervals[2]]
        elif i01 == 3:
            return [intervals[0], intervals[1], (intervals[2] + 1) % 12]
        return triad


def minimal_voice_leading(chord_a: list[int], chord_b: list[int]) -> VoiceLeading:
    return VoiceLeader().minimal_cost(chord_a, chord_b)


def voice_leading_smoothness(chord_a: list[int], chord_b: list[int]) -> float:
    return VoiceLeader().smoothness(chord_a, chord_b)
