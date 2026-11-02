"""Jazz chord voicing construction and voice leading."""

from __future__ import annotations

from enum import Enum


class VoicingType(Enum):
    CLOSE = "close"
    DROP2 = "drop2"
    DROP3 = "drop3"
    ROOTLESS_A = "rootless_a"
    ROOTLESS_B = "rootless_b"
    SHELL = "shell"
    QUARTAL = "quartal"
    SPREAD = "spread"


_CHORD_INTERVALS: dict[str, list[int]] = {
    "maj7":   [0, 4, 7, 11],
    "dom7":   [0, 4, 7, 10],
    "min7":   [0, 3, 7, 10],
    "min7b5": [0, 3, 6, 10],
    "dim7":   [0, 3, 6, 9],
    "aug7":   [0, 4, 8, 10],
    "min_maj7": [0, 3, 7, 11],
    "dom7b9": [0, 4, 7, 10, 13],
    "dom7s9": [0, 4, 7, 10, 15],
    "dom7s11": [0, 4, 7, 10, 18],
    "dom13":  [0, 4, 7, 10, 14, 21],
}


def voice_chord(root: int, quality: str, voicing: VoicingType = VoicingType.CLOSE) -> list[int]:
    intervals = _CHORD_INTERVALS.get(quality, _CHORD_INTERVALS["dom7"])
    close = [root + iv for iv in intervals]
    if voicing == VoicingType.CLOSE:
        return close
    if voicing == VoicingType.DROP2:
        return drop2_voicing(close)
    if voicing == VoicingType.ROOTLESS_A:
        return rootless_voicing(close, variant="A")
    if voicing == VoicingType.ROOTLESS_B:
        return rootless_voicing(close, variant="B")
    if voicing == VoicingType.SHELL:
        return shell_voicing(root, quality)
    if voicing == VoicingType.QUARTAL:
        return quartal_voicing(root)
    return close


def rootless_voicing(close_voicing: list[int], variant: str = "A") -> list[int]:
    if len(close_voicing) < 4:
        return close_voicing
    if variant == "A":
        return close_voicing[1:]
    return [close_voicing[1], close_voicing[3]] + close_voicing[4:] + [close_voicing[2] + 12]


def drop2_voicing(close_voicing: list[int]) -> list[int]:
    if len(close_voicing) < 4:
        return close_voicing
    result = list(close_voicing)
    second_from_top = result[-2]
    del result[-2]
    result.insert(0, second_from_top - 12)
    return result


def shell_voicing(root: int, quality: str) -> list[int]:
    intervals = _CHORD_INTERVALS.get(quality, _CHORD_INTERVALS["dom7"])
    third = intervals[1] if len(intervals) > 1 else 4
    seventh = intervals[3] if len(intervals) > 3 else 10
    return [root, root + third, root + seventh]


def quartal_voicing(root: int, n_notes: int = 4) -> list[int]:
    result = [root]
    for _ in range(n_notes - 1):
        result.append(result[-1] + 5)
    return result


def voice_leading_cost(voicing_a: list[int], voicing_b: list[int]) -> float:
    if not voicing_a or not voicing_b:
        return 0.0
    min_len = min(len(voicing_a), len(voicing_b))
    total = sum(abs(voicing_a[i] - voicing_b[i]) for i in range(min_len))
    total += abs(len(voicing_a) - len(voicing_b)) * 7
    return total / max(min_len, 1)
