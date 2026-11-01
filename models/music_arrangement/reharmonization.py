"""Reharmonization techniques for arrangement."""

from __future__ import annotations


def tritone_substitution(root: int) -> int:
    return (root + 6) % 12


def backdoor_progression(target_root: int) -> list[int]:
    bvii = (target_root - 2) % 12
    iv = (target_root + 5) % 12
    return [iv, bvii, target_root]


def modal_interchange(
    root: int,
    original_quality: str,
) -> list[tuple[int, str]]:
    options = []
    if original_quality == "maj7":
        options.append((root, "min7"))
        options.append(((root + 8) % 12, "maj7"))
        options.append(((root + 5) % 12, "min7"))
    elif original_quality == "min7":
        options.append((root, "maj7"))
        options.append(((root + 3) % 12, "maj7"))
    elif original_quality == "dom7":
        options.append(((root + 6) % 12, "dom7"))
        options.append((root, "min7"))
    return options


def chromatic_mediant(root: int, quality: str) -> list[tuple[int, str]]:
    results = []
    for shift in [3, 4, 8, 9]:
        new_root = (root + shift) % 12
        results.append((new_root, quality))
    return results


def reharmonize_melody_note(
    melody_note: int,
    original_root: int,
    original_quality: str,
) -> list[tuple[int, str]]:
    pc = melody_note % 12
    options = []
    for candidate_root in range(12):
        interval = (pc - candidate_root) % 12
        if interval in (0, 4, 7, 11):
            options.append((candidate_root, "maj7"))
        if interval in (0, 3, 7, 10):
            options.append((candidate_root, "min7"))
        if interval in (0, 4, 7, 10):
            options.append((candidate_root, "dom7"))
    options = [(r, q) for r, q in options if not (r == original_root and q == original_quality)]
    return options[:8]
