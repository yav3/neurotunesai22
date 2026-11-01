"""Verbal/textual description of musical content for screen readers."""
from __future__ import annotations


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_INTERVAL_NAMES = {
    0: "unison", 1: "minor second", 2: "major second", 3: "minor third",
    4: "major third", 5: "perfect fourth", 6: "tritone", 7: "perfect fifth",
    8: "minor sixth", 9: "major sixth", 10: "minor seventh",
    11: "major seventh", 12: "octave",
}

_DYNAMICS = [
    (0, "silent"), (16, "ppp"), (32, "pp"), (48, "p"),
    (64, "mp"), (80, "mf"), (96, "f"), (112, "ff"), (127, "fff"),
]

_CHORD_TEMPLATES = {
    (0, 4, 7): "major triad",
    (0, 3, 7): "minor triad",
    (0, 3, 6): "diminished triad",
    (0, 4, 8): "augmented triad",
    (0, 4, 7, 11): "major seventh",
    (0, 3, 7, 10): "minor seventh",
    (0, 4, 7, 10): "dominant seventh",
    (0, 3, 6, 9): "diminished seventh",
    (0, 3, 6, 10): "half-diminished seventh",
}

_DURATION_WORDS = {
    4.0: "whole note", 2.0: "half note", 1.5: "dotted quarter",
    1.0: "quarter note", 0.75: "dotted eighth", 0.5: "eighth note",
    0.25: "sixteenth note",
}


def describe_pitch(midi_pitch: int) -> str:
    pc = midi_pitch % 12
    octave = midi_pitch // 12 - 1
    name = _NOTE_NAMES[pc]
    if midi_pitch == 60:
        return "middle C"
    if octave == 4:
        return f"{name} above middle C" if pc > 0 else "middle C"
    return f"{name}{octave}"


def describe_interval(semitones: int) -> str:
    semitones = abs(semitones)
    if semitones <= 12:
        return _INTERVAL_NAMES.get(semitones, f"{semitones} semitones")
    octaves = semitones // 12
    remainder = semitones % 12
    base = _INTERVAL_NAMES.get(remainder, f"{remainder} semitones")
    if remainder == 0:
        return f"{octaves} octaves" if octaves > 1 else "octave"
    return f"{base} plus {octaves} octave{'s' if octaves > 1 else ''}"


def describe_rhythm(durations: list[float], bpm: float) -> str:
    if not durations:
        return "silence"
    parts = []
    for d in durations:
        closest = min(_DURATION_WORDS.keys(), key=lambda k: abs(k - d))
        if abs(closest - d) < 0.1:
            parts.append(_DURATION_WORDS[closest])
        else:
            ms = d * 60000.0 / max(bpm, 1.0)
            parts.append(f"{ms:.0f}ms")
    total_beats = sum(durations)
    return f"{len(parts)} notes over {total_beats:.1f} beats: {', '.join(parts)}"


def describe_dynamics(velocity: int) -> str:
    v = max(0, min(127, velocity))
    result = "silent"
    for threshold, label in _DYNAMICS:
        if v >= threshold:
            result = label
    return result


def describe_chord(pitches: list[int]) -> str:
    if not pitches:
        return "silence"
    if len(pitches) == 1:
        return describe_pitch(pitches[0])
    pcs = sorted(set(p % 12 for p in pitches))
    root = pcs[0]
    intervals = tuple((pc - root) % 12 for pc in pcs)
    root_name = _NOTE_NAMES[root]
    quality = _CHORD_TEMPLATES.get(intervals)
    if quality:
        return f"{root_name} {quality}"
    return f"{root_name} chord ({', '.join(_NOTE_NAMES[pc] for pc in pcs)})"


def describe_progression(chords: list[list[int]]) -> str:
    if not chords:
        return "no chords"
    descriptions = [describe_chord(c) for c in chords]
    return " → ".join(descriptions)


def generate_audio_guide(
    pitches: list[int],
    durations: list[float],
    velocities: list[int] | None = None,
) -> list[str]:
    if velocities is None:
        velocities = [64] * len(pitches)
    guide = []
    for i, (p, d, v) in enumerate(zip(pitches, durations, velocities)):
        closest_dur = min(_DURATION_WORDS.keys(), key=lambda k: abs(k - d))
        dur_word = _DURATION_WORDS.get(closest_dur, f"{d} beats")
        dyn = describe_dynamics(v)
        pitch_desc = describe_pitch(p)
        parts = [pitch_desc, dur_word, dyn]
        if i > 0:
            interval = abs(p - pitches[i - 1])
            direction = "up" if p > pitches[i - 1] else "down" if p < pitches[i - 1] else "repeated"
            parts.append(f"{direction} {describe_interval(interval)}")
        guide.append(", ".join(parts))
    return guide
